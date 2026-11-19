# langgraph-llm-compiler

Implementazione modulare dell'architettura **LLM Compiler** con orchestrazione tramite **LangGraph** e supporto a modelli locali via **Ollama** (testata con `qwen3:8b`).

Il progetto è progettato con tre obiettivi primari: **correttezza formale**, **riproducibilità** e **valutazione sperimentale comparativa** (latenza, parallelismo effettivo, speedup rispetto a baseline sequenziali, affidabilità e confronti tra framework/architetture).

---

## Architettura

La pipeline separa rigorosamente pianificazione, validazione/compilazione, esecuzione concorrente, sintesi e orchestrazione del workflow:

```mermaid
flowchart LR
    R["ToolRegistry"] -->|catalogo prompt| P["Planner (LLM)"]
    P -->|ExecutionPlan| C["PlanCompiler"]
    R -->|contratti I/O| C
    C -->|CompiledDAG| E["Executor (asyncio event-driven)"]
    R -->|handler + schemi| E
    E -->|ExecutionResult| J["Joiner (LLM + classificazione deterministica)"]
    C -. "v2: CompilationIssue" .-> P
    J -. "v3: replan + risultati parziali" .-> P
```

### Stato dei moduli

| Modulo | Stato | Responsabilità |
| :--- | :--- | :--- |
| `planner` | Implementato | Genera un `ExecutionPlan` tipizzato via structured output (`json_schema`), con `BaseChatModel` iniettato e `PlannerOutputError` con `raw_output`. |
| `compiler` | Implementato | `ToolRegistry` (unica *source of truth*), validazione semantica in singola passata (6 regole) e costruzione di un `CompiledDAG` read-only. |
| `executor` | Implementato | Scheduler `asyncio` event-driven con `ready_queue`, risoluzione `input_bindings`/`output_path`, validazione runtime I/O, timeout, gestione best-effort + `skipped` e telemetria per task. |
| `joiner` | Implementato (v1) | Sintesi della risposta finale nella lingua della richiesta utente originale (`JoinerOutput`), con classificazione deterministica di `status` (`complete`/`partial`/`failed`) e `missing_information` (`JoinerResult`). |
| `graph` | Implementato (v1) | `StateGraph` LangGraph lineare con uscite condizionali su errore (`planner`, `compiler`, `executor`), stato `CompilerState`, `WorkflowError` e `WorkflowMetrics` (`component_latency_ms`). |

---

## Scelte architetturali e motivazioni

### 1. Parametri unificati (`inputs`) e dipendenze implicite tramite `InputBinding`
In `planner/schemas.py`, `TaskSpec` utilizza un **singolo dizionario `inputs: dict[str, InputBinding | Any]`** (`union_mode="left_to_right"`) in cui ogni parametro contiene o un valore statico oppure un riferimento `InputBinding(task_id, output_path)` — vietando campi separati `arguments`, `input_bindings` e `dependencies` (`extra="forbid"`).
* Il DAG viene derivato direttamente dai binding (`task.bindings`), eliminando divergenze tra dipendenze dichiarate e dati consumati.
* Unificare valori statici e binding nello stesso dizionario rende **strutturalmente impossibile per costruzione** la collisione di chiavi tra parametri statici e dinamici (errore molto frequente sui modelli locali 8B quando si separano `arguments` e `input_bindings`).

### 2. Separazione tra validazione sintattica (Pydantic) e semantica (`PlanCompiler`)
Gli schemi Pydantic in `planner/schemas.py` verificano solo la struttura locale dei modelli (`frozen=True`, `extra="forbid"`, `NonEmptyStr`). Tutti gli invarianti globali vengono verificati da `PlanCompiler` in **un'unica passata**, raccogliendo tutte le `CompilationIssue` in una `PlanCompilationError`:
1. Unicità dei `task_id` (`duplicate_task_id`).
2. Esistenza di ogni `tool_name` nel `ToolRegistry` (`unknown_tool`).
3. Esistenza dei task referenziati ed esclusione di auto-dipendenze (`missing_dependency`, `self_dependency`).
4. Assenza di cicli con ricostruzione del percorso (`cycle_detected`) e calcolo di ordine topologico e livelli paralleli (algoritmo di Kahn).
5. Validazione statica di `input_schema` (parametri mancanti, sconosciuti, valori statici non validi) e di `output_path` su `output_schema`.

Questa separazione distingue nettamente errori di parsing JSON da errori semantici di compilazione e consente di inviare feedback strutturato al `Planner` nel futuro loop di self-correction.

### 3. `ToolRegistry` come unica *source of truth*
Ogni strumento è descritto una sola volta tramite `ToolSpec(name, description, func, input_schema, output_schema)`:
* `ToolRegistry.to_planner_catalog()` genera automaticamente il catalogo testuale per il prompt del `Planner`.
* `PlanCompiler` usa gli stessi schemi per validare parametri e `output_path` prima dell'esecuzione.
* `Executor` usa gli stessi schemi per la validazione runtime prima e dopo l'invocazione del tool.

### 4. Scheduler event-driven in `asyncio` (e perché il DAG non è un `StateGraph` dinamico)
L'`Executor` è completamente indipendente da LangGraph ed esegue il `CompiledDAG` con uno scheduler **event-driven / ready-queue** (`asyncio.create_task` + `asyncio.wait(..., return_when=FIRST_COMPLETED)` + `dag.is_ready()`):
* **Nessuna barriera wave-based / BSP**: il runtime Pregel di LangGraph avanza per superstep sincroni (modello *Bulk Synchronous Parallel*), costringendo tutti i nodi del superstep corrente a terminare prima di avviare il successivo.
* **Concorrenza controllata (`max_concurrency`)**: default `8`, `None` per concorrenza illimitata, `1` come **baseline sequenziale deterministico** sullo stesso `Executor` (a parità di overhead).
* **Tool sync e async**: `async def` invocati con `await`, `def` sincroni eseguiti via `asyncio.to_thread()` per non bloccare l'event loop.
* **Timeout configurabili (`task_timeout_s`)**: su tool `async def` annullano la coroutine via `asyncio.wait_for`; su tool sincroni in `asyncio.to_thread` interrompono l'attesa registrando un errore di timeout (in Python un thread OS già avviato non può essere terminato forzatamente, quindi è bene evitare tool sincroni con side-effect non idempotenti nei test di timeout).

#### Verifica sperimentale: Event-driven vs LangGraph `StateGraph` dinamico (Pregel BSP)
Per verificare quantitativamente questa scelta senza varianza LLM, `experiments/benchmark_scheduling.py` (risultati salvati in `results/benchmark_scheduling.json`) esegue **gli stessi identici `CompiledDAG`** con tre motori (mediana su 3 ripetizioni):

| Topologia DAG | Nodi / Livelli | Sequenziale (`max=1`) | LangGraph Pregel BSP *(+compile)* | Executor Event-driven (`max=8`) | Speedup BSP | Speedup Event-driven | Riduzione latenza vs BSP |
| :--- | :---: | ---: | ---: | ---: | :---: | :---: | :---: |
| `symmetric_fanout` *(controllo equità)* | 5 / 2 | 522.2 ms | 164.4 ms *(+2.8 ms)* | **160.9 ms** | 3.18x | **3.25x** | -2.1% *(parità)* |
| `asymmetric_branches` *(T1:50→T3:250 vs T2:300→T4:50 → T5:50)* | 5 / 3 | 702.5 ms | 606.1 ms *(+3.0 ms)* | **401.8 ms** | 1.16x | **1.75x** | **-33.7%** (-204 ms) |
| `staggered_multi_branch` *(profilo reale di `test2`)* | 8 / 3 | 903.7 ms | 466.4 ms *(+4.0 ms)* | **421.8 ms** | 1.94x | **2.14x** | **-9.6%** (-45 ms) |
| `deep_asymmetric_pipeline` *(4 livelli asimmetrici)* | 7 / 4 | 663.3 ms | 548.3 ms *(+3.8 ms)* | **383.0 ms** | 1.21x | **1.73x** | **-30.1%** (-165 ms) |

Il confronto diretto delle timeline su `asymmetric_branches` (stessa scala temporale `0 → 606 ms`) rende visibile la **barriera di superstep BSP**: in LangGraph Pregel `T3` resta fermo per ~252 ms dopo il completamento di `T1` in attesa della fine di `T2`, mentre nell'`Executor` event-driven `T3` parte immediatamente a `51 ms` e gira in parallelo a `T2`:

```text
LangGraph Pregel BSP (606 ms):
  T1 │██································│   1 →  52 ms
  T2 │████████████████··················│   1 → 302 ms
  T3 │·················██████████████···│ 304 → 555 ms  <-- fermo 252 ms alla barriera BSP
  T4 │·················██···············│ 304 → 354 ms
  T5 │·······························██·│ 555 → 606 ms

Executor Event-driven (402 ms):
  T1 │██································│   0 →  50 ms
  T2 │████████████████··················│   0 → 301 ms
  T3 │··██████████████··················│  51 → 301 ms  <-- parte subito al termine di T1
  T4 │················███···············│ 301 → 352 ms
  T5 │···················███············│ 352 → 402 ms
```

* Su DAG simmetrici (`symmetric_fanout`) i due motori si equivalgono (confermando l'equità del benchmark e un overhead Pregel contenuto di ~3.5 ms + ~2.8 ms di compilazione dinamica).
* Su DAG asimmetrici — tipici delle chiamate a strumenti/API con latenze diverse — lo scheduler event-driven abbatte la latenza di esecuzione fino al **-33.7%** rispetto a un `StateGraph` dinamico LangGraph.

### 5. Gestione errori best-effort, propagazione `SKIPPED` e telemetria
In caso di errore di un task, l'`Executor` **non interrompe** i rami indipendenti:
* Il task fallito registra `status="error"`, `error_type`, `error` e `error_stage` (`binding`, `input_validation`, `tool_execution`, `output_validation`, `timeout`), distinguendo errori di binding, validazione runtime, timeout ed eccezioni del tool.
* Tutti i discendenti raggiungibili vengono marcati immediatamente `status="skipped"` (propagazione transitiva) con `started_at_ms=None`, `finished_at_ms=None` e `skip_reason`.
* Tutti i risultati (`TaskExecutionResult`, `ExecutionResult`) sono modelli Pydantic `frozen=True` privi di oggetti `Exception` raw, quindi direttamente serializzabili per lo stato di LangGraph e i report sperimentali.

### 6. Separazione deterministica nel `Joiner` (`JoinerOutput` vs `JoinerResult`)
Nel `Joiner` l'LLM genera solo `JoinerOutput(answer)` (nella lingua della richiesta utente originale, strettamente ancorato agli output `success`). Lo stato di completezza (`complete` / `partial` / `failed`) e `missing_information` (in ordine topologico, usando `task.description` o `tool_name` come fallback) sono calcolati **deterministicamente** dal codice a partire da `ExecutionResult` e impacchettati in `JoinerResult`.

### 7. Ruolo di LangGraph
LangGraph viene riservato al livello di **orchestrazione macro stateful** (`graph/`):
* gestione dei cicli di controllo (`planner <-> compiler` in v2, `joiner -> planner` in v3) tramite conditional edges e stato tipizzato;
* streaming, checkpointing e ispezionabilità passo-passo;
* interfaccia uniforme (`ainvoke`/`astream`) per il confronto sperimentale *apples-to-apples* con altre architetture (es. ReAct).

### 8. Rigore metodologico sulle metriche
Per evitare ambiguità nei benchmark, teniamo distinte due metriche:
* **Effective parallelism**: $\sum_i T_i / T_{\text{parallel}}$ (parallelismo medio osservato durante una singola esecuzione).
* **Measured speedup**: $T_{\text{sequential}} / T_{\text{parallel}}$ (misurato rieseguendo lo stesso `CompiledDAG` con `max_concurrency=1` come baseline sequenziale).

---

## Struttura del repository

```text
langgraph-llm-compiler/
├── pyproject.toml
├── src/
│   └── langgraph_llm_compiler/
│       ├── llm.py                 # Factory ChatOllama
│       ├── planner/
│       │   ├── schemas.py         # InputBinding, TaskSpec, ExecutionPlan
│       │   └── planner.py         # Planner (plan / aplan), SYSTEM_PROMPT, PlannerOutputError
│       ├── compiler/
│       │   ├── registry.py        # ToolSpec, ToolRegistry
│       │   ├── dag.py             # CompilationIssue, PlanCompilationError, CompiledDAG
│       │   └── compiler.py        # PlanCompiler (validazione + Kahn scheduling)
│       ├── executor/
│       │   ├── results.py         # TaskExecutionResult, ExecutionResult
│       │   ├── bindings.py        # resolve_output_path, resolve_task_inputs
│       │   └── executor.py        # Executor (scheduler event-driven async)
│       ├── joiner/
│       │   ├── schemas.py         # JoinerOutput, JoinerResult, JoinerStatus
│       │   └── joiner.py          # Joiner (async join), classificazione deterministica, JoinerOutputError
│       └── graph/
│           ├── state.py           # CompilerState, WorkflowError, WorkflowMetrics, update_metrics
│           ├── nodes.py           # 4 nodi async sottili (planner, compiler, executor, joiner)
│           └── builder.py         # build_compiler_graph() -> CompiledStateGraph
├── tests/
│   ├── test_planner_schemas.py
│   ├── test_planner.py
│   ├── test_compiler.py
│   ├── test_executor.py
│   ├── test_joiner.py
│   └── test_graph.py
├── experiments/
│   ├── test_planner_qwen3.py      # Pipeline diretta Python (Registry -> Planner -> Compiler -> Executor -> Joiner)
│   ├── test_graph_qwen3.py        # Esperimento A (live LangGraph vs Direct) + Esperimento B (overhead isolato)
│   ├── test2.py                   # Trace visivo fase-per-fase su LangGraph + export PNG/Mermaid in results/
│   ├── benchmark_v1_baseline.py   # Suite baseline v1 (8 query, reasoning=False vs True -> results/*.json)
│   └── benchmark_scheduling.py    # Benchmark deterministico Event-driven vs LangGraph Pregel BSP vs Seq
└── results/                       # Output sperimentali, benchmark JSON e diagrammi PNG/Mermaid
```

---

## Setup e comandi

Requisiti: Python `>=3.12`, [`uv`](https://docs.astral.sh/uv/) e [Ollama](https://ollama.com/) (solo per gli esperimenti live con modelli locali).

```bash
# Installa le dipendenze (inclusi pytest, ruff e grandalf per draw_ascii)
uv sync

# Esegui la suite di test unitari (non richiede Ollama)
uv run pytest -v

# Controllo lint con ruff
uv run ruff check src tests experiments
```

### Esperimenti e benchmark

```bash
# Benchmark deterministico di scheduling (Event-driven vs LangGraph Pregel BSP, non richiede Ollama)
uv run python experiments/benchmark_scheduling.py --repeats 3

# Benchmark isolato dell'overhead di orchestrazione LangGraph (non richiede Ollama)
uv run python experiments/test_graph_qwen3.py --mode overhead --overhead-iters 50

# Trace visivo fase-per-fase con qwen3:8b + export PNG e report Mermaid in results/
uv run python experiments/test2.py --no-reasoning

# Suite baseline v1 (8 query, reasoning=False vs True, richiede Ollama + qwen3:8b)
uv run python experiments/benchmark_v1_baseline.py --reasoning-modes both
```

---

## Roadmap

- [x] **Planner**: schemi Pydantic, structured output, prompt 1-shot, test unitari.
- [x] **Compiler**: `ToolRegistry`, validazione semantica in singola passata, `CompiledDAG`, test unitari.
- [x] **Executor**: scheduler event-driven, binding resolution, validazione runtime I/O, best-effort + `skipped`, timeout, test unitari.
- [x] **Joiner v1**: sintesi risposta finale nella lingua dell'utente (`JoinerOutput`), stato e `missing_information` deterministici (`JoinerResult`), test unitari.
- [x] **LangGraph v1**: `StateGraph` lineare con uscite condizionali su errore (`planner`, `compiler`, `executor`), `WorkflowMetrics`, test unitari (`test_graph.py`) ed esperimenti comparativi (`test_graph_qwen3.py`).
- [ ] **v2 — Compile-time self-correction**: ciclo `compiler -> planner` con `CompilationIssue` e budget di retry.
- [ ] **v3 — Runtime replanning**: ciclo `joiner -> planner` con risultati parziali e `max_replans`.
- [ ] **Esperimenti & Benchmark**: suite riproducibile in `experiments/` e `results/` (parallelismo, latenza, affidabilità, confronto con ReAct e altri framework).
