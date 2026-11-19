# langgraph-llm-compiler

Implementazione modulare, verificata formalmente e benchmarkabile dell'architettura **LLM Compiler** con orchestrazione stateful tramite **LangGraph** e supporto a modelli locali via **Ollama** (testata end-to-end con `qwen3:8b`).

Il progetto è progettato con tre obiettivi primari: **correttezza formale** (contratti di input/output validati staticamente e a runtime), **riproducibilità** (suite di test unitari deterministiche e benchmark salvati in JSON) e **valutazione sperimentale comparativa** (latenza, parallelismo effettivo, speedup misurato rispetto a baseline sequenziali, affidabilità e confronti tra motori di scheduling e configurazioni LLM).

---

## Indice

1. [Cos'è un LLM Compiler e quale problema risolve](#1-cosè-un-llm-compiler-e-quale-problema-risolve)
2. [Architettura completa e passaggio di testimone tra moduli indipendenti](#2-architettura-completa-e-passaggio-di-testimone-tra-moduli-indipendenti)
   - [2.1 Modulo 0 — `ToolRegistry` (il catalogo e i contratti)](#21-modulo-0--toolregistry-il-catalogo-e-i-contratti)
   - [2.2 Modulo 1 — `Planner` (dal testo libero al piano strutturato)](#22-modulo-1--planner-dal-testo-libero-al-piano-strutturato)
   - [2.3 Modulo 2 — `PlanCompiler` (dal piano non fidato al `CompiledDAG` certificato)](#23-modulo-2--plancompiler-dal-piano-non-fidato-al-compileddag-certificato)
   - [2.4 Modulo 3 — `Executor` (dal `CompiledDAG` all'esecuzione parallela)](#24-modulo-3--executor-dal-compileddag-allesecuzione-parallela)
   - [2.5 Modulo 4 — `Joiner` (dai risultati tecnici alla risposta finale)](#25-modulo-4--joiner-dai-risultati-tecnici-alla-risposta-finale)
   - [2.6 Modulo 5 — `graph` / LangGraph (il coordinatore sopra ai moduli)](#26-modulo-5--graph--langgraph-il-coordinatore-sopra-ai-moduli)
3. [Scelte architetturali e funzionamento interno (in dettaglio)](#3-scelte-architetturali-e-funzionamento-interno-in-dettaglio)
   - [3.1 Schema unificato `inputs` e dipendenze implicite (Strada A)](#31-schema-unificato-inputs-e-dipendenze-implicite-strada-a)
   - [3.2 `ToolRegistry` come unica *source of truth*](#32-toolregistry-come-unica-source-of-truth)
   - [3.3 Validazione sintattica (Pydantic) vs semantica (`PlanCompiler`) e algoritmo di Kahn](#33-validazione-sintattica-pydantic-vs-semantica-plancompiler-e-algoritmo-di-kahn)
   - [3.4 Scheduler `asyncio` event-driven vs LangGraph `StateGraph` dinamico (Pregel BSP)](#34-scheduler-asyncio-event-driven-vs-langgraph-stategraph-dinamico-pregel-bsp)
   - [3.5 Esecuzione best-effort, propagazione `SKIPPED` e telemetria serializzabile](#35-esecuzione-best-effort-propagazione-skipped-e-telemetria-serializzabile)
   - [3.6 Separazione deterministica nel `Joiner` (`JoinerOutput` vs `JoinerResult`)](#36-separazione-deterministica-nel-joiner-joineroutput-vs-joinerresult)
   - [3.7 Orchestrazione macro con LangGraph: da v1 lineare a v2 compile-time self-correction](#37-orchestrazione-macro-con-langgraph-da-v1-lineare-a-v2-compile-time-self-correction)
   - [3.8 Rigore metodologico delle metriche](#38-rigore-metodologico-delle-metriche)
4. [Risultati sperimentali e analisi critica](#4-risultati-sperimentali-e-analisi-critica)
   - [4.1 Overhead puro di orchestrazione LangGraph](#41-overhead-puro-di-orchestrazione-langgraph)
   - [4.2 Baseline v1 su suite standard (4 tool, 8 query)](#42-baseline-v1-su-suite-standard-4-tool-8-query)
   - [4.3 Benchmark deterministico di scheduling (Event-driven vs Pregel BSP)](#43-benchmark-deterministico-di-scheduling-event-driven-vs-pregel-bsp)
   - [4.4 Hard Suite 10 tool: matrice 2x2 (v1 vs v2 e reasoning)](#44-hard-suite-10-tool-matrice-2x2-v1-vs-v2-e-reasoning)
   - [4.5 Cosa cambia con i retry (v2): vantaggi, svantaggi e guida alla scelta](#45-cosa-cambia-con-i-retry-v2-vantaggi-svantaggi-e-guida-alla-scelta)
5. [Struttura del repository](#5-struttura-del-repository)
6. [Setup, verifica e comandi per gli esperimenti](#6-setup-verifica-e-comandi-per-gli-esperimenti)
7. [Roadmap](#7-roadmap)

---

## 1. Cos'è un LLM Compiler e quale problema risolve

Negli agenti tradizionali basati su tool-calling iterativo (come **ReAct**), il modello linguistico (LLM) procede un passo alla volta in un ciclo sequenziale:

$$\text{LLM (Ragionamento)} \rightarrow \text{Tool 1} \rightarrow \text{LLM (Osservazione)} \rightarrow \text{Tool 2} \rightarrow \text{LLM} \rightarrow \dots \rightarrow \text{Risposta}$$

Questo approccio presenta tre colli di bottiglia strutturali quando una richiesta richiede più strumenti (API, database, calcoli concatenati):

1. **Latenza moltiplicata dalle chiamate LLM**: per eseguire $N$ tool dipendenti o indipendenti servono fino a $N+1$ round-trip verso l'LLM, che è quasi sempre il componente più lento della pipeline.
2. **Nessun parallelismo naturale sui flussi complessi**: anche se la richiesta chiede dati su 4 Paesi diversi e poi 4 calcoli derivati, un agente iterativo spesso li esegue in serie (o in piccoli batch rigidi) senza un vero grafo delle dipendenze dati.
3. **Nessuna verifica statica prima dell'esecuzione**: se un parametro o un nome di tool è sbagliato al passo 4, l'errore emerge solo *dopo* aver già pagato il costo e la latenza dei passi 1, 2 e 3.

L'architettura **LLM Compiler** (ispirata a *An LLM Compiler for Parallel Function Calling*, Kim et al., 2024) risolve questi limiti separando **pianificazione**, **verifica formale**, **esecuzione parallela** e **sintesi**:

* Il **Planner** (1 chiamata LLM) traduce la richiesta in un **ExecutionPlan** completo: un grafo aciclico diretto (**DAG**, *Directed Acyclic Graph*) in cui ogni nodo è una chiamata a un tool e ogni arco è un collegamento esplicito tra l'output di un task e l'input di un altro.
* Il **PlanCompiler** (100% codice deterministico, < 1 ms) valida staticamente l'intero piano *prima* di chiamare qualsiasi strumento: verifica che tutti i tool esistano, che gli argomenti rispettino i contratti Pydantic, che non ci siano cicli e che tutti i percorsi nei dati di output siano validi. Se trova errori, in **v2** li restituisce strutturati al Planner per una correzione mirata (*compile-time self-correction*).
* L'**Executor** (motore asincrono `asyncio` event-driven) lancia in parallelo tutti i task privi di dipendenze e avvia automaticamente ogni task successivo nel millisecondo esatto in cui tutti i suoi prerequisiti sono pronti.
* Il **Joiner** (1 chiamata LLM + logica deterministica) sintetizza una risposta chiara nella lingua dell'utente basandosi esclusivamente sugli output riusciti, affiancandola a uno stato formale (`complete` / `partial` / `failed`) calcolato dal codice.

Nel caso felice, anche un workflow con 10–15 chiamate a strumenti viene risolto con **sole 2 chiamate LLM totali** (Planner + Joiner) e la massima parallelizzazione possibile dei tool.

> **In parole semplici:**
> Immagina di dover cucinare una cena con più portate. Un agente tradizionale (tipo ReAct) è come un cuoco che legge una riga della ricetta, esce a comprare il primo ingrediente, torna a casa, rilegge la ricetta, riesce a comprare il secondo ingrediente, mette a bollire l'acqua, aspetta che la pasta sia cotta *prima* ancora di iniziare a tagliare le verdure per il sugo.
>
> Un **LLM Compiler**, invece, si comporta come uno chef organizzato con una brigata:
> 1. Legge tutta la richiesta e scrive subito il **piano completo della cena** (cosa serve e in che ordine).
> 2. Prima di accendere i fornelli, un **ispettore** controlla la lista: verifica che tutti gli attrezzi esistano in cucina e che non ci siano errori logici.
> 3. La **squadra ai fornelli** fa partire subito in parallelo tutto ciò che non dipende da altro (acqua sul fuoco e taglio verdure contemporaneamente), e passa subito al passo successivo appena un ingrediente è pronto.
> 4. Alla fine, il **capo sala** porta il piatto in tavola spiegando chiaramente il risultato (e dicendo onestamente se un ingrediente non era disponibile).

---

## 2. Architettura completa e passaggio di testimone tra moduli indipendenti

Una delle scelte progettuali più importanti di questo repository è che **ogni modulo è completamente separato e autosufficiente**: ognuno ha un solo compito, un contratto di input/output tipizzato, una propria suite di test unitari dedicata e **non conosce i dettagli interni degli altri moduli**.

Puoi usare `Planner` da solo per generare piani, `PlanCompiler` da solo per validare un piano scritto a mano, `Executor` da solo su un `CompiledDAG` senza nessun LLM e senza LangGraph, o collegarli tutti tramite il modulo `graph`.

```mermaid
flowchart LR
    R["ToolRegistry"] -->|catalogo prompt| P["Planner (LLM)"]
    P -->|ExecutionPlan| C["PlanCompiler"]
    R -->|contratti I/O| C
    C -->|CompiledDAG| E["Executor (asyncio event-driven)"]
    R -->|handler + schemi| E
    E -->|ExecutionResult| J["Joiner (LLM + stato deterministico)"]
    C -. "v2: CompilationIssue + previous_plan" .-> P
    J -. "v3 roadmap: replan + risultati parziali" .-> P
```

### Riepilogo dei moduli

| Modulo | File principali | Usa LLM? | Stato | Responsabilità unica |
| :--- | :--- | :---: | :---: | :--- |
| `ToolRegistry` (`compiler/registry.py`) | `registry.py` | No | Implementato | Custodisce i `ToolSpec` (funzioni + schemi Pydantic I/O) e genera il catalogo per il Planner. |
| `planner` | `schemas.py`, `planner.py` | **Sì** | Implementato (v2) | Trasforma la richiesta testuale (e l'eventuale feedback di retry) in un `ExecutionPlan` sintatticamente valido. |
| `compiler` | `dag.py`, `compiler.py` | No | Implementato | Verifica staticamente tutte le regole semantiche del piano e produce un `CompiledDAG` immutabile (o tutte le `CompilationIssue`). |
| `executor` | `results.py`, `bindings.py`, `executor.py` | No | Implementato | Esegue il `CompiledDAG` in parallelo con scheduler `asyncio` event-driven e produce `ExecutionResult`. |
| `joiner` | `schemas.py`, `joiner.py` | **Sì** + codice | Implementato (v1) | Sintetizza `answer` (LLM) e calcola deterministicamente `status` e `missing_information` (`JoinerResult`). |
| `graph` | `state.py`, `nodes.py`, `builder.py` | No (orchestrazione) | Implementato (v2) | Collega i moduli in uno `StateGraph` LangGraph, gestisce errori, metriche e il ciclo di retry v2 `compiler -> planner`. |

---

Per vedere **esattamente** cosa fa ogni modulo, cosa riceve, cosa produce, quali garanzie offre e **dove finisce la sua responsabilità**, seguiamo lo stesso identico esempio reale dall'inizio alla fine (scenario `partial_failure_recovery`):

> **Richiesta utente:** *"Fetch statistics for Italy and Atlantis. Compute Italy's population density (population divided by area_km2) and compute the difference between Italy's population and Atlantis's population (Italy minus Atlantis)."*

---

### 2.1 Modulo 0 — `ToolRegistry` (il catalogo e i contratti)

Prima ancora che arrivi la richiesta, il sistema ha bisogno di sapere quali strumenti esistono. Questo è l'unico compito di `ToolRegistry` (`compiler/registry.py`).

* **Input:** un elenco di `ToolSpec`, ciascuno composto da nome, descrizione, funzione Python (`def` o `async def`), `input_schema` (Pydantic) e `output_schema` (Pydantic):
  ```python
  registry = ToolRegistry([
      ToolSpec(
          name="country_stats",
          description="Fetch demographic and economic statistics for a country.",
          func=get_country_stats,
          input_schema=CountryStatsInput,    # country: str
          output_schema=CountryStatsOutput,  # population: int, area_km2: float, ...
      ),
      ToolSpec(
          name="calculator",
          description="Perform basic arithmetic operations.",
          func=run_calculator,
          input_schema=CalculatorInput,      # operation: Literal['add','subtract','multiply','divide'], a: float, b: float
          output_schema=CalculatorOutput,    # result: float
      ),
  ])
  ```
* **Cosa fa:** registra i tool vietando duplicati involontari, verifica che gli schemi siano davvero sottoclassi di `BaseModel` e che gli handler siano `callable`. Su richiesta, genera automaticamente la descrizione testuale per il prompt (`registry.to_planner_catalog()`).
* **Output (due viste degli stessi dati):**
  1. **Verso il `Planner` (`dict[str, str]`):**
     ```text
     country_stats: Fetch demographic and economic statistics for a country.
       Args: country (str).
       Output: {"population": int, "area_km2": float, "gdp_billion_usd": float, "unemployment_pct": float}.
     calculator: Perform basic arithmetic operations.
       Args: operation ('add' | 'subtract' | 'multiply' | 'divide'), a (float), b (float).
       Output: {"result": float}.
     ```
  2. **Verso `PlanCompiler` ed `Executor`:** lookup diretto dei `ToolSpec` con i modelli Pydantic originali e le funzioni eseguibili.
* **Garanzie offerte a valle:**
  * Nomi univoci e non vuoti, handler eseguibili, schemi I/O coerenti e condivisi da tutti i moduli senza duplicazioni.
* **Fine della responsabilità:** `ToolRegistry` **non chiama mai** l'LLM, **non valida** piani e **non esegue** alcuna funzione. Espone i contratti e passa il testimone al **`Planner`**.

> **In parole semplici:**
> Il `ToolRegistry` è la cassetta degli attrezzi con l'etichetta attaccata su ogni strumento: *"questo è il cacciavite, serve per questo, accetta questi pezzi e produce questo risultato"*. Non fa lavori e non decide nulla: si limita a mostrare il catalogo ufficiale a chi dovrà pianificare, controllare o eseguire.

---

### 2.2 Modulo 1 — `Planner` (dal testo libero al piano strutturato)

Ora entra in scena il `Planner` (`planner/planner.py`). È il **primo modulo che usa un LLM**.

* **Input:**
  * `request: str` (la frase in linguaggio naturale dell'utente);
  * `tool_catalog: dict[str, str]` (prodotto da `ToolRegistry.to_planner_catalog()`);
  * *(solo nei retry v2)* `previous_plan: ExecutionPlan` e `feedback: str` con gli errori segnalati dal compilatore.
* **Cosa fa:** costruisce il prompt di sistema (con le regole del formato e 3 esempi astratti) + la richiesta utente, e chiama l'LLM con `with_structured_output(ExecutionPlan, method="json_schema", include_raw=True)`.
* **Output:** un oggetto Pydantic gerarchico `ExecutionPlan` immutabile:
  ```json
  {
    "objective": "Compute Italy's population density and population difference between Italy and Atlantis",
    "tasks": [
      {
        "task_id": "task_1",
        "tool_name": "country_stats",
        "description": "Get stats for Italy",
        "inputs": {"country": "Italy"}
      },
      {
        "task_id": "task_2",
        "tool_name": "country_stats",
        "description": "Get stats for Atlantis",
        "inputs": {"country": "Atlantis"}
      },
      {
        "task_id": "task_3",
        "tool_name": "calculator",
        "description": "Compute Italy's population density",
        "inputs": {
          "operation": "divide",
          "a": {"task_id": "task_1", "output_path": "population"},
          "b": {"task_id": "task_1", "output_path": "area_km2"}
        }
      },
      {
        "task_id": "task_4",
        "tool_name": "calculator",
        "description": "Calculate population difference (Italy - Atlantis)",
        "inputs": {
          "operation": "subtract",
          "a": {"task_id": "task_1", "output_path": "population"},
          "b": {"task_id": "task_2", "output_path": "population"}
        }
      }
    ]
  }
  ```
* **Garanzie offerte a valle:**
  * **Solo garanzie sintattiche/strutturali** (fornite da Pydantic): il risultato è un `ExecutionPlan` ben formato, immutabile (`frozen=True`), con `objective` e `tasks` non vuoti, nessun campo extra sconosciuto, e i binding ben formati già convertiti in oggetti `InputBinding(task_id, output_path)`.
  * Se l'LLM produce JSON rotto o fuori schema, il modulo solleva `PlannerOutputError` allegando `raw_output`.
* **Fine della responsabilità:** il `Planner` **non garantisce** che i `tool_name` esistano davvero, che gli argomenti siano del tipo giusto, che gli `output_path` esistano o che non ci siano cicli tra task. Ha tradotto l'intenzione umana in una proposta di piano strutturata: la sua responsabilità finisce qui e passa il testimone al **`PlanCompiler`**.

> **In parole semplici:**
> Il `Planner` è l'architetto che ascolta cosa vuoi costruire e disegna la bozza del progetto su carta millimetrata: *"passo 1 fai questo, passo 2 fai quest'altro, passo 3 prendi il risultato del passo 1 e dividilo..."*. Garantisce che il disegno sia leggibile e segua il formato standard del foglio, ma **non** è suo compito verificare con il righello se tutte le misure tornano: quello tocca all'ispettore successivo.

---

### 2.3 Modulo 2 — `PlanCompiler` (dal piano non fidato al `CompiledDAG` certificato)

Il `PlanCompiler` (`compiler/compiler.py`) riceve la proposta del `Planner` come **input non fidato** e la sottopone a verifica formale. È **100% deterministico (zero LLM, < 1 ms)**.

* **Input:**
  * `plan: ExecutionPlan` (prodotto dal `Planner`);
  * `registry: ToolRegistry` (per confrontare ogni task con i contratti ufficiali).
* **Cosa fa:** esegue in **un'unica passata** tutte le verifiche semantiche (`duplicate_task_id`, `unknown_tool`, `malformed_binding`, `self_dependency`, `missing_dependency`, `cycle_detected`, validazione statica di `input_schema` e attraversamento di `output_path` su `output_schema`). Se trova anche una sola violazione, solleva `PlanCompilationError` con l'elenco completo di tutte le `CompilationIssue`. Altrimenti applica l'**algoritmo di Kahn** per calcolare ordine topologico e livelli paralleli.
* **Output:** un `CompiledDAG` read-only (`frozen=True` + `MappingProxyType`):
  ```text
  CompiledDAG (valido, immutabile, 0.27 ms):
    nodes             : task_1, task_2, task_3, task_4
    root_tasks        : ('task_1', 'task_2')
    leaf_tasks        : ('task_3', 'task_4')
    topological_order : ('task_1', 'task_2', 'task_3', 'task_4')
    levels (ASAP)     :
      L0 │ [task_1: country_stats]  [task_2: country_stats]
      L1 │ [task_3: calculator]     [task_4: calculator]
    dependencies      :
      task_3 <- {task_1}   (task_1.population -> a, task_1.area_km2 -> b)
      task_4 <- {task_1, task_2}  (task_1.population -> a, task_2.population -> b)
  ```
* **Garanzie offerte a valle:**
  * **Garanzie semantiche e topologiche complete**: ogni `task_id` è univoco; ogni `tool_name` esiste nel `ToolRegistry`; ogni `InputBinding` punta a un task esistente e a un `output_path` realmente presente nello schema di output upstream; il grafo è garantito **aciclico**; tutti gli argomenti statici rispettano tipi, obbligatorietà ed enum di `input_schema`.
* **Fine della responsabilità:** il `PlanCompiler` **non esegue alcun tool**, **non fa I/O** e **non sa** se un dato esisterà a runtime (ad esempio `"Atlantis"` è una `str` sintatticamente e staticamente valida, quindi il piano è corretto: sarà il tool a scoprire a runtime che Atlantide non è nel database). La sua responsabilità finisce qui e passa il testimone all'**`Executor`**.

> **In parole semplici:**
> Il `PlanCompiler` è l'ingegnere collaudatore: prende la bozza dell'architetto, controlla ogni singola misura con il catalogo degli attrezzi e verifica che non ci siano circoli viziosi (*"A aspetta B che aspetta A"*). Se c'è anche un solo errore, boccia il piano spiegando esattamente cosa sistemare; se è tutto in regola, mette il timbro "approvato e sigillato" (`CompiledDAG`) e lo passa alla squadra operativa.

---

### 2.4 Modulo 3 — `Executor` (dal `CompiledDAG` all'esecuzione parallela)

L'`Executor` (`executor/executor.py`) riceve un `CompiledDAG` già certificato e lo esegue realmente. È **100% deterministico/asincrono (zero LLM, zero dipendenza da LangGraph)**.

* **Input:**
  * `dag: CompiledDAG` (prodotto dal `PlanCompiler`);
  * `registry: ToolRegistry` (per invocare gli handler e validare I/O a runtime);
  * configurazione di esecuzione: `max_concurrency` (default `8`), `task_timeout_s` (opzionale).
* **Cosa fa:** avvia in parallelo i `root_tasks` tramite `asyncio` (`ready_queue` + `asyncio.wait(..., FIRST_COMPLETED)`). Per ogni task pronto: risolve gli `InputBinding` leggendo i campi reali dagli output upstream (`resolve_task_inputs`), valida gli input risolti con `input_schema`, invoca il tool (`await` o `asyncio.to_thread`), valida il risultato con `output_schema`. Se un task fallisce, isola il guasto e marca immediatamente `skipped` tutti i suoi discendenti, lasciando correre i rami indipendenti.
* **Output:** un `ExecutionResult` immutabile e serializzabile in JSON, con telemetria per ogni singolo task:
  ```text
  ExecutionResult (wall: 141.8 ms | sum_task_duration: 192.3 ms | effective_parallelism: 1.36x):
    task_1 (country_stats) -> status="success" |   0 → 101 ms
      output = {"population": 58990000, "area_km2": 301340.0, "gdp_billion_usd": 2300.0, ...}
    task_2 (country_stats) -> status="error"   |   0 →  50 ms
      error_stage="tool_execution", error_type="ValueError",
      error="Country 'Atlantis' not found in dataset."
    task_3 (calculator)    -> status="success" | 101 → 142 ms
      output = {"result": 195.758943...}
    task_4 (calculator)    -> status="skipped" | started_at_ms=None, finished_at_ms=None
      skip_reason="Skipped because upstream task 'task_2' failed."
  ```
* **Garanzie offerte a valle:**
  * Ogni task del DAG ha esattamente un `TaskExecutionResult` con stato conclusivo (`success` / `error` / `skipped`), nessuna eccezione non gestita propaga all'esterno e nessun oggetto `Exception` raw inquina i dati.
  * Nessun task `success` ha violato il proprio `input_schema` o `output_schema`; nessun task dipendente da un nodo fallito è stato eseguito con dati corrotti.
* **Fine della responsabilità:** l'`Executor` **non sa** quale fosse la richiesta originale dell'utente, **non scrive** testo in linguaggio naturale e **non decide** come spiegare i risultati all'umano. La sua responsabilità finisce qui e passa il testimone al **`Joiner`**.

> **In parole semplici:**
> L'`Executor` è la squadra operativa sul campo: prende il progetto approvato e mette subito al lavoro tutti gli operai che possono partire contemporaneamente (`task_1` sull'Italia e `task_2` su Atlantide). Quando l'operaio su Atlantide scopre che il Paese non esiste, la squadra non ferma il cantiere: finisce comunque il calcolo sull'Italia (`task_3`), depenna il calcolo impossibile su Atlantide (`task_4`) e compila il registro di cantiere con risultati, errori e cronometro.

---

### 2.5 Modulo 4 — `Joiner` (dai risultati tecnici alla risposta finale)

Il `Joiner` (`joiner/joiner.py`) trasforma i dati tecnici grezzi dell'`Executor` nella risposta finale per l'utente. Combina **codice deterministico** (per lo stato formale) e **un LLM** (solo per la redazione testuale).

* **Input:**
  * `request: str` (la richiesta originale dell'utente, necessaria per la lingua e il contesto);
  * `execution_result: ExecutionResult` (prodotto dall'`Executor`);
  * `plan: ExecutionPlan | None` (opzionale, per usare le descrizioni umane `task.description` al posto dei nomi grezzi dei tool).
* **Cosa fa:**
  1. **Parte deterministica:** calcola `status` (`"complete"` / `"partial"` / `"failed"`) e costruisce `missing_information` ordinato topologicamente.
  2. **Parte LLM (LLM #2):** invoca l'LLM con `with_structured_output(JoinerOutput)` chiedendogli *soltanto* `answer: str`, ancorato ai soli task `success`, nella lingua della richiesta e senza esporre ID interni come `task_3`.
* **Output:** un oggetto `JoinerResult` immutabile:
  ```json
  {
    "status": "partial",
    "missing_information": [
      "Get stats for Atlantis — error (ValueError): Country 'Atlantis' not found in dataset.",
      "Calculate population difference (Italy - Atlantis) — skipped: Skipped because upstream task 'task_2' failed."
    ],
    "answer": "The population density of Italy is approximately 195.76 people per square kilometer. However, the population difference between Italy and Atlantis could not be calculated because statistics for Atlantis were not found in the dataset."
  }
  ```
* **Garanzie offerte a valle:**
  * `status` e `missing_information` sono **matematicamente coerenti** con `ExecutionResult` (non dipendono mai dall'umore dell'LLM);
  * `answer` è una stringa non vuota formattata per l'utente finale.
* **Fine della responsabilità:** il `Joiner` (in v1) **non ritenta** l'esecuzione e **non modifica** il piano: il suo lavoro termina consegnando la risposta strutturata e completa.

> **In parole semplici:**
> Il `Joiner` è il portavoce: legge il registro tecnico del cantiere e scrive un messaggio chiaro, educato e onesto nella tua lingua, senza annoiarti con codici interni tipo `task_3`. Allo stesso tempo, il timbro ufficiale *"completo / parziale / fallito"* e la lista esatta di cosa è mancato vengono stampati in automatico dal computer, così non possono mai esserci imprecisioni.

---

### 2.6 Modulo 5 — `graph` / LangGraph (il coordinatore sopra ai moduli)

Infine, il modulo `graph` (`graph/state.py`, `graph/nodes.py`, `graph/builder.py`) **non re-implementa nessuna logica** dei moduli precedenti: li avvolge in 4 nodi sottili dentro uno `StateGraph` di LangGraph.

* **Input:** lo stato iniziale `{"request": "..."}` passato a `await graph.ainvoke(...)` o `graph.astream(...)`.
* **Cosa fa:** gestisce il passaggio di stato (`CompilerState`) da un modulo all'altro, cronometra ogni fase in `WorkflowMetrics`, intercetta eventuali errori applicativi in `WorkflowError` e — in **v2** (`max_compile_retries > 0`) — gestisce il ciclo condizionale `compiler -> planner` quando il `PlanCompiler` boccia un piano.
* **Output:** lo stato finale `CompilerState` completo di:
  * `plan`, `compiled_dag`, `execution_result`, `joiner_result`, `final_answer`, `status`;
  * `error` (eventuale `WorkflowError`), `compile_error_history`;
  * `metrics` (`planning_latency_ms`, `compilation_latency_ms`, `execution_latency_ms`, `joining_latency_ms`, `llm_calls`, `compile_attempts`, `compile_retries`, `plan_valid`, `component_latency_ms`).
* **Garanzie offerte a valle:**
  * Terminazione garantita (grazie al budget `max_compile_retries`), osservabilità step-by-step via `astream`, e separazione netta tra logica dei componenti e orchestrazione del flusso.

> **In parole semplici:**
> Il modulo `graph` è il **nastro trasportatore intelligente** (con un supervisore) che collega le postazioni di lavoro descritte sopra: prende la scatola prodotta da un reparto, cronometra quanto ci ha messo, la passa al reparto successivo e — se l'ispettore (`PlanCompiler`) rimanda indietro il foglio perché c'è un errore — lo riporta automaticamente al `Planner` per una seconda passata, tenendo il conto dei tentativi. Se un giorno vuoi sostituire LangGraph con un altro orchestratore, tutti i moduli sotto (`ToolRegistry`, `Planner`, `PlanCompiler`, `Executor`, `Joiner`) continuano a funzionare esattamente allo stesso modo senza cambiare una riga.

---

## 3. Scelte architetturali e funzionamento interno (in dettaglio)

### 3.1 Schema unificato `inputs` e dipendenze implicite (Strada A)

In `src/langgraph_llm_compiler/planner/schemas.py`, ogni task è definito da `TaskSpec`:

* `task_id: NonEmptyStr`
* `tool_name: NonEmptyStr`
* `description: NonEmptyStr`
* `inputs: dict[str, TaskInputValue]` dove:
  ```python
  class InputBinding(BaseModel):
      model_config = ConfigDict(frozen=True, extra="forbid")
      task_id: NonEmptyStr
      output_path: NonEmptyStr

  TaskInputValue = Annotated[InputBinding | Any, Field(union_mode="left_to_right")]
  ```

Tutti i modelli hanno `frozen=True` (immutabili) e `extra="forbid"` (rifiutano campi non previsti come i vecchi `arguments`, `input_bindings` o `dependencies`). Le proprietà derivate `task.static_inputs` e `task.bindings` filtrano automaticamente `inputs` separando i valori statici dagli `InputBinding`.

**Perché questa scelta ("Strada A") è stata decisiva:**
1. **Zero divergenze tra dipendenze e dataflow**: il grafo delle dipendenze è derivato *unicamente* dai `task_id` presenti in `task.bindings`. Non esiste un campo `dependencies` separato che l'LLM potrebbe dimenticare di aggiornare o tenere disallineato.
2. **Eliminazione per costruzione delle collisioni di chiavi**: in una versione preliminare con dizionari separati (`arguments` per valori statici + `input_bindings` per riferimenti dinamici), `qwen3:8b` inseriva spesso valori segnaposto come `"a": 0` in `arguments` *mentre* dichiarava `"a": {"task_id": "task_1", ...}` in `input_bindings`, provocando errori di collisione in circa 2/3 dei DAG complessi. Con un **singolo dizionario `inputs`** una chiave può contenere *o* un valore statico *o* un `InputBinding`: la collisione è matematicamente impossibile.
3. **Precedenza esplicita `union_mode="left_to_right"`**: Pydantic prova *prima* a validare qualsiasi valore come `InputBinding`; solo se non corrisponde ricade in `Any` (valore statico).
4. **Prompt neutro anti-overfitting (`planner/planner.py`)**: gli esempi nel `SYSTEM_PROMPT` usano esclusivamente nomi di tool fittizi e astratti (`fetch_record`, `compare_values`, `scale_value`). Nei test preliminari su `qwen3:8b`, inserire il nome reale `"calculator"` in un esempio few-shot induceva il modello a copiare letteralmente argomenti dell'esempio (es. `"b": 1`) nelle chiamate reali a `calculator`. L'uso di tool astratti ha eliminato completamente questo bias.
5. **Structured output con debuggabilità**: il Planner invoca `llm.with_structured_output(ExecutionPlan, method="json_schema", include_raw=True)`. Se il modello emette JSON malformato o non conforme, `PlannerOutputError` conserva sempre `raw_output` per l'analisi post-mortem.

> **In parole semplici:**
> Prima avevamo tre fogli separati che l'intelligenza artificiale doveva compilare a mano: *"valori fissi"*, *"collegamenti ai passi precedenti"* e *"elenco delle dipendenze"*. Un modello piccolo (8 miliardi di parametri) faceva spesso confusione, scrivendo `0` sul primo foglio e il collegamento sul secondo, mandando in conflitto il sistema.
>
> Con lo **schema unificato `inputs`**, abbiamo messo tutto in un'unica tabella: per ogni parametro c'è una sola casella. O ci scrivi un numero/testo fisso (es. `"Italy"`), oppure ci metti una freccia (es. *"prendi `population` dal `task_1`"*). In questo modo è fisicamente impossibile contraddirsi, e il sistema capisce da solo chi dipende da chi semplicemente guardando le frecce.

---

### 3.2 `ToolRegistry` come unica *source of truth*

In `src/langgraph_llm_compiler/compiler/registry.py`, ogni strumento viene registrato una sola volta come `ToolSpec`:

```python
ToolSpec(
    name="country_stats",
    description="Fetch demographic and economic statistics for a country.",
    func=get_country_stats,
    input_schema=CountryStatsInput,
    output_schema=CountryStatsOutput,
)
```

Da questa singola registrazione derivano automaticamente tre usi sincronizzati:
1. **Prompt del Planner**: `ToolRegistry.to_planner_catalog()` serializza nomi, descrizioni, campi di input (con tipi, vincoli `Literal`/enum e obbligatorietà) e campi di output (inclusi eventuali campi annidati multi-livello) direttamente nel prompt.
2. **Validazione statica nel PlanCompiler**: gli stessi `input_schema` e `output_schema` vengono usati per verificare argomenti statici e `output_path` prima di eseguire alcunché.
3. **Validazione runtime nell'Executor**: prima e dopo l'esecuzione di ogni handler, l'Executor valida gli input risolti con `input_schema` e il valore di ritorno con `output_schema`.

> **In parole semplici:**
> Il `ToolRegistry` è come l'unico manuale ufficiale degli attrezzi del laboratorio. Sia chi pianifica il lavoro (Planner), sia chi lo controlla (Compiler), sia chi usa gli attrezzi (Executor) legge sempre la *stessa identica pagina*. Se cambi un parametro a uno strumento in un solo punto del codice, tutta la pipeline si aggiorna automaticamente senza rischiare discrepanze.

---

### 3.3 Validazione sintattica (Pydantic) vs semantica (`PlanCompiler`) e algoritmo di Kahn

La pipeline separa rigorosamente due classi di errori:
* **Errori sintattici (`PlannerOutputError`)**: controllati da Pydantic sul JSON prodotto dall'LLM (campi mancanti, stringhe vuote `NonEmptyStr`, campi extra proibiti, JSON troncato).
* **Errori semantici (`PlanCompilationError`)**: controllati da `PlanCompiler.compile(plan)` sull'intero grafo in **un'unica passata** (*non* fail-fast). Anziché interrompersi al primo errore, il compilatore raccoglie tutte le `CompilationIssue(code, message, task_id, field)` in un solo report completo, così nel loop v2 il Planner può correggere più problemi in un unico tentativo.

Le regole verificate in singola passata da `PlanCompiler` sono:

| Codice `CompilationIssue` | Cosa verifica |
| :--- | :--- |
| `duplicate_task_id` | Unicità di tutti i `task_id` nel piano. |
| `unknown_tool` | Esistenza di ogni `tool_name` nel `ToolRegistry`. |
| `malformed_binding` | Intercetta dizionari in `inputs` che contengono `"task_id"` ma hanno fallito il parsing `InputBinding` (es. refuso `'output, path'` o chiave inventata `'divided_by'`). Senza questo controllo specifico, il fallback `Any` della union produrrebbe un messaggio fuorviante come *"Input should be a valid number"*; il compilatore invece emette `malformed_binding` indicando esattamente le chiavi ammesse (`'task_id'`, `'output_path'`) e sopprime il falso positivo `invalid_static_argument`. |
| `self_dependency` | Divieto per un task di fare riferimento al proprio stesso `task_id`. |
| `missing_dependency` | Esistenza di ogni `task_id` referenziato da un `InputBinding`. |
| `cycle_detected` | Assenza di cicli nel grafo delle dipendenze, con ricostruzione esplicita del percorso del ciclo (es. `task_1 -> task_2 -> task_1`). |
| `missing_required_argument` / `unknown_argument` / `invalid_static_argument` | Validazione statica dei parametri contro `ToolSpec.input_schema` (verificando anche gli enum `Literal` e i tipi dei valori statici, pur ignorando momentaneamente i campi che arriveranno via binding a runtime). |
| `invalid_output_path` | Attraversamento ricorsivo del percorso punteggiato (es. `economy.gdp.nominal_billion_usd`) lungo la gerarchia di modelli Pydantic dell'`output_schema` del task upstream. |

Se non emerge alcuna issue, il compilatore esegue l'**algoritmo di Kahn** (ordinamento topologico con calcolo dei livelli *ASAP — As Soon As Possible*) e costruisce un `CompiledDAG` completamente **immutabile** (`frozen=True` + `MappingProxyType` per tutte le mappe interne) che espone:
* `topological_order`: sequenza deterministica dei task;
* `levels`: raggruppamento per profondità logica (utile per analisi e visualizzazione, *non* usato come barriera di esecuzione);
* `root_tasks` e `leaf_tasks`;
* `is_ready(task_id, completed_ids)`: predicato $O(\text{in-degree})$ usato dallo scheduler event-driven.

> **In parole semplici:**
> Immagina un insegnante molto preciso che corregge il tema prima di consegnarlo alla giuria. Invece di fermarsi al primo errore di ortografia e ridarti il foglio indietro dieci volte, legge *tutto* il piano dall'inizio alla fine e ti restituisce una lista numerata con ogni singolo problema: *"qui hai usato un attrezzo che non esiste, qui hai scritto un collegamento sbagliato, qui il passo 2 aspetta il passo 3 che aspetta il passo 2"*.
>
> In più, se il modello sbaglia a scrivere un collegamento (magari aggiungendo una parola inventata come `divided_by`), il compilatore riconosce che stava provando a fare un collegamento e glielo dice chiaramente invece di rispondergli *"mi aspettavo un numero"*. Solo quando il foglio non ha più neanche un segno rosso viene promosso a piano esecutivo immutabile (`CompiledDAG`).

---

### 3.4 Scheduler `asyncio` event-driven vs LangGraph `StateGraph` dinamico (Pregel BSP)

Una scelta architetturale fondamentale del progetto è mantenere l'`Executor` (`src/langgraph_llm_compiler/executor/executor.py`) **completamente indipendente da LangGraph**, eseguendo il `CompiledDAG` tramite uno scheduler **event-driven con ready-queue** in `asyncio`:

1. Tutti i `root_tasks` entrano subito nella `ready_queue`.
2. Finché ci sono task in `ready_queue` e slot liberi rispetto a `max_concurrency` (default `8`; `None` per concorrenza illimitata; `1` per ottenere una **baseline sequenziale equa** sullo stesso codice), lo scheduler avvia `asyncio.create_task(...)`.
3. Lo scheduler attende con `asyncio.wait(running, return_when=asyncio.FIRST_COMPLETED)`: **nel millisecondo esatto in cui termina anche un solo task**, ne controlla i soli successori diretti tramite `dag.is_ready(...)` e mette immediatamente in esecuzione quelli le cui dipendenze sono tutte soddisfatte.
4. **Supporto trasparente a tool sync e async**: le coroutine `async def` (rilevate con `inspect.iscoroutinefunction`) sono eseguite con `await`; le funzioni `def` sincrone vengono delegate a `asyncio.to_thread()` per non bloccare l'event loop.
5. **Timeout per task (`task_timeout_s`)**: su tool `async def`, `asyncio.wait_for` cancella effettivamente la coroutine; su tool `def` sincroni in `asyncio.to_thread`, interrompe l'attesa e registra `error_stage="timeout"` (nota tecnica Python: un thread OS già avviato non è terminabile forzatamente, quindi è buona norma usare tool `async def` o idempotenti quando si testano timeout stretti).

**Perché NON abbiamo trasformato il `CompiledDAG` in un `StateGraph` dinamico di LangGraph?**
Perché il runtime **Pregel** di LangGraph segue il modello **Bulk Synchronous Parallel (BSP)**: l'esecuzione avanza per *superstep* discreti separati da una **barriera di sincronizzazione globale**. Tutti i nodi lanciati nello stesso superstep devono completare prima che il superstep successivo possa iniziare. In presenza di tool/API con latenze diverse (caso normale nel mondo reale), un ramo veloce resta bloccato alla barriera ad aspettare un ramo lento completamente indipendente.

#### Verifica sperimentale: Event-driven vs LangGraph Pregel BSP (`benchmark_scheduling.py`)

Per misurare questa differenza in modo oggettivo (zero chiamate LLM, stessi identici `CompiledDAG` eseguiti su 3 motori, mediana su 3 ripetizioni; report completo in `results/benchmark_scheduling.json`):

| Topologia DAG | Nodi / Livelli | Sequenziale (`max=1`) | LangGraph Pregel BSP *(+compile)* | Executor Event-driven (`max=8`) | Speedup BSP | Speedup Event-driven | Riduzione latenza vs BSP |
| :--- | :---: | ---: | ---: | ---: | :---: | :---: | :---: |
| `symmetric_fanout` *(controllo equità)* | 5 / 2 | 522.2 ms | 164.4 ms *(+2.8 ms)* | **160.9 ms** | 3.18x | **3.25x** | -2.1% *(parità)* |
| `asymmetric_branches` *(T1:50→T3:250 vs T2:300→T4:50 → T5:50)* | 5 / 3 | 702.5 ms | 606.1 ms *(+3.0 ms)* | **401.8 ms** | 1.16x | **1.75x** | **-33.7%** (-204 ms) |
| `staggered_multi_branch` *(profilo reale di `test2`)* | 8 / 3 | 903.7 ms | 466.4 ms *(+4.0 ms)* | **421.8 ms** | 1.94x | **2.14x** | **-9.6%** (-45 ms) |
| `deep_asymmetric_pipeline` *(4 livelli asimmetrici)* | 7 / 4 | 663.3 ms | 548.3 ms *(+3.8 ms)* | **383.0 ms** | 1.21x | **1.73x** | **-30.1%** (-165 ms) |

Il diagramma Gantt affiancato su `asymmetric_branches` (stessa scala temporale `0 → 606 ms`) mostra visivamente l'effetto della barriera BSP:

```text
LangGraph Pregel BSP (606 ms):
  T1 │██································│   1 →  52 ms
  T2 │████████████████··················│   1 → 302 ms
  T3 │·················██████████████···│ 304 → 555 ms  <-- fermo 252 ms alla barriera BSP!
  T4 │·················██···············│ 304 → 354 ms
  T5 │·······························██·│ 555 → 606 ms

Executor Event-driven (402 ms):
  T1 │██································│   0 →  50 ms
  T2 │████████████████··················│   0 → 301 ms
  T3 │··██████████████··················│  51 → 301 ms  <-- parte subito al termine di T1!
  T4 │················███···············│ 301 → 352 ms
  T5 │···················███············│ 352 → 402 ms
```

* Sul caso simmetrico (`symmetric_fanout`) i due motori pareggiano (`164.4 ms` vs `160.9 ms`), dimostrando che il benchmark è equo e non artificialmente sbilanciato.
* Sui DAG asimmetrici, eliminare la barriera BSP riduce il tempo di esecuzione fino al **-33.7%** (-204 ms su 606 ms) e risparmia anche ~3–4 ms di compilazione dinamica del grafo a ogni richiesta.

> **In parole semplici:**
> Immagina una staffetta con due corsie parallele. Nella corsia A il primo corridore è velocissimo (50 secondi) e il secondo è più lento (250 secondi); nella corsia B il primo è lento (300 secondi) e il secondo è veloce (50 secondi).
> * Nel modello a **"ondate" (Pregel BSP di LangGraph)**, il giudice di gara non lascia partire *nessun* secondo corridore finché *entrambi* i primi corridori non hanno finito. Così il secondo corridore della corsia A resta fermo ai blocchi per 250 secondi a guardare la corsia B!
> * Nel nostro modello **a eventi (Event-driven)**, appena il primo corridore della corsia A arriva al cambio, il suo compagno parte **all'istante** senza aspettare la corsia B. Risultato: la gara finisce in un terzo del tempo in meno!

---

### 3.5 Esecuzione best-effort, propagazione `SKIPPED` e telemetria serializzabile

In `src/langgraph_llm_compiler/executor/results.py` e `executor.py`, ogni task produce un `TaskExecutionResult` immutabile (`frozen=True`):

* `status`: `"success" | "error" | "skipped"`
* `output`: payload validato tramite `ToolSpec.output_schema` (solo se `success`)
* `error`, `error_type`, `error_stage`: compilati in caso di errore, con `error_stage` che distingue esattamente dove si è verificato il guasto:
  * `"binding"` (fallimento nella risoluzione di un `output_path` a runtime)
  * `"input_validation"` (i dati risolti a runtime violano `input_schema`, es. divisione per zero o tipi imprevisti)
  * `"tool_execution"` (eccezione lanciata dal tool stesso, es. risorsa non trovata)
  * `"output_validation"` (il tool ha restituito un dato non conforme a `output_schema`)
  * `"timeout"` (superato `task_timeout_s`)
* `skip_reason`: compilato se un task upstream è fallito
* `duration_ms`, `started_at_ms`, `finished_at_ms`: telemetria precisa (`None` per i timestamp dei task `skipped`, perché non sono mai stati avviati)

**Politica best-effort e propagazione transitiva:**
* Se un task fallisce, l'Executor **non interrompe** i rami indipendenti del DAG.
* Tutti i discendenti raggiungibili dal task fallito (diretti e indiretti, a qualsiasi profondità) vengono marcati immediatamente `status="skipped"` senza mai occupare slot di esecuzione.
* Nessun oggetto `Exception` raw viene conservato nei risultati: `ExecutionResult` è 100% serializzabile in JSON, rendendolo sicuro per lo stato di LangGraph, i log sperimentali e futuri checkpointer.

> **In parole semplici:**
> Se durante una giornata di commissioni scopri che la panetteria è chiusa, non annulli dal calendario anche la farmacia e la posta: porti a casa tutto ciò che non dipendeva dal pane (esecuzione *best-effort*), depenni automaticamente dalla lista "fare i panini" perché senza pane non si possono fare (propagazione *skipped*), e sul taccuino annoti l'ora esatta e il motivo preciso di ogni intoppo.

---

### 3.6 Separazione deterministica nel `Joiner` (`JoinerOutput` vs `JoinerResult`)

In `src/langgraph_llm_compiler/joiner/`, separiamo nettamente ciò che è **generativo** da ciò che è **deterministico**:

* **Responsabilità dell'LLM (`JoinerOutput`)**: genera *unicamente* il campo testuale `answer: str` seguendo 4 regole esplicite di sistema:
  1. **Lingua**: rispondere nella stessa lingua della richiesta utente originale.
  2. **Grounding stretto**: riportare solo fatti e numeri presenti negli output `success`; mai inventare o stimare risultati mancanti.
  3. **Trasparenza**: esplicitare chiaramente quali parti non sono state completate e perché.
  4. **Stile user-friendly**: non esporre mai identificativi interni (`task_1`, `task_4`) né nomi grezzi di funzioni/tool nella risposta finale.
* **Responsabilità del codice deterministico (`JoinerResult`)**:
  * `classify_execution_status(execution_result)` calcola `status`:
    * `"complete"` se tutti i task sono `success`;
    * `"failed"` se zero task sono `success`;
    * `"partial"` in tutti i casi misti.
  * `extract_missing_information(execution_result, plan)` costruisce l'elenco ordinato topologicamente dei task falliti o saltati, usando la descrizione leggibile `task.description` (con fallback a `tool_name`) e il motivo strutturato.

> **In parole semplici:**
> All'intelligenza artificiale chiediamo solo quello che sa fare meglio: scrivere un testo chiaro, naturale e ben scritto nella tua lingua. Ma il "semaforo" che dice se il lavoro è **completo**, **parziale** o **fallito** — e la lista ufficiale di cosa è mancato — li calcola il programma con regole matematiche rigide. In questo modo il modello non può mai "fare il furbo" dichiarando *tutto ok!* quando in realtà un pezzo si è rotto.

---

### 3.7 Orchestrazione macro con LangGraph: da v1 lineare a v2 compile-time self-correction

LangGraph viene usato **esclusivamente al livello di orchestrazione macro stateful** (`src/langgraph_llm_compiler/graph/`), dove brillano i suoi punti di forza: gestione pulita dei cicli di controllo tramite conditional edges, stato condiviso tipizzato (`CompilerState`), streaming passo-passo (`astream`) e interfaccia uniforme per confrontare la nostra architettura con altri agenti (es. ReAct).

```mermaid
flowchart TD
    START(["START"]) --> P["planner"]
    P -->|"ExecutionPlan OK"| C["compiler"]
    P -->|"PlannerOutputError"| END_ERR(["END (failed)"])
    C -->|"CompiledDAG OK"| E["executor"]
    C -->|"PlanCompilationError & retries < budget (v2)"| P
    C -->|"PlanCompilationError & budget esaurito"| END_ERR
    E -->|"ExecutionResult OK"| J["joiner"]
    E -->|"Errore infrastrutturale"| END_ERR
    J --> END_OK(["END (complete / partial / failed)"])
```

Dettagli chiave dell'implementazione in `graph/state.py`, `graph/nodes.py` e `graph/builder.py`:

* **4 nodi async sottili (`nodes.py`)**: ciascun nodo wrappa il proprio componente core (`Planner`, `PlanCompiler`, `Executor`, `Joiner`) misurandone la latenza via `time.perf_counter()`, aggiornando `WorkflowMetrics` e intercettando eccezioni applicative in `WorkflowError` (preservando però esplicitamente `asyncio.CancelledError` affinché la cancellazione di task asincroni non venga mai inghiottita).
* **Retrocompatibilità v1 → v2**: `build_compiler_graph(..., max_compile_retries: int = 0)` mantiene di default `0` (comportamento v1 puramente lineare). Impostando `max_compile_retries > 0` (es. `2`), il grafo abilita il ciclo `compiler -> planner`:
  1. Al fallimento di `PlanCompiler`, il nodo compiler formatta tutte le `CompilationIssue` in `compile_feedback` e aggiunge l'elenco dei messaggi a `compile_error_history`.
  2. Il router condizionale verifica se `compile_retries < max_compile_retries` (`compile_retries = max(0, compile_attempts - 1)`). Se c'è ancora budget, instrada verso `planner`.
  3. Il nodo `planner` rileva `compile_feedback` e richiama `planner.aplan(request, previous_plan=..., feedback=...)`, chiedendo all'LLM una riparazione mirata del piano precedente.
  4. Le latenze `planning_latency_ms` e `compilation_latency_ms` sono **cumulative** su tutti i tentativi, e `llm_calls` conta ogni chiamata reale.
  5. `metrics.plan_valid` è **tri-stato**: `None` se fallisce il Planner prima della compilazione, `False` se il Compiler boccia il piano, `True` non appena il piano compila.
* **Nessun parametro fittizio (`checkpointer`)**: `build_compiler_graph` non espone ancora un parametro `checkpointer` perché la serializzazione di `CompiledDAG` (che usa `MappingProxyType`) tramite `MemorySaver`/`SqliteSaver` non è stata ancora testata formalmente; verrà introdotto solo insieme ai relativi test di persistenza.

> **In parole semplici:**
> LangGraph è il **direttore d'orchestra** che sta sopra ai quattro reparti. Tiene un quaderno condiviso dove sono annotati la richiesta, il piano, gli eventuali errori e i tempi cronometrati di ogni reparto.
> * Nella **versione 1**, se l'Ispettore (Compiler) trovava anche un solo errore nel piano, il direttore fermava subito tutto e dichiarava fallimento.
> * Nella **versione 2**, il direttore prende il foglio con le correzioni in rosso dell'Ispettore e lo rimanda al Pianificatore dicendo: *"Questo era il tuo piano, ecco esattamente cosa non va, riprova!"*. Per sicurezza imposta un limite massimo di tentativi (es. 2 retry), così se il modello si incarta il sistema si ferma comunque in modo pulito e conserva tutta la storia degli errori per farci capire cosa è successo.

---

### 3.8 Rigore metodologico delle metriche

Nei benchmark di sistemi paralleli è facile confondere metriche simili ma concettualmente diverse. In questo progetto le separiamo in modo esplicito:

* **Effective parallelism** (calcolato in `ExecutionResult`):
  $$\text{effective\_parallelism} = \frac{\sum_{i} T_i}{T_{\text{parallel}}}$$
  Indica il grado medio di sovrapposizione dei task osservato durante una singola esecuzione (somma delle durate individuali divisa per il tempo wall-clock parallelo).
* **Measured speedup** (calcolato nei benchmark):
  $$\text{measured\_speedup} = \frac{T_{\text{sequential (max\_concurrency=1)}}}{T_{\text{parallel}}}$$
  È lo speedup *reale* misurato rieseguendo lo stesso identico `CompiledDAG` sullo stesso `Executor` con `max_concurrency=1` come baseline sequenziale, eliminando qualsiasi bias di implementazione.
* **Overhead puro di orchestrazione LangGraph**:
  $$\text{langgraph\_overhead\_ms} = T_{\text{graph\_wall}} - \text{metrics.component\_latency\_ms}$$
  Dove `component_latency_ms` è la somma esatta dei tempi spesi dentro i 4 nodi (`planning + compilation + execution + joining`). La differenza isola al centesimo di millisecondo il costo del runtime LangGraph (dispatch nodi, transizioni stato, routing condizionale).

> **In parole semplici:**
> Per essere scientificamente onesti, non ci accontentiamo di dire *"sembra veloce"*: cronometriamo sempre il **confronto diretto** tra fare lo stesso identico lavoro un passo alla volta (`max_concurrency=1`) e farlo in parallelo, e separiamo il tempo passato a lavorare davvero dal tempo speso dal "direttore d'orchestra" (LangGraph) per spostare le carte da una scrivania all'altra.

---

## 4. Risultati sperimentali e analisi critica

Tutti gli esperimenti live sono stati eseguiti localmente tramite **Ollama** con il modello **`qwen3:8b`** (`temperature=0.0`, `joiner reasoning=False`). Tutti i dataset di output completi in JSON sono archiviati in `results/`.

### 4.1 Overhead puro di orchestrazione LangGraph

Script: `experiments/test_graph_qwen3.py --mode overhead --overhead-iters 50`

Per isolare il costo di LangGraph dalla varianza non deterministica dell'LLM, abbiamo confrontato la chiamata diretta in Python dei 4 componenti contro `build_compiler_graph(...)` su **50 iterazioni deterministiche** (modelli fake e tool a 0 ms):

| Configurazione | Mediana | Media | Note |
| :--- | ---: | ---: | :--- |
| Pipeline diretta Python | `0.351 ms` | `0.368 ms` | 4 chiamate sequenziali dirette |
| LangGraph `StateGraph` (`graph_wall_time`) | `2.186 ms` | `2.230 ms` | Include dispatch nodi e gestione stato Pregel |
| **Overhead puro LangGraph** | **`+1.83 ms`** | **`+1.86 ms`** | Nei run live su `qwen3:8b`: `~3.8–4.1 ms` (`< 0.1%` del tempo totale) |

**Conclusione:** usare LangGraph come orchestratore macro aggiunge meno di 2 millisecondi a vuoto (~4 ms nei run live con payload completi) — completamente trascurabile rispetto ai secondi dell'LLM, a fronte di un enorme guadagno in osservabilità, streaming e gestione dei cicli v2/v3.

---

### 4.2 Baseline v1 su suite standard (4 tool, 8 query)

Script: `experiments/benchmark_v1_baseline.py --reasoning-modes both`
Report: `results/v1_baseline_20261119T192608Z.json`

La suite standard comprende 4 tool (`country_stats`, `calculator`, `percent_change`, `currency_convert`) e 8 query di difficoltà crescente (`easy`, `medium`, `hard`, `resilience` con fallimento parziale).

Prima del refactor allo schema unificato `inputs` (Strada A), `qwen3:8b` falliva circa 2/3 dei DAG complessi per collisioni tra `arguments` e `input_bindings`. Dopo Strada A, persino **v1 senza alcun retry (`max_compile_retries=0`)** raggiunge la saturazione completa (**16/16 = 100%**):

| Modalità Planner | Plan Valid | Status OK | Accuratezza Numerica | Med Plan | Med Join | Med Wall | Med Speedup (Max) | Med Overhead LG |
| :--- | :---: | :---: | :---: | ---: | ---: | ---: | :---: | ---: |
| `v1` + `reasoning=False` | **8/8 (100%)** | **8/8 (100%)** | **8/8 (100%)** | **3.44 s** | 0.65 s | **4.24 s** | 1.41x (**2.15x**) | `3.91 ms` |
| `v1` + `reasoning=True` | **8/8 (100%)** | **8/8 (100%)** | **8/8 (100%)** | 7.19 s | 0.62 s | 7.99 s | 1.41x (**2.14x**) | `3.85 ms` |

**Conclusione:** su cataloghi compatti (4 tool) e schemi chiari, lo schema unificato `inputs` rende `qwen3:8b` talmente affidabile da saturare la suite già al primo tentativo; in questo regime `reasoning=False` è strettamente preferibile perché dimezza la latenza (`4.24 s` vs `7.99 s`) a parità di accuratezza (100%).

---

### 4.3 Benchmark deterministico di scheduling (Event-driven vs Pregel BSP)

Script: `experiments/benchmark_scheduling.py --repeats 3` — Report: `results/benchmark_scheduling.json`

Come dettagliato nella [Sezione 3.4](#34-scheduler-asyncio-event-driven-vs-langgraph-stategraph-dinamico-pregel-bsp), a parità di `CompiledDAG` lo scheduler `asyncio` event-driven pareggia con LangGraph Pregel BSP sui DAG simmetrici (`160.9 ms` vs `164.4 ms`, -2.1%) e abbatte la latenza di esecuzione fino al **-33.7%** (`401.8 ms` vs `606.1 ms` su `asymmetric_branches`, **-30.1%** su `deep_asymmetric_pipeline`) sui DAG asimmetrici.

---

### 4.4 Hard Suite 10 tool: matrice 2x2 (v1 vs v2 e reasoning)

Script: `experiments/benchmark_v2_hard_suite.py` (con e senza `--reasoning`)
Report: `results/hard_suite_v1_vs_v2_20261119T221908Z.json` (`reasoning=False`) e `results/hard_suite_v1_vs_v2_20261119T224204Z.json` (`reasoning=True`)

Poiché la suite da 4 tool era saturata al 100%, per misurare l'impatto reale del **compile-time self-correction (v2)** abbiamo progettato una **Hard Suite** realistica composta da:
* **10 tool totali**: 6 strumenti di dominio (`macro_profile`, `energy_profile`, `fx_quote`, `tax_estimator`, `math_op`, `pct_diff`) + **4 distrattori plausibili** (`weather_forecast`, `stock_quote`, `timezone_offset`, `unit_convert`) progettati per indurre scelte errate di tool;
* **Output annidati fino a 3 livelli** (es. `economy.gdp.nominal_billion_usd`, `demographics.population_millions`, `metrics.renewable_share_pct`);
* **Enum stretti** (`op: 'add'|'sub'|'mul'|'div'`, `country_code: 'IT'|'DE'|'FR'|'ES'|'JP'`);
* **6 query complesse (`h01`–`h06`)** verificate da un **grader numerico automatico** (che controlla la presenza di tutti i valori attesi entro tolleranza di arrotondamento nella risposta finale del Joiner).

#### Matrice completa 2×2 dei risultati

| Configurazione | Plan Valid | Accuratezza Numerica Esatta | Casi recuperati via retry | Retry totali | LLM Calls | Med Wall | Max Wall | Med Speedup (Max) |
| :--- | :---: | :---: | :---: | :---: | :---: | ---: | ---: | :---: |
| **`v1` (`retries=0`) + `reasoning=False`** | 2/6 (**33%**) | 1/6 (**17%**) | 0 | 0 | 8 | **5.28 s** | **10.74 s** | 2.43x (2.63x) |
| **`v2` (`retries=2`) + `reasoning=False`** | 5/6 (**83%**) | 3/6 (**50%**) | **3** | 5 | 16 | **11.64 s** | **14.38 s** | 2.27x (2.64x) |
| **`v1` (`retries=0`) + `reasoning=True`** | 4/6 (**67%**) | 4/6 (**67%**) | 0 | 0 | 10 | 34.03 s | 445.99 s | 2.22x (2.74x) |
| **`v2` (`retries=2`) + `reasoning=True`** | **6/6 (100%)** | **6/6 (100%)** | **3** | 4 | 16 | 26.32 s | 131.59 s | 2.15x (**2.78x**) |

#### Anatomia dei casi più significativi (dai log JSON)

* **`h01_deep_nested_fx` (trappola del distrattore `unit_convert`, `reasoning=False`)**: al tentativo 1 il modello ha provato a convertire valute (`USD -> EUR`) usando il distrattore `unit_convert` (che accetta solo `km`, `mi`, `kg`, `lb`). Il compilatore ha emesso 4 `invalid_static_argument` esplicitando i valori ammessi: in **v2**, al retry 1 il modello ha abbandonato `unit_convert`, è passato a `fx_quote` + `math_op`, e ha prodotto tutti e 3 i numeri esatti (`4163`, `3781.2`, `381.8`).
* **`h02_co2_per_million` (riparazione di `malformed_binding` e `unknown_tool`)**:
  * a `reasoning=False`, al tentativo 1 il modello ha inventato una chiave `divided_by` dentro l'`InputBinding`. La nostra regola dedicata `malformed_binding` lo ha intercettato con un feedback preciso e **v2** ha riparato il piano in 1 retry (`5.34`, `8.07`, `51.16%` esatti).
  * a `reasoning=True`, al tentativo 1 il modello aveva addirittura allucinato tre tool inesistenti (`algebraic_manipulation`, `arithmetic_operations`, `division`); **v2** lo ha riportato sui binari giusti al retry 1 (`9` nodi, `4` livelli, `speedup 2.51x`).
* **`h03_renewable_twh_groups` (il limite strutturale della validazione statica vs DAG da 14 nodi)**:
  * a `reasoning=False`, il piano compilava al primo colpo (10 nodi, 3 livelli, `speedup 2.64x`), ma conteneva un **errore logico puro**: moltiplicava `total_twh * renewable_share_pct` dimenticando `/ 100` (ottenendo `25,275 TWh` invece di `252.75 TWh`). Poiché tutti i tipi erano validi (`float * float`), un compilatore statico *non può* rilevare questo errore.
  * a `reasoning=True` + **v2**, il modello ha impostato la formula corretta con `/ 100` in un DAG da **14 nodi su 4 livelli** (`speedup 2.78x`): al tentativo 1 mancavano alcuni campi `op`/`country`, al retry 1 aveva usato `output_path="result"` invece di `"value"` su `math_op`, e al **retry 2** ha riparato tutto ottenendo i risultati esatti al centesimo (`252.75 TWh` e `393.9 TWh`).
* **`h04_tax_fx_chain` (*fixation bias* vs *runaway thinking loop*)**:
  * a `reasoning=False`, vedendo il distrattore `unit_convert` nel `previous_plan`, il modello ha mostrato un fenomeno di **fixation bias**, riproponendo `unit_convert` su tutti e 3 i tentativi finché `max_compile_retries=2` non ha fermato il grafo in modo pulito (`14.38 s`).
  * a `reasoning=True`, in **v1** il modello è caduto in un *runaway thinking loop* da **`445.99 s`** terminato con `PlannerOutputError` (output troncato); in **v2**, dopo un singolo `invalid_output_path` (`'value'` su `fx_quote`), ha recuperato al retry 1 in `28.31 s` producendo tutti e 4 i numeri esatti (`184000`, `114080`, `104880`, `9200`).

---

### 4.5 Cosa cambia con i retry (v2): vantaggi, svantaggi e guida alla scelta

#### Vantaggi misurati di v2 (compile-time self-correction)
1. **Salto netto di affidabilità su cataloghi complessi**: a `reasoning=False` la `Plan validity` sale da **33% → 83%** (+50 punti percentuali) e l'accuratezza numerica triplica (**17% → 50%**); a `reasoning=True` si raggiunge **6/6 (100%)** sia in validità che in accuratezza numerica.
2. **Zero overhead sul caso felice (*zero-cost on happy path*)**: se un piano compila al primo tentativo (`compile_retries = 0`), v2 non aggiunge alcuna chiamata LLM né latenza extra rispetto a v1.
3. **Ortogonalità rispetto a `reasoning=True`**: le due tecniche risolvono classi di errore diverse e complementari:
   * **`v2` (retry deterministici)** elimina gli **errori di contratto/schema** (`unknown_tool`, `unknown_argument`, `malformed_binding`, `invalid_output_path`, enum errati su distrattori).
   * **`reasoning=True`** elimina gli **errori logici puri** (es. dimenticare `/ 100` in `h03` e `h06`), invisibili a qualsiasi compilatore statico.
4. **Terminazione garantita e tracciabilità**: il budget `max_compile_retries` impedisce loop infiniti e registra l'intera `compile_error_history` per diagnosi e analisi.

#### Svantaggi e limiti emersi sperimentalmente
1. **Costo in latenza e chiamate LLM sui casi falliti**: ogni retry comporta una nuova chiamata completa al `Planner` (sulla Hard Suite le chiamate LLM totali passano da `8 → 16` a `reasoning=False` e da `10 → 16` a `reasoning=True`, raddoppiando la mediana wall-clock da `5.28 s` a `11.64 s` a `reasoning=False`).
2. **Invisibilità degli errori semantici/matematici puri**: se il piano rispetta tutti i tipi Pydantic ma applica una formula sbagliata, `PlanCompiler` lo approva comunque.
3. **Fixation bias sui modelli 8B senza reasoning**: includere `previous_plan` nel prompt di retry aiuta nella maggior parte dei casi a correggere chirurgicamente un singolo campo, ma occasionalmente può "ancorare" il modello su un tool distrattore sbagliato (`h04` a `reasoning=False`).
4. **Tail latency di `reasoning=True` su modelli 8B**: senza un tetto esplicito (`num_predict` o timeout), il thinking mode può occasionalmente generare catene di pensiero molto lunghe (`131.59 s` su `h02` in v2, `445.99 s` su `h04` in v1).

#### Guida pratica: quale configurazione scegliere?

| Scenario d'uso | Configurazione consigliata | Perché |
| :--- | :--- | :--- |
| Catalogo piccolo (≤ 5 tool), schemi semplici | `v2 (retries=1–2)` + `reasoning=False` | 100% di accuratezza, ~4 s di latenza, rete di sicurezza gratuita sul caso felice. |
| Catalogo ampio (10+ tool), priorità a **latenza bassa e prevedibile** | `v2 (retries=2)` + `reasoning=False` | Tempi sempre sotto i 15 s (`Max Wall 14.38 s`), 83% di piani validi senza rischi di *runaway thinking*. |
| Catalogo ampio e formule complesse, priorità a **massima accuratezza** | `v2 (retries=2)` + `reasoning=True` *(con limite `num_predict`/timeout)* | 100% di piani validi e 100% di accuratezza numerica (6/6), proteggendo la tail latency con un cap sui token di pensiero. |

> **In parole semplici:**
> Ecco cosa ci hanno insegnato tutti gli esperimenti in quattro frasi:
> 1. **LangGraph è praticamente gratis**: il costo del "direttore d'orchestra" è meno di 4 millesimi di secondo (un battito di ciglia).
> 2. **Il nostro esecutore a eventi fa risparmiare davvero tempo**: quando gli strumenti hanno tempi di risposta diversi, finisce fino a un terzo prima rispetto a uno schema a ondate rigide.
> 3. **I retry del compilatore (v2) e il "pensiero profondo" (`reasoning=True`) sono come due paia di occhiali diversi**: i retry correggono gli errori di distrazione (nomi di strumenti sbagliati, collegamenti scritti male) senza rallentare mai i casi già giusti; il pensiero profondo corregge gli errori di ragionamento matematico (come ricordarsi di dividere una percentuale per 100), ma fa riflettere il modello molto più a lungo e ogni tanto rischia di farlo "incartare" nei suoi stessi pensieri se non gli metti una sveglia.
> 4. **Insieme fanno la differenza**: sui test facili basta la modalità veloce per fare 100%, ma sui test difficili con tanti strumenti-trappola è l'abbinamento **v2 + pensiero profondo** a portare il punteggio dal 17% fino al **100% perfetto (6 su 6)**!

---

## 5. Struttura del repository

```text
langgraph-llm-compiler/
├── pyproject.toml                     # Dipendenze, configurazione ruff e pytest (testpaths = ["tests"])
├── README.md                          # Questa documentazione tecnica + guida in parole semplici
├── src/
│   └── langgraph_llm_compiler/
│       ├── llm.py                     # Factory ChatOllama (configurazione modello, temperature, reasoning)
│       ├── planner/
│       │   ├── schemas.py             # NonEmptyStr, InputBinding, TaskSpec (inputs unificato), ExecutionPlan
│       │   └── planner.py             # Planner (plan / aplan + retry feedback), SYSTEM_PROMPT neutro, PlannerOutputError
│       ├── compiler/
│       │   ├── registry.py            # ToolSpec, ToolRegistry (source of truth + to_planner_catalog)
│       │   ├── dag.py                 # CompilationIssue, PlanCompilationError, CompiledDAG (immutabile)
│       │   └── compiler.py            # PlanCompiler (validazione semantica in singola passata + Kahn scheduling)
│       ├── executor/
│       │   ├── results.py             # TaskExecutionResult, ExecutionResult (frozen, serializzabili)
│       │   ├── bindings.py            # BindingResolutionError, resolve_output_path, resolve_task_inputs
│       │   └── executor.py            # Executor (scheduler asyncio event-driven, best-effort, timeout)
│       ├── joiner/
│       │   ├── schemas.py             # JoinerStatus, JoinerOutput (LLM), JoinerResult (app)
│       │   └── joiner.py              # Joiner (async join), classificazione deterministica, JoinerOutputError
│       └── graph/
│           ├── state.py               # CompilerState, WorkflowError, WorkflowMetrics, update_metrics
│           ├── nodes.py               # 4 nodi async sottili (planner, compiler, executor, joiner)
│           └── builder.py             # build_compiler_graph(..., max_compile_retries=0) -> CompiledStateGraph
├── tests/                             # 57 test unitari deterministici (zero dipendenza da Ollama)
│   ├── test_planner_schemas.py
│   ├── test_planner.py
│   ├── test_compiler.py
│   ├── test_executor.py
│   ├── test_joiner.py
│   └── test_graph.py
├── experiments/
│   ├── test_planner_qwen3.py          # Pipeline diretta Python (Registry -> Planner -> Compiler -> Executor -> Joiner)
│   ├── test_graph_qwen3.py            # Esperimento A (live LangGraph vs Direct) + Esperimento B (overhead isolato)
│   ├── test2.py                       # Trace visivo fase-per-fase via astream + export PNG/Mermaid in results/
│   ├── benchmark_v1_baseline.py       # Suite baseline v1 (4 tool, 8 query, reasoning=False vs True)
│   ├── benchmark_v2_hard_suite.py     # Hard suite 10 tool con grading numerico automatico: v1 vs v2
│   └── benchmark_scheduling.py        # Benchmark deterministico Event-driven vs LangGraph Pregel BSP vs Seq
└── results/                           # Output sperimentali salvati (JSON, PNG dello StateGraph, report Mermaid)
```

---

## 6. Setup, verifica e comandi per gli esperimenti

**Requisiti:** Python `>=3.12`, [`uv`](https://docs.astral.sh/uv/) e [Ollama](https://ollama.com/) con `qwen3:8b` (`ollama pull qwen3:8b`, necessario solo per gli esperimenti live con LLM).

### Installazione, test unitari e lint (non richiedono Ollama)

```bash
# Installa tutte le dipendenze (inclusi pytest, ruff e grandalf per draw_ascii)
uv sync

# Esegui la suite completa di 57 test unitari deterministici
uv run pytest -v

# Controllo statico stile e qualità con ruff
uv run ruff check src tests experiments
```

### Benchmark deterministici (non richiedono Ollama)

```bash
# Confronto Event-driven vs LangGraph Pregel BSP vs Sequenziale su 4 topologie DAG
uv run python experiments/benchmark_scheduling.py --repeats 3

# Misura isolata dell'overhead di orchestrazione LangGraph (50 iterazioni deterministiche)
uv run python experiments/test_graph_qwen3.py --mode overhead --overhead-iters 50
```

### Esperimenti live con `qwen3:8b` (richiedono Ollama attivo)

```bash
# Trace visivo passo-passo su LangGraph (v2, default --max-compile-retries 2) + export PNG e Mermaid in results/
uv run python experiments/test2.py --no-reasoning

# Suite baseline v1 (4 tool, 8 query, confronto reasoning=False vs True)
uv run python experiments/benchmark_v1_baseline.py --reasoning-modes both

# Hard suite 10 tool: confronto v1 (0 retry) vs v2 (2 retry) con grading numerico automatico
uv run python experiments/benchmark_v2_hard_suite.py
uv run python experiments/benchmark_v2_hard_suite.py --reasoning
```

---

## 7. Roadmap

- [x] **Planner**: schemi Pydantic (`inputs` unificato Strada A), structured output `json_schema`, prompt neutro anti-overfitting, test unitari.
- [x] **Compiler**: `ToolRegistry`, validazione semantica in singola passata (incluso `malformed_binding` e `invalid_output_path` multi-livello), algoritmo di Kahn, `CompiledDAG` immutabile, test unitari.
- [x] **Executor**: scheduler `asyncio` event-driven, binding resolution, validazione runtime I/O, best-effort + propagazione `skipped`, `max_concurrency`, `task_timeout_s`, telemetria serializzabile, test unitari.
- [x] **Joiner v1**: sintesi risposta finale nella lingua dell'utente (`JoinerOutput`), `status` e `missing_information` deterministici (`JoinerResult`), test unitari.
- [x] **LangGraph v1**: `StateGraph` lineare con uscite condizionali su errore, `WorkflowMetrics`, benchmark overhead isolato e benchmark deterministico di scheduling (Event-driven vs Pregel BSP).
- [x] **v2 — Compile-time self-correction**: ciclo `compiler -> planner` con `previous_plan`, `compile_feedback` (`CompilationIssue`), `compile_error_history`, latenze cumulative, `max_compile_retries` retrocompatibile e Hard Suite 10 tool (matrice 2×2).
- [ ] **v3 — Runtime replanning**: ciclo `joiner -> planner` con risultati parziali, strumenti di fallback e budget `max_replans`.
- [ ] **Fase 6 — Benchmark comparativi cross-architettura**: confronto riproducibile *apples-to-apples* tra LLM Compiler (v1/v2/v3) e agenti iterativi (es. LangGraph ReAct) su latenza, chiamate LLM, token e accuratezza.
