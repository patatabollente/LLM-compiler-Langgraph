# Laboratorio Didattico: LLM Compiler v3 vs LangGraph ReAct

- **Modello:** `qwen3:8b` (`temperature=0.0`, `reasoning=False`)
- **Obiettivo:** osservare passo dopo passo come ragionano ed eseguono gli stessi identici tool.


---

## Scenario 1 — Il Grafo Asimmetrico (Parallelismo, Dataflow e Chiamate LLM)


**Obiettivo:** Capire perché LLM Compiler fa 2 sole chiamate LLM ed esegue i rami senza barriere mentre ReAct deve fermarsi a ogni livello per ripassare dall'LLM.

```text
  Ramo A: fetch_order_summary (0.3s) ──▶ run_risk_audit (1.8s) ──┐
                                                                  ├──▶ build_executive_score (0.3s)
  Ramo B: fetch_warehouse_stock (1.8s) ─▶ compute_restock_cost (0.3s) ┘
  (Lavoro sequenziale totale = 4.5s | Ottimo event-driven = 2.4s | Ottimo a turni = 3.9s)
```

**Richiesta identica per entrambi:**
> Valuta l'ordine 'ORD-42' e lo stock del prodotto 'SKU-X9'. Recupera i dati dell'ordine e calcola il suo risk_score con run_risk_audit; in parallelo recupera lo stock di 'SKU-X9' e calcola il restock_cost_eur con compute_restock_cost. Infine combina risk_score e restock_cost_eur usando build_executive_score e riporta tutti i valori (risk_score, restock_cost_eur, priority_index e recommendation) in italiano.

### Parte A — LLM Compiler v3 (passo dopo passo)




**[Step 1 · Round 1] PLANNER (chiamata LLM)**
- Obiettivo: `Evaluate order 'ORD-42' and product stock 'SKU-X9', retrieve order data and calculate its risk_score using run_risk_audit; in parallel retrieve stock data for 'SKU-X9' and calculate restock_cost_eur using compute_restock_cost. Finally combine risk_score and restock_cost_eur using build_executive_score and report all values (risk_score, restock_cost_eur, priority_index and recommendation) in Italian.`
- Ha pianificato **5 task** in una sola chiamata LLM:
  - `task_1` → `fetch_order_summary` (order_id='ORD-42')
  - `task_2` → `run_risk_audit` (order_value_eur ◀── task_1.order_value_eur, customer_tier_code ◀── task_1.customer_tier_code)
  - `task_3` → `fetch_warehouse_stock` (sku='SKU-X9')
  - `task_4` → `compute_restock_cost` (missing_units ◀── task_3.missing_units, unit_cost_eur ◀── task_3.unit_cost_eur)
  - `task_5` → `build_executive_score` (risk_score ◀── task_2.risk_score, restock_cost_eur ◀── task_4.restock_cost_eur)
  - *Nota didattica: i simboli `◀──` sono `InputBinding`: dichiarano il dataflow prima che parta un solo tool.*

**[Step 2 · Round 1] PLAN COMPILER (0 LLM, 0.13 ms)**
- **APPROVATO**: `CompiledDAG` valido (5 nodi, 3 livelli, 0 errori statici)
  - Livello `L0`: `task_1:fetch_order_summary, task_3:fetch_warehouse_stock`
  - Livello `L1`: `task_2:run_risk_audit, task_4:compute_restock_cost`
  - Livello `L2`: `task_5:build_executive_score`

**[Step 3 · Round 1] EXECUTOR (0 LLM, scheduler asyncio event-driven)**
- Wall-clock DAG: `2.40s` | Somma durate task: `4.51s` | Parallelismo effettivo: `1.87x`
  - `[OK]` `task_1` (`fetch_order_summary`) `0.00s → 0.30s` → `{'order_id': 'ORD-42', 'order_value_eur': 12400.0, 'customer_tier_code': 'TIER-B'}`
  - `[OK]` `task_3` (`fetch_warehouse_stock`) `0.00s → 1.80s` → `{'sku': 'SKU-X9', 'missing_units': 35, 'unit_cost_eur': 48.0}`
  - `[OK]` `task_2` (`run_risk_audit`) `0.30s → 2.10s` → `{'risk_score': 40.73}`
  - `[OK]` `task_4` (`compute_restock_cost`) `1.80s → 2.10s` → `{'restock_cost_eur': 1976.6}`
  - `[OK]` `task_5` (`build_executive_score`) `2.10s → 2.40s` → `{'priority_index': 76.91, 'recommendation': 'ESCALATE_IMMEDIATELY'}`

**[Step 4 · Round 1] JOINER (chiamata LLM + stato deterministico)**
- **DECISIONE**: `action='finish'` | `status='complete'`

**Risposta finale LLM Compiler v3:**
> L'ordine 'ORD-42' ha un risk_score di 40.73. Il prodotto 'SKU-X9' ha un restock_cost_eur di 1976.6 EUR. Il priority_index è 76.91 e la raccomandazione è 'ESCALATE_IMMEDIATELY'.


### Parte B — LangGraph ReAct (turno per turno)




**[Turno 1 · AGENT] (chiamata LLM #1)**
- Decide sul momento di chiamare **2 tool**:
  - `fetch_order_summary({'order_id': 'ORD-42'})`
  - `fetch_warehouse_stock({'sku': 'SKU-X9'})`
  - *Nota didattica: non può ancora pianificare i passi successivi perché non ha un `InputBinding` esplicito tra tool.*

**[Turno 1 · TOOLS] (esegue il batch e attende tutto il batch)**
  - `[OK]` `fetch_order_summary` → `{"customer_tier_code": "TIER-B", "order_id": "ORD-42", "order_value_eur": 12400.0}`
  - `[OK]` `fetch_warehouse_stock` → `{"missing_units": 35, "sku": "SKU-X9", "unit_cost_eur": 48.0}`
  - *Nota didattica: anche se `fetch_order_summary` finisce in 0.30s, il turno resta bloccato finché non termina anche `fetch_warehouse_stock` (1.80s).*

**[Turno 2 · AGENT] (chiamata LLM #2)**
- Decide sul momento di chiamare **2 tool**:
  - `run_risk_audit({'customer_tier_code': 'TIER-B', 'order_value_eur': 12400})`
  - `compute_restock_cost({'missing_units': 35, 'unit_cost_eur': 48})`

**[Turno 2 · TOOLS] (esegue il batch e attende tutto il batch)**
  - `[OK]` `run_risk_audit` → `{"risk_score": 40.73}`
  - `[OK]` `compute_restock_cost` → `{"restock_cost_eur": 1976.6}`

**[Turno 3 · AGENT] (chiamata LLM #3)**
- Decide sul momento di chiamare **1 tool**:
  - `build_executive_score({'risk_score': 40.73, 'restock_cost_eur': 1976.6})`

**[Turno 3 · TOOLS] (esegue il batch e attende tutto il batch)**
  - `[OK]` `build_executive_score` → `{"priority_index": 76.91, "recommendation": "ESCALATE_IMMEDIATELY"}`

**[Turno 4 · AGENT] (chiamata LLM #4)**
- Nessuna altra tool call → scrive la risposta finale.

**Risposta finale LangGraph ReAct:**
> L'analisi dell'ordine e dello stock è completata. Ecco i risultati:
> 
> - **risk_score**: 40.73  
> - **restock_cost_eur**: 1976.60 EUR  
> - **priority_index**: 76.91  
> - **recommendation**: ESCALATE_IMMEDIATELY  
> 
> Si consiglia di agire immediatamente per gestire l'ordine e il restock del prodotto.


### Confronto visivo e lezione didattica

```text
TIMELINE REALE A CONFRONTO (stessa scala: 0 -> 11.58 s)
Legenda: ▓ = chiamata LLM | █ = esecuzione Tool OK | ✖ = Tool in errore

  LLM Compiler v3 (0 → 11.58 s)
    [LLM ] Compiler LLM #1          │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓·················│  0.05s →  7.55s (7.50s)
    [TOOL] fetch_order(ORD-42)      │·······························█················│  7.56s →  7.86s (0.30s)
    [TOOL] fetch_stock(SKU-X9)      │·······························███████··········│  7.56s →  9.36s (1.80s)
    [TOOL] run_risk_audit           │································████████········│  7.86s →  9.66s (1.80s)
    [TOOL] compute_restock_cost     │······································██········│  9.36s →  9.66s (0.30s)
    [TOOL] build_executive_score    │········································█·······│  9.67s →  9.97s (0.30s)
    [LLM ] Compiler LLM #2          │·········································▓▓▓▓▓▓·│  9.97s → 11.58s (1.61s)

  LangGraph ReAct (0 → 11.58 s)
    [LLM ] ReAct LLM #1             │▓▓··············································│  0.00s →  0.70s (0.70s)
    [TOOL] fetch_order(ORD-42)      │··██············································│  0.71s →  1.01s (0.30s)
    [TOOL] fetch_stock(SKU-X9)      │··████████······································│  0.71s →  2.51s (1.80s)
    [LLM ] ReAct LLM #2             │··········▓▓▓▓··································│  2.51s →  3.56s (1.04s)
    [TOOL] compute_restock_cost     │··············██································│  3.56s →  3.86s (0.30s)
    [TOOL] run_risk_audit           │··············████████··························│  3.56s →  5.36s (1.80s)
    [LLM ] ReAct LLM #3             │······················▓▓························│  5.37s →  5.88s (0.51s)
    [TOOL] build_executive_score    │························█·······················│  5.89s →  6.19s (0.30s)
    [LLM ] ReAct LLM #4             │·························▓▓▓▓▓▓·················│  6.19s →  7.55s (1.36s)
```

| Metrica | LLM Compiler v3 | LangGraph ReAct | Differenza |
| :--- | ---: | ---: | :--- |
| Risultato esatto | `EXACT` | `EXACT` | Parità |
| Chiamate LLM | `2` | `4` | -2 chiamate |
| Tempo I/O Tool (unione reale) | `2.40 s` | `3.91 s` | -1.50 s |
| Tempo totale LLM | `9.11 s` | `3.62 s` | +5.49 s |
| Wall-clock totale | `11.58 s` | `7.55 s` | +4.03 s |
| Token (In / Out / Tot) | `2571/620/3191` | `3671/262/3933` | -742 tot |
| Validazione statica pre-run | `Sì (PlanCompiler)` | `No (scopre a runtime)` | DAG certificato prima di eseguire |
| Retry / Replan | `0 compile / 0 replan` | `Loop nativo a turni` | — |


**Cosa ci insegna questo scenario:**
1. **Parallelismo event-driven reale**: LLM Compiler ha eseguito l'intero DAG di tool in `2.40s` contro `3.91s` di ReAct (`+1.50s` risparmiati sull'I/O), perché `run_risk_audit` è partito appena finito `fetch_order_summary` senza aspettare `fetch_warehouse_stock`.
2. **Meno chiamate LLM**: `2` chiamate LLM (`Planner` + `Joiner`) contro `4` in ReAct.
3. **Trade-off token (Output JSON vs Input ripetuti)**: il Planner genera più token di output (`620` vs `262`), ma evita di rispedire tutta la cronologia a ogni turno (`2571` input vs `3671`).


---

## Scenario 2 — L'Imprevisto a Runtime (Errore 503 + Tool di Backup)


**Obiettivo:** Confrontare il recupero strutturato di LLM Compiler v3 (best-effort + skipped + replan_context) con l'adattamento reattivo turno-per-turno di ReAct.

```text
  shipping_quote_primary('EU-NORTH')  ──▶ OK (42.50 EUR)
  shipping_quote_primary('LEGACY-99') ──▶ ERRORE 503! (backup: 87.50 EUR)
  calculate_invoice_total(42.50, 87.50) ──▶ subtotal 130.00 + fee 9.50 = 139.50 EUR
```

**Richiesta identica per entrambi:**
> Recupera le tariffe di spedizione per le zone 'EU-NORTH' e 'LEGACY-99' usando prima shipping_quote_primary (se fallisce, usa il backup disponibile), poi calcola la fattura ufficiale con calculate_invoice_total e riporta subtotal_eur, insurance_fee_eur e grand_total_eur in italiano.

### Parte A — LLM Compiler v3 (passo dopo passo)




**[Step 1 · Round 1] PLANNER (chiamata LLM)**
- Obiettivo: `Recupera le tariffe di spedizione per le zone 'EU-NORTH' e 'LEGACY-99' usando prima shipping_quote_primary (se fallisce, usa il backup disponibile), poi calcola la fattura ufficiale con calculate_invoice_total e riporta subtotal_eur, insurance_fee_eur e grand_total_eur in italiano.`
- Ha pianificato **6 task** in una sola chiamata LLM:
  - `task_1` → `shipping_quote_primary` (zone_code='EU-NORTH')
  - `task_2` → `shipping_quote_primary` (zone_code='LEGACY-99')
  - `task_3` → `shipping_quote_backup` (zone_code='EU-NORTH')
  - `task_4` → `shipping_quote_backup` (zone_code='LEGACY-99')
  - `task_5` → `calculate_invoice_total` (cost_a_eur ◀── task_1.shipping_eur, cost_b_eur ◀── task_2.shipping_eur)
  - `task_6` → `calculate_invoice_total` (cost_a_eur ◀── task_3.shipping_eur, cost_b_eur ◀── task_4.shipping_eur)
  - *Nota didattica: i simboli `◀──` sono `InputBinding`: dichiarano il dataflow prima che parta un solo tool.*

**[Step 2 · Round 1] PLAN COMPILER (0 LLM, 0.12 ms)**
- **APPROVATO**: `CompiledDAG` valido (6 nodi, 2 livelli, 0 errori statici)
  - Livello `L0`: `task_1:shipping_quote_primary, task_2:shipping_quote_primary, task_3:shipping_quote_backup, task_4:shipping_quote_backup`
  - Livello `L1`: `task_5:calculate_invoice_total, task_6:calculate_invoice_total`

**[Step 3 · Round 1] EXECUTOR (0 LLM, scheduler asyncio event-driven)**
- Wall-clock DAG: `0.70s` | Somma durate task: `1.86s` | Parallelismo effettivo: `2.64x`
  - `[OK]` `task_1` (`shipping_quote_primary`) `0.00s → 0.35s` → `{'zone_code': 'EU-NORTH', 'shipping_eur': 42.5, 'carrier': 'FastEuro'}`
  - `[ERR]` `task_2` (`shipping_quote_primary`) `0.00s → 0.35s` → `503 Service Unavailable for zone 'LEGACY-99' on primary carrier. Use 'shipping_quote_backup' instead.`
  - `[OK]` `task_3` (`shipping_quote_backup`) `0.00s → 0.45s` → `{'zone_code': 'EU-NORTH', 'shipping_eur': 46.0, 'carrier': 'SafeBackup'}`
  - `[OK]` `task_4` (`shipping_quote_backup`) `0.00s → 0.45s` → `{'zone_code': 'LEGACY-99', 'shipping_eur': 87.5, 'carrier': 'SafeBackup'}`
  - `[OK]` `task_6` (`calculate_invoice_total`) `0.45s → 0.70s` → `{'subtotal_eur': 133.5, 'insurance_fee_eur': 9.68, 'grand_total_eur': 143.18}`
  - `[SKIP]` `task_5` (`calculate_invoice_total`) → *Skipped because upstream task 'task_2' failed.*

**[Step 4 · Round 1] JOINER (chiamata LLM + stato deterministico)**
- **DECISIONE**: `action='replan'` → Avvio **Round 2**
- Motivo: *La tariffa di spedizione per la zona 'LEGACY-99' è stata recuperata correttamente tramite shipping_quote_backup, ma la fattura ufficiale non è stata calcolata a causa del fallimento di task_2. Ora che la tariffa per 'LEGACY-99' è disponibile, è necessario richiamare calculate_invoice_total con cost_a_eur=42.5 (da EU-NORTH) e cost_b_eur=87.5 (da LEGACY-99) per ottenere subtotal_eur, insurance_fee_eur e grand_total_eur.*
- *Nota didattica: i risultati già riusciti vengono salvati in `replan_context`, così il Round 2 non ripeterà le chiamate già riuscite.*

**[Step 5 · Round 2] PLANNER (chiamata LLM)**
- Obiettivo: `Calcola la fattura ufficiale utilizzando le tariffe di spedizione per 'EU-NORTH' e 'LEGACY-99' e riporta subtotal_eur, insurance_fee_eur e grand_total_eur in italiano.`
- Ha pianificato **1 task** in una sola chiamata LLM:
  - `task_1` → `calculate_invoice_total` (cost_a_eur=42.5, cost_b_eur=87.5)
  - *Nota didattica: i simboli `◀──` sono `InputBinding`: dichiarano il dataflow prima che parta un solo tool.*

**[Step 6 · Round 2] PLAN COMPILER (0 LLM, 0.16 ms)**
- **APPROVATO**: `CompiledDAG` valido (1 nodi, 1 livelli, 0 errori statici)
  - Livello `L0`: `task_1:calculate_invoice_total`

**[Step 7 · Round 2] EXECUTOR (0 LLM, scheduler asyncio event-driven)**
- Wall-clock DAG: `0.25s` | Somma durate task: `0.25s` | Parallelismo effettivo: `1.00x`
  - `[OK]` `task_1` (`calculate_invoice_total`) `0.00s → 0.25s` → `{'subtotal_eur': 130.0, 'insurance_fee_eur': 9.5, 'grand_total_eur': 139.5}`

**[Step 8 · Round 2] JOINER (chiamata LLM + stato deterministico)**
- **DECISIONE**: `action='finish'` | `status='complete'`

**Risposta finale LLM Compiler v3:**
> Le informazioni relative alla fattura ufficiale sono le seguenti: il subtotal_eur è di 130,00 euro, l'insurance_fee_eur è di 9,50 euro e il grand_total_eur è di 139,50 euro.


### Parte B — LangGraph ReAct (turno per turno)




**[Turno 1 · AGENT] (chiamata LLM #1)**
- Decide sul momento di chiamare **2 tool**:
  - `shipping_quote_primary({'zone_code': 'EU-NORTH'})`
  - `shipping_quote_primary({'zone_code': 'LEGACY-99'})`

**[Turno 1 · TOOLS] (esegue il batch e attende tutto il batch)**
  - `[OK]` `shipping_quote_primary` → `{"carrier": "FastEuro", "shipping_eur": 42.5, "zone_code": "EU-NORTH"}`
  - `[ERR]` `shipping_quote_primary` → `Error: RuntimeError("503 Service Unavailable for zone 'LEGACY-99' on primary carrier. Use 'shipping_quote_backup' instead.")
 Please fix your mistakes.`

**[Turno 2 · AGENT] (chiamata LLM #2)**
- Decide sul momento di chiamare **2 tool**:
  - `shipping_quote_backup({'zone_code': 'LEGACY-99'})`
  - `calculate_invoice_total({'cost_a_eur': 42.5, 'cost_b_eur': 35})`

**[Turno 2 · TOOLS] (esegue il batch e attende tutto il batch)**
  - `[OK]` `shipping_quote_backup` → `{"carrier": "SafeBackup", "shipping_eur": 87.5, "zone_code": "LEGACY-99"}`
  - `[OK]` `calculate_invoice_total` → `{"grand_total_eur": 84.38, "insurance_fee_eur": 6.88, "subtotal_eur": 77.5}`

**[Turno 3 · AGENT] (chiamata LLM #3)**
- Nessuna altra tool call → scrive la risposta finale.

**Risposta finale LangGraph ReAct:**
> Le tariffe di spedizione e i dettagli della fattura ufficiale sono i seguenti:
> 
> - **Subtotale (subtotal_eur):** 77,50 €  
> - **Frequenza assicurativa (insurance_fee_eur):** 6,88 €  
> - **Totale (grand_total_eur):** 84,38 €  
> 
> Questi valori sono calcolati utilizzando i costi di spedizione per le zone 'EU-NORTH' e 'LEGACY-99'.


### Confronto visivo e lezione didattica

```text
TIMELINE REALE A CONFRONTO (stessa scala: 0 -> 15.69 s)
Legenda: ▓ = chiamata LLM | █ = esecuzione Tool OK | ✖ = Tool in errore

  LLM Compiler v3 (0 → 15.69 s)
    [LLM ] Compiler LLM #1          │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓························│  0.00s →  8.17s (8.16s)
    [TOOL] primary(EU-NORTH)        │·························█······················│  8.17s →  8.53s (0.35s)
    [ERR ] primary(LEGACY-99)       │·························✖······················│  8.17s →  8.53s (0.35s)
    [TOOL] backup(EU-NORTH)         │·························█······················│  8.17s →  8.62s (0.45s)
    [TOOL] backup(LEGACY-99)        │·························█······················│  8.17s →  8.62s (0.45s)
    [TOOL] calculate_invoice_total  │··························█·····················│  8.63s →  8.88s (0.25s)
    [LLM ] Compiler LLM #2          │···························▓▓▓▓▓▓▓··············│  8.88s → 11.32s (2.43s)
    [LLM ] Compiler LLM #3          │··································▓▓▓▓▓▓▓▓······│ 11.32s → 13.87s (2.55s)
    [TOOL] calculate_invoice_total  │··········································█·····│ 13.88s → 14.13s (0.25s)
    [LLM ] Compiler LLM #4          │···········································▓▓▓▓·│ 14.14s → 15.68s (1.54s)

  LangGraph ReAct (0 → 15.69 s)
    [LLM ] ReAct LLM #1             │▓▓··············································│  0.00s →  0.73s (0.73s)
    [TOOL] primary(EU-NORTH)        │··█·············································│  0.74s →  1.09s (0.35s)
    [ERR ] primary(LEGACY-99)       │··✖·············································│  0.74s →  1.09s (0.35s)
    [LLM ] ReAct LLM #2             │···▓▓···········································│  1.09s →  1.82s (0.73s)
    [TOOL] calculate_invoice_total  │·····█··········································│  1.83s →  2.08s (0.25s)
    [TOOL] backup(LEGACY-99)        │·····█··········································│  1.83s →  2.28s (0.45s)
    [LLM ] ReAct LLM #3             │······▓▓▓▓▓·····································│  2.28s →  3.85s (1.57s)
```

| Metrica | LLM Compiler v3 | LangGraph ReAct | Differenza |
| :--- | ---: | ---: | :--- |
| Risultato esatto | `EXACT` | `MISS` | Vince chi è EXACT |
| Chiamate LLM | `4` | `3` | +1 chiamate |
| Tempo I/O Tool (unione reale) | `0.95 s` | `0.80 s` | +0.15 s |
| Tempo totale LLM | `14.69 s` | `3.03 s` | +11.66 s |
| Wall-clock totale | `15.69 s` | `3.85 s` | +11.84 s |
| Token (In / Out / Tot) | `6183/1005/7188` | `2182/226/2408` | +4780 tot |
| Validazione statica pre-run | `Sì (PlanCompiler)` | `No (scopre a runtime)` | DAG certificato prima di eseguire |
| Retry / Replan | `0 compile / 1 replan` | `Loop nativo a turni` | — |


**Cosa ci insegna questo scenario:**
1. **Gestione sicura del guasto in LLM Compiler v3**: `calculate_invoice_total` viene marcato `SKIPPED`, il risultato già riuscito di `EU-NORTH` viene preservato in `replan_context`, e il Round 2 esegue solo il backup + il totale.
2. **Punto di forza naturale di ReAct**: su imprevisti runtime, ReAct si adatta al turno successivo con soli `226` token di output totali (contro `1005` di Compiler v3).
3. **Sintesi onesta**: LLM Compiler v3 brilla sui flussi multi-step paralleli e asimmetrici (Scenario 1); ReAct brilla nei flussi brevi, veloci ed esplorativi (Scenario 2).

