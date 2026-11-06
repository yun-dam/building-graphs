# Study 1 (Updated, v2). GNN-Grounded Question Answering over Building Knowledge Graphs: Evolving BrickQA

> This update revises `research_plan_1.md` based on the four works in `relevantWorks/`:
>
> - **BrickQA**: Ko et al., BuildSys '26. Our current system.
> - **Build2SPARQL**: Jung, 2026. The training corpus.
> - **BuildingQA**: Mulayim et al., BuildSys '25. The final benchmark.
> - **BSchema**: "Enabling Zero-Shot KGQA in Smart Buildings…", Elsevier preprint, 2026. The new state of the art on BuildingQA.
>
> It also draws on the BrickQA code and per-query CSVs, the BuildingQA repo (metrics, baseline CSVs, `bschema/` outputs), and the Build2SPARQL v1.0 release.
>
> **What changed in v2, after reading BSchema:**
>
> 1. BSchema, not BrickQA, is now the system to beat.
> 2. The proposed method moves from a dynamic-tool design to a hybrid of a static summary, a learned question-conditioned selection over it, and dynamic tools.
> 3. The GNN is positioned as a learned, question-conditioned generalization of BSchema's Weisfeiler–Lehman-style structural summarization.
> 4. Baselines, LLMs, targets, failure taxonomy and the case study are realigned.

---

## 0. Executive Summary

**Decision.** Evolve BrickQA; do not restart from scratch. Its *context mechanism* changes:

- **Old context** (raw first-N triples plus tool exploration) is replaced by **BSchema structural summaries**.
- **A trained GNN selects and expands** those summaries per question.
- BrickQA's decomposition, execution-feedback loop and validators are kept.
- Framework name: **BRIDGE** (**B**uilding **R**etrieval via **I**nductive, **D**omain-general **G**raph **E**ncoding).

**Why, from a benchmark point of view (details in §2):**

1. **The state of the art has moved.** On BuildingQA, Row-Matching F1, question-weighted:

   | System | LLM | All 4 buildings | 3 Brick buildings | Cost per question |
   |---|---|---:|---:|---:|
   | BSchema | Gemma 4 31B Thinking | **0.59** | 0.69 | $0.006 |
   | BSchema tool-based variant | Gemma 4 31B Thinking | 0.50 | — | — |
   | BrickQA 5000T | Gemini 2.5 Flash | not evaluated on B59 | 0.597 | $0.026 |
   | ReAct (best) | o3-mini | 0.38 | — | — |

   The paper below must beat BSchema, not only BrickQA.
2. **The systems are complementary per building.** Row-F1 for each:

   | Building | BrickQA 5000T | BSchema (Gemma) |
   |---|---:|---:|
   | TUC | **0.989** | 0.88 |
   | DFLEX | **0.676** | 0.66 |
   | Mortar/BLDG11 | 0.405 | **0.63** (tool-based: 0.69) |
   | LBNL/B59 (223P) | — | **0.30** |

   BrickQA's iterative, decomposition-driven loop wins where the graph is small. BSchema's static topology map wins where topology is complex or unfamiliar (Mortar, 223P). A system that combines both has a credible path past 0.59.
3. **BSchema names our method as its open problem.**
   - Its main limitation is that summaries must fit in context. LBNL already had to be compressed to θ = 0.3, with θ chosen per building by graph size, and summaries for large or less repetitive buildings will not fit.
   - Its stated future work is a "hybrid … seeding an agent with a compact structural summary while granting it exploration tools".
   - A trained, question-conditioned GNN that picks which parts of the summary matter, and expands to instances on demand, is exactly that hybrid, with a learned selector instead of a hand-set θ.
4. **Starting from scratch is not justified by the benchmark.** The two strongest systems are both agentic loops with execution feedback. An end-to-end fine-tuned generator would learn Build2SPARQL's simple surface distribution:
   - 81.6% linear queries;
   - a mean of 3.9 BGP triples.

   BuildingQA's expert queries are much larger: up to 53 triple patterns and up to 16 projections. Learned *grounding* (where things are in this building) transfers; learned *query strings* do not.

**Primary claim to test.** A question-conditioned GNN over BSchema summaries and instance graphs, trained only on Build2SPARQL, does three things:

1. It reduces vocabulary hallucination and topological mismatch beyond static BSchema.
2. It removes the context-size ceiling, with no per-building θ.
3. It plans answer structure.

Together these give the best BuildingQA results on all four buildings at BSchema-level cost.

---

## 1. Evidence Base

### 1.1 BuildingQA results to date (Row-F1 unless noted)

| System (LLM) | DFLEX | Mortar/BLDG11 | TUC | LBNL/B59 (223P) | 3 Brick bldgs, question-weighted | All 4, question-weighted | Exact-F1 | $/question |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ReAct 5000T (o3-mini) | 0.61 | 0.27 | 0.86 | 0.06 | — | 0.378 | 0.00 | ~0.064 |
| BrickQA 5000T (Gemini 2.5 Flash, 1 run) | **0.676** | 0.405 | **0.989** | n/a | 0.597 (macro 0.690) | n/a | 0.437 (macro) | 0.026 |
| BrickQA 100T (Gemini 2.5 Flash) | 0.472 | 0.361 | 0.933 | n/a | 0.510 | n/a | 0.390 | 0.005 |
| BSchema + validator (Gemma 4 31B Thinking) | 0.66 | 0.63 | 0.88 | **0.30** | **0.690** (macro 0.723) | **0.595** | ~0.00 | **0.006** |
| BSchema tool-based variant (Gemma 4) | 0.67 | **0.69** | 0.51 | 0.04 | — | 0.498 | 0.00 | 0.019 |
| Validator-only (Gemma 4) | 0.32 | 0.52 | 0.05 | 0.04 | — | 0.290 | 0.01 | 0.023 |
| BSchema (Nemotron Super 3) | 0.40 | 0.34 | 0.33 | 0.07 | — | 0.283 | 0.04 | 0.004 |

**Observations that shape the design:**

- **Confounds in the BrickQA vs BSchema comparison.** Model, number of runs and building coverage all differ (Gemini 2.5 Flash, 1 run, 3 buildings vs Gemma 4 31B, about 6 runs, 4 buildings). P0 (§9) removes these confounds by re-running both systems under the same LLMs.
- **Exact-Match F1.** BrickQA has non-trivial Exact-F1 (0.44 macro); BSchema has about 0. Answer-structure planning is a differentiator worth keeping.
- **Where BSchema wins on LBNL.** The gain (0.30 vs 0.04) comes mostly from *budget exhaustion* in the tool-based agent: 60–74 tool calls, about 246k tokens, and only 15% of runs complete. Static topology removes this. Our dynamic part must stay bounded in turns and tokens.
- **The TUC example (TUC_005).** Static topology suppresses "sounds-right" predicate substitution. The tool-based agent wrote `?sensor isPointOf ?eq . ?eq hasLocation ?zone` instead of the real `?zone hasPart ?space . ?space isLocationOf/hasPoint ?temp`.

### 1.2 Unified failure taxonomy (BSchema §3 + BrickQA §5)

BSchema analysed 2,632 BuildingQA baseline queries, of which 1,455 failures could be diagnosed:

| Failure | Share (BSchema) | Also reported in BrickQA as | Proposed fix in BRIDGE |
|---|---:|---|---|
| Vocabulary hallucination (term) | **79.4%** | (implicit in sensor linking) | GNN schema linker returns **exact URIs present in this graph**; ontology validator |
| Namespace hallucination (subset) | 30.1% | — | Prefix rebinding from the graph's actual namespaces (versioned Brick, QUDT, 223P); URIs passed whole, never re-typed |
| Topological mismatch | **15.6%** | Semantic traversal failure (76–100% of BLDG11 failures) | GNN **path retrieval** over BSchema / instance graph; path validator (first failing triple pattern, as in BSchema's validator) |
| Incorrect SPARQL syntax | 8.5% | — | BrickQA syntax fixer + validator (kept) |
| Identifier guessing | 6.0% | — | Instance linking via GNN-ranked `search_graph`; member-graph lookup |
| Misunderstanding of request (not measurable from KG) | — | Arity mismatch (78–100%), granularity mismatch | **Projection planner** + BrickQA decomposition; hierarchy-aware schema linking |

### 1.3 Data facts that constrain the design

**Build2SPARQL v1.0**

- Scale and composition:
  - 201 buildings (180 Brick, 21 223P).
  - 6,136 queries and 30,680 questions, in registers Q1–Q5.
  - Patterns: 81.6% linear, 13.3% branching, and small shares of aggregation, filtered, UNION and OPTIONAL.
  - The release ships a 173/28 `split.json`, but it is **not used**: it leaks duplicates across train/test. We define our own leakage-free split (§3.4).
- **No graphs and no answers are included.** The source graphs must be downloaded (Mortar, Brick examples, open223, BuildGraph on Zenodo), and gold results produced by execution. The TBox must be loaded, because 98% of queries use `rdf:type/rdfs:subClassOf*`.
- **Leakage:**
  - `bldg11` (= BuildingQA Mortar) is **excluded**.
  - **B59 audit (done).** `lbnl-bdg3-1` is a different building from B59: 533 vs 76 PhysicalSpaces, 1,144 vs 1,510 ConnectionPoints. `lbnl-example-radiant` is a small example model. No Build2SPARQL building duplicates TUC or DFLEXLIBS.
  - **Exact duplicates found** (identical query-signature sets; one copy dropped):
    - HPL = NIST-HPL;
    - IBAL = NIST-IBAL;
    - bldg30 = smc;
    - multiple-submeters = virtual-meter.

**BuildingQA**

- 4 buildings and 188 questions: TUC 30, DFLEX 36, Mortar 76, LBNL B59 46.
- No aggregation queries.
- Many questions ask for timeseries IDs or external references.
- Primary metric: Row-F1 (`metrics.py`). A prediction with fewer columns than the gold scores 0.

**BSchema artifacts already in the repo** (`data/BuildingQA/bschema/`)

- Summaries exist at θ ∈ {0, 0.3, 0.5, 0.7, full} for all 4 eval buildings and for `bldg_brick_anon` / `bldg_223p_anon`.
- Example: LBNL is 38,982 triples, reduced to 1,320 at θ = 0.3 and 20,699 at full.
- BSchema also outputs a **member graph** that maps summary nodes back to instances. This lets a summary-level GNN expand to instances.

---

## 2. Decision Analysis: Keep BrickQA, or Start a New Framework?

### 2.1 Options

| Option | Description | Expected BuildingQA outcome | Risk |
|---|---|---|---|
| A. BrickQA + BSchema context (no learning) | Replace first-N triples with BSchema at a fixed θ | Probably about BSchema-level (0.59–0.65) | Low novelty; inherits the context ceiling and per-building θ; no GNN |
| B. **Evolve BrickQA → BRIDGE (chosen)** | BrickQA loop + BSchema summary + **trained GNN** that selects and expands summary/instance context per question + projection planner | Targets the remaining gap on every building (LBNL 0.30, Mortar 0.63) and keeps BrickQA's TUC/DFLEX and Exact-F1 strengths | Moderate: GNN supervision via executed Build2SPARQL queries (§3.3) |
| C. New end-to-end framework (GNN soft-token LLM fine-tuned on Build2SPARQL) | One model encodes the graph and decodes SPARQL | Strong in distribution; weak on long expert queries and 223P | High: distribution shift; loses execution feedback; LLM fine-tuning cost |
| D. Adopt BSchema's agent as the base, add GNN | BSchema + validator + GNN selection | Similar to B on accuracy | Builds our paper on another group's system; loses BrickQA's decomposition and Exact-F1 advantage; weaker story as a continuation of our own line of work |

### 2.2 Why B, on benchmark grounds

1. **It combines the per-building winners.** Neither existing system is best everywhere (§1.1). B keeps BrickQA's loop (TUC, DFLEX, Exact-F1) and imports BSchema's static topology (Mortar, LBNL).
2. **The GNN has a precise, defensible role.**
   - BSchema's iterative reclassification (extract the 1-hop class graph → group equivalent ones → relabel → repeat) is a **1-WL / color-refinement procedure**. It is non-learned, question-agnostic and thresholded.
   - Message-passing GNNs are exactly the learned, differentiable counterpart of 1-WL.
   - BRIDGE therefore replaces "one summary for every question at a hand-set θ" with "a **question-conditioned** soft summary": GNN relevance over BSchema nodes, expanded through the member graph to instances only where needed.
   - This is novel relative to BSchema. It also directly removes the per-building θ selection, which BSchema made by graph size and which is effectively a test-time hyperparameter.
3. **It scales where BSchema cannot.** For large or less repetitive buildings (our real large office building; `bldg_brick_anon` with 173k ABox triples, whose BSchema at θ = 0.5 is about 25k triples), the GNN selects a token-bounded subset. This addresses BSchema's acknowledged limitation.
4. **Option C stays as a comparison arm** (§6.2). Option A is the **key ablation**: "BrickQA + BSchema, no GNN" isolates the value of learning.

### 2.3 What changes in the BrickQA code base

| Component | Fate |
|---|---|
| `brick_decomposer.py` | **Keep**. Sensor-linking proposals are re-ranked and verified by the GNN schema linker, and URIs are snapped to the graph's actual vocabulary. |
| `brick_agent.py` controller | **Keep**, with a **turn and token budget** (lesson from BSchema's LBNL analysis). Gains GNN-backed tools and a pre-loaded, question-conditioned BSchema context block. |
| `search_brick` (substring) | **Replace** with GNN- and embedding-ranked `search_graph`. |
| Prompt TTL samples, LLM building description | **Replace** with question-conditioned BSchema selection. Raw TTL is kept only for baseline reproduction. |
| Validators | **Keep and extend**: path validator (BFS to the first failing triple pattern, as in BSchema), ontology/namespace validator, arity validator. |
| LLM backend (Gemini via Vertex only) | **Generalize**: Gemini 2.5 Flash (BrickQA continuity) + Gemma 4 31B Thinking (BSchema continuity, open weights). |

---

## 3. Data Plan

### 3.1 Roles of each dataset

| Dataset | Role |
|---|---|
| Build2SPARQL train (154 buildings, 4,670 queries; own split, §3.4) | **GNN training**, few-shot pool |
| Build2SPARQL val (9 real buildings, 256 queries) | Model and budget selection |
| Build2SPARQL test (33 buildings, 1,012 queries, incl. 6 223P) | Test on unseen buildings and an unseen building type; retriever-only metrics |
| BuildingQA (4 buildings, 188 questions) | **Final benchmark**; never used for training or selection; all 4 buildings reported |
| BSchema summaries (repo + regenerated for all Build2SPARQL buildings) | Summary-level graph view for the GNN (§4.1) |
| Own real large office building (Brick + 223P) | Case study and external validation (§7) |
| `bldg_brick_anon` / `bldg_223p_anon` | Scale stress test (no QA). `bldg_223p_anon` is the only 223P graph in the repo besides B59, and it has no QA pairs. Independent of our own building (confirmed). |

### 3.2 Graph preprocessing

1. **Download and resolve** all source graphs, and log any that fail. Load the matching TBox version.
2. **Build three views per building:**
   - **(a) ABox instance graph**, using `remove-ontology.py`.
   - **(b) BSchema summary graph + member graph.** Run `create-bschema.py` for every Build2SPARQL building at θ ∈ {0, 0.3, 0.7, 1.0}. This is cheap: near-linear time, seconds to minutes per building.
   - **(c) TBox class hierarchy** with definitions.
3. **Normalize node text.** Split opaque identifiers, keep labels and external-reference IDs, and record each graph's **actual namespaces** for prefix rebinding.

### 3.3 GNN supervision from Build2SPARQL (by execution, no manual labels)

For each (question, gold SPARQL, building):

1. **Execute** the gold SPARQL to get the gold result rows and answer nodes.
2. **Bind all BGP variables** to get the gold *instance* evidence subgraph. Map it through the member graph to get the gold **BSchema-node evidence**.
3. **Label from the query AST:**
   - gold schema terms (exact URIs);
   - the gold class-level path skeleton;
   - the gold projection set.
4. **Seed mentions** come from Q4 (instance-grounded) questions, for instance-linking labels.

**Filters:**

- drop queries with zero rows;
- down-weight implausible 223P connection-topology queries;
- cap per-building query counts.

**Augmentation for BuildingQA-like structure.** Synthesize multi-projection, branching and OPTIONAL queries over each building's class index, using Build2SPARQL's own mining procedure with larger projection sets. This creates projection-planner and long-path supervision.

### 3.4 Splits

We define our own **leakage-free, group-level split**. It is reproducible with `data/splits/make_split.py` and written to `data/splits/brickqa_g_split.json` (seed 42).

**Construction rules**

1. **Exclude** BuildingQA buildings (`bldg11`).
2. **Deduplicate** exact duplicates (query-signature Jaccard = 1.0), keeping one copy. Dropped: NIST-HPL, NIST-IBAL, smc, virtual-meter.
3. **Group** buildings that must never be separated:
   - **Data-driven:** union-find over pairs with query-signature Jaccard ≥ 0.8. Signature = SPARQL with variable names normalized.
   - **Source families:**
     - BuildGraph by building type (all vintages and instances). Types linked by a near-duplicate pair are merged, which gives four groups:
       - hospital + large office + medium office;
       - hotel + outpatient;
       - small office + strip mall;
       - primary school.
     - Guideline 36 (Brick; 223P), meter examples, PNNL (3 buildings), LBNL (2 buildings).
     - Data-driven pairs: bldg13 + bldg4, bldg14 + bldg3.
4. **Stratify** by ontology/corpus (BuildGraph, Mortar-Brick, Brick-reference, 223P).
   - Groups are assigned greedily toward ~70/10/20 by query count.
   - The 223P stratum is assigned by family, so test covers distinct real-building sources.

**Resulting split** (196 buildings after exclusion and deduplication)

| Split | Buildings | Queries | BuildGraph | Mortar-Brick | Brick-ref | 223P |
|---|---:|---:|---:|---:|---:|---:|
| Train | 154 | 4,670 | 105 bldgs (4 types' groups) | 30 | 8 | 11 (Guideline 36 ×7, HPL, nrel-example, design-patterns, rvav) |
| Val | 9 | 256 | 0 | 5 | 2 | 2 (IBAL, nist-bdg1-1) |
| Test | 33 | 1,012 | 15 (primary school, an unseen type) | 10 | 2 | 6 (PNNL ×3, LBNL ×2, scb-vrf) |

**Verification**

- No building appears in two splits.
- **0** building pairs with Jaccard ≥ 0.8 cross splits.
- The maximum test–train similarity among real buildings is 0.74 (bldg20–bldg44). This is the normal level of shared Mortar modeling conventions, not duplication.

**Design notes**

- Validation has no synthetic buildings on purpose: model selection is driven by real buildings, which are closer to BuildingQA.
- The 223P test set grows from 1 clean building (official split) to 6 buildings from 3 independent families. This makes RQ4c measurable. Leave-group-out cross-validation over the 223P families stays an optional robustness check.
- All 5 registers of a query stay with their building's split.
- **Pilot.** For cost-capped early LLM runs, a stratified ~80-building subset of train/val/test is drawn from this split, which preserves group integrity.
- **BuildingQA isolation.** BuildingQA questions, SPARQL and answers are never used for training, prompting or hyperparameter selection. That includes θ: unlike BSchema, θ is **not** tuned per BuildingQA building.

---

## 4. Proposed Method: BRIDGE

### 4.0 Pipeline

```
Question ─► (1) Decomposition (BrickQA)
         ─► (2) GNN over BSchema summary graph (question-conditioned)
                 ├─ schema linking: exact URIs of relevant (structural) classes & predicates
                 ├─ path retrieval: top-k class-level paths between linked classes
                 └─ projection planning: which classes/attributes to SELECT
         ─► (3) Budgeted context assembly: selected BSchema subgraph (+ member-graph instance samples)
                 — replaces both first-N triples and the full static BSchema
         ─► (4) BrickQA loop (turn/token-bounded) with GNN tools:
                 search_graph · suggest_paths · expand(node) · execute_sparql · stop
         ─► (5) Validation: syntax · namespace/ontology · path (first failing triple) · arity · temporal
         ─► SPARQL + answer table
```

### 4.1 Graph encoder: a learned, question-conditioned BSchema

**Two-level graph.**

- The **summary level** consists of BSchema nodes (structural classes, e.g. `VAV_version_2`). Each node carries its base class, a member count and attribute signatures. Edges are the summarized predicates.
- The **instance level** is reached through the member graph, and only for nodes the summary-level GNN ranks highly.

This keeps message passing small, typically 10²–10⁴ nodes, even for buildings with 400k triples.

**Message passing.** Conditional message passing in the NBFNet / ULTRA family:

- node states are initialized from the question embedding at seed nodes (linked classes and mentions);
- relation-specific messages are passed for 4–6 layers;
- the model is inductive over entities, so unseen buildings need no retraining.

**Relation transfer.** An ULTRA-style relation graph (relation co-occurrence patterns), initialized from predicate-name text embeddings. This is the lever for **Brick → 223P transfer**: `s223:connectedTo` / `hasConnectionPoint` / `hasDomainSpace` chains.

**Node features.**

- frozen text embeddings of the class name, the definition from the TBox, and labels;
- BSchema statistics: member count, degree signature, and depth in the class hierarchy.

**Relation to WL.** The encoder can represent BSchema's partition: 1-WL color classes are a fixed point of message passing. It *weights* that partition by question relevance, which is the learned generalization of θ.

**Optional pretraining.**

- Self-supervised link prediction and BSchema-partition prediction over all ~200 building graphs.
- Initialize from an ULTRA / GFM-RAG checkpoint where licenses allow.

### 4.2 Heads

| Head | Output | Training signal | Failure targeted |
|---|---|---|---|
| Schema linker | Ranked **exact URIs** (classes, predicates, namespaces) present in this graph | Gold schema terms (BCE + listwise) | Vocabulary + namespace hallucination (79.4% / 30.1%) |
| Path retriever | Top-k class-level paths between linked classes, with member counts; plus a SubgraphRAG-style triple scorer for instance expansion | Gold path skeleton (contrastive, sibling-path hard negatives); gold evidence triples | Topological mismatch (15.6%), BrickQA traversal failures |
| Projection planner | The set of classes/attributes to project + cardinality | Gold projection sets + synthetic multi-projection augmentation | Arity and granularity mismatch; Exact-F1 |
| Instance linker | Ranked instances for explicit mentions (via member graph) | Q4 seed mentions | Identifier guessing |

### 4.3 Budgeted context assembly (the "hybrid")

- **Budget.** The context is filled greedily by GNN score under a token budget B. B is a single global value chosen on Build2SPARQL validation; it is not set per building.
- **Format.** The selected BSchema subgraph is serialized in BSchema's own Turtle format, for comparability, plus 1–3 member instances per selected node.
- **Fallback for exploration.** If the agent needs more, it calls `expand` / `suggest_paths`, which are GNN-scored and therefore stay cheap.
- **Why this addresses both earlier failures:**
  - It avoids BSchema's context-ceiling failure, because a large building is selected rather than truncated.
  - It avoids the tool-based method's budget-exhaustion failure, because the topology is front-loaded and exploration is guided.

### 4.4 Validation additions

- **Path validator.** BFS over the BGP to report the first triple pattern that yields no results (BSchema's design), plus GNN-ranked alternative paths.
- **Namespace / ontology validator.** Rebinds prefixes to the graph's namespaces and rejects terms absent from the TBox or ABox.
- **Arity validator.** Checks the projection count against the planner. Under-projection scores 0 in `metrics.py`.
- **Kept from BrickQA:** syntax, temporal, coverage and join checks.
- **Turn and token caps,** with repairs logged.

### 4.5 Optional stretch: learned generation

A LoRA-tuned open LLM, conditioned on GNN-selected context and optionally on graph soft-tokens (G-Retriever / GNP). It is evaluated as an alternative generator inside the same loop (Option C arm).

---

## 5. Research Questions (revised)

- **RQ1. Accuracy.** Does BRIDGE beat BSchema, BrickQA and the BuildingQA baselines on Row/Entity/Exact-F1 across all four buildings, under the **same LLMs**?
- **RQ2. Value of learning.**
  - At equal context budget, does a trained, question-conditioned GNN selection beat static BSchema (best θ, chosen on validation), embedding kNN and PCST?
  - How much does each head (schema linker, path retriever, projection planner) contribute?
- **RQ3. Static vs dynamic, revisited.** Does a learned hybrid dominate both pure static (BSchema) and pure dynamic (tool-based / BrickQA) context in accuracy and cost, across graph sizes?
- **RQ4. Generalization.** How well does the GNN transfer to:
  - (a) unseen Build2SPARQL buildings;
  - (b) BuildingQA expert queries;
  - (c) 223P (B59);
  - (d) our real large office building?
- **RQ5. Complexity.** How does performance vary with pattern type, BGP size, projection count and question register?
- **RQ6. Scale.** How do accuracy, tokens and latency change as graph size grows, up to `bldg_brick_anon` (428k triples) and our building, where full BSchema no longer fits in context?
- **RQ7. Ontology invariance (case study).** Do the Brick and 223P models of the same real building yield equivalent answers?

---

## 6. Experimental Design

### 6.1 Common settings

- **LLMs:**
  - **Gemma 4 31B Thinking**: open weights, the same model as BSchema. This is the primary comparison.
  - **Gemini 2.5 Flash**: the same model as BrickQA.
  - Optionally **Nemotron Super 3**, to test robustness on a weak SPARQL model, as BSchema did.
- **Runs:** 6 per configuration, matching BSchema's protocol. Report mean ± std, plus building-level bootstrap confidence intervals.
- **Metrics:** BuildingQA `metrics.py`, unchanged. Report:
  - **question-weighted** values (the BSchema / BuildingQA convention) and **macro** values (the BrickQA Table 2 convention), both labelled;
  - completion rate, defined as a non-empty final answer;
  - context-window failures;
  - tokens (prompt / completion), $ per question, tool calls and wall-clock time.

### 6.2 Systems compared (all re-run under the same LLMs where code is available)

| System | Purpose |
|---|---|
| ReAct (100/5000/noKG), DA-KGQA | Published BuildingQA baselines (repo CSVs) |
| Validator-only | Lower bound (BSchema condition 1) |
| BSchema + validator (θ per paper: 0.7, LBNL 0.3) | **Current SOTA**, reproduced |
| BSchema tool-based (MCP suite) | Dynamic-exploration reference |
| BrickQA (NoKG / 100T / 5000T) | Our prior system, now including **B59** |
| BrickQA + static BSchema (θ chosen on Build2SPARQL val) | **Option A**: the "no-learning" hybrid |
| BrickQA + embedding kNN / PCST (G-Retriever) / SubgraphRAG retrieval | Non-learned and light-learned retrieval at equal budget |
| Build2SPARQL 3-shot retrieved | Data-driven few-shot baseline |
| **BRIDGE (full)** | Proposed |
| BRIDGE soft-token generator | Option C arm |

### 6.3 Ablations

1. GNN selection → static BSchema at θ ∈ {0, 0.3, 0.7, 1.0}, at equal budget. This is the central "value of learning" test.
2. Summary-level GNN only, vs adding member-graph instance expansion.
3. Remove the schema linker, the path retriever or the projection planner, one at a time.
4. GNN untrained (text similarity only) at the same budget.
5. ULTRA relation graph → fixed relation embeddings. This is the 223P transfer test.
6. Training data: Brick-only vs Brick+223P (effect on B59).
7. Budget B sweep: about 1k to 30k tokens. Plot the accuracy and cost curves next to BSchema θ.
8. Remove the decomposition; remove the turn cap.
9. Pretrained vs scratch GNN.

### 6.4 Analyses

- Per building, per question source (human, LLM_1–5), per Build2SPARQL register and pattern, BGP size and projection count.
- **Error taxonomy using BSchema's automated categories,** for direct comparability:
  - syntax (rdflib parser);
  - vocabulary and namespace hallucination;
  - topological mismatch;
  - identifier guessing;
  - plus BrickQA's arity and granularity categories;
  - plus retrieval miss: gold evidence not in the selected context, measured on Build2SPARQL.
- **Retriever-only metrics on Build2SPARQL test:**
  - gold-URI recall@k;
  - gold-path hit@k;
  - evidence recall at budget B;
  - context size vs static BSchema.

---

## 7. Case Study: Our Real Large Office Building (Brick + ASHRAE 223P)

This is the strongest external-validity asset. It is a real, large building, unseen by all training data and benchmarks, and it is modeled in **both** ontologies.

### 7.1 Preparation

1. **Independence (confirmed).** This building does not overlap with `bldg_brick_anon` / `bldg_223p_anon`, with BSchema's Berkeley FDD case-study building, or with any BuildingQA or Build2SPARQL building. It is therefore a clean, fully held-out test.
   - It is also one of only two sources of **223P QA evidence outside B59**, the other being Build2SPARQL's 21 223P buildings. BuildingQA's 223P signal rests on a single building with 46 questions.
2. **Validation.** Run SHACL validation and compute graph statistics: triples, entities, points, equipment, locations and BSchema size at each θ.
   - This building is likely in the regime where full BSchema does not fit in context. That makes it the **key test of RQ6**.
3. **Expert QA set.**
   - 40–60 operator questions, with **paired gold SPARQL for the Brick and 223P models**.
   - Coverage: BuildingQA-style shapes (multi-hop, multi-projection, UNION/OPTIONAL), aggregation, and timeseries / BACnet reference retrieval.
   - Plus 5-register paraphrases.
   - The set is frozen before any BRIDGE run.
4. **Timeseries (if BAS trends are available).** 10–20 operator analytics and FDD questions with **ground-truth answers**: known fault windows, affected units, threshold counts.
   - This makes the analysis quantitative, unlike BSchema's single qualitative FDD case.
   - It also follows BrickQA's LBNL FDD case study.

### 7.2 Uses

| Use | Measurement |
|---|---|
| External validation (RQ4d) | Row/Entity/Exact-F1: BrickQA vs BSchema vs BRIDGE |
| Scale (RQ6) | Accuracy, tokens and latency when full BSchema exceeds the context budget; GNN selection vs forced-θ BSchema |
| Ontology invariance (RQ7) | Answer agreement between the Brick and 223P runs, with entities aligned through shared BACnet/point IDs |
| Analytics / FDD | Accuracy on ground-truthed timeseries tasks |
| Qualitative | 3–5 traces showing GNN-selected context, retrieved paths, projection plan and repairs, including failures |

---

## 8. Evaluation Targets

These are hypotheses, not commitments.

- **BuildingQA, all 4 buildings, question-weighted Row-F1:** above **0.59**, the BSchema / Gemma 4 SOTA, using the same LLM. Mean of 6 runs, with confidence intervals that exclude BSchema.
- **Per building:**
  - LBNL above 0.30;
  - Mortar above 0.63, and ideally at least the 0.69 of the tool-based variant;
  - TUC and DFLEX no lower than BrickQA's 0.989 and 0.676 respectively, under the same LLM.
- **Exact-F1:** well above BSchema's ~0.00, via the projection planner. BrickQA's macro value of 0.437 is the reference.
- **Cost:** no more than about $0.006 per question with Gemma 4, which is BSchema parity.
- **Large-graph regime** (our building and `bldg_brick_anon`):
  - BRIDGE keeps accuracy where full BSchema overflows the context;
  - no per-building θ is needed.
- **Build2SPARQL test:** above the paper's best 3-shot retrieved result (Qwen 2.5 32B: Exact-Match 65.2%, Row-F1 0.659).

---

## 9. Work Plan and Decision Gates

| Phase | Weeks | Deliverables | Gate |
|---|---|---|---|
| P0 Reproduction (confound removal) | 1–3 | Re-run BrickQA (all 4 buildings, incl. B59) and BSchema + validator under **Gemma 4 31B and Gemini 2.5 Flash**, 6 runs each; reproduce the tool-based and validator-only conditions | **G0:** BSchema reproduced within ±0.03 of its reported results; a clean head-to-head table |
| P1 Data | 2–5 | Graph downloads; leakage audit; executed gold results; BSchema + member graphs for all Build2SPARQL buildings; evidence labels mapped to summary nodes | **G1:** ≥ 90% of train queries execute with non-empty results and map to BSchema nodes |
| P1b Quick win | 4–5 | **Option A** (BrickQA + static BSchema, θ chosen on validation) on BuildingQA validation proxies | Establishes the no-learning hybrid baseline |
| P2 GNN v1 | 5–9 | Summary-level GNN (schema linker + path retriever), trained from scratch; retriever-only metrics | **G2:** at equal budget, gold-URI/path recall beats static BSchema at the best θ, and beats kNN/PCST |
| P2b Pretraining | 7–10 | Self-supervised / ULTRA initialization; instance-level expansion | — |
| P3 Integration | 9–12 | Budgeted context assembly, GNN tools, validators, projection planner, turn caps | **G3:** gain over Option A on Build2SPARQL validation buildings |
| P4 Benchmarking | 12–14 | Frozen config on BuildingQA: all systems × 2–3 LLMs × 6 runs; ablations | — |
| P5 Case study | 10–15 (parallel) | Real-building QA set (frozen early), scale tests, Brick-vs-223P invariance, timeseries/FDD | — |
| P6 Writing | 15–17 | Paper, code, GNN weights, executed Build2SPARQL gold answers (where licenses allow) | — |

**Fallback.** If G2 fails:

- ship Option A, plus the projection planner, the namespace/path validators and the turn caps;
- report the GNN as a negative result, including the analysis of where static BSchema already saturates retrieval.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| BSchema already captures most retrievable signal, leaving little gain for the GNN on 4 small buildings | Measure the gain where it should matter most (LBNL, the large real building, `bldg_brick_anon`); report retriever metrics on 33 Build2SPARQL test buildings for statistical power |
| Model confound (Gemma vs Gemini) blurs the comparison | P0 re-runs all systems under the same LLMs |
| Build2SPARQL graph availability or version mismatch | Pin TBox versions; drop and log non-executing queries |
| Distribution shift, Build2SPARQL → BuildingQA | Learn grounding and planning, not generation; multi-projection augmentation; keep the LLM loop |
| Leakage (bldg11, LBNL/NIST duplicates of B59) | Exclusion list; graph-hash and class-index deduplication |
| Train/test leakage inside Build2SPARQL: the official split separates exact duplicates (HPL/NIST-HPL, IBAL/NIST-IBAL), and BuildGraph types are near-identical (hotel vs outpatient, Jaccard 0.93) | Our own group-level split (§3.4): deduplicate, group near-duplicates and source families, verify 0 cross-split pairs ≥ 0.8. The 223P test has 6 buildings from 3 independent families. |
| 223P evidence on BuildingQA rests on one building (B59, 46 questions) | Report B59 with confidence intervals and per-question detail; corroborate with the Build2SPARQL 223P test buildings and the 223P model of our real building |
| Small benchmark (188 questions) and noisy results | 6 runs, bootstrap confidence intervals, per-building reporting, Build2SPARQL test + case study as complementary evidence |
| Ground-truth ambiguity (BuildingQA, BSchema Fig. 6: valid alternative routes) | Rely on Row/Entity-F1; manually review "plausible but scored wrong" cases |
| Agent loops exhausting budgets (tool-based LBNL failure) | Front-loaded context, turn and token caps, completion rate reported |
| Being out-scooped: BSchema authors pursue the same hybrid | Move quickly on P0–P2; differentiate through the *learned* question-conditioned selection, Build2SPARQL-scale supervision, the projection planner and the dual-ontology real-building case study |

---

## 11. Expected Contributions

1. **BRIDGE.** A hybrid building-KGQA agent in which a **trained, question-conditioned GNN** generalizes structural graph summarization (BSchema / 1-WL) into per-question context selection. It removes the context-size ceiling and per-building tuning of static summaries.
2. **State-of-the-art results on all four BuildingQA buildings, under controlled LLMs.** This includes a stronger ASHRAE 223P result and improved Exact-Match F1, with full ablations isolating the value of learning versus static summaries.
3. **A recipe that turns a text-to-SPARQL corpus into graph supervision.** Build2SPARQL is converted into URI-, path-, evidence- and projection-level labels at both instance and summary level. The executed gold answers are released.
4. **A real large-building case study with parallel Brick and 223P models.** It covers scale beyond the context budget, ontology-invariant QA, and ground-truthed timeseries/FDD tasks.
