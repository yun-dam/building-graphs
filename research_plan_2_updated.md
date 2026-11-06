# Study 2 (Updated). Ontology-Aware Inductive GNNs for Building Knowledge Graph Completion

> This update revises `research_plan_2.md`. It draws on:
>
> - the four works in `relevantWorks/`:
>   - **BrickQA** (Ko et al., BuildSys '26): our paper;
>   - **Build2SPARQL** (Jung, 2026): the training corpus;
>   - **BuildingQA** (Mulayim et al., BuildSys '25);
>   - **Graph Learning on Cyber-Physical KGs** (Anwar, Saka & Fierro, SenSys '25 poster): the closest prior work;
> - the Build2SPARQL v1.0 release in `data/`;
> - `research_plan_1_updated.md`, which defines Study 1 (BRIDGE).
>
> It makes five changes to the original plan:
>
> 1. **GNN at the core, made state of the art.** R-GCN drops from "the encoder" to a baseline. The main model is an inductive, query-conditioned GNN in the NBFNet / A\*Net / ULTRA family, with three ontology components built in.
> 2. **Unseen-building generalization becomes the main setting.** Same-building recovery stays, but as a secondary setting. A new building is what practitioners actually face.
> 3. **Realistic missingness.** Random edge dropping is joined by cold-start and subsystem-level missingness that mimic how real building models are incomplete.
> 4. **Name leakage is controlled.** Entity names like `AHU_1_SAT` reveal relations, so name-based baselines and a name-hard subset are added.
> 5. **Our own large office building becomes the external validation and case study.** It is hand-modeled in both Brick and ASHRAE 223P.

---

## 0. Executive Summary

**Task.** Given:

- an observed, incomplete building knowledge graph (Brick or ASHRAE 223P), and
- its ontology,

rank candidate missing relations, e.g. `(Zone_Air_Temp_Sensor_17, isPointOf, ?)` or `(AHU_2, feeds, ?)`, and return calibrated scores that a modeler can review.

**Why a new method is needed.**

- Building KGs are small, sparse, very heterogeneous and site-specific.
- Classic KG embeddings (TransE, DistMult, ComplEx) are transductive. They cannot score entities of a new building.
- The closest prior work (Anwar et al., SenSys '25) trains one heterogeneous GNN per graph, with a separate subnetwork for each (class, edge, class) combination and extra parent-class subnetworks:
  - parameters grow from about 1.1×10⁸ to 1.7×10⁸;
  - training time doubles;
  - accuracy reaches a median of 62% on the 45 Mortar graphs.

  Their result confirms that the class hierarchy helps. Their way of adding it does not scale and does not transfer across ontologies.

**Proposed model (working name: BOLT, Building Ontology-aware Link predictor with Transferable relations).** It has three parts, all sharing parameters across buildings and ontologies:

1. **Entity-inductive reasoning.** Conditional message passing (NBFNet / A\*Net): node states are computed relative to the query head, so no entity embeddings are needed.
2. **Relation-inductive reasoning.** An ULTRA-style relation graph, enriched with ontology axioms (inverse, sub-property, domain/range). Brick and 223P predicates live in one representation space.
3. **Ontology injection without per-type subnetworks:**
   - a shared **class-hierarchy encoder** over the TBox gives every node a type embedding whose size does not depend on the number of classes;
   - a **soft compatibility prior** with hierarchical back-off (leaf → parent → grandparent) enters both the score and the loss.

**Data.**

- **Training:** the source graphs of the 201 Build2SPARQL buildings, i.e. 120 BuildGraph synthetic and 81 real (Mortar, Brick reference, 223P examples).
- **SPARQL use:** the Build2SPARQL SPARQL queries are used only as a *downstream evaluation instrument* (does completion restore query answers?). The natural-language questions remain unused.
- **External validation:** our own hand-modeled real large office (Brick + 223P). It never enters training or model selection.

**Primary claim to test.** A single inductive, ontology-aware GNN, trained across many buildings:

- recovers hidden relations in **unseen** buildings better than KG embeddings, R-GCN/CompGCN, the Anwar et al. architecture, and a general KG foundation model (ULTRA);
- makes fewer ontology-incompatible predictions;
- degrades more gracefully as observed links decrease;
- transfers from Brick to 223P on the same physical building.

---

## 1. Evidence Base and Positioning

### 1.1 What the related works tell us

| Work | Relevant finding | Implication for Study 2 |
|---|---|---|
| Anwar, Saka & Fierro (SenSys '25) | Ontology-informed heterogeneous GNN (GraphSAGE / GAT) with parent-class layers beats Node2Vec on Mortar. Brick has 1,253 classes × 54 edge labels, giving about 42M possible relation types. Each type gets its own subnetwork, so the parameter count explodes. Evaluation: train on one graph, test across graphs, accuracy only. | Direct baseline (re-implemented). We keep the idea (hierarchy aids generalization) but replace per-type subnetworks with shared, parameter-efficient encoders, and evaluate with ranking metrics and controlled leakage. |
| Build2SPARQL (Jung, 2026) | 201 KGs, 81 real and 120 synthetic. Real graphs are irregular and vary widely in "system completeness". There is a roughly 20-point gap between synthetic and real held-out performance. Building-level splits are required. Names KG completeness assessment as a use case. | Training corpus. Results are reported **separately for real buildings**. Synthetic graphs serve as structural augmentation, not as a proxy for real difficulty. |
| BrickQA (ours, BuildSys '26) | Lists among its limitations that "real-world building knowledge graphs are often incomplete or inconsistently modeled". One case-study failure came from a modeling mismatch (`Speed_Status` vs `On_Off_Status`). | Study 2 addresses the input-side cause of QA failures. The integration experiment (§10) measures whether completion helps QA. |
| BuildingQA (Mulayim et al., BuildSys '25) | Relational structure missing from a graph makes correct QA impossible, whatever the QA method. | Motivates the downstream "query-answer recovery" metric (§7.3). |

### 1.2 Data facts that constrain the design

These facts were checked against the local Build2SPARQL v1.0 release.

- **Scale.** 201 building files. The `corpus` field reads `buildgraph` 120, `mortar` 59 and `brick_reference` 22; the `mortar` label also covers the 223P models. By schema: 180 Brick and 21 223P.
- **The original "80 buildings" corresponds to the 81 real buildings.** This update keeps them as the primary evaluation population and adds the 120 synthetic graphs as training augmentation.
- **The source graphs are not in the release.** They must be downloaded from:
  - BuildGraph: Zenodo 10.5281/zenodo.20015123;
  - Mortar: brickschema.org/ttl/mortar;
  - open223: models.open223.info;
  - Brick `examples/`.

  This is shared with Study 1, P1.
- **Predicate frequency in Build2SPARQL queries** (number of queries using the predicate) is a proxy for which relations matter downstream:

  | Brick | Queries | 223P | Queries |
  |---|---:|---|---:|
  | `hasPoint` / `isPointOf` | 3,142 / 582 | `cnx` | 567 |
  | `feeds` / `isFedBy` | 1,868 / 137 | `connectsAt`, `connectsThrough`, `connectedThrough` | 165, 128, 58 |
  | `hasPart` / `isPartOf` | 758 / 85 | `hasConnectionPoint` / `isConnectionPointOf` | 94 / 96 |
  | `controls` | 44 | `hasProperty` | 164 |
  | `hasLocation` / `isLocationOf` | 2 / 17 | `observes`, `actuatedByProperty` | 32, 29 |
  | `meters` / `isMeteredBy`, `hasSubMeter` | 15 / 7, 4 | `contains`, `hasDomainSpace`, `encloses` | 21, 4, 4 |

- **Near-duplicates.** These must be grouped so they never straddle splits:
  - `HPL`/`NIST-HPL` and `IBAL`/`NIST-IBAL`;
  - the 7 `guideline36-2021-A-*` variants;
  - the 5 BuildGraph instances in each archetype × vintage cell.
- **Names carry answers.** Synthetic instance names encode their relations, e.g. `Zone_F3_Z02.Occupancy_Sensor` names its zone. Mortar point names often embed equipment tags, e.g. `AHU_1_SAT`. A name-matching heuristic could therefore look strong for trivial reasons (§8).

### 1.3 Positioning statement

To our knowledge, this is the first study of **inductive, cross-building and cross-ontology** link prediction on building KGs. It uses current KG-reasoning GNNs and quantifies separately what ontology types, hierarchy and constraints each contribute. The literature check is part of P0, and the novelty claim must be re-confirmed there.

---

## 2. Research Questions (revised)

- **RQ1. Model.** Does an ontology-aware inductive GNN (BOLT) outperform:
  - KG embeddings;
  - relational GNNs (R-GCN, CompGCN);
  - the Anwar et al. heterogeneous GNN;
  - a general KG foundation model (ULTRA, zero-shot and fine-tuned);

  on missing-link recovery in **unseen** buildings?
- **RQ2. Ontology ablation.** What are the separate contributions of:
  - (a) type features;
  - (b) class hierarchy;
  - (c) domain/range and shape-based compatibility;
  - (d) ontology-relation enrichment of the relation graph?

  And is it better to inject the ontology as *features*, as *graph structure* (class nodes in the message-passing graph) or as *score/loss priors*?
- **RQ3. Robustness to incompleteness.** Does ontology information reduce the degradation:
  - (a) as observed links decrease (sparsity ladder);
  - (b) under realistic missingness patterns (cold-start points, missing subsystems);
  - (c) when `rdf:type` itself is partially missing?
- **RQ4. Semantic validity.** Does BOLT reduce ontology-incompatible predictions without just acting as a candidate filter? Are its scores calibrated enough to support review?
- **RQ5. Transfer.** How well does a model trained across buildings transfer:
  - (a) synthetic → real;
  - (b) Brick → 223P, including on the *same physical building*?
- **RQ6. Real-building utility.** On our hand-modeled large office:
  - How much modeling effort does GNN-assisted completion save?
  - Do completed graphs restore downstream SPARQL answers?

---

## 3. Data Plan

### 3.1 Roles of each dataset

| Dataset | Role | Notes |
|---|---|---|
| Build2SPARQL source graphs, synthetic (120, BuildGraph) | Training and pretraining augmentation | Held-out archetype test for the synthetic track only |
| Build2SPARQL source graphs, real (81: Mortar, Brick reference, 223P) | **Primary train/validation/test population** (grouped cross-validation) | Results reported separately for real buildings |
| Build2SPARQL SPARQL queries (no NL) | **Downstream evaluation only** (§7.3); optionally used to weight target relations | NL questions are not used anywhere. Queries are not a training signal. |
| Brick 1.4 and ASHRAE 223P TBoxes and SHACL shapes | Class hierarchy, relation axioms, compatibility prior | Versions pinned. Mortar graphs migrated to one Brick version. |
| Own real large office (Brick + 223P, hand-modeled) | **External validation + case study** (§9) | Never used for training, model selection or prior construction |
| `bldg_brick_anon` / `bldg_223p_anon` (BuildingQA repo) | Scale test (144k–428k triples) | Check whether they derive from our building, as in Study 1 §7.1 |

### 3.2 Graph preprocessing

Same pipeline as Study 1 §3.2, with link-prediction-specific additions:

1. **Download and validate.**
   - Resolve every `building_id` and pin the ontology version.
   - Run SHACL validation and log each building's violation count. These counts are a covariate in the analysis.
2. **Split ABox from TBox.**
   - The instance graph keeps instance–instance edges plus `rdf:type`.
   - The TBox is held separately: class hierarchy, property axioms, SHACL shapes.
   - Literals, units and timeseries references are kept as optional node attributes, never as prediction targets.
3. **Canonicalize inverse pairs.**
   - Merge each pair into one canonical direction: `hasPoint`↔`isPointOf`, `feeds`↔`isFedBy`, `hasPart`↔`isPartOf`, `hasLocation`↔`isLocationOf`, `meters`↔`isMeteredBy`, `hasConnectionPoint`↔`isConnectionPointOf`, and `connectsAt`/`connectsThrough`↔`connectedThrough`.
   - Inverse edges are added back inside the model, never as data. One fact is one triple, so it cannot leak through its inverse (§8).
4. **Materialize inferences after splitting, never before.**
   - Reasoning (e.g. `brickschema` expansion, 223P `connected` inference from `cnx`) runs only on the observed training graph of each split.
   - Triples that are *entailed* by observed triples plus the ontology are tagged as "rule-derivable". They are evaluated separately (§7.4).
5. **Two granularity levels for 223P.**
   - **Raw:** connection-point level (`hasConnectionPoint`, `cnx`, `connectsThrough`).
   - **Abstracted:** equipment–equipment `connected` / `connectedTo`, derived from the raw level.

   Both are predicted and reported. The abstracted level aligns with Brick `feeds`, which enables the cross-ontology study.
6. **Normalize types.**
   - Multi-typed nodes keep all their types.
   - Untyped nodes get an `UNKNOWN` type, and their count is reported.

### 3.3 Prediction targets

Five relation families, defined by function and mapped to predicates (canonical direction):

| Family | Brick | 223P |
|---|---|---|
| F1 Point ↔ Equipment/Space (telemetry association) | `hasPoint` | `hasProperty`, `observes`, `actuatedByProperty` |
| F2 Equipment → Equipment (flow/topology) | `feeds` | `connected`/`connectedTo` (abstracted); `cnx`, `connectsThrough` (raw) |
| F3 Composition | `hasPart` | `contains`, `hasConnectionPoint` |
| F4 Spatial | `hasLocation` | `hasDomainSpace`, `encloses`, `hasPhysicalLocation` |
| F5 Control / metering | `controls`, `meters`, `hasSubMeter` | (where present) |

- **Eligibility.** A predicate is a target in a building only if it has at least 20 instances there. Below that threshold it stays in the message-passing graph, but is not a target.
- **Priority.** F1 and F2 are the headline families: they are the most frequent and the most queried (Table §1.2), and the most often missing in practice. F4 is sparse in Build2SPARQL and gets most of its evaluation weight from the real-building case study.

### 3.4 Splits

All splits are **building-level**. Near-duplicate groups (§1.2) are always kept together.

| Setting | Population | Split |
|---|---|---|
| **S1 Unseen buildings (main)** | 81 real, grouped into about 70 groups | **5-fold grouped cross-validation**, stratified by source (Mortar / reference / 223P). Within each training fold, about 10% of buildings are used for validation. The 120 synthetic buildings are always in training. |
| S1-compat | `split.json` (173 train / 28 test) | Reported once, for comparability with Study 1 and Build2SPARQL. The 28 test buildings are 15 BuildGraph hospital + 13 real. |
| S1-synth | BuildGraph | Hold out one whole archetype (all 3 vintages × 5 instances) per fold. This measures synthetic-to-new-archetype transfer. |
| **S2 Same building (secondary)** | Each real building | Hidden-link split 80/10/10 of target links, stratified by relation. Validation and test links are removed from message passing. |
| S3 External | Our large office | Never seen. Protocols M1–M5 (§3.5) applied to both models. |

The original 56/8/16 split of the 80 buildings is kept as a **pilot configuration** for fast iteration before running full cross-validation.

### 3.5 Missingness protocols

For each test building, "observed" and "hidden" parts are created under five protocols:

| Protocol | What is hidden | Real-world analogue |
|---|---|---|
| **M1 Random** | 10/20/30% of target links, uniform within each relation | Generic omissions; this is the original plan |
| **M2 Relation-block** | All links of one family inside one subsystem or floor, e.g. every `hasLocation` on floor 3 | A modeling pass that was never done for part of the building |
| **M3 Cold-start entity** | All links of 10–30% of point (or equipment) nodes. Their `rdf:type` is kept. | New BAS points imported with tag-derived types and no relationships. This is the most common practical case. |
| **M4 Subsystem** | All downstream F2/F1 links of a whole AHU or plant loop | A system that was added but never connected |
| **M5 Type noise** | M1 plus removal of `rdf:type` on 10–30% of nodes, or coarsening of types to their parent class | Partially typed or under-specified models |

**Sparsity ladder (RQ3a).**

- The test set is fixed (M1 at 10%).
- The remaining non-target observed links are subsampled to {100, 80, 60, 40, 20}%.
- 5 repeats per level.

---

## 4. Proposed Method: BOLT

### 4.0 Pipeline

```
Observed ABox  +  TBox (class hierarchy, property axioms, SHACL shapes)
   │
   ├─► (A) Class-hierarchy encoder (shared, TBox-level)  ──► type embedding τ(v) for every node
   ├─► (B) Ontology-enriched relation graph encoder (ULTRA-style) ──► query-conditioned relation embeddings
   ├─► (C) Conditional message passing over the building graph (NBFNet / A*Net) ──► h_v | (head, relation)
   ├─► (D) Score = neural score + gated ontology compatibility prior (hierarchical back-off)
   └─► (E) Calibration + conformal candidate sets + path explanations ──► ranked review list
```

Each component is chosen to remove a specific obstacle:

- **(C)** handles unseen entities;
- **(B)** handles unseen and cross-ontology relations;
- **(A)** handles the long tail of classes without per-type parameters;
- **(D)** turns the ontology's semantics into a learnable, soft constraint;
- **(E)** makes the output usable for human review.

### 4.1 (C) Entity-inductive backbone: conditional message passing

- **Query and initialization.** For a query (h, r, ?), set the boundary condition h_v⁰ = 𝟙[v = h] · q_r + W_τ τ(v).
  - The second term is the **type-aware initialization**. Vanilla NBFNet initializes non-head nodes at zero and therefore cannot see classes.
- **Message passing.** Relation-conditioned messages (DistMult-style m = h_u ⊙ w_r(q)), PNA or sum aggregation, layer norm and short-cut connections; L = 6 layers, d = 64.
  - Each layer is a generalized Bellman–Ford step, so the model can reason over the multi-hop paths that characterize building topologies, e.g. AHU → VAV → Zone → Point.
- **Scale.** On large graphs (real large office, `bldg_*_anon`), use **A\*Net**-style learned priority pruning, or k-hop subgraph extraction around the head. Latency is reported.
- **Why this family.** NBFNet, A\*Net and ULTRA are among the strongest inductive KG-completion models (an up-to-date leaderboard check is part of P0). They need no entity embeddings, they produce path-based explanations, and they are cheap at the scale of building graphs, which have hundreds to tens of thousands of nodes.

### 4.2 (B) Relation-inductive encoder enriched with the ontology

- **Relation graph.** Following ULTRA, build a graph whose nodes are the relations present in the building. Its edges are the four interaction types (head-to-head, head-to-tail, tail-to-head, tail-to-tail), weighted by co-occurrence.
  - A small GNN over this graph, conditioned on the query relation, yields relation embeddings.
  - Relations are therefore defined by *how they interact*, not by an ID. This is what lets one model serve Brick and 223P and handle predicates unseen in training.
- **Ontology enrichment (new).** Add TBox-derived edges to the relation graph:
  - `inverseOf`;
  - `subPropertyOf`;
  - "shares domain" and "shares range", computed through the class hierarchy;
  - a Brick ↔ 223P alignment where a mapping exists (e.g. Brick `feeds` ≈ 223P `connectedTo` at the abstracted level). The alignment is used **only** as a relation-graph edge, never as data.
- **Initialization.** Start from a public ULTRA checkpoint (pretrained on general KGs) where the license allows; otherwise start from scratch. Both are compared (§5).

### 4.3 (A) Class-hierarchy encoder (replacing per-type subnetworks)

- **Hierarchy encoding.** Encode the TBox `rdfs:subClassOf` DAG with a shared encoder, so the parameter count is independent of the number of classes (Brick has about 1,300). Two candidates are compared:
  1. a GNN over the class DAG with ancestor aggregation (default);
  2. order-preserving embeddings: hyperbolic (Poincaré/Lorentz) or box embeddings, which encode subsumption geometrically.
- **Class features.** Structural features (depth, number of ancestors, number of descendants) plus an optional embedding of the class label.
  - The class label is ontology vocabulary, not instance text, so it is leakage-free.
  - It enables zero-shot handling of classes unseen in training.
  - This is an ablation (§6), not a requirement.
- **Node type embedding.** τ(v) is a mean or attention pool over the embeddings of v's asserted types. `UNKNOWN` maps to a learned vector.
- **Auxiliary task.** Masked type prediction: predict the class of nodes whose type is hidden in M5. This forces τ to be consistent with the structure and gives robustness when types are missing.

This is the parameter-efficient counterpart of Anwar et al.'s parent-class layers: ancestors inform every node through a shared encoder instead of through extra subnetworks.

### 4.4 (D) Scoring with a gated ontology-compatibility prior

The score is

s(h, r, t) = MLP([h_t ; q_r ; τ(t)]) + g_r · log π(r | C(h), C(t)),

where the prior π combines two sources:

1. **Explicit axioms.** `rdfs:domain`/`rdfs:range` and SHACL property shapes (`sh:class`, `sh:node` on the path r) from Brick 1.4 and 223P.
2. **Empirical class-level schema.** Counts of (class, property, class) in the *training* buildings only. This is the same index Build2SPARQL mines.

Further details of the prior:

- **Hierarchical back-off.** π is estimated at leaf level and smoothed toward parent and grandparent levels, as in Kneser–Ney-style back-off. An unseen leaf pair (e.g. a new sensor subclass) therefore inherits its parent's compatibility.
- **Learned gate g_r ≥ 0, per relation.** The model learns how much to trust the prior. Brick leaves many relations loosely constrained, so the gate matters.

**Why soft.** Domain and range carry *inferential* semantics, and real models contain missing or wrong types, so the prior is never used as a hard filter. A hard filter is evaluated only as a baseline (Plain + filter, §5).

### 4.5 Training

- **Objective.** 1-to-N tail (and head) scoring with binary cross-entropy, using self-adversarial negative weighting.
- **Negative mix:**
  - uniform negatives;
  - **type-compatible hard negatives**: the same class as the true tail, e.g. a sibling VAV on the same AHU;
  - **structural hard negatives**: within 2–3 hops of the head.

  Known positives in the observed graph, and all their inverses, are filtered out of the negatives.
- **Semantic regularizer.** A penalty on probability mass assigned to candidates whose π falls below a threshold. Its weight is annealed, so it guides early training without overriding the data later.
- **Cardinality awareness.** Functional relations (e.g. a point is usually `isPointOf` exactly one equipment; a VAV is fed by one AHU) use a softmax over candidates. Non-functional relations use independent sigmoids. Functionality is estimated from the training buildings.
- **Three-stage schedule:**
  1. *(optional)* ULTRA initialization;
  2. **multi-building self-supervised pretraining**: random-edge link prediction over all training buildings, synthetic and real, Brick and 223P;
  3. supervised fine-tuning on the target families under protocols M1–M5, mixed.
- **Test-time adaptation (TTA).** A new building's *observed* edges are a free training signal. Run a few self-supervised steps on them before predicting its hidden edges. TTA is reported as a separate row, since it costs compute at deployment.
- **Budget.** About 1–5M parameters. One GPU-day per cross-validation fold is expected; this is to be confirmed in P2.

### 4.6 (E) Outputs for review

- **Calibration.** Temperature scaling on validation buildings; report the expected calibration error (ECE).
- **Conformal candidate sets.** For each query, a set guaranteed to contain the true link with probability 1−α (split-conformal over validation buildings). Report the set size at α = 0.1. This is the "review list".
- **Path explanations.** The top contributing paths for each prediction, a native feature of NBFNet-family models. For example: "Sensor_17 → isPointOf? VAV_3, because Sensor_17 shares a zone with VAV_3's damper command". These are used in the case study (§9).
- **Optional structured decoding.** A joint assignment of cold-start points to equipment under cardinality constraints, via Hungarian matching or a small ILP, compared with independent top-1 choices.

---

## 5. Systems Compared

| # | System | Inductive (new buildings)? | Purpose |
|---|---|---|---|
| B0 | **Ontology prior only** (π with back-off) | Yes | Shows how far type statistics alone go |
| B1 | **Name heuristic** (token, prefix and edit similarity between entity names) | Yes | Exposes name leakage; a strong practical baseline |
| B2 | AnyBURL (rule mining) | Partly (rules transfer) | Symbolic baseline; rule-derivable links |
| B3 | TransE / DistMult / ComplEx / RotatE | **No**: S2 only | Classic KG-embedding baselines |
| B4 | R-GCN, CompGCN (with type features) | With features only | Relational GNNs; the original plan's encoder |
| B5 | Heterogeneous GNN (HGT / heterogeneous GraphSAGE) | With features only | Heterogeneous GNN baseline |
| B6 | **Anwar et al.** heterogeneous GNN with parent-class layers (re-implementation) | Partly | Closest prior work; also compared on parameters and training time |
| B7 | NBFNet / A\*Net (no ontology) | Entities only | "Plain" state-of-the-art inductive GNN |
| B8 | **ULTRA**: zero-shot, and fine-tuned | Entities + relations | General KG foundation model |
| B9 | Plain (B7) + hard ontology filter | — | Separates representation gains from filtering gains |
| **P** | **BOLT** (full), with and without TTA | Entities + relations | Proposed |
| P+name | BOLT + character n-gram name features | — | Practical "name-aware" track, reported separately |

**Optional.** An LLM link predictor (open-weight LLM given a serialized k-hop neighborhood plus candidate list). Reviewers in 2026 will ask for it. It uses names, so it belongs to the name-aware track.

**Rules for all systems:**

- the same splits, candidate sets, filtering and negatives;
- hyperparameters tuned with equal budgets on validation buildings only;
- systems that cannot handle new entities (B3) are run only in S2 and labeled as such.

---

## 6. Ablations (RQ2)

| Ablation | Removes / changes |
|---|---|
| −type | τ(v) removed from the initialization and the score |
| −hierarchy | τ computed from leaf class one-hot, with no ancestors |
| −prior | g_r = 0; semantic regularizer off |
| −relation enrichment | ULTRA relation graph without TBox edges |
| −aux type task | No masked type prediction |
| −pretraining / −ULTRA init | Fine-tune only from scratch |
| −synthetic | Train on real buildings only |
| **Injection mode** | Ontology as features (default) vs as graph structure (class nodes plus `rdf:type`/`subClassOf` edges in message passing) vs as prior only |
| Hierarchy encoder | DAG-GNN vs hyperbolic vs box |
| Class-label text | On vs off |

**Controls.** Every ablation is run with both full-entity and type-constrained candidate sets (§7.1). This shows whether a gain comes from better representations or merely from a smaller effective candidate set.

---

## 7. Evaluation

### 7.1 Ranking metrics (primary)

- **Metrics.** Filtered MRR and Hits@1/3/10, for both head and tail prediction.
- **Two candidate protocols, always reported side by side:**
  - **full**: all entities in the building;
  - **type-constrained**: entities compatible under π at the parent level.

  Good full-protocol scores obtained by ranking type-impossible candidates last are therefore visible as such.
- **Aggregation.** Macro-average over buildings (primary), plus micro-average. Broken down by:
  - relation family (F1–F5);
  - source: synthetic / Mortar / reference / 223P;
  - ontology: Brick / 223P;
  - missingness protocol (M1–M5).

### 7.2 Semantic and review-oriented metrics (RQ4)

- **Sem@k.** Fraction of the top-k predictions that are compatible with the ontology (explicit axioms), and the violation rate.
- **AUPRC** on a balanced set that includes hard negatives, plus **ECE**.
- **Conformal coverage and mean set size** at α = 0.1.
- **Review efficiency.** Recall of hidden links as a function of the number of candidates a reviewer inspects (recall@budget curve).

Ontology compatibility is *not* treated as correctness. Sem@k is reported next to accuracy, never in place of it.

### 7.3 Downstream query-answer recovery (uses Build2SPARQL SPARQL, no NL)

For each test building, every Build2SPARQL query of that building is executed three times:

- (a) on the complete graph, which is the gold answer;
- (b) on the incomplete graph;
- (c) on the incomplete graph plus links accepted from BOLT (top-1 per query for functional relations; threshold chosen on validation).

The metric is the **answer-row recall restored**, (c − b) / (a − b), together with the row precision of (c). This uses the KG-grounded correctness of Build2SPARQL's queries to measure *how much the completion matters for real information needs*. It also links Study 2 to Study 1.

### 7.4 Breakdowns that guard the interpretation

- **Rule-derivable vs non-derivable** hidden links (§3.2, step 4).
- **Name-easy vs name-hard** links: the name-hard subset contains links where B1 ranks the true answer outside its top-3.
- Real vs synthetic.
- Building size and SHACL violation count, used as regression covariates.

### 7.5 Statistics

- 5 random seeds × 5 cross-validation folds.
- Building-level bootstrap 95% confidence intervals.
- Paired Wilcoxon signed-rank tests across buildings for P vs each baseline, with Holm correction.
- Effect sizes are reported alongside p-values.

---

## 8. Data Leakage and Evaluation Controls

The original eight controls are kept, with additions marked **(new)**.

1. Validation and test links are removed from both the training graph and the message-passing graph.
2. Inverse pairs are canonicalized (§3.2, step 3); the inverse of a hidden link is hidden as well. **(new)** Equivalent 223P paths are hidden jointly: hiding equipment-level `connected` also hides the connection-point chain that entails it, and vice versa.
3. **(new)** Reasoning and materialization run only after the split, on observed triples. Transitive closures (e.g. `hasPart+`, `feeds+`) that would re-create a hidden link are blocked.
4. Rule-derivable links are reported separately.
5. Hard negatives are used, not only type-impossible easy negatives.
6. Candidate sets and filtering are identical across systems.
7. Predictions not in the gold graph are not counted as confirmed. They are reviewed only in the case study (§9, C5).
8. **(new) Name leakage.**
   - The main model sees no entity names; entities are anonymous.
   - Names are allowed only in the separately reported name-aware track.
   - B1 and the name-hard subset quantify how much is explainable by names alone.
9. **(new) Building leakage.**
   - Near-duplicate groups stay together.
   - The empirical prior π and the calibration are fit on training and validation buildings only.
   - The real large office is never used for prior estimation, model selection or calibration.
10. **(new) Cross-study leakage.** If BOLT's pretrained encoder is reused by Study 1 (§10), its pretraining set must exclude the BuildingQA buildings, i.e. drop `bldg11` and any duplicates found in Study 1's P1 audit.

---

## 9. Case Study and External Validation: Our Real Large Office (Brick + ASHRAE 223P)

The hand-modeled building is the strongest external-validity asset in the study:

- it is real, large and curated;
- it is unseen by all training data;
- it is modeled **twice**, so the *same physical system* can be studied under two ontologies;
- because it was modeled by hand, its gold graph is far more reliable than that of the Mortar graphs, which were generated automatically.

### 9.1 Preparation (frozen before any model is run on it)

1. **Validation and statistics.**
   - Run SHACL validation on both models.
   - Report statistics (triples, entities, points, equipment, spaces, relations per family) in the format of BrickQA Table 1.
2. **Overlap check.** Confirm whether `bldg_brick_anon` / `bldg_223p_anon` in the BuildingQA repository derive from this building. If they do, treat them as the same facility in both studies.
3. **Cross-model entity alignment.** Map Brick ↔ 223P entities through shared BACnet/point identifiers and equipment tags. Record the alignment coverage.
4. **Test protocol.** Fix and record the hidden sets for M1–M5 (seeded) before any run.
5. **Optional: modeling history.** If intermediate snapshots of the modeling process exist (file versions, dates), keep them for a temporal split.

### 9.2 Experiments

| ID | Experiment | Measurement | RQ |
|---|---|---|---|
| C1 | **External inductive test**: BOLT and the baselines, trained on Build2SPARQL graphs only, applied to both models under M1–M5, with and without TTA | Ranking metrics; the gap to Build2SPARQL real-building cross-validation | RQ1, RQ3 |
| C2 | **Sim-to-real**: compare with the 15 BuildGraph *large office* instances; train on synthetic only vs real only vs both | The synthetic-to-real gap on a matched archetype | RQ5a |
| C3 | **Cross-ontology transfer**: train on Brick-only buildings → test on the 223P model, and vice versa; then train jointly | Performance drop vs joint training; contribution of relation-graph enrichment (Brick↔223P alignment edges) | RQ5b |
| C4 | **Ontology invariance**: for aligned entity pairs, compare predictions on the Brick model vs the 223P model (F1 and F2 at the abstracted level) | Agreement rate and rank correlation, the link-prediction counterpart of Study 1's RQ6 | RQ5b |
| C5 | **Modeling-assistant simulation** (M3 at 100% for points): keep the equipment, spaces and typed points; delete all point relations; simulate a reviewer who inspects the ranked candidates for each point and accepts the true one | Fraction of point relations recovered vs number of candidates inspected, against B0/B1. If modeling time was logged, convert to estimated person-hours saved. | RQ6 |
| C6 | **Expert review of beyond-gold predictions**: the top-N high-confidence predictions that are *not* in the gold model are reviewed by the modeler against drawings and BAS records. 20% random low-score candidates are mixed in, and reviewers do not see the scores. | Labels: truly missing / plausible / wrong. This estimates how many "naturally missing" relations even a curated model has. | RQ4, RQ6 |
| C7 | **Temporal split** (if history exists): train or adapt on an early snapshot, predict links added later | Ranking metrics under real, not simulated, incompleteness | RQ3 |
| C8 | **Scale**: full-graph inference on the large office and on `bldg_*_anon` | Latency, memory and parameter count vs B6 (Anwar et al.) | RQ1 |
| C9 | **Qualitative**: 3–5 walk-throughs with path explanations, including at least one failure | — | — |

### 9.3 Optional data-driven extension

If BAS trend data is available, add a time-series similarity feature, such as the correlation between a point and an equipment's command or status. Study its complementarity with structure for F1 links. This is clearly marked as an extension, so that the core study stays purely graph and ontology based.

---

## 10. Relationship to Study 1

| | Study 1 (BRIDGE) | Study 2 (BOLT) |
|---|---|---|
| Input | NL question + building KG | Building KG + ontology |
| Supervision | Build2SPARQL questions and SPARQL, converted into graph labels | Graph structure only (self-supervised + hidden-link) |
| Output | SPARQL + answer table | Ranked, calibrated missing links |
| GNN | Question-conditioned retriever (NBFNet/ULTRA-style) | Relation-query-conditioned link predictor (NBFNet/A\*Net/ULTRA-style) |

Synergies between the two studies:

1. **Shared preprocessing.** Graph acquisition, TBox handling and the class index are built once (Study 1, P1).
2. **Shared pretraining.** BOLT's multi-building self-supervised pretraining (§4.5, stage 2) is the same pretraining Study 1 plans as P2b. Study 2 produces and releases this "building-KG encoder"; Study 1 consumes it. The cross-study leakage rule in §8, item 10 applies.
3. **Integration experiment (after both are evaluated independently).**
   - On the real large office: degrade the graph (M3/M4), then run Study 1's frozen real-building QA set on:
     - (i) the complete graph;
     - (ii) the degraded graph;
     - (iii) the degraded graph completed by BOLT (accepted links only, using the validation threshold).
   - Measure the Row-F1 recovered.
   - This is the end-to-end story: *completion makes building knowledge accessible again*.

---

## 11. Work Plan and Decision Gates

| Phase | Weeks | Deliverables | Gate |
|---|---|---|---|
| P0 Literature + reproduction | 1–2 | Up-to-date check of inductive KGC state of the art and building-KGC prior work; reproduce NBFNet and ULTRA on a public inductive benchmark | **G0:** reproduced numbers within reported ranges |
| P1 Data (shared with Study 1) | 2–4 | Graph downloads, version pinning, SHACL statistics, canonicalization, near-duplicate grouping, target eligibility table, M1–M5 generators, executed Build2SPARQL gold answers | **G1:** at least 70 real buildings usable, with at least one F1 or F2 target each |
| P2 Baselines | 4–6 | B0–B9 under S1 / S2 (pilot split first); name-hard subset defined | **G2:** NBFNet (B7) beats B0 and B1 on the name-hard subset. If not, the task framing is revisited, e.g. by moving toward cold-start-only evaluation. |
| P3 BOLT v1 | 6–10 | Components A–D, training schedule, ablations on the pilot split | **G3:** BOLT beats B7 and B8 on validation buildings under the type-constrained protocol, which rules out filtering-only gains |
| P4 Full evaluation | 10–13 | 5-fold × 5-seed cross-validation, M1–M5, sparsity ladder, downstream query recovery, statistics | — |
| P5 Case study | 9–15 (parallel; protocol frozen in week 9) | C1–C9 on the real large office; expert review | — |
| P6 Integration | 14–15 | Study 1 × Study 2 experiment (§10) | Both studies frozen |
| P7 Writing + release | 15–17 | Paper; code; pretrained building-KG encoder; split and missingness generators | — |

**Fallback.** If G3 fails, the study still contributes:

- the first leakage-controlled, cross-building benchmark for building KG completion (splits, M1–M5, name-hard subset, query-recovery metric);
- a systematic comparison of state-of-the-art inductive GNNs with an analysis of where ontology helps and where it does not.

---

## 12. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Some source graphs are unavailable or version-mismatched | Pin ontology versions; migrate Brick versions; log and drop unusable buildings. 81 real buildings leave margin above G1's 70. |
| Name leakage inflates every model | Anonymous entities in the main track; B1 and the name-hard subset; name-aware results reported separately |
| The ontology prior just acts as a filter | Type-constrained protocol, B9 (plain + filter), and the −prior ablation are always reported together |
| Synthetic graphs dominate training and do not transfer | Real-only reporting; −synthetic ablation; C2 sim-to-real on a matched archetype |
| Few 223P buildings (21, many tiny) | Joint Brick + 223P training through the relation graph; C3/C4 on the large 223P model of our own building |
| Real "gold" graphs are themselves incomplete, so true predictions are scored as false | C6 blind expert review; type-constrained and filtered metrics; the hand-modeled building serves as a high-fidelity gold |
| Large graphs exceed memory | A\*Net pruning or k-hop subgraphs; C8 reports latency and memory |
| ULTRA checkpoint licensing or compatibility | Pretraining from scratch on the building corpus is the default; ULTRA initialization is an ablation |
| Overlap between our building and `bldg_*_anon` | Resolve in §9.1, step 2, before any experiment |

---

## 13. Expected Contributions

1. **BOLT.** To our knowledge, the first entity- and relation-inductive, ontology-aware GNN for building KG completion. It replaces per-type subnetworks with a shared hierarchy encoder and a gated, back-off compatibility prior, and one model serves both Brick and ASHRAE 223P.
2. **A leakage-controlled benchmark protocol for building KG completion,** built on the Build2SPARQL source graphs:
   - building-level grouped splits;
   - five realistic missingness protocols;
   - a name-hard subset;
   - a downstream SPARQL answer-recovery metric.
3. **A quantified account of what the ontology contributes:** types, hierarchy, constraints and relation axioms, as features vs structure vs priors, under sparsity, missing types and synthetic-to-real or Brick-to-223P shift.
4. **A real-building validation** on a hand-modeled large office with parallel Brick and 223P models. It includes cross-ontology invariance, a modeling-effort simulation and a blind expert review of predicted missing relations.
5. **A pretrained building-KG encoder** and an end-to-end demonstration, together with Study 1, that graph completion restores the answers to building questions.
