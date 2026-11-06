# Project Proposal: BRIDGE — Question-Conditioned Graph Neural Retrieval for Question Answering over Building Knowledge Graphs

## 1. Motivation

**Why building KGs matter, and why they go unused.** Buildings account for a large share of energy use. Analytics such as fault detection and optimal control could cut that waste, but they need to know *which* sensor belongs to *which* equipment and space [CITE].

Semantic ontologies such as **Brick** [CITE] and **ASHRAE 223P** [CITE] capture exactly this information as a knowledge graph (KG): equipment, spaces and points, linked by relations such as `feeds` and `hasPoint`. Yet the people closest to buildings, facility managers and operators, rarely know SPARQL. Rich semantic models therefore stay largely inaccessible [CITE: BrickQA; BuildingQA].

Natural-language question answering over building KGs (KGQA) would remove this barrier.

**Why existing methods fall short.** KGQA methods built for open-domain graphs transfer poorly to buildings. On the BuildingQA benchmark, the best general-purpose agents reach only 0.38 Row-Matching F1 [CITE]. Building KGs are hard for four reasons:

- They use a specialized vocabulary and opaque labels (e.g. `AHU_1_SAT`).
- Answers require multi-hop paths through a dense, repetitive topology (sensor → VAV → zone).
- Each building is modeled differently, and Brick and 223P use different relation vocabularies.
- There is no per-building training data, so a method must work on buildings it has never seen.

**Why a graph ML approach.** Domain-specific LLM agents raised the state of the art to 0.59 Row-F1:

- **BrickQA** [CITE], our prior work, uses dynamic graph exploration.
- **BSchema** [CITE] uses precomputed structural summaries.

Both are zero-shot and contain **no learned graph component**. Their remaining errors are *graph grounding* errors:

- **vocabulary hallucination** (79% of failed queries);
- **topological mismatch** (16%), i.e. assuming a direct edge where the building uses a multi-hop path [CITE: BSchema].

These are exactly the decisions a GNN can learn: which nodes, relations and paths in *this* graph are relevant to *this* question. Two developments make learning feasible now:

- the release of **Build2SPARQL**, about 200 buildings with about 30k question–query pairs [CITE];
- **inductive GNNs** that transfer to unseen graphs and unseen relation types [CITE: NBFNet; ULTRA].

We propose **BRIDGE** (**B**uilding **R**etrieval via **I**nductive, **D**omain-general **G**raph **E**ncoding), which trains a question-conditioned GNN retriever on many buildings and uses it to ground LLM-based KGQA on unseen ones.

## 2. Application Domain

### 2.1 Datasets

| Dataset | Role | Size | Public? |
|---|---|---|---|
| **Build2SPARQL v1.0** [CITE: Jung, 2026] | Training, validation, test | 201 buildings (180 Brick, 21 223P), 6,136 SPARQL queries, 30,680 questions (5 paraphrase registers per query), 6 query patterns | **Yes.** Zenodo, CC BY-NC 4.0 |
| Source building graphs (Mortar [CITE], Brick examples, open223 [CITE], BuildGraph [CITE]) | Graphs on which the queries execute | 4 to 932 instances per building (largest benchmark graph about 63k triples) | **Yes.** Public repositories / Zenodo |
| **BuildingQA** [CITE: Mulayim et al., BuildSys 2025] | Final held-out benchmark | 4 real buildings (3 Brick, 1 223P), 27 expert queries, 188 questions | **Yes.** GitHub |

**Our split.** The split shipped with Build2SPARQL puts exact duplicate buildings (e.g. `HPL` and `NIST-HPL`) on both sides of train/test, so we built our own **leakage-free group-level split**. We removed the BuildingQA building (`bldg11`) and exact duplicates, grouped near-duplicate buildings (query-signature Jaccard ≥ 0.8) and same-source families, and stratified the groups by ontology. No near-duplicate pair crosses splits:

| Split | Buildings | Queries | Questions | Includes |
|---|---:|---:|---:|---|
| Train | 154 | 4,670 | 23,350 | — |
| Validation | 9 | 256 | 1,280 | Real buildings only |
| Test | 33 | 1,012 | 5,060 | 6 223P buildings from 3 independent families; 1 unseen synthetic building type |

**Release.** Both source datasets are already public. Before submission we will publicly release the split and its script, the executed gold answer tables, the derived supervision labels (§2.2), and all code.

[TODO: if the hand-modeled real office building is included as a case study, confirm it can be released (e.g. anonymized), or keep it out of the reported results.]

### 2.2 Task and Metrics

**End task.** The input is a question *q* and a building graph *G = (V, R, E)*. The output is a SPARQL query, whose execution result is an answer table.

**Graph ML task (our focus): question-conditioned retrieval.** Score every node v ∈ V and every edge (u, r, v) ∈ E for relevance to *q*. Then return the top-k relational paths, which are passed to an LLM that writes the SPARQL.

**Supervision comes from execution, with no manual labeling.** Executing each gold SPARQL and binding its triple-pattern variables marks the gold **evidence nodes, edges and class-level paths**, giving about 23k labeled (question, subgraph) pairs.

**Retrieval metrics** (Build2SPARQL test): evidence-node and evidence-edge Recall@k, gold-path Hit@k, schema-linking F1 over ontology terms, and context size.

**End-to-end metrics** (BuildingQA, official `metrics.py`). After aligning the predicted table *P*'s columns to the gold table *G* by the best permutation:

- **Row-Matching F1 (primary):** the harmonic mean of row precision |P ∩ G| / |P| and row recall |P ∩ G| / |G|.
- **Entity-Set F1** (per column, over unique values), **Exact-Match F1** (no permutation) and **Arity F1** (number of columns).

The current best Row-F1 on all four buildings is 0.59 (BSchema [CITE]).

### 2.3 Why These Datasets

- **Many graphs, one task.** Every building is a different graph, so the task tests **inductive** learning on unseen graphs. Build2SPARQL is the only building-KG corpus large enough (about 200 graphs, about 30k questions) to train a GNN.
- **Two ontologies.** Brick and 223P have largely disjoint relation vocabularies, which tests generalization to **unseen relation types**.
- **Free, exact supervision.** Every query executes, so subgraph relevance labels are exact and cost nothing.
- **A hard, standard benchmark.** BuildingQA has expert questions (up to 53 triple patterns, 16 output columns) and published baselines.

## 3. Graph ML Techniques

### 3.1 Model

We use **conditional message passing** from **NBFNet** [CITE: Zhu et al., NeurIPS 2021] and its fully-inductive extension **ULTRA** [CITE: Galkin et al., ICLR 2024]. We adapt it to natural-language queries and path retrieval, in the spirit of **GNN-RAG** [CITE: Mavromatis & Karypis, 2025] and **SubgraphRAG** [CITE: Li et al., ICLR 2025].

```
 question q ──► text encoder ──► q ∈ R^d ─────────────┐
                                                       ▼
 building KG G ──► (a) relation graph G_R ──► relation GNN ──► relation embeddings {w_r}
        │                                                         │
        └──► seed nodes S (linked classes / mentions) ──► (b) entity GNN conditioned on (q, S)
                                                                   │
                                     node scores s(v), edge scores s(u,r,v)
                                                                   ▼
                              (c) top-k relational paths ──► LLM ──► SPARQL ──► answer
```

**(a) Relation graph (ULTRA).** Relations become nodes. Two relations are connected by one of four meta-relations, according to how they share entities: head-to-head, head-to-tail, tail-to-head, tail-to-tail. A small GNN over this graph produces an embedding **w**_r for every relation *of the given building*. No relation embedding table is learned, so unseen relations such as 223P's `connectedTo` still receive representations.

**(b) Question-conditioned entity GNN (NBFNet / generalized Bellman–Ford).** Node states are *pairwise* with the query. The boundary condition injects the question at the seed nodes:

$$h^{(0)}_q(v) = \mathbb{1}[v \in S]\, W_0\, \mathbf{q}$$

$$h^{(t)}_q(v) = \mathrm{LN}\Big(\mathrm{AGG}\big(\{\, h^{(t-1)}_q(u) \odot \mathbf{w}_r^{(t)} : (u, r, v) \in E \,\} \cup \{\, h^{(0)}_q(v) \,\}\big)\Big) + h^{(t-1)}_q(v)$$

Here the messages are DistMult-style, AGG is sum or PNA aggregation, and the shortcut keeps the previous state. After L = 4–6 layers, nodes and edges are scored as

$$s(v) = \sigma\big(\mathrm{MLP}([h^{(L)}_q(v);\, \mathbf{q}])\big), \qquad s(u, r, v) = \sigma\big(\mathrm{MLP}([h^{(L)}_q(u);\, \mathbf{w}_r;\, h^{(L)}_q(v)])\big).$$

**(c) Path retrieval.** We take the top-k nodes by s(v) and extract the highest-scoring paths from the seed nodes to them. Path score is the sum of log edge scores, found with a beam search over the scored graph. Paths are lifted to the class level, e.g. `VAV —feeds→ HVAC_Zone`, and verbalized for the LLM.

**Training.** The loss is $\mathcal{L} = \mathrm{BCE}_{\text{node}} + \mathrm{BCE}_{\text{edge}} + \lambda\, \mathcal{L}_{\text{path}}$. The BCE terms use the execution-derived evidence labels. $\mathcal{L}_{\text{path}}$ is contrastive: the gold class-level path is the positive, and sibling paths with the same endpoints are hard negatives. An optional warm start uses ULTRA-style link-prediction pretraining over all ~200 graphs. The implementation uses PyTorch Geometric.

**Baselines**, all at equal context budget: text-embedding kNN over triples, PCST subgraph selection (G-Retriever) [CITE], an R-GCN node classifier [CITE], SubgraphRAG's triple scorer, and static BSchema summaries [CITE]. **Ablations:** learned relation embeddings instead of the relation graph, unconditional message passing, and no path loss.

### 3.2 Why These Models Fit These Datasets

- **Unseen graphs.** NBFNet/ULTRA have no entity embeddings, so they are inductive over entities by design. Transductive KG embeddings (TransE, RotatE) cannot be applied to a new building.
- **Unseen relations.** ULTRA's relation graph derives relation embeddings from structure. A model trained mostly on Brick can therefore transfer to 223P, the hardest BuildingQA building (best Row-F1 0.30).
- **Expressivity on repetitive graphs.** Building KGs repeat the same motif thousands of times (VAV → zone → sensor). Standard message passing is bounded by 1-WL [CITE: Xu et al., 2019], so structurally identical nodes get identical embeddings and the model cannot tell *which* VAV the question is about. Conditional message passing acts as a labeling trick: it computes representations relative to the question's seeds, which is provably more expressive [CITE: Huang et al., NeurIPS 2023]. The current state of the art, BSchema, is a non-learned 1-WL-style color refinement; our model is its learned, question-conditioned counterpart.
- **Paths match the downstream need.** SPARQL is built from relational paths, and topological mismatch is a path error, so scoring and returning paths targets the dominant failure directly.
- **Scale.** Model size does not depend on graph size. The cost per question is O(L·|E|·d), and it can be restricted to a k-hop region around the seeds.
