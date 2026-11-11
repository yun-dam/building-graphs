# Retriever baselines: does question conditioning matter?

This folder runs controlled comparisons between BRIDGE and **question-agnostic** GNN retrievers. It imports from `bridge/` and does not modify it.

## Setup

Each baseline encodes the graph once with no question, `h = Enc(G)`, and only sees the question in the output heads. BRIDGE instead conditions every message-passing step on the question, NBFNet-style. Everything else is shared with BRIDGE: the node, edge, answer and path heads, the losses, the 154/9/33 building split, the training loop, early stopping, and `bridge.eval_retriever`. The encoder is the only difference.

| model | encoder | params |
|---|---|---:|
| BRIDGE v1 | question-conditioned, 6 entity + 6 relation layers | 308k |
| CompGCN, 2 layers | CompGCN (mult), relation-aware, direction-aware | 239k |
| CompGCN, 6 layers | same, matched to BRIDGE's depth and size | 305k |
| GCN, no relations, 6 layers | same layer, relation types ignored | 305k |
| MLP on node text | no message passing | 206k |

All models use d=64 and BRIDGE's default hyperparameters, with 3 seeds (0, 1, 2), trained on CPU.

## Results (test split, mean ± std over 3 seeds)

**All 33 test buildings**

| model | node R@10 | edge R@50 | path Hit@1 | path Hit@10 | answer F1 |
|---|---:|---:|---:|---:|---:|
| **BRIDGE v1** | **0.851 ± .015** | **0.827 ± .014** | **0.763 ± .005** | **0.959 ± .006** | **0.691 ± .032** |
| CompGCN, 2 layers | 0.744 ± .039 | 0.469 ± .012 | 0.749 ± .021 | 0.922 ± .007 | 0.627 ± .059 |
| CompGCN, 6 layers | 0.651 ± .047 | 0.465 ± .005 | 0.666 ± .031 | 0.917 ± .017 | 0.513 ± .045 |
| GCN, no relations, 6 layers | 0.719 ± .032 | 0.486 ± .010 | 0.676 ± .027 | 0.884 ± .012 | 0.573 ± .038 |
| MLP on node text | 0.605 ± .094 | 0.457 ± .014 | 0.627 ± .047 | 0.816 ± .013 | 0.498 ± .132 |
| text-kNN (no training) | 0.737 | 0.560 | 0.423 | 0.788 | 0.367 |

**ASHRAE 223P buildings (s223)**

| model | node R@10 | edge R@50 | path Hit@10 |
|---|---:|---:|---:|
| **BRIDGE v1** | 0.583 ± .029 | **0.496 ± .049** | **0.839 ± .025** |
| CompGCN, 2 layers | 0.325 ± .069 | 0.191 ± .029 | 0.697 ± .039 |
| text-kNN (no training) | **0.665** | 0.413 | 0.703 |

The full tables with all metrics and all strata are in [results/retriever_eval_seeds.md](results/retriever_eval_seeds.md).

## Takeaways

1. **Question conditioning matters most for edges.** BRIDGE's edge R@50 is 0.83. Every question-agnostic encoder lands at 0.46–0.49, with spreads of about ±0.01. A shared encoding finds relevant nodes but cannot pick out which edges a given question needs.
2. **Top-1 path ranking is close to a tie.** The 2-layer CompGCN is 1.4 points behind BRIDGE on Hit@1, within its own seed spread of ±0.021. The advantage shows up in recall and in the Hit@10 tail.
3. **Depth hurts without conditioning.** At BRIDGE's size and depth, CompGCN loses about 9 points of node R@10 compared with 2 layers, while BRIDGE benefits from 6 layers. This motivates the expressivity study.
4. **Relation awareness alone does not decide it.** At 6 layers, the GCN that ignores relations is ahead of CompGCN on node recall and behind it on Hit@10.
5. **223P transfer is still open.** BRIDGE trails text-kNN on node R@10 for the 223P buildings. The question-agnostic encoders lose about half their performance there.
6. **Answer F1 is the noisiest metric** (MLP: ±0.13), so small F1 differences should not be read as real.

## Results by question bucket

`eval_buckets.py` tags each test question five ways:

- SPARQL pattern
- gold path hops
- answer-set size
- paraphrase source
- stratum

It then reports per-bucket metrics as the mean over 3 seeds. Text-kNN appears as a single number because it has no training. Below, "best baseline" means the best of the four GNN/MLP baselines and text-kNN.

The full tables are in [results/bucket_eval.md](results/bucket_eval.md). The per-question rows, for re-slicing, are in `results/bucket_rows.jsonl.gz`.

| bucket | n | BRIDGE edge R@50 | best baseline | Δ | BRIDGE Hit@1 | best baseline | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| hops = 1 | 3020 | 0.859 | 0.624 (kNN) | **+0.235** | 0.804 | 0.763 (CompGCN-2) | +0.041 |
| hops = 2 | 1435 | 0.825 | 0.593 (GCN) | **+0.232** | 0.697 | 0.730 (CompGCN-2) | −0.033 |
| hops = 3+ | 305 | 0.850 | 0.760 (CompGCN-6) | +0.091 | 0.674 | 0.707 (CompGCN-2) | −0.033 |
| answers 11–100 | 2685 | 0.854 | 0.484 (GCN) | **+0.370** | 0.832 | 0.847 (CompGCN-2) | −0.015 |
| answers >100 | 1160 | 0.610 | 0.511 (kNN) | +0.099 | 0.530 | 0.440 (CompGCN-6) | **+0.090** |
| linear | 3785 | 0.867 | 0.580 (kNN) | **+0.287** | 0.802 | 0.785 | +0.017 |
| branching | 705 | 0.839 | 0.645 (GCN) | +0.194 | 0.619 | 0.657 | −0.038 |
| filtered / aggregation / optional | 140 / 200 / 70 | 0.29 / 0.64 / 0.56 | kNN ties or wins | ≈0 / +0.03 / −0.11 | | | |
| stratum s223 | 1310 | 0.496 | 0.413 (kNN) | +0.083 | 0.468 | 0.340 (kNN) | **+0.128** |
| stratum mortar-brick | 1210 | 0.955 | 0.754 (kNN) | +0.200 | 0.787 | 0.893 (CompGCN-2) | **−0.106** |

**What the buckets show**

1. **The edge-recall gap does not grow with hops; it shrinks.** It is +0.23 at 1 and 2 hops but +0.09 at 3+ hops. The gap is largest for mid-sized answer sets (11–100) and for linear patterns. The hypothesis that "conditioning helps most on multi-hop questions" is not what the data shows.
2. **On top-1 path ranking, the 2-layer CompGCN beats BRIDGE on multi-hop questions** (2 and 3+ hops, branching patterns) and on the mortar-brick stratum (−0.11). BRIDGE wins Hit@1 on large answer sets (>100) and on 223P (+0.13).
3. **On SPARQL modifiers, text-kNN is as good as or better than every learned model on node recall.** This covers the filtered, aggregation, union and optional patterns, about 10% of questions. Filtered questions are hard for every learned model: edge R@50 is 0.29 at best. These buckets have 70–200 questions each, so treat them as directional.
4. **All models are robust across paraphrase sources.** LLM_5_v3 is the hardest source for every model, and BRIDGE drops the most there (edge R@50 0.80 vs 0.84 for the other sources).

**Caveat on the edge comparison.** The shared edge head scores an edge from `[h_s, w_r, h_t]`. In BRIDGE those states depend on the question. In the encode-then-match baselines they do not, so a baseline's edge scores depend on the question only through which nodes enter the top-64 candidate region. This shows up as baseline edge R@50 being constant across paraphrase sources (0.469–0.470).

The edge gap therefore measures "conditioning in the encoder vs no question in the edge head". A fairer baseline would feed `q` into the edge MLP. Until that runs, read the edge-recall gap as an upper bound on the effect of encoder conditioning. Node recall and path Hit@1 are not affected, because those heads already see `q`.

## Reproduce

Run these from the repo root after the `bridge` data pipeline (`labels`, `graph_build`, `data`):

```bash
experiments/seed_chain.sh 1
```

This trains BRIDGE plus the four baselines for one seed and evaluates them on test. For seed 0, the names have no `_s0` suffix. To aggregate:

```bash
python -m experiments.aggregate_seeds runs/exp/retriever_eval_baselines.json runs/exp/retriever_eval_baselines_s1.json runs/exp/retriever_eval_baselines_s2.json
```

## Files

| path | what it holds |
|---|---|
| `encoders.py` | CompGCN / GCN / MLP encoders with BRIDGE's heads |
| `train_baseline.py`, `eval_baseline.py` | training and evaluation, mirroring `bridge.train` / `bridge.eval_retriever` |
| `aggregate_seeds.py`, `seed_chain.sh` | mean ± std over seeds; the per-seed run |
| `eval_buckets.py` | per-question-bucket evaluation (`--from-rows` re-aggregates without rerunning models) |
| `results/bucket_eval.{json,md}`, `results/bucket_rows.jsonl.gz` | bucket results; per-question rows |
| `results/retriever_eval_s{0,1,2}.json` | per-seed, per-stratum test metrics |
| `results/retriever_eval_seeds.{json,md}` | aggregated results |
| `results/retriever_eval_v1_repro_with_knn_pcst.json` | BRIDGE seed 0 plus the text-kNN and PCST baselines (from `bridge.eval_retriever`) |
| `results/train_logs/*.jsonl` | per-epoch validation curves |

**Notes**

- Checkpoints (`runs/`) are not committed.
- The test set has 1005 queries, not the report's 1000, because it uses Brick 1.4.4 rather than 1.4.3.
- On s223, compgcn_l6 has node R@50 below its R@10, so recall drops at the larger cutoff. This also happens with his PCST baseline, so it probably comes from how the evaluator averages per question. It hasn't been investigated.
