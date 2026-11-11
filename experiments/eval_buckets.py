"""Per-question-bucket retriever evaluation: where does question conditioning help?

Re-runs bridge.eval_retriever's per-question metrics (eval_bridge for learned models; the same
text-kNN scoring as bridge.eval_retriever.main) and tags every test question with buckets:

  pattern   SPARQL pattern type from Build2SPARQL (linear, branching, filtered, aggregation, union, optional)
  hops      length of the gold class-level path (1, 2, 3+; "none" if no gold path)
  answers   size of the gold answer set (2-10, 11-100, >100)
  source    which paraphrase generator wrote the question text (Build2SPARQL `source`)
  stratum   buildgraph / mortar-brick / brick-reference / s223

python -m experiments.eval_buckets --seeds 0 1 2            # runs all models, writes per-question rows
python -m experiments.eval_buckets --from-rows               # re-aggregate only
Output: runs/exp/bucket_rows.jsonl, runs/exp/bucket_eval.json, markdown tables on stdout.
"""
import argparse
import collections
import json
import statistics
import time

import torch

from bridge import kg
from bridge.eval_retriever import PREP, KNNBaseline, embed, eval_bridge, f1, path_texts, rec_at, summarize
from bridge.retrieve import InfGraph, Prefixes, load_model
from experiments import encoders

ROOT = kg.ROOT
OUT = ROOT / "runs" / "exp"
BASELINES = ["compgcn_l2", "compgcn_l6", "gcn_norel_l6", "mlp_text"]
BUCKETS = ["pattern", "hops", "answers", "source", "stratum"]
METRICS = ["node_recall@10", "edge_recall@50", "path_hit@1", "path_hit@10", "answer_f1"]


def buckets(s, source, stratum):
    h = len(s["path"]["paths"][0][1]) if s["path"] else None
    a = len(s["ans"])
    return {"pattern": s["pattern"],
            "hops": "none" if h is None else str(h) if h < 3 else "3+",
            "answers": "1" if a <= 1 else "2-10" if a <= 10 else "11-100" if a <= 100 else ">100",
            "source": source, "stratum": stratum}


def knn_rows(ig, samples, qemb, dev):
    knn, res = KNNBaseline(ig), []
    for s in samples:
        q = qemb[s["q_idx"]].to(dev).float()
        eo, no, _ = knn.rank(q)
        r = {"nr": rec_at(no.tolist(), s["nodes"], (10, 50)), "er": rec_at(eo.tolist(), s["edges"], (50, 200)),
             "ans_f1": f1(no[:10].tolist(), s["ans"])}
        if s["path"] and len(s["path"]["paths"]) > 1:
            sc = embed(path_texts(ig, s["path"]["paths"])).to(dev) @ q
            r["rank"] = int((sc > sc[0]).sum()) + 1
        res.append(r)
    return res


def run(seeds, dev, path):
    man = kg.manifest()
    qemb = torch.load(PREP / "qemb.pt", weights_only=False)["emb"]
    models = {}
    for sd in seeds:
        sfx = "" if sd == 0 else f"_s{sd}"
        models[("bridge_v1", sd)] = load_model(ROOT / "runs" / "ckpt" / f"v1_repro{sfx}.pt", dev)[0]
        for m in BASELINES:
            models[(m, sd)] = encoders.load(ROOT / "runs" / "ckpt_exp" / f"{m}{sfx}.pt", dev)[0]
    out, t0 = open(path, "w"), time.time()
    for b in kg.split()["test"]:
        g = torch.load(ROOT / "runs" / "graphs" / f"{b}.pt", weights_only=False)
        prep = torch.load(PREP / f"{b}.pt", weights_only=False)
        g["rel_ids"] = prep["rel_ids"]
        ig = InfGraph(g, Prefixes(ROOT / man[b]["path"]), dev)
        ig.t["is_class"] = prep["is_class"].to(dev)
        lab = json.loads((ROOT / "runs" / "labels" / f"{b}.json").read_text())
        src = {q["query_id"]: q["sources"] for q in lab["queries"]}
        samples = prep["samples"]
        tags = [buckets(s, src[s["qid"]][s["reg"] - 1], man[b]["stratum"]) for s in samples]
        runs = [((m, sd), eval_bridge(model, ig, samples, qemb, dev, False)) for (m, sd), model in models.items()]
        runs.append((("text-kNN", 0), knn_rows(ig, samples, qemb, dev)))
        for (m, sd), res in runs:
            for s, t, r in zip(samples, tags, res):
                out.write(json.dumps({"model": m, "seed": sd, "building": b, "qid": s["qid"], "reg": s["reg"],
                                      **t, **r}) + "\n")
        print(f"{b} ({man[b]['stratum']}) done {time.time()-t0:.0f}s", flush=True)
    out.close()


def aggregate(path):
    rows = collections.defaultdict(list)  # (bucket, value, model, seed) -> rows
    for line in open(path):
        r = json.loads(line)
        r["nr"] = {int(k): v for k, v in r["nr"].items()}
        r["er"] = {int(k): v for k, v in r["er"].items()}
        for bk in BUCKETS + ["all"]:
            rows[(bk, "all" if bk == "all" else r[bk], r["model"], r["seed"])].append(r)
    per_seed = {k: summarize(v) for k, v in rows.items()}
    res = collections.defaultdict(lambda: collections.defaultdict(dict))  # bucket -> value -> model -> stats
    groups = collections.defaultdict(list)
    for (bk, val, m, sd), s in per_seed.items():
        groups[(bk, val, m)].append(s)
    for (bk, val, m), ss in groups.items():
        st = {"n": ss[0]["n_questions"], "n_path": ss[0]["n_path_questions"], "seeds": len(ss)}
        for c in METRICS:
            xs = [s[c] for s in ss if s[c] is not None]
            if xs:
                st[c] = {"mean": statistics.mean(xs), "std": statistics.stdev(xs) if len(xs) > 1 else 0.0}
        res[bk][val][m] = st
    return res


ORDER = {"hops": ["1", "2", "3+", "none"], "answers": ["1", "2-10", "11-100", ">100"],
         "pattern": ["linear", "branching", "filtered", "aggregation", "union", "optional"]}
MODELS = ["bridge_v1", "compgcn_l2", "compgcn_l6", "gcn_norel_l6", "mlp_text", "text-kNN"]


def tables(res, metrics=("edge_recall@50", "path_hit@1", "node_recall@10")):
    lines = []
    for bk in ["all"] + BUCKETS:
        vals = ORDER.get(bk, sorted(res[bk]))
        vals = [v for v in vals if v in res[bk]] + [v for v in sorted(res[bk]) if v not in vals]
        for c in metrics:
            lines += [f"\n### {bk} — {c}\n", "| bucket | n | " + " | ".join(MODELS) + " | BRIDGE − best baseline |",
                      "|---|---:|" + "---:|" * (len(MODELS) + 1)]
            for v in vals:
                ms = res[bk][v]
                n = ms["bridge_v1"]["n_path" if c.startswith("path") else "n"]
                cell = lambda m: (f"{ms[m][c]['mean']:.3f} ± {ms[m][c]['std']:.3f}" if ms[m]["seeds"] > 1
                                  else f"{ms[m][c]['mean']:.3f}") if m in ms and c in ms[m] else "–"
                base = [ms[m][c]["mean"] for m in MODELS[1:] if m in ms and c in ms[m]]
                gap = f"{ms['bridge_v1'][c]['mean'] - max(base):+.3f}" if base and c in ms["bridge_v1"] else "–"
                lines.append(f"| {v} | {n} | " + " | ".join(cell(m) for m in MODELS) + f" | {gap} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--from-rows", action="store_true")
    ap.add_argument("--rows", default=str(OUT / "bucket_rows.jsonl"))
    ap.add_argument("--out", default=str(OUT / "bucket_eval.json"))
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not a.from_rows:
        run(a.seeds, torch.device(a.device), a.rows)
    res = aggregate(a.rows)
    json.dump(res, open(a.out, "w"), indent=1)
    print(tables(res))


if __name__ == "__main__":
    main()
