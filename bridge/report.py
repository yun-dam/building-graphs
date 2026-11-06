"""Collect results -> reports/bridge_results.json, reports/figures/*.png, error-analysis table.

The narrative report (reports/bridge_report.md) is written from these numbers.
python -m bridge.report
"""
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
REP = ROOT / "reports"
FIG = REP / "figures"
BLDS = ["TUC_building", "dflexlibs_multizone", "bldg11", "b59"]
BNAME = {"TUC_building": "TUC", "dflexlibs_multizone": "DFLEX", "bldg11": "Mortar/BLDG11", "b59": "LBNL/B59"}
PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6


def load(p, default=None):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


# ----------------------------------------------------------------------------- error analysis
def graph_vocab(building):
    import pyoxigraph as ox
    from bridge import kg
    ttl = ROOT / "data" / "BuildingQA" / "eval_buildings" / f"{building}.ttl"
    voc = set()
    for t in ox.parse(path=str(ttl), format=ox.RdfFormat.TURTLE):
        voc.add(t.predicate.value)
        if t.predicate.value == kg.RDF_TYPE and isinstance(t.object, ox.NamedNode):
            voc.add(t.object.value)
    tb = kg.TBOX_223P if building == "b59" else kg.TBOX_BRICK
    tbox_terms = {t.subject.value for t in kg.tbox_triples(tb)}
    return voc, tbox_terms


def expand_terms(sparql):
    pre = dict(re.findall(r"PREFIX\s+(\w*):\s*<([^>]+)>", sparql or "", re.I))
    from bridge.labels import STD_PREFIXES
    for k, v in STD_PREFIXES.items():
        pre.setdefault(k, v)
    body = re.sub(r"PREFIX[^\n]*\n", "", sparql or "", flags=re.I)
    body = re.sub(r'"[^"]*"', " ", body)
    terms = set()
    for p, l in re.findall(r"(?<![\w?$<])([A-Za-z][\w\-]*):([A-Za-z_][\w\-]*)", body):
        if p in pre and p in ("brick", "s223", "ref"):
            terms.add(pre[p] + l)
    return terms


def categorize(rec, sc, gold_sparql, ctx_block, vocab_cache):
    b = rec["building"]
    if b not in vocab_cache:
        vocab_cache[b] = graph_vocab(b)
    voc, tbox = vocab_cache[b]
    if not rec.get("final_sparql") or rec.get("error") or (rec.get("rows_eval") is None and rec.get("reexec_error")):
        return "execution error"
    used = expand_terms(rec["final_sparql"])
    halluc = [t for t in used if t not in voc and t not in tbox]
    if halluc:
        return "vocabulary hallucination"
    if sc["arity_f1"] < 1.0:
        return "arity mismatch"
    gold_terms = expand_terms(gold_sparql)
    if ctx_block is not None:
        from bridge.retrieve import Prefixes
        missing = []
        for t in gold_terms:
            ln = re.split(r"[#/]", t)[-1]
            if ln not in ctx_block:
                missing.append(t)
        if missing:
            return "retrieval miss"
    if sc["row_f1"] < 1.0:
        return "topological mismatch"
    return "other"


def error_analysis(scores, n=20, root=RUNS / "e2e_bkgqa"):
    per = scores["per_question"]
    runs = sorted(k for k in per if k.startswith("E2/"))
    if not runs:
        return None
    run = runs[0]
    gold_q = {}
    for f in sorted((ROOT / "data" / "BuildingQA" / "Benchmark_QA_pairs").glob("*_combined.json")):
        for rec in json.loads(f.read_text(encoding="utf-8")):
            for q in rec["queries"]:
                gold_q[q["query_id"]] = q["sparql_query"]
    ctx = load(RUNS / "e2e" / "contexts_E2.json", {})
    recs = {}
    for line in (root / "E2" / run.split("/")[1] / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        recs[r["qid"]] = r
    failed = sorted([k for k, s in per[run].items() if s["row_f1"] < 1.0], key=lambda k: (per[run][k]["row_f1"], k))
    # deterministic spread over buildings: round-robin by building
    byb = {}
    for k in failed:
        byb.setdefault(recs[k]["building"], []).append(k)
    pick = []
    while len(pick) < n and any(byb.values()):
        for b in BLDS:
            if byb.get(b) and len(pick) < n:
                pick.append(byb[b].pop(0))
    vc, rows = {}, []
    for k in pick:
        cat = categorize(recs[k], per[run][k], gold_q[k.split("#")[0]], ctx.get(k, {}).get("block"), vc)
        rows.append({"qid": k, "building": recs[k]["building"], "row_f1": per[run][k]["row_f1"], "category": cat})
    all_cats = Counter()
    for k in failed:
        all_cats[categorize(recs[k], per[run][k], gold_q[k.split("#")[0]], ctx.get(k, {}).get("block"), vc)] += 1
    return {"run": run, "sample": rows, "sample_counts": dict(Counter(r["category"] for r in rows)),
            "all_failed_counts": dict(all_cats), "n_failed": len(failed)}


# ----------------------------------------------------------------------------- figures
def fig_buildings(scores):
    cfgs = [c for c in ("E0", "E1", "E2") if c in scores["configs"]]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    w = 0.8 / max(1, len(cfgs))
    colors = {"E0": "#9aa5b1", "E1": "#e0a458", "E2": "#2f6db3"}
    labels = {"E0": "E0 NoKG", "E1": "E1 text-kNN", "E2": "E2 BRIDGE"}
    for i, c in enumerate(cfgs):
        ag = scores["configs"][c]["agg"]
        m = [ag.get(f"{b}_row_f1", {}).get("mean", 0) for b in BLDS]
        s = [ag.get(f"{b}_row_f1", {}).get("std", 0) for b in BLDS]
        ax.bar(np.arange(len(BLDS)) + (i - (len(cfgs) - 1) / 2) * w, m, w, yerr=s, label=labels[c],
               color=colors[c], capsize=2)
    ax.set_xticks(range(len(BLDS)))
    ax.set_xticklabels([BNAME[b] for b in BLDS])
    ax.set_ylabel("Row-matching F1")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.set_title("BuildingQA Row-F1 per building (Gemini 2.5 Flash)", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "rowf1_per_building.png", dpi=160)
    plt.close(fig)


def fig_recall(ret):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    groups = [("Evidence nodes", [("node_recall@10", 10), ("node_recall@50", 50)]),
              ("Evidence edges", [("edge_recall@50", 50), ("edge_recall@200", 200)]),
              ("Gold path", [("path_hit@1", 1), ("path_hit@5", 5), ("path_hit@10", 10)])]
    colors = {"main": "#2f6db3", "text-kNN": "#e0a458", "PCST": "#5a9e6f"}
    for ax, (title, ks) in zip(axes, groups):
        for m in ("main", "text-kNN", "PCST"):
            if m not in ret:
                continue
            ys = [ret[m]["all"].get(k) for k, _ in ks]
            if any(y is None for y in ys):
                continue
            ax.plot([x for _, x in ks], ys, marker="o", label="BRIDGE" if m == "main" else m, color=colors[m])
        ax.set_xscale("log")
        ax.set_xticks([x for _, x in ks])
        ax.set_xticklabels([str(x) for _, x in ks])
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("k")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Recall / Hit@k")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Retriever on Build2SPARQL test (33 unseen buildings)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "retriever_recall_at_k.png", dpi=160)
    plt.close(fig)


# ----------------------------------------------------------------------------- compute/cost
def compute_cost():
    tin = tout = 0
    for f in list(RUNS.glob("e2e*/*/*/records/*.json")) + list(RUNS.glob("dev*/*/*/*/records/*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        tin += r.get("tokens_in", 0)
        tout += r.get("tokens_out", 0)
    return {"gemini_tokens_in": tin, "gemini_tokens_out": tout, "gemini_usd_est": tin * PRICE_IN + tout * PRICE_OUT}


def training_curves():
    out = {}
    for f in sorted((RUNS / "ckpt").glob("*.jsonl")):
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        ep = [r for r in recs if "epoch" in r]
        if not ep:
            continue
        best = max(ep, key=lambda r: r["val"]["path_hit@10"] + 1e-3 * r["val"]["node_recall@50"])
        cfg = next((r["config"] for r in recs if "config" in r), {})
        out[f.stem] = {"epochs_run": len(ep), "best_epoch": best["epoch"], "best_val": best["val"],
                       "train_minutes": round(ep[-1]["elapsed_s"] / 60, 1),
                       "config": {k: cfg.get(k) for k in ("d", "ent_layers", "rel_layers", "lambda_path",
                                                          "hard_seeding", "no_relation_graph", "init")}}
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    ret = load(RUNS / "retriever_eval.json", {})
    scores = load(RUNS / "e2e_bkgqa" / "scores.json", {"configs": {}, "per_question": {}})
    scores_brickqa = load(RUNS / "e2e" / "scores.json", {"configs": {}})
    dev = {ag: load(RUNS / "dev" / ag / "scores.json", {"configs": {}}).get("configs", {}) for ag in ("brickqa", "bkgqa")}
    res = {"retriever": ret, "buildingqa": scores.get("configs", {}),
           "buildingqa_original_brickqa": scores_brickqa.get("configs", {}), "dev_set": dev,
           "training": training_curves(), "scale_check": load(RUNS / "scale_check.json"),
           "cost": compute_cost(), "leakage_pre": (RUNS / "leakage_check.txt").read_text() if (RUNS / "leakage_check.txt").exists() else None,
           "leakage_e2e": (RUNS / "leakage_check_e2e.txt").read_text() if (RUNS / "leakage_check_e2e.txt").exists() else None,
           "metrics_check": (RUNS / "metrics_check.txt").read_text() if (RUNS / "metrics_check.txt").exists() else None}
    if scores.get("per_question"):
        res["error_analysis"] = error_analysis(scores)
        fig_buildings(scores)
    if ret:
        fig_recall(ret)
    for c in res["buildingqa"].values():
        c.pop("per_run_full", None)
    (REP / "bridge_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({"cost": res["cost"], "training": {k: (v["best_epoch"], round(v["best_val"]["path_hit@10"], 3)) for k, v in res["training"].items()},
                      "error_analysis": (res.get("error_analysis") or {}).get("sample_counts")}, indent=1))


if __name__ == "__main__":
    main()
