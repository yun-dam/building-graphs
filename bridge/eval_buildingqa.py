"""Step 8 driver: BuildingQA end-to-end (E0 NoKG, E1 text-kNN context, E2 BRIDGE context).

python -m bridge.eval_buildingqa contexts --ckpt runs/ckpt/main.pt      # precompute E1/E2 blocks
python -m bridge.eval_buildingqa run --config E2 --run 1 [--workers 4] [--force] [--limit N]
python -m bridge.eval_buildingqa score                                    # -> runs/e2e/scores.json
python -m bridge.eval_buildingqa verify-metrics
Global options (before the sub-command): --agent {brickqa,bkgqa} --dataset {buildingqa,dev}

Each question is answered by a fresh OS process (bridge.run_one_question).
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BQ = ROOT / "data" / "BuildingQA"
E2E = ROOT / "runs" / "e2e"
PRICE_IN, PRICE_OUT = 0.15 / 1e6, 0.60 / 1e6
QUESTION_TIMEOUT = 1500
sys.path.insert(0, str(BQ))


from bridge import qsets  # noqa: E402

CFG = {"agent": "brickqa", "dataset": "buildingqa", "tag": "", "store": "default", "refiner": False,
       "ctx_dir": None, "out_root": None}


def root():
    if CFG.get("out_root"):
        return Path(CFG["out_root"])
    return qsets.out_root(CFG["dataset"], CFG["agent"])


def ctx_dir():
    return Path(CFG["ctx_dir"]) if CFG.get("ctx_dir") else qsets.contexts_dir(CFG["dataset"])


def questions():
    return list(qsets.questions(CFG["dataset"]))


safe = qsets.safe


# ----------------------------------------------------------------------------- contexts
def cmd_contexts_v3(a):
    """E3 blocks: BRIDGE v3 verbalization (paths + instance triples + answer candidates + literals)."""
    import torch
    from bridge.retrieve import bq_graph, bridge_context_v3, load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(a.ckpt, dev)
    e3, by_b = {}, collections.defaultdict(list)
    for q in questions():
        by_b[q["building"]].append(q)
    for b, items in by_b.items():
        ig = bq_graph(b, dev) if CFG["dataset"] == "buildingqa" else dev_graph(b, dev)
        for q in items:
            blk, info = bridge_context_v3(model, ig, q["text"])
            e3[q["qkey"]] = {"block": blk, "tokens": info["tokens"], "latency_s": info["latency_s"],
                             "n_paths": info["n_paths"], "n_triples": info["n_triples"], "class_paths": info["class_paths"]}
        print(b, len(items), "E3 tokens", np.mean([e3[q["qkey"]]["tokens"] for q in items]).round(0), flush=True)
    cdir = qsets.contexts_dir(CFG["dataset"])
    (cdir / "contexts_E3.json").write_text(json.dumps(e3, ensure_ascii=False), encoding="utf-8")


def cmd_contexts_v4(a):
    """E4 blocks: whole-building schema skeleton + E3 (BRIDGE v3) detail."""
    import torch
    from bridge.retrieve import bq_graph, e4_context, load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(a.ckpt, dev)
    e4, by_b = {}, collections.defaultdict(list)
    for q in questions():
        by_b[q["building"]].append(q)
    for b, items in by_b.items():
        ig = bq_graph(b, dev) if CFG["dataset"] == "buildingqa" else dev_graph(b, dev)
        for q in items:
            blk, info = e4_context(model, ig, q["text"])
            e4[q["qkey"]] = {"block": blk, "tokens": info["tokens"], "skeleton_tokens": info["skeleton_tokens"],
                             "class_paths": info["class_paths"]}
        print(b, len(items), "E4 tokens", np.mean([e4[q["qkey"]]["tokens"] for q in items]).round(0), flush=True)
    cdir = qsets.contexts_dir(CFG["dataset"])
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "contexts_E4.json").write_text(json.dumps(e4, ensure_ascii=False), encoding="utf-8")
    # E5 ablation: skeleton only (question-independent, no learned retrieval)
    from bridge.retrieve import e5_context
    e5 = {}
    for b, items in by_b.items():
        ig = bq_graph(b, dev) if CFG["dataset"] == "buildingqa" else dev_graph(b, dev)
        blk, info = e5_context(ig)
        for q in items:
            e5[q["qkey"]] = {"block": blk, "tokens": info["tokens"]}
    (cdir / "contexts_E5.json").write_text(json.dumps(e5, ensure_ascii=False), encoding="utf-8")


def cmd_contexts_oracle(a):
    """E6 (dev sets only): gold-evidence context, an upper bound for retrieval."""
    import torch
    from bridge.retrieve import oracle_context
    assert CFG["dataset"] != "buildingqa", "oracle contexts are a dev-only diagnostic"
    dev = torch.device("cpu")
    e6, by_b = {}, collections.defaultdict(list)
    for q in questions():
        by_b[q["building"]].append(q)
    for b, items in by_b.items():
        ig = dev_graph(b, dev)
        lab = {r["query_id"]: r for r in json.loads((ROOT / "runs" / "labels" / f"{b}.json").read_text())["queries"]}
        for q in items:
            blk, info = oracle_context(ig, lab[q["qid"].split("::")[1]])
            e6[q["qkey"]] = {"block": blk, "tokens": info["tokens"]}
        print(b, len(items), "E6 tokens", np.mean([e6[q["qkey"]]["tokens"] for q in items]).round(0), flush=True)
    (qsets.contexts_dir(CFG["dataset"]) / "contexts_E6.json").write_text(json.dumps(e6, ensure_ascii=False), encoding="utf-8")


def cmd_contexts_v7(a):
    """E7 blocks: whole-building skeleton + GNN-compiled executable query skeletons (with results)."""
    import pyoxigraph as ox
    import torch
    from bridge import kg
    from bridge.retrieve import bq_graph, e7_context, load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(a.ckpt, dev)
    e7, by_b = {}, collections.defaultdict(list)
    for q in questions():
        by_b[q["building"]].append(q)
    for b, items in by_b.items():
        ig = bq_graph(b, dev) if CFG["dataset"] == "buildingqa" else dev_graph(b, dev)
        store = kg.make_store(items[0]["ttl"], items[0]["tbox"], abox_format=ox.RdfFormat.TURTLE)
        for q in items:
            blk, info = e7_context(model, ig, q["text"], store)
            e7[q["qkey"]] = {"block": blk, "tokens": info["tokens"], "n_candidates": info["n_candidates"],
                             "candidates": [c["sparql"] for c in info["candidates"]]}
        print(b, len(items), "E7 tokens", np.mean([e7[q["qkey"]]["tokens"] for q in items]).round(0), flush=True)
    cdir = qsets.contexts_dir(CFG["dataset"])
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "contexts_E7.json").write_text(json.dumps(e7, ensure_ascii=False), encoding="utf-8")


def cmd_contexts(a):
    import torch
    from bridge.retrieve import TripleIndex, bq_graph, bridge_context, knn_context, load_model
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_model(a.ckpt, dev)
    qs = questions()
    e1, e2 = {}, {}
    by_b = collections.defaultdict(list)
    for q in qs:
        by_b[q["building"]].append(q)
    for b, items in by_b.items():
        ig = bq_graph(b, dev) if CFG["dataset"] == "buildingqa" else dev_graph(b, dev)
        idx = TripleIndex(ig)
        for q in items:
            blk, info = bridge_context(model, ig, q["text"])
            e2[q["qkey"]] = {"block": blk, "tokens": info["tokens"], "latency_s": info["latency_s"],
                             "n_paths": info["n_paths"], "class_paths": info["class_paths"]}
            blk, info = knn_context(idx, ig, q["text"])
            e1[q["qkey"]] = {"block": blk, "tokens": info["tokens"], "latency_s": info["latency_s"],
                             "n_triples": info["n_triples"]}
        print(b, len(items), "E2 tokens", np.mean([e2[q["qkey"]]["tokens"] for q in items]).round(0),
              "E1 tokens", np.mean([e1[q["qkey"]]["tokens"] for q in items]).round(0), flush=True)
    cdir = qsets.contexts_dir(CFG["dataset"])
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "contexts_E1.json").write_text(json.dumps(e1, ensure_ascii=False), encoding="utf-8")
    (cdir / "contexts_E2.json").write_text(json.dumps(e2, ensure_ascii=False), encoding="utf-8")
    (cdir / "contexts_meta.json").write_text(json.dumps({"ckpt": str(a.ckpt), "config": cfg}, default=str))


def dev_graph(b, dev):
    """Build2SPARQL test building graph (built in Step 3) for dev-set contexts."""
    import torch
    from bridge import kg
    from bridge.retrieve import InfGraph, Prefixes
    g = torch.load(ROOT / "runs" / "graphs" / f"{b}.pt", weights_only=False)
    g["rel_ids"] = torch.load(ROOT / "runs" / "prep" / f"{b}.pt", weights_only=False)["rel_ids"]
    return InfGraph(g, Prefixes(ROOT / kg.manifest()[b]["path"]), dev)


# ----------------------------------------------------------------------------- run
def run_one(config, run, qkey, force):
    rec = root() / f"{config}{CFG['tag']}" / str(run) / "records" / f"{safe(qkey)}.json"
    if rec.exists() and not force:
        return qkey, "skip", 0
    env = dict(os.environ, GOOGLE_CLOUD_LOCATION="us-central1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    cmd = [sys.executable, "-m", "bridge.run_one_question", "--config", config, "--run", str(run), "--qkey", qkey,
           "--agent", CFG["agent"], "--dataset", CFG["dataset"], "--tag", CFG["tag"], "--store", CFG["store"]]
    if CFG["refiner"]:
        cmd.append("--refiner")
    if CFG.get("ctx_dir"):
        cmd += ["--ctx-dir", str(CFG["ctx_dir"])]
    if CFG.get("out_root"):
        cmd += ["--out-root", str(CFG["out_root"])]
    if force:
        cmd.append("--force")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=QUESTION_TIMEOUT)
    except subprocess.TimeoutExpired:
        write_timeout_record(config, run, qkey, time.time() - t0)
        return qkey, f"fail: wall-clock timeout {QUESTION_TIMEOUT}s (killed, scored 0)", time.time() - t0
    status = "ok" if p.returncode == 0 and rec.exists() else f"fail rc={p.returncode} {p.stderr[-300:]}"
    return qkey, status, time.time() - t0


def write_timeout_record(config, run, qkey, dt):
    """The question process was killed: record it as a failed answer (no SPARQL), keeping its token usage."""
    rdir = root() / f"{config}{CFG['tag']}" / str(run)
    log = rdir / "logs" / f"{safe(qkey)}.jsonl"
    lines = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    start = next((l for l in lines if l.get("event") == "start"), {})
    q = next(x for x in questions() if x["qkey"] == qkey)
    rec = {"qkey": safe(qkey), "qid": qkey, "config": config, "run": str(run), "building": q["building"],
           "group": q["group"], "agent": CFG["agent"], "dataset": CFG["dataset"],
           "question": q["text"], "source": q["source"], "pid": start.get("pid"), "agent_start": start.get("agent_start"),
           "context_tokens": 0, "final_sparql": None, "rows_agent": [], "rows_eval": None,
           "reexec_error": "no sparql", "error": f"wall-clock timeout {QUESTION_TIMEOUT}s; process killed",
           "latency_s": round(dt, 1), "n_tool_calls": sum(1 for l in lines if "response" in l),
           "tokens_in": sum(l.get("tokens_in", 0) for l in lines), "tokens_out": sum(l.get("tokens_out", 0) for l in lines),
           "llm_calls": sum(1 for l in lines if "response" in l), "llm_retries": 0}
    if config in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"):
        rec["context_tokens"] = json.loads((ctx_dir() / f"contexts_{config}.json").read_text(encoding="utf-8"))[qkey]["tokens"]
    (rdir / "records").mkdir(parents=True, exist_ok=True)
    (rdir / "records" / f"{safe(qkey)}.json").write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def cost_so_far():
    tin = tout = 0
    for f in (list((ROOT / "runs").glob("e2e*/*/*/records/*.json")) + list((ROOT / "runs").glob("dev*/*/*/*/records/*.json"))
              + list((ROOT / "runs" / "hybrid" / "e2e").glob("*/*/*/records/*.json"))):
        r = json.loads(f.read_text(encoding="utf-8"))
        tin += r.get("tokens_in", 0)
        tout += r.get("tokens_out", 0)
    return tin, tout, tin * PRICE_IN + tout * PRICE_OUT


def cmd_run(a):
    qs = questions()
    if a.limit:
        qs = qs[: a.limit]
    print(f"{a.config} run {a.run}: {len(qs)} questions, {a.workers} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(a.workers) as ex:
        redo = set()
        if a.rerun_errors:
            for q in qs:
                p = root() / f"{a.config}{CFG['tag']}" / str(a.run) / "records" / f"{safe(q['qkey'])}.json"
                if p.exists() and a.rerun_errors in (json.loads(p.read_text(encoding="utf-8")).get("error") or ""):
                    redo.add(q["qkey"])
            print(f"re-running {len(redo)} records whose error contains {a.rerun_errors!r}: {sorted(redo)}", flush=True)
            qs = [q for q in qs if q["qkey"] in redo]
        futs = [ex.submit(run_one, a.config, a.run, q["qkey"], a.force or q["qkey"] in redo) for q in qs]
        for f in futs:
            qkey, status, dt = f.result()
            done += 1
            if status != "skip":
                tin, tout, usd = cost_so_far()
                print(f"[{done}/{len(qs)}] {qkey} {status[:200]} {dt:.0f}s | total ${usd:.2f}", flush=True)
                if usd > a.max_usd:
                    print("COST CAP REACHED; stopping", flush=True)
                    ex.shutdown(cancel_futures=True)
                    break
    # consolidate results.jsonl
    rdir = root() / f"{a.config}{CFG['tag']}" / str(a.run)
    with open(rdir / "results.jsonl", "w", encoding="utf-8") as f:
        for q in questions():
            p = rdir / "records" / f"{safe(q['qkey'])}.json"
            if p.exists():
                f.write(p.read_text(encoding="utf-8").strip() + "\n")


# ----------------------------------------------------------------------------- scoring
def score_record(r, gold_rows):
    import metrics as M
    pred = r.get("rows_eval")
    if pred is None:
        pred = r.get("rows_agent") or []
    if not pred:
        return {"arity_f1": 0.0, "entity_f1": 0.0, "row_f1": 0.0, "exact_f1": 0.0}
    em = M.get_entity_and_row_matching_f1(gold_rows, pred)
    return {"arity_f1": float(M.get_arity_matching_f1(pred, gold_rows)), "entity_f1": float(em["entity_set_f1"]),
            "row_f1": float(em["row_matching_f1"]), "exact_f1": float(M.get_exact_match_f1(gold_rows, pred))}


def _score_job(t):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return score_record(t[1], t[2])


def cmd_score(a):
    gold = qsets.gold(CFG["dataset"])
    qs = {q["qkey"]: q for q in questions()}
    E2E = root()
    per = {}  # (config, run) -> {qkey: scores}
    cache_f = E2E / "score_cache.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    todo = []
    for res in sorted(E2E.glob("*/*/results.jsonl")):
        cfg, run = res.parent.parent.name, res.parent.name
        if run == "test":
            continue
        for line in res.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            key = f"{cfg}/{run}/{r['qid']}/{r.get('agent_start')}"
            if key not in cache:
                todo.append((key, r, gold[r["qid"].split("#")[0]]))
    if todo:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(8) as ex:
            for i, (key, sc) in enumerate(zip([t[0] for t in todo], ex.map(_score_job, todo, chunksize=2))):
                cache[key] = sc
                if i % 50 == 0:
                    cache_f.write_text(json.dumps(cache))
        cache_f.write_text(json.dumps(cache))
    for res in sorted(E2E.glob("*/*/results.jsonl")):
        cfg, run = res.parent.parent.name, res.parent.name
        if run == "test":
            continue
        d = {}
        for line in res.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            key = f"{cfg}/{run}/{r['qid']}/{r.get('agent_start')}"
            s = dict(cache[key])
            s.update({"tokens_in": r.get("tokens_in", 0), "tokens_out": r.get("tokens_out", 0),
                      "latency_s": r.get("latency_s"), "tool_calls": r.get("n_tool_calls"),
                      "error": r.get("error"), "context_tokens": r.get("context_tokens", 0)})
            d[r["qid"]] = s
        per[(cfg, run)] = d
    cache_f.write_text(json.dumps(cache))
    metrics = ["row_f1", "entity_f1", "exact_f1", "arity_f1"]
    blds = sorted({q["group"] for q in qs.values()})
    out = {"configs": {}}
    for cfg in sorted({c for c, _ in per}):
        runs = sorted(r for c, r in per if c == cfg)
        rs = []
        for run in runs:
            d = per[(cfg, run)]
            row = {"n": len(d)}
            for m in metrics:
                row[f"qw_{m}"] = float(np.mean([d[k][m] for k in d])) if d else 0
                pb = [np.mean([d[k][m] for k in d if qs[k]["group"] == b]) for b in blds
                      if any(qs[k]["group"] == b for k in d)]
                row[f"macro_{m}"] = float(np.mean(pb)) if pb else 0
                for b in blds:
                    v = [d[k][m] for k in d if qs[k]["group"] == b]
                    row[f"{b}_{m}"] = float(np.mean(v)) if v else None
                for src in sorted({q["source"] for q in qs.values()}):
                    v = [d[k][m] for k in d if qs[k]["source"] == src]
                    row[f"src_{src}_{m}"] = float(np.mean(v)) if v else None
            tin = sum(d[k]["tokens_in"] for k in d)
            tout = sum(d[k]["tokens_out"] for k in d)
            row.update({"tokens_in": tin, "tokens_out": tout, "usd": tin * PRICE_IN + tout * PRICE_OUT,
                        "mean_latency_s": float(np.mean([d[k]["latency_s"] or 0 for k in d])),
                        "mean_tool_calls": float(np.mean([d[k]["tool_calls"] or 0 for k in d])),
                        "mean_context_tokens": float(np.mean([d[k]["context_tokens"] or 0 for k in d])),
                        "n_errors": sum(1 for k in d if d[k]["error"])})
            rs.append(row)
        agg = {}
        for key in rs[0]:
            vals = [r[key] for r in rs if r.get(key) is not None]
            if vals and isinstance(vals[0], (int, float)):
                agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)) if len(vals) > 1 else 0.0}
        out["configs"][cfg] = {"runs": runs, "per_run": rs, "agg": agg}
    out["per_question"] = {f"{c}/{r}": d for (c, r), d in per.items()}
    (E2E / "scores.json").write_text(json.dumps(out, indent=1))
    for cfg, v in out["configs"].items():
        ag = v["agg"]
        print(cfg, v["runs"], "Row qw %.3f±%.3f macro %.3f | Ent %.3f Exact %.3f Arity %.3f | $%.2f" % (
            ag["qw_row_f1"]["mean"], ag["qw_row_f1"]["std"], ag["macro_row_f1"]["mean"],
            ag["qw_entity_f1"]["mean"], ag["qw_exact_f1"]["mean"], ag["qw_arity_f1"]["mean"], ag["usd"]["mean"]))
        print("   per bldg row:", {b: round(ag[f"{b}_row_f1"]["mean"], 3) for b in blds if f"{b}_row_f1" in ag})


def cmd_verify_metrics(a):
    import copy
    import metrics as M
    gold = json.loads((ROOT / "runs" / "data" / "buildingqa_gold.json").read_text())
    keys = ["TUC_001", "DFLEXLIBS_002", "MORTAR_007", "LBNL_003", "MORTAR_008"]
    lines = []
    for k in keys:
        g = gold[k]["rows"]
        em = M.get_entity_and_row_matching_f1(g, copy.deepcopy(g))
        lines.append(f"gold-vs-gold {k}: arity={M.get_arity_matching_f1(g, g):.3f} entity={em['entity_set_f1']:.3f} "
                     f"row={em['row_matching_f1']:.3f} exact={M.get_exact_match_f1(g, copy.deepcopy(g)):.3f}")
    for i, k in enumerate(keys):
        g = gold[k]["rows"]
        c = copy.deepcopy(g)
        kind = ["drop half rows", "drop a column", "corrupt values", "add junk rows", "empty prediction"][i]
        if i == 0:
            c = c[: max(1, len(c) // 2)] if len(c) > 1 else [{kk: {"value": "x"} for kk in c[0]}]
        elif i == 1:
            col = list(c[0])[-1]
            c = [{kk: v for kk, v in r.items() if kk != col} for r in c]
        elif i == 2:
            for r in c:
                for kk in r:
                    r[kk] = {"value": r[kk]["value"] + "_corrupt"}
        elif i == 3:
            c = c + [{kk: {"value": f"junk{j}"} for kk in c[0]} for j in range(len(c))]
        else:
            c = []
        if c:
            em = M.get_entity_and_row_matching_f1(g, c)
            lines.append(f"corrupted ({kind}) {k}: arity={M.get_arity_matching_f1(c, g):.3f} entity={em['entity_set_f1']:.3f} "
                         f"row={em['row_matching_f1']:.3f} exact={M.get_exact_match_f1(g, c):.3f}")
        else:
            lines.append(f"corrupted ({kind}) {k}: scored as 0 by our wrapper (no rows)")
    txt = "\n".join(lines)
    print(txt)
    (ROOT / "runs" / "metrics_check.txt").write_text(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="brickqa", choices=["brickqa", "bkgqa"])
    ap.add_argument("--dataset", default="buildingqa", choices=["buildingqa", "dev", "dev223", "devlit"])
    ap.add_argument("--tag", default="")
    ap.add_argument("--store", default="default", choices=["default", "Oxigraph"])
    ap.add_argument("--refiner", action="store_true")
    ap.add_argument("--ctx-dir", default=None, help="contexts directory override (default: qsets.contexts_dir)")
    ap.add_argument("--out-root", default=None, help="results root override (default: qsets.out_root)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("contexts"); c.add_argument("--ckpt", default=str(ROOT / "runs" / "ckpt" / "main.pt"))
    c3 = sub.add_parser("contexts-v3"); c3.add_argument("--ckpt", default=str(ROOT / "runs" / "ckpt" / "main.pt"))
    sub.add_parser("contexts-oracle")
    c7 = sub.add_parser("contexts-v7"); c7.add_argument("--ckpt", default=str(ROOT / "runs" / "ckpt" / "main.pt"))
    c4 = sub.add_parser("contexts-v4"); c4.add_argument("--ckpt", default=str(ROOT / "runs" / "ckpt" / "main.pt"))
    r = sub.add_parser("run")
    r.add_argument("--config", required=True); r.add_argument("--run", required=True)
    r.add_argument("--workers", type=int, default=4); r.add_argument("--force", action="store_true")
    r.add_argument("--limit", type=int); r.add_argument("--rerun-errors", default=None); r.add_argument("--max-usd", type=float, default=900.0)
    sub.add_parser("score")
    sub.add_parser("verify-metrics")
    a = ap.parse_args()
    absol = lambda x: (str(Path(x) if Path(x).is_absolute() else ROOT / x) if x else None)  # noqa: E731
    CFG.update(agent=a.agent, dataset=a.dataset, tag=a.tag, store=a.store, refiner=a.refiner,
               ctx_dir=absol(a.ctx_dir), out_root=absol(a.out_root))
    {"contexts": cmd_contexts, "contexts-v3": cmd_contexts_v3, "contexts-v4": cmd_contexts_v4, "contexts-oracle": cmd_contexts_oracle, "contexts-v7": cmd_contexts_v7, "run": cmd_run, "score": cmd_score, "verify-metrics": cmd_verify_metrics}[a.cmd](a)


if __name__ == "__main__":
    main()
