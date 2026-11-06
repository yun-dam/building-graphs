"""Answer exactly ONE BuildingQA question in a fresh process (isolated session).

python -m bridge.run_one_question --config E2 --run 1 --qkey TUC_001#1 [--agent bkgqa] [--dataset dev] [--force]

Writes {root}/{config}/{run}/logs/{qkey}.jsonl (every Gemini call) and
{root}/{config}/{run}/records/{qkey}.json (the per-question record);
root = runs/e2e (BrickQA), runs/e2e_bkgqa (BKGQA) or runs/dev/{agent} (dev set).
"""
import argparse
import io
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BQ = ROOT / "data" / "BuildingQA"
E2E = ROOT / "runs" / "e2e"


from bridge.qsets import contexts_dir, out_root, question, safe  # noqa: E402


def reexecute(sparql, ttl, tbox, timeout=180):
    """Execute the final SPARQL on ABox+TBox (same store construction as the gold answers)."""
    import re
    import threading
    import pyoxigraph as ox
    from bridge import kg
    from bridge.labels import ttl_prefix_block, with_prefixes
    store = kg.make_store(ttl, tbox, abox_format=ox.RdfFormat.TURTLE)
    q = with_prefixes(sparql, ttl_prefix_block(ttl))
    q = re.sub(r"<bldg-59#>", "<urn:bldg-59#>", q)  # BrickQA's relative default prefix
    box = {}

    def run():
        try:
            sols = store.query(q, use_default_graph_as_union=True)
            names = [v.value for v in sols.variables]
            out = []
            for sol in sols:
                d = {n: {"value": kg.term_str(sol[n])} for n in names if sol[n] is not None}
                if d:
                    out.append(d)
            box["rows"] = out
        except Exception as e:  # noqa: BLE001
            box["err"] = f"{type(e).__name__}: {e}"[:300]

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        return None, "timeout"
    return box.get("rows"), box.get("err")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--qkey", required=True)
    ap.add_argument("--agent", default="brickqa", choices=["brickqa", "bkgqa"])
    ap.add_argument("--dataset", default="buildingqa", choices=["buildingqa", "dev", "dev223", "devlit"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="", help="suffix of the run directory, e.g. x / xr")
    ap.add_argument("--store", default="default", choices=["default", "Oxigraph"])
    ap.add_argument("--refiner", action="store_true")
    ap.add_argument("--ctx-dir", default=None, help="read contexts_{config}.json from here (default: qsets.contexts_dir)")
    ap.add_argument("--out-root", default=None, help="results root (default: qsets.out_root)")
    a = ap.parse_args()
    if a.out_root:   # absolute paths: the agent later chdirs into runs/e2e/_cwd
        a.out_root = str(Path(a.out_root) if Path(a.out_root).is_absolute() else ROOT / a.out_root)
    if a.ctx_dir:
        a.ctx_dir = str(Path(a.ctx_dir) if Path(a.ctx_dir).is_absolute() else ROOT / a.ctx_dir)
    rdir = (Path(a.out_root) if a.out_root else out_root(a.dataset, a.agent)) / f"{a.config}{a.tag}" / str(a.run)
    rec_path = rdir / "records" / f"{safe(a.qkey)}.json"
    if rec_path.exists() and not a.force:
        print(f"exists: {rec_path} (use --force)")
        return 0
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    q = question(a.dataset, a.qkey)
    building, question_text, source = q["building"], q["text"], q["source"]
    ctx_block, ctx_tokens = None, 0
    if a.config in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"):
        ctx = json.loads(((Path(a.ctx_dir) if a.ctx_dir else contexts_dir(a.dataset)) / f"contexts_{a.config}.json").read_text(encoding="utf-8"))[a.qkey]
        ctx_block, ctx_tokens = ctx["block"], ctx["tokens"]
    agent_start = datetime.now().isoformat(timespec="microseconds")
    assert os.environ.get("GOOGLE_CLOUD_PROJECT"), "set GOOGLE_CLOUD_PROJECT"
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    from bridge import brickqa_adapter as ad
    log = ad.CallLog(rdir / "logs" / f"{safe(a.qkey)}.jsonl")
    log.write({"event": "start", "pid": os.getpid(), "agent_start": agent_start, "config": a.config,
               "run": a.run, "qkey": a.qkey, "building": building, "agent": a.agent, "dataset": a.dataset})
    ad.prepare_cwd()
    ad.install_patches(log)
    t0 = time.time()
    rec = {"qkey": safe(a.qkey), "qid": a.qkey, "config": a.config, "run": a.run, "building": building,
           "group": q["group"], "agent": a.agent, "dataset": a.dataset, "tag": a.tag, "store": a.store,
           "refiner": a.refiner, "question": question_text, "source": source, "pid": os.getpid(), "agent_start": agent_start,
           "context_tokens": ctx_tokens}
    stdout = io.StringIO()
    try:
        with redirect_stdout(stdout):
            if a.agent == "bkgqa":
                agent = ad.make_bkgqa_agent(q["ttl"], ctx_block, rdir / "agent_logs", store=a.store,
                                            projection_refiner=a.refiner)
            else:
                agent = ad.make_agent(q["ttl"], ctx_block)
            state, final = agent.run(question_text, verbose=True)
        rec["final_sparql"] = final.sparql if final else None
        rec["rows_agent"] = [{k: {"value": v} for k, v in r.items()} for r in (final.execution_result or [])] if final else []
        rec["agent_status"] = str(final.execution_status) if final else None
        rec["n_actions"] = len(state.actions)
        rec["n_tool_calls"] = sum(1 for x in state.actions if x.action_name not in ("stop", "error"))
        rec["error"] = None
    except Exception as e:  # noqa: BLE001
        rec.update({"final_sparql": None, "rows_agent": [], "error": f"{type(e).__name__}: {e}"[:500],
                    "traceback": traceback.format_exc()[-2000:]})
    rec["latency_s"] = round(time.time() - t0, 1)
    rec["rows_eval"], rec["reexec_error"] = (None, "no sparql")
    if rec.get("final_sparql"):
        try:
            rec["rows_eval"], rec["reexec_error"] = reexecute(rec["final_sparql"], q["ttl"], q["tbox"])
        except Exception as e:  # noqa: BLE001
            rec["rows_eval"], rec["reexec_error"] = None, f"{type(e).__name__}: {e}"[:300]
    rec["tokens_in"], rec["tokens_out"] = log.tokens_in, log.tokens_out
    rec["llm_calls"], rec["llm_retries"] = log.calls, log.retries
    rec["rows_agent"] = rec["rows_agent"][:20000]
    if rec["rows_eval"] is not None:
        rec["rows_eval"] = rec["rows_eval"][:20000]
    log.write({"event": "end", "latency_s": rec["latency_s"], "error": rec["error"]})
    (rdir / "logs" / f"{safe(a.qkey)}.stdout.txt").write_text(stdout.getvalue()[-200000:], encoding="utf-8")
    rec_path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: rec[k] for k in ("qid", "config", "run", "latency_s", "tokens_in", "tokens_out", "error")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
