"""v2-style labels of the Build2SPARQL queries, rebuilt on the CLEANED hybrid graphs.

Identical to bridge.v2.labels_v2 (same sampling, timeouts, bridge.v2.labeler.label_query) except that
the ABox is runs/hybrid/nt/{b}.nt (ontology-term filtered) + TBox.

python -m bridge.hybrid.labels_h [--buildings ...] [--split train val test] [--procs 8] [--force]
Output: runs/hybrid/labels/{b}.json

python -m bridge.hybrid.labels_h --synth-v2      # re-label the v2 short synthetic queries (runs/v2/synth,
                                                 # incl. their paraphrases) on the hybrid graphs
Output: runs/hybrid/synth_v2/{b}.json  (queries whose labels became empty are dropped)
"""
import argparse
import json
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor

from bridge.hybrid import common as H
from bridge.labels import MAX_Q, TIMEOUT, building_meta
from bridge.v2 import common as C2
from bridge.v2.labeler import Ctx, label_query, s223_weight


def _worker(b, skip, conn):
    try:
        ctx = Ctx(b, nt=H.clean_nt(b), strat=C2.stratum(b))
        meta = building_meta(b)
        out, stats = [], {}
        for q in meta["queries"]:
            if q["query_id"] in skip:
                stats["timeout"] = stats.get("timeout", 0) + 1
                continue
            conn.send(("progress", q["query_id"]))
            rec, why = label_query(ctx, q["sparql_query"], TIMEOUT)
            if rec is None:
                k = why.split(":")[0]
                stats[k] = stats.get(k, 0) + 1
                continue
            stats["ok"] = stats.get("ok", 0) + 1
            qs = sorted(q["questions"], key=lambda x: x["question_number"])
            r = {"query_id": q["query_id"], "pattern_type": q["pattern_type"], "source": "b2s",
                 "sparql_query": q["sparql_query"], "weight": s223_weight(ctx.stratum, q["sparql_query"]),
                 "questions": [x["question_text"] for x in qs], "sources": [x["source"] for x in qs]}
            r.update(rec)
            out.append(r)
        n_exec = len(out)
        if len(out) > MAX_Q:
            out = sorted(random.Random(0).sample(out, MAX_Q), key=lambda r: r["query_id"])
        conn.send(("done", {"building": b, "stratum": ctx.stratum, "graph_variant": C2.graph_variant(b),
                            "split": C2.split_of(b), "cleaned": True, "n_queries": len(meta["queries"]),
                            "n_exec": n_exec, "n_kept": len(out), "stats": stats, "queries": out}))
    except Exception as e:  # noqa: BLE001
        conn.send(("fail", f"{type(e).__name__}: {e}"))
    conn.close()


def run_building(b):
    skip = set()
    t0 = time.time()
    while True:
        parent, child = mp.Pipe(duplex=False)
        p = mp.Process(target=_worker, args=(b, tuple(skip), child), daemon=True)
        p.start()
        child.close()
        cur, last = None, time.time()
        while True:
            if parent.poll(5):
                try:
                    kind, payload = parent.recv()
                except EOFError:
                    p.join()
                    return {"building": b, "fail": "worker died"}
                if kind == "progress":
                    cur, last = payload, time.time()
                elif kind == "done":
                    p.join()
                    payload["seconds"] = round(time.time() - t0, 1)
                    return payload
                else:
                    p.join()
                    return {"building": b, "fail": payload}
            elif cur and time.time() - last > TIMEOUT + 30:
                p.kill()
                skip.add(cur)
                print(f"  [{b}] hard-timeout on {cur}; restarting", flush=True)
                break
            elif not p.is_alive() and not parent.poll(0):
                return {"building": b, "fail": "worker exited"}


def main(buildings=None, splits=("train", "val", "test"), procs=8, force=False):
    H.LDIR.mkdir(parents=True, exist_ok=True)
    todo = [b for b in (buildings or H.b2s_buildings(splits)) if force or not (H.LDIR / f"{b}.json").exists()]
    diffs = []
    with ThreadPoolExecutor(procs) as ex:
        for res in ex.map(run_building, todo):
            b = res["building"]
            if "fail" in res:
                (H.LDIR / f"{b}.fail.json").write_text(json.dumps(res))
                print(f"{b}: FAIL {res['fail']}", flush=True)
                continue
            (H.LDIR / f"{b}.json").write_text(json.dumps(res))
            old = C2.LDIR / f"{b}.json"
            if old.exists():
                o = json.loads(old.read_text())
                if o["n_exec"] != res["n_exec"]:
                    diffs.append((b, o["n_exec"], res["n_exec"]))
    print(f"labelled {len(todo)} buildings; n_exec differs from runs/v2/labels for {len(diffs)}: {diffs}")


def _relabel_synth_job(b):
    try:
        src = json.loads((C2.SDIR / f"{b}.json").read_text())
        ctx = Ctx(b, nt=H.clean_nt(b), strat=C2.stratum(b))
        out, drop = [], 0
        for q in src["queries"]:
            rec, why = label_query(ctx, q["sparql_query"], 30)
            if rec is None or not rec["Y_ans"]:
                drop += 1
                continue
            out.append({**{k: v for k, v in q.items() if k not in rec}, **rec})
        return dict(src, queries=out, cleaned=True, relabel_dropped=drop)
    except Exception as e:  # noqa: BLE001
        return {"building": b, "fail": f"{type(e).__name__}: {e}"}


def main_synth_v2(procs=8):
    from concurrent.futures import ProcessPoolExecutor
    out_dir = H.H / "synth_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [b for b in H.b2s_buildings(("train", "val")) if (C2.SDIR / f"{b}.json").exists()]
    n = d = 0
    with ProcessPoolExecutor(procs) as ex:
        for b, r in zip(todo, ex.map(_relabel_synth_job, todo)):
            if "fail" in r:
                print(b, "FAIL", r["fail"], flush=True)
                continue
            (out_dir / f"{b}.json").write_text(json.dumps(r))
            n += len(r["queries"]); d += r["relabel_dropped"]
    print(f"v2 synth relabelled on hybrid graphs: {len(todo)} buildings, {n} queries kept, {d} dropped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--split", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--synth-v2", action="store_true")
    a = ap.parse_args()
    if a.synth_v2:
        main_synth_v2(a.procs)
    else:
        main(a.buildings, tuple(a.split), a.procs, a.force)
