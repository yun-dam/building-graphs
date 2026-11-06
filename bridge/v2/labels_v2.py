"""BRIDGE v2 labels for the Build2SPARQL queries (train/val for training; test for evaluation only).

Same execution as bridge.labels (pyoxigraph, ABox in a named graph + TBox, union default graph,
60 s timeout, SELECT * rewrite capped at 5000 rows, <= 60 queries/building sampled with seed 0),
on the graph variant (raw|compiled) chosen by v1, but literal bindings map to attribute nodes
(see bridge.v2.labeler).

python -m bridge.v2.labels_v2 [--buildings ...] [--split train val test] [--procs 8] [--force]
Output: runs/v2/labels/{b}.json
"""
import argparse
import json
import multiprocessing as mp
import random
import time
from concurrent.futures import ThreadPoolExecutor

from bridge.labels import MAX_Q, TIMEOUT, building_meta
from bridge.v2 import common as C
from bridge.v2.labeler import Ctx, label_query, s223_weight


def _worker(b, skip, conn):
    try:
        ctx = Ctx(b)
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
                 "sparql_query": q["sparql_query"],
                 "weight": s223_weight(ctx.stratum, q["sparql_query"]),
                 "questions": [x["question_text"] for x in qs], "sources": [x["source"] for x in qs]}
            r.update(rec)
            out.append(r)
        n_exec = len(out)
        if len(out) > MAX_Q:
            out = sorted(random.Random(0).sample(out, MAX_Q), key=lambda r: r["query_id"])
        conn.send(("done", {"building": b, "stratum": ctx.stratum, "graph_variant": C.graph_variant(b),
                            "split": C.split_of(b), "n_queries": len(meta["queries"]), "n_exec": n_exec,
                            "n_kept": len(out), "stats": stats, "queries": out}))
    except Exception as e:  # noqa: BLE001
        conn.send(("fail", f"{type(e).__name__}: {e}"))
    conn.close()


def run_building(b):
    """Child process per building; a query hanging past the timeout kills it and is skipped."""
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
    from bridge.v2.graph_build_v2 import buildings_for
    C.LDIR.mkdir(parents=True, exist_ok=True)
    todo = [b for b in (buildings or buildings_for(splits))
            if force or not (C.LDIR / f"{b}.json").exists()]
    with ThreadPoolExecutor(procs) as ex:
        for res in ex.map(run_building, todo):
            b = res["building"]
            if "fail" in res:
                (C.LDIR / f"{b}.fail.json").write_text(json.dumps(res))
                print(f"{b}: FAIL {res['fail']}", flush=True)
                continue
            (C.LDIR / f"{b}.json").write_text(json.dumps(res))
            nl = sum(1 for q in res["queries"] if q["lit_vars"])
            print(f"{b}: {res['n_exec']}/{res['n_queries']} exec, kept {res['n_kept']} "
                  f"(lit-proj {nl}) {res['stats']} {res['seconds']}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--split", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.buildings, tuple(a.split), a.procs, a.force)
