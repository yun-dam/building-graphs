"""devlit: a BuildingQA-like dev set (literal columns, 4-7 hop query graphs) for agent/context decisions.

Queries are mined like bridge.hybrid.synth_long (on the cleaned graphs) but from Build2SPARQL TEST
buildings (VAL buildings only to fill a stratum that TEST cannot fill; signatures of the long-synth
val queries are excluded). Evaluation only - never used for training or early stopping.
  * 15 queries per stratum (s223 / mortar-brick / buildgraph / brick-reference), round-robin over
    buildings, seed 0; >= 1 literal column (where the stratum has literals at all), 4-7 hops;
  * gold rows = execution of the gold SPARQL on the ORIGINAL TTL (compiled variant where v1 labels chose
    it) + TBox, exactly like bridge.dev_set; 1..2000 rows; queries with blank-node values are dropped.
  * `text` is the template question for now (`text_template` keeps it); `para` (NOT run here) replaces
    `text` by one Gemini paraphrase (register drawn with seed 0) and keeps `text_template`.

python -m bridge.hybrid.dev_sets build            -> runs/dev/dev_setlit.json (+ stats)
python -m bridge.hybrid.dev_sets stats
python -m bridge.hybrid.dev_sets para [--yes-call-gemini]   (dry run by default)
"""
import argparse
import collections
import json
import math
import random
import zlib

import pyoxigraph as ox

from bridge import kg
from bridge.dev_set import gold_rows
from bridge.hybrid import common as H
from bridge.hybrid.synth_long import hybrid_ctx, mine
from bridge.v2 import common as C2

OUT = H.ROOT / "runs" / "dev" / "dev_setlit.json"
PER_STRATUM = 15
PER_BUILDING_MINE = 14
HOP_W_DEV = {4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0}
STRATA = ("s223", "mortar-brick", "buildgraph", "brick-reference")


def val_long_sigs(b):
    f = H.SRAW / f"{b}.json"
    return {q["signature"] for q in json.loads(f.read_text())["queries"]} if f.exists() else set()


def has_bnode(rows):
    return any(v["value"].startswith(("_:", "urn:bnode:")) for r in rows for v in r.values())


def pool_for(b):
    ctx = hybrid_ctx(b)
    seed = zlib.crc32(("devlit:" + b).encode())
    qs, st = mine(ctx, PER_BUILDING_MINE, HOP_W_DEV, seed, require_lit=True, exclude_sigs=val_long_sigs(b),
                  prefix="LIT")
    if not qs:   # strata without literal attributes: fall back to IRI-valued attribute columns
        qs, st = mine(ctx, PER_BUILDING_MINE, HOP_W_DEV, seed, require_lit=False, exclude_sigs=val_long_sigs(b),
                      prefix="LIT")
    store = kg.make_store(H.ttl_of(b), kg.tbox_path(ctx.stratum), abox_format=ox.RdfFormat.TURTLE)
    out = []
    for q in qs:
        rows = gold_rows(store, q["sparql_query"])
        if not rows or len(rows) > 2000 or has_bnode(rows):
            continue
        out.append((q, rows))
    return out, dict(st)


def build():
    sp = kg.split()
    rng = random.Random(0)
    items, mined = [], {}
    for stratum in STRATA:
        picked = []
        for split in ("test", "val"):
            if len(picked) >= PER_STRATUM:
                break
            blds = [b for b in sp[split] if C2.stratum(b) == stratum]
            per_b = {}
            for b in blds:
                pool, st = pool_for(b)
                mined[b] = {"split": split, "valid_gold": len(pool), "stats": st}
                rng.shuffle(pool)
                pool.sort(key=lambda x: x[0]["n_lit_cols"] == 0)       # stable: literal queries first
                per_b[b] = pool
            while len(picked) < PER_STRATUM and any(per_b.values()):
                for b in sorted(per_b):
                    if per_b[b] and len(picked) < PER_STRATUM:
                        picked.append((b, *per_b[b].pop(0)))
        for b, q, rows in picked:
            qid = f"{b}::{q['query_id']}"
            items.append({"qkey": f"{qid}#1", "qid": qid, "building": b, "group": stratum, "text": q["template"],
                          "text_template": q["template"], "source": "template",
                          "ttl": str(H.ttl_of(b).relative_to(H.ROOT)), "stratum": stratum,
                          "pattern_type": q["pattern_type"], "gold_sparql": q["sparql_query"], "gold_rows": rows,
                          "split": C2.split_of(b), "n_hops": q["n_hops"], "span": q["span"],
                          "n_cols": q["n_cols"], "n_lit_cols": q["n_lit_cols"], "signature": q["signature"]})
        print(stratum, sum(i["stratum"] == stratum for i in items), flush=True)
    OUT.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    (H.H / "devlit_mining.json").write_text(json.dumps(mined, indent=1))
    print("saved", OUT, len(items))
    stats()


def stats():
    items = json.loads(OUT.read_text(encoding="utf-8"))
    print(f"devlit: {len(items)} questions, {len({i['building'] for i in items})} buildings "
          f"({collections.Counter(i['split'] for i in items)})")
    for st in STRATA + ("ALL",):
        xs = [i for i in items if st == "ALL" or i["stratum"] == st]
        if not xs:
            continue
        print(f"  {st:16s} n={len(xs):3d} hops={dict(sorted(collections.Counter(i['n_hops'] for i in xs).items()))} "
              f"span={dict(sorted(collections.Counter(i['span'] for i in xs).items()))} "
              f"literal-col share={sum(i['n_lit_cols'] for i in xs) / max(1, sum(i['n_cols'] for i in xs)):.2f} "
              f"q with literal col={sum(i['n_lit_cols'] > 0 for i in xs)} "
              f"gold rows median={sorted(len(i['gold_rows']) for i in xs)[len(xs) // 2]}")


def para(yes, budget):
    items = json.loads(OUT.read_text(encoding="utf-8"))
    todo = [(f"D|{i['qid']}", i["text_template"], i["gold_sparql"]) for i in items]
    n_chunks = math.ceil(len(todo) / 8)
    print(f"{len(todo)} devlit questions -> {n_chunks} Gemini 2.5 Flash requests (est. ${n_chunks * 0.002:.3f})")
    if not yes:
        print("DRY RUN: no Gemini request made. Re-run with --yes-call-gemini.")
        return
    from bridge.v2 import synth as S2
    C2.V2 = H.H          # cache runs/hybrid/para_cache.jsonl, usage runs/hybrid/gemini_usage.jsonl (runtime only)
    got = S2.paraphrase_all(todo, budget)
    rng = random.Random(0)
    n = 0
    for i in items:
        reg = rng.choice(["technical", "plain"])
        p = got.get(f"D|{i['qid']}")
        if p:
            i["text"], i["source"] = p[reg], f"gemini_{reg}"
            n += 1
    OUT.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    print(f"paraphrased {n}/{len(items)}; regenerate the devlit contexts (E1/E5/E9) afterwards")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "stats", "para"])
    ap.add_argument("--yes-call-gemini", action="store_true")
    ap.add_argument("--budget", type=float, default=0.5)
    a = ap.parse_args()
    {"build": build, "stats": stats, "para": lambda: para(a.yes_call_gemini, a.budget)}[a.cmd]()
