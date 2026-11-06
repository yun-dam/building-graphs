"""Leakage-free dev set for agent comparison: 60 questions from Build2SPARQL *test* buildings
(never used for training/selection), 15 per stratum, one random register per query (seed 0).
Gold rows = gold SPARQL executed on the original TTL (+ compiled variant where labels chose it) + TBox.

python -m bridge.dev_set  ->  runs/dev/dev_set.json
"""
import json
import random
import re
from pathlib import Path

import pyoxigraph as ox

from bridge import kg
from bridge.labels import run_with_timeout

ROOT = kg.ROOT
OUT = ROOT / "runs" / "dev" / "dev_set.json"
PER_STRATUM = 15
MAX_GOLD_ROWS = 2000


def ttl_for(b, man):
    lab = json.loads((ROOT / "runs" / "labels" / f"{b}.json").read_text())
    p = ROOT / man[b]["path"]
    return p.with_suffix(".compiled.ttl") if lab.get("graph_variant") == "compiled" else p


def gold_rows(store, sparql):
    rows, err = run_with_timeout(store, sparql, timeout=120)
    if err or not rows:
        return None
    names = [v.value for v in rows[0].variables] if hasattr(rows[0], "variables") else None
    if names is None:
        m = re.search(r"SELECT\s+(DISTINCT\s+)?(.+?)\s+WHERE", sparql, re.I | re.S)
        names = re.findall(r"\?(\w+)", m.group(2))
    out = []
    for sol in rows:
        d = {}
        for n in dict.fromkeys(names):
            try:
                x = sol[n]
            except Exception:  # noqa: BLE001
                x = None
            if x is not None:
                d[n] = {"value": kg.term_str(x)}
        if d:
            out.append(d)
    return out


def main(name="dev_set", splits=("test",), strata=None, per_stratum=PER_STRATUM, exclude=None):
    global PER_STRATUM
    PER_STRATUM = per_stratum
    out_path = OUT.with_name(f"{name}.json")
    man = kg.manifest()
    test = [b for sp in splits for b in kg.split()[sp]]
    rng = random.Random(0)
    excluded = set()
    if exclude:
        excluded = {it["qid"] for it in json.loads(OUT.with_name(f"{exclude}.json").read_text(encoding="utf-8"))}
    by_stratum = {}
    for b in test:
        if strata and man[b]["stratum"] not in strata:
            continue
        by_stratum.setdefault(man[b]["stratum"], []).append(b)
    items = []
    for stratum, blds in sorted(by_stratum.items()):
        pool = []
        for b in blds:
            lab = json.loads((ROOT / "runs" / "labels" / f"{b}.json").read_text())
            meta = {q["query_id"]: q for q in json.loads(
                (ROOT / "data" / "Build2SPARQL_v1.0" / f"{b}.json").read_text(encoding="utf-8"))[0]["queries"]}
            for rec in lab["queries"]:
                if (rec["n_rows"] <= MAX_GOLD_ROWS and rec["pattern_type"] != "aggregation"
                        and f"{b}::{rec['query_id']}" not in excluded):
                    pool.append((b, rec, meta[rec["query_id"]]))
        rng.shuffle(pool)
        # spread over buildings: round-robin
        per_b = {}
        for x in pool:
            per_b.setdefault(x[0], []).append(x)
        picked = []
        while len(picked) < PER_STRATUM and any(per_b.values()):
            for b in sorted(per_b):
                if per_b[b] and len(picked) < PER_STRATUM:
                    picked.append(per_b[b].pop())
        stores = {}
        for b, rec, meta in picked:
            if b not in stores:
                stores[b] = kg.make_store(ttl_for(b, man), kg.tbox_path(stratum), abox_format=ox.RdfFormat.TURTLE)
            g = gold_rows(stores[b], meta["sparql_query"])
            if not g:
                continue
            reg = rng.randrange(5)
            q = meta["questions"][reg]
            items.append({"qkey": f"{b}::{rec['query_id']}#{q['question_number']}", "qid": f"{b}::{rec['query_id']}",
                          "building": b, "group": stratum, "text": q["question_text"], "source": q["source"],
                          "ttl": str(ttl_for(b, man).relative_to(ROOT)), "stratum": stratum,
                          "pattern_type": rec["pattern_type"], "gold_sparql": meta["sparql_query"], "gold_rows": g})
        print(stratum, len([i for i in items if i["stratum"] == stratum]), flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, ensure_ascii=False))
    print("saved", out_path, len(items))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="dev_set")
    ap.add_argument("--splits", nargs="*", default=["test"])
    ap.add_argument("--strata", nargs="*")
    ap.add_argument("--per-stratum", type=int, default=PER_STRATUM)
    ap.add_argument("--exclude")
    a = ap.parse_args()
    main(a.name, tuple(a.splits), a.strata, a.per_stratum, a.exclude)
