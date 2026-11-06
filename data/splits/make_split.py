"""Leakage-free building-level split of Build2SPARQL v1.0 for BRIDGE.

Rules
1. Exclude BuildingQA evaluation buildings (bldg11 = BuildingQA Mortar).
2. Exact duplicates (identical query-signature sets, Jaccard = 1.0): keep one canonical copy.
3. Near duplicates / same source family are grouped; a group never spans splits.
   - Data-driven: union-find over building pairs with query-signature Jaccard >= 0.8.
   - Source families: BuildGraph by building type (all vintages and instances; types
     linked by a near-duplicate pair are merged),
     Guideline 36 (Brick and 223P separately), meter examples, PNNL, LBNL, NIST-HPL, NIST-IBAL.
4. Stratify by ontology/corpus: buildgraph, mortar-brick, brick-reference, s223.
   Targets ~70/10/20 (train/val/test) by query count, greedy over seeded-shuffled groups.
   The s223 stratum is assigned by hand so that test covers distinct real-building families.

Usage: python data/splits/make_split.py
"""
import glob
import itertools
import json
import os
import random
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "..", "Build2SPARQL_v1.0")
OUT = os.path.join(ROOT, "brickqa_g_split.json")
SEED = 42
NEAR_DUP = 0.8
TARGET = {"test": 0.20, "val": 0.10}

EXCLUDE = {"bldg11": "BuildingQA evaluation building (Mortar)"}
# Exact duplicates (Jaccard = 1.0): dropped copy -> kept canonical
DEDUP = {
    "NIST-HPL": "HPL",
    "NIST-IBAL": "IBAL",
    "virtual-meter": "multiple-submeters",
    "smc": "bldg30",
}
FAMILIES = {
    "g36-brick": ["g36-ahu-a9", "g36-combined-ahu-vav", "g36-vav-a2"],
    "meters": ["building_meter", "main-and-submeter", "multiple-submeters", "single-meter"],
    "g36-223p": [f"guideline36-2021-A-{i}" for i in (1, 2, 3, 4, 7, 8, 9)],
    "pnnl": ["pnnl-bdg1-2", "pnnl-bdg2-1", "pnnl-bdg3-2"],
    "lbnl": ["lbnl-bdg3-1", "lbnl-example-radiant"],
}
# s223 stratum (21 buildings, 19 after dedup): hand assignment by family
S223_ASSIGN = {
    "test": ["pnnl", "lbnl", "scb-vrf"],
    "val": ["IBAL", "nist-bdg1-1"],
}


def signature_set(queries):
    sigs = set()
    for q in queries:
        s = q["sparql_query"]
        s = s[s.find("SELECT"):] if "SELECT" in s else s
        s = re.sub(r"\?\w+", "?v", s)
        sigs.add(re.sub(r"\s+", " ", s).strip())
    return sigs


def load():
    meta = {}
    for f in glob.glob(os.path.join(SRC, "*.json")):
        name = os.path.basename(f)[:-5]
        if name == "split":
            continue
        rec = json.load(open(f, encoding="utf-8"))[0]
        qs = rec["queries"]
        meta[name] = {
            "corpus": rec["corpus"],
            "n_queries": len(qs),
            "s223": any("s223" in q["sparql_query"] for q in qs),
            "sigs": signature_set(qs),
        }
    return meta


def stratum(name, m):
    if m["s223"]:
        return "s223"
    if m["corpus"] == "buildgraph":
        return "buildgraph"
    if m["corpus"] == "mortar":
        return "mortar-brick"
    return "brick-reference"


def build_groups(names, meta):
    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    for a, b in itertools.combinations(names, 2):
        sa, sb = meta[a]["sigs"], meta[b]["sigs"]
        if len(sa & sb) / len(sa | sb) >= NEAR_DUP:
            # never merge across strata; BuildGraph types with near-identical
            # buildings (e.g. hotel/outpatient) end up in one group
            if stratum(a, meta[a]) == stratum(b, meta[b]):
                union(a, b)
    for members in FAMILIES.values():
        members = [m for m in members if m in parent]
        for m in members[1:]:
            union(members[0], m)
    for n in names:
        if n.startswith("BG_"):
            btype = n[3:].rsplit("_", 2)[0]
            parent.setdefault("BGTYPE_" + btype, "BGTYPE_" + btype)
            union(n, "BGTYPE_" + btype)

    groups = defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    return list(groups.values())


def group_label(members):
    for fam, fm in FAMILIES.items():
        if set(members) & set(fm):
            return fam
    if members[0].startswith("BG_"):
        types = sorted({m[3:].rsplit("_", 2)[0] for m in members})
        return "BG_" + "+".join(types)
    return "+".join(sorted(members))


def assign(groups, meta):
    rng = random.Random(SEED)
    split = {}
    by_stratum = defaultdict(list)
    for g in groups:
        by_stratum[stratum(g[0], meta[g[0]])].append(g)

    for st, gs in by_stratum.items():
        if st == "s223":
            for g in gs:
                label = group_label(g)
                part = next((p for p, labels in S223_ASSIGN.items() if label in labels), "train")
                for n in g:
                    split[n] = part
            continue
        total = sum(meta[n]["n_queries"] for g in gs for n in g)
        gs = sorted(gs, key=lambda g: sorted(g)[0])
        rng.shuffle(gs)
        filled = {"test": 0, "val": 0}
        for g in gs:
            q = sum(meta[n]["n_queries"] for n in g)
            part = "train"
            for p in ("test", "val"):
                if filled[p] < TARGET[p] * total and filled[p] + q <= 1.3 * TARGET[p] * total:
                    part = p
                    break
            if part != "train":
                filled[part] += q
            for n in g:
                split[n] = part
    return split


def main():
    meta = load()
    names = sorted(n for n in meta if n not in EXCLUDE and n not in DEDUP)
    groups = build_groups(names, meta)
    split = assign(groups, meta)

    out = {
        "description": "Leakage-free building-level split of Build2SPARQL v1.0 for BRIDGE (see make_split.py).",
        "seed": SEED,
        "near_duplicate_jaccard": NEAR_DUP,
        "excluded": EXCLUDE,
        "deduplicated": {k: f"exact duplicate of {v}" for k, v in DEDUP.items()},
        "train": sorted(n for n, p in split.items() if p == "train"),
        "val": sorted(n for n, p in split.items() if p == "val"),
        "test": sorted(n for n, p in split.items() if p == "test"),
        "groups": {group_label(g): sorted(g) for g in groups if len(g) > 1},
        "stats": {},
    }
    for part in ("train", "val", "test"):
        rows = defaultdict(lambda: [0, 0])
        for n in out[part]:
            st = stratum(n, meta[n])
            rows[st][0] += 1
            rows[st][1] += meta[n]["n_queries"]
        out["stats"][part] = {
            "buildings": len(out[part]),
            "queries": sum(meta[n]["n_queries"] for n in out[part]),
            "by_stratum": {k: {"buildings": v[0], "queries": v[1]} for k, v in sorted(rows.items())},
        }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["stats"], indent=2))
    for part in ("val", "test"):
        print(part, out[part])


if __name__ == "__main__":
    main()
