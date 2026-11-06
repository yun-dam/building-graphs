"""Shared helpers for BRIDGE v2 (paths, graph variant, attribute-node ids).

Only Build2SPARQL train/val/test buildings are touched; nothing under data/BuildingQA except
the Brick / 223P ontology TBox files (same as v1, via bridge.kg).
"""
import collections
import hashlib
import json

import pyoxigraph as ox

from bridge import kg

ROOT = kg.ROOT
V2 = ROOT / "runs" / "v2"
GDIR = V2 / "graphs"
LDIR = V2 / "labels"
SDIR = V2 / "synth"
SRAW = V2 / "synth_raw"
PREP = V2 / "prep"
CKPT = V2 / "ckpt"
V1_LABELS = ROOT / "runs" / "labels"
MAX_ATTR_PER_SUBJ = 12
MAX_LIT_CHARS = 160


def graph_variant(b):
    """raw|compiled, as selected by v1 labelling (more executable gold queries)."""
    f = V1_LABELS / f"{b}.json"
    return json.loads(f.read_text())["graph_variant"] if f.exists() else "raw"


def nt_path(b):
    man = kg.manifest()
    ttl = ROOT / man[b]["path"].replace("\\", "/")
    if graph_variant(b) == "compiled":
        return kg.nt_cache(b + "@compiled", ttl.with_suffix(".compiled.ttl"))
    return kg.nt_cache(b, ttl)


def stratum(b):
    return kg.manifest()[b]["stratum"]


def split_of(b):
    sp = kg.split()
    for k in ("train", "val", "test"):
        if b in sp[k]:
            return k
    return None


def attr_id(s, p, lit):
    h = hashlib.sha1(f"{s}\x00{p}\x00{lit}".encode("utf-8")).hexdigest()[:20]
    return f"lit:{h}"


def is_attr(n):
    return n.startswith("lit:")


def skip_literal(s, p):
    return s.startswith(kg.OWL) or p.startswith(kg.OWL)


def kept_literals(abox):
    """subject -> [(p, lexical)] kept as attribute nodes: rdfs:label first, then (p, value) order, cap 12."""
    per = collections.defaultdict(set)
    for t in abox:
        if isinstance(t.object, ox.Literal):
            s, p = t.subject.value, t.predicate.value
            if skip_literal(s, p):
                continue
            per[s].add((p, t.object.value))
    out = {}
    for s, items in per.items():
        items = sorted(items, key=lambda x: (x[0] != kg.LABEL, x[0], x[1]))
        out[s] = items[:MAX_ATTR_PER_SUBJ]
    return out


def kept_attr_ids(abox):
    return {attr_id(s, p, v) for s, its in kept_literals(abox).items() for p, v in its}


def read_abox(nt):
    return list(ox.parse(path=str(nt), format=ox.RdfFormat.N_TRIPLES))
