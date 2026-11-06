"""Paths and helpers shared by BRIDGE-Hybrid (all outputs under runs/hybrid/).

Inputs are read-only: Build2SPARQL graphs via runs/data/nt (v1 skolemized cache, graph variant as
chosen by v1 labelling) and BuildingQA eval graphs (data/BuildingQA/eval_buildings/*.ttl, used ONLY
to build inference graphs / contexts).
"""
import collections
import json
import zlib

import pyoxigraph as ox
import torch

from bridge import kg
from bridge.v2 import common as C2

ROOT = kg.ROOT
H = ROOT / "runs" / "hybrid"
NT = H / "nt"
GDIR = H / "graphs"            # v2 format (attribute nodes), Build2SPARQL buildings
GDIR_BQ = H / "graphs_bq"      # v2 format, BuildingQA eval graphs
G1DIR = H / "graphs_v1"        # v1 format (E1 kNN / E5 skeleton), dev + BuildingQA graphs
LDIR = H / "labels"            # v2-style labels of Build2SPARQL queries on hybrid graphs
SRAW = H / "synth_long_raw"    # long-path synthetic queries (template questions)
SDIR = H / "synth_long"        # + paraphrases (after `synth_long para`)
PREP = H / "prep"
CKPT = H / "ckpt"
CTX = H / "contexts"
BQ_DIR = ROOT / "data" / "BuildingQA" / "eval_buildings"
BQ_BUILDINGS = ["TUC_building", "b59", "bldg11", "bldg_223p_anon", "bldg_brick_anon", "dflexlibs_multizone"]


def bq_stratum(b):
    return "s223" if b in ("b59", "bldg_223p_anon") else "brick"


def clean_nt(name):
    """Cleaned skolemized N-Triples: name = Build2SPARQL building id, or bq_<id> for BuildingQA graphs."""
    return NT / f"{name}.nt"


def b2s_buildings(splits=("train", "val", "test")):
    sp = kg.split()
    return [b for k in splits for b in sp[k]]


def read_nt(path):
    return list(ox.parse(path=str(path), format=ox.RdfFormat.N_TRIPLES))


def literal_map(nt_path):
    """node IRI -> [(predicate IRI, lexical)] (same format as bridge.retrieve.literal_attrs)."""
    lits = collections.defaultdict(list)
    for t in ox.parse(path=str(nt_path), format=ox.RdfFormat.N_TRIPLES):
        if isinstance(t.object, ox.Literal):
            lits[t.subject.value].append((t.predicate.value, t.object.value))
    return lits


def rel_ids(g):
    return torch.tensor([zlib.crc32(n.encode()) % 4096 for n in g["rel_names"]], dtype=torch.long)


def ttl_of(b):
    """Original TTL used by the agent / gold execution (compiled variant where v1 labels chose it)."""
    if b in BQ_BUILDINGS:
        return BQ_DIR / f"{b}.ttl"
    p = ROOT / kg.manifest()[b]["path"].replace("\\", "/")
    return p.with_suffix(".compiled.ttl") if C2.graph_variant(b) == "compiled" else p


def stratum(b):
    return bq_stratum(b) if b in BQ_BUILDINGS else C2.stratum(b)


def inf_graph(b, dev, fmt="v2"):
    """InfGraph on the CLEANED hybrid graph of building b (Build2SPARQL id or BuildingQA id).
    literal attributes (_lits) are read from the cleaned N-Triples, not the v1 cache."""
    from bridge.retrieve import InfGraph, Prefixes
    if fmt == "v2":
        f = (GDIR_BQ if b in BQ_BUILDINGS else GDIR) / f"{b}.pt"
    else:
        f = G1DIR / f"{b}.pt"
    g = torch.load(f, weights_only=False)
    g["rel_ids"] = rel_ids(g)
    ig = InfGraph(g, Prefixes(ttl_of(b)), dev)
    if "is_attr" in g:
        ig.is_attr = g["is_attr"]
    else:
        ig.is_attr = torch.zeros(len(g["node_ids"]), dtype=torch.bool)
    ig._lits = literal_map(clean_nt(("bq_" + b) if b in BQ_BUILDINGS else b))
    return ig


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
