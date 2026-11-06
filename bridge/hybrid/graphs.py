"""Rebuild GNN graphs from the CLEANED N-Triples (runs/hybrid/nt).

  v2 format (attribute nodes; bridge.v2.graph_build_v2.build): all Build2SPARQL buildings
      -> runs/hybrid/graphs/{b}.pt ; the 6 BuildingQA eval graphs -> runs/hybrid/graphs_bq/{b}.pt
  v1 format (bridge.graph_build.build; for the E1 kNN and E5 skeleton comparison contexts):
      Build2SPARQL val+test buildings and the 6 BuildingQA graphs -> runs/hybrid/graphs_v1/{b}.pt

python -m bridge.hybrid.graphs [--what v2 v2bq v1] [--buildings ...] [--force]
"""
import argparse
import time

import torch

from bridge import graph_build as GB1
from bridge.hybrid import common as H
from bridge.v2 import common as C2
from bridge.v2 import graph_build_v2 as GB2


def _save(g, out, variant):
    g["graph_variant"] = variant
    g["cleaned"] = True
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(g, out)


def main(what=("v2", "v2bq", "v1"), buildings=None, force=False):
    jobs = []
    if "v2" in what:
        jobs += [(b, b, C2.stratum(b), H.GDIR / f"{b}.pt", GB2.build, C2.graph_variant(b)) for b in H.b2s_buildings()]
    if "v2bq" in what:
        jobs += [(b, "bq_" + b, H.bq_stratum(b), H.GDIR_BQ / f"{b}.pt", GB2.build, "raw") for b in H.BQ_BUILDINGS]
    if "v1" in what:
        jobs += [(b, b, C2.stratum(b), H.G1DIR / f"{b}.pt", GB1.build, C2.graph_variant(b))
                 for b in H.b2s_buildings(("val", "test"))]
        jobs += [(b, "bq_" + b, H.bq_stratum(b), H.G1DIR / f"{b}.pt", GB1.build, "raw") for b in H.BQ_BUILDINGS]
    for b, name, strat, out, fn, variant in jobs:
        if buildings and b not in buildings:
            continue
        if out.exists() and not force:
            continue
        t0 = time.time()
        g = fn(name, H.clean_nt(name), strat)
        _save(g, out, variant)
        print(f"{out.parent.name}/{b}: nodes={len(g['node_ids'])} attr={g.get('n_attr', 0)} cls={int(g['is_class'].sum())} "
              f"edges={g['edge_index'].shape[1]} rels={len(g['rel_names'])} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", nargs="*", default=["v2", "v2bq", "v1"])
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(tuple(a.what), a.buildings, a.force)
