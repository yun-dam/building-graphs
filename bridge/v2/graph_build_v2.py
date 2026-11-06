"""BRIDGE v2 graph: v1 graph (bridge.graph_build.build) + literal ATTRIBUTE NODES.

Every kept literal triple (s, p, "lit") becomes a node  lit:<sha1(s,p,lit)>  with text
"<pred local name>: <literal>" and an edge s -p-> attr (+ inverse, like every edge type).
Cap 12 literal triples per subject (rdfs:label first). Literal text is also kept in the owner's
text, as in v1. Node order: instances | attribute nodes | classes, so n_inst counts instances +
attribute nodes and class nodes remain the tail (bridge.data relies on that). Extra keys:
is_attr [N] bool, n_attr.

python -m bridge.v2.graph_build_v2 [--buildings ...] [--split train val test] [--force]
Output: runs/v2/graphs/{b}.pt
"""
import argparse
import collections
import time

import pyoxigraph as ox
import torch

from bridge import kg
from bridge.graph_build import embed, prefix_of, rel_text, split_name, tbox_info
from bridge.v2 import common as C


def relation_graph(ei, et):
    """ULTRA meta-relation graph (h2h, h2t, t2h, t2t) - same construction as v1."""
    heads, tails = collections.defaultdict(set), collections.defaultdict(set)
    for s, d, r in zip(ei[0].tolist(), ei[1].tolist(), et.tolist()):
        heads[s].add(r)
        tails[d].add(r)
    meta = {k: set() for k in range(4)}
    for v in set(heads) | set(tails):
        H, T = heads.get(v, ()), tails.get(v, ())
        for a in H:
            for b in H:
                if a != b:
                    meta[0].add((a, b))
            for b in T:
                meta[1].add((a, b))
        for a in T:
            for b in H:
                meta[2].add((a, b))
            for b in T:
                if a != b:
                    meta[3].add((a, b))
    rs, rd, rt = [], [], []
    for k, pairs in meta.items():
        for a, b in sorted(pairs):
            rs.append(a); rd.append(b); rt.append(k)
    return torch.tensor([rs, rd], dtype=torch.long), torch.tensor(rt, dtype=torch.long)


def build(building, nt_path, stratum, max_lit_chars=C.MAX_LIT_CHARS):
    tb_triples = kg.tbox_triples(kg.tbox_path(stratum))
    abox = C.read_abox(nt_path)
    anc, parents = kg.subclass_closure(tb_triples + abox)
    tlabels, tdefs = tbox_info(tb_triples)
    kept = C.kept_literals(abox)

    inst, lits, labels = {}, collections.defaultdict(list), {}
    types = collections.defaultdict(set)
    triples = []
    for t in abox:
        s, p, o = t.subject.value, t.predicate.value, t.object
        if p == kg.SUBCLASS or s.startswith(kg.OWL) or p.startswith(kg.OWL):
            continue
        if isinstance(o, ox.Literal):
            inst.setdefault(s, None)
            if p == kg.LABEL:
                labels.setdefault(s, o.value)
            else:
                lits[s].append(f"{split_name(kg.local_name(p))} {o.value}")
            continue
        if p == kg.RDF_TYPE:
            if o.value.startswith(kg.OWL):
                continue
            inst.setdefault(s, None)
            types[s].add(o.value)
            continue
        inst.setdefault(s, None)
        inst.setdefault(o.value, None)
        triples.append((s, p, o.value))
    classes = set()
    for ts in types.values():
        for c in ts:
            classes |= {a for a in anc.get(c, {c}) if not a.startswith(kg.OWL)}
    inst_ids = [n for n in inst if n not in classes]
    inst_set = set(inst_ids)
    # attribute nodes
    attr_ids, attr_text, attr_triples = [], [], []
    for s in inst_ids:
        for p, v in kept.get(s, ()):
            a = C.attr_id(s, p, v)
            attr_ids.append(a)
            attr_text.append(f"{split_name(kg.local_name(p))}: {v}"[: max_lit_chars + 40])
            attr_triples.append((s, p, a))
    cls_ids = sorted(classes)
    node_ids = inst_ids + attr_ids + cls_ids
    idx = {n: i for i, n in enumerate(node_ids)}
    assert len(idx) == len(node_ids), "duplicate node ids"
    n_inst = len(inst_ids) + len(attr_ids)

    def primary(n):
        ts = types.get(n)
        if not ts:
            return None
        return sorted(ts, key=lambda c: (-len(anc.get(c, {c})), c))[0]

    prim = ([idx.get(primary(n), -1) if primary(n) else -1 for n in inst_ids]
            + [-1] * len(attr_ids) + [-1] * len(cls_ids))

    preds = sorted({p for _, p, _ in triples} | {p for _, p, _ in attr_triples} | {kg.RDF_TYPE, kg.SUBCLASS})
    pid = {p: i for i, p in enumerate(preds)}
    src, dst, rel = [], [], []

    def add(s, p, o):
        src.append(idx[s]); dst.append(idx[o]); rel.append(2 * pid[p])

    for s, p, o in triples:
        if s in idx and o in idx:
            add(s, p, o)
    for s, p, a in attr_triples:
        add(s, p, a)
    for s, ts in types.items():
        if s in inst_set:
            for c in ts:
                if c in idx:
                    add(s, kg.RDF_TYPE, c)
    for c in cls_ids:
        for pa in parents.get(c, ()):
            if pa in idx and pa != c:
                add(c, kg.SUBCLASS, pa)
    E = len(src)
    ei = torch.tensor([src + dst, dst + src], dtype=torch.long)
    et = torch.tensor(rel + [r + 1 for r in rel], dtype=torch.long)

    texts = []
    for n in inst_ids:
        parts = []
        if not n.startswith("urn:bnode:"):
            parts.append(split_name(kg.local_name(n)))
        if n in labels:
            parts.append(labels[n])
        pc = primary(n)
        if pc:
            parts.append(split_name(kg.local_name(pc)))
        if lits.get(n):
            parts.append(" ; ".join(lits[n])[:max_lit_chars])
        texts.append(" | ".join(x for x in parts if x) or "entity")
    texts += attr_text
    for c in cls_ids:
        name = tlabels.get(c) or split_name(kg.local_name(c))
        texts.append(f"{prefix_of(c)} class {name}. {tdefs.get(c, '')}"[:300])
    rel_names, rel_texts = [], []
    for p in preds:
        rel_names += [p, "^" + p]
        rel_texts += [rel_text(p), rel_text(p, inverse=True)]
    rg, rgt = relation_graph(ei, et)
    return {
        "building": building, "stratum": stratum, "version": 2,
        "node_ids": node_ids, "n_inst": n_inst, "n_attr": len(attr_ids),
        "is_class": torch.tensor([0] * n_inst + [1] * len(cls_ids), dtype=torch.bool),
        "is_attr": torch.tensor([0] * len(inst_ids) + [1] * len(attr_ids) + [0] * len(cls_ids), dtype=torch.bool),
        "primary_type": torch.tensor(prim, dtype=torch.long),
        "texts": texts, "x": embed(texts).half(),
        "edge_index": ei, "edge_type": et, "n_fwd_edges": E,
        "rel_names": rel_names, "rel_texts": rel_texts, "rel_x": embed(rel_texts).half(),
        "rel_graph": rg, "rel_graph_type": rgt,
    }


def buildings_for(splits):
    sp = kg.split()
    return [b for k in splits for b in sp[k]]


def main(buildings=None, splits=("train", "val", "test"), force=False):
    C.GDIR.mkdir(parents=True, exist_ok=True)
    todo = buildings or buildings_for(splits)
    for b in todo:
        out = C.GDIR / f"{b}.pt"
        if out.exists() and not force:
            continue
        t0 = time.time()
        g = build(b, C.nt_path(b), C.stratum(b))
        g["graph_variant"] = C.graph_variant(b)
        torch.save(g, out)
        print(f"{b}: nodes={len(g['node_ids'])} attr={g['n_attr']} cls={int(g['is_class'].sum())} "
              f"edges={g['edge_index'].shape[1]} rels={len(g['rel_names'])} {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--split", nargs="*", default=["train", "val", "test"])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.buildings, tuple(a.split), a.force)
