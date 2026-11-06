"""Step 3: build a per-building graph for the GNN.

Nodes: ABox instances (incl. skolemized external-reference nodes) + ontology class nodes
(classes used + ancestors). Edges: ABox predicates, rdf:type, rdfs:subClassOf, plus an inverse
for every edge type (relation id 2k = forward, 2k+1 = inverse). Literals are appended to the
owning node's text. Features: frozen all-MiniLM-L6-v2 (384-d).

Output: runs/graphs/{building}.pt  (dict of tensors + id lists)
"""
import argparse
import collections
import json
import re
from pathlib import Path

import pyoxigraph as ox
import torch

from bridge import kg

ROOT = kg.ROOT
OUT = ROOT / "runs" / "graphs"
ENC_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_ENC = None


def encoder():
    global _ENC
    if _ENC is None:
        from sentence_transformers import SentenceTransformer
        _ENC = SentenceTransformer(ENC_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    return _ENC


@torch.no_grad()
def embed(texts, bs=512):
    if not texts:
        return torch.zeros(0, 384)
    v = encoder().encode(texts, batch_size=bs, convert_to_tensor=True, normalize_embeddings=True,
                         show_progress_bar=False)
    return v.float().cpu()


def split_name(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    s = re.sub(r"[_\-\.\/:#]+", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def prefix_of(iri):
    if iri.startswith(kg.BRICK):
        return "brick"
    if iri.startswith(kg.S223):
        return "s223"
    if "brickschema.org/schema/Brick/ref" in iri:
        return "ref"
    if iri.startswith("http://www.w3.org/1999/02/22-rdf-syntax-ns#"):
        return "rdf"
    if iri.startswith("http://www.w3.org/2000/01/rdf-schema#"):
        return "rdfs"
    if "qudt.org" in iri:
        return "qudt"
    return ""


def rel_text(iri, inverse=False):
    t = split_name(kg.local_name(iri))
    p = prefix_of(iri)
    t = f"{p} {t}".strip()
    return f"inverse of {t}" if inverse else t


def tbox_info(tbox_triples):
    labels, defs = {}, {}
    for t in tbox_triples:
        pv = t.predicate.value
        if isinstance(t.object, ox.Literal):
            if pv == kg.SKOS_DEF or (pv == kg.RDFS_COMMENT and t.subject.value not in defs):
                defs[t.subject.value] = t.object.value
            elif pv == kg.LABEL:
                labels.setdefault(t.subject.value, t.object.value)
    return labels, defs


def build(building, nt_path, stratum, max_lit_chars=160):
    tb_triples = kg.tbox_triples(kg.tbox_path(stratum))
    abox = list(ox.parse(path=str(nt_path), format=ox.RdfFormat.N_TRIPLES))
    anc, parents = kg.subclass_closure(tb_triples + abox)
    tlabels, tdefs = tbox_info(tb_triples)

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
    # class nodes: used classes + ancestors (excluding owl:*)
    classes = set()
    for ts in types.values():
        for c in ts:
            classes |= {a for a in anc.get(c, {c}) if not a.startswith(kg.OWL)}
    # instances that are also classes (rare punning) are kept as classes
    inst_ids = [n for n in inst if n not in classes]
    cls_ids = sorted(classes)
    node_ids = inst_ids + cls_ids
    idx = {n: i for i, n in enumerate(node_ids)}
    n_inst = len(inst_ids)

    def primary(n):
        ts = types.get(n)
        if not ts:
            return None
        return sorted(ts, key=lambda c: (-len(anc.get(c, {c})), c))[0]

    prim = [idx.get(primary(n), -1) if primary(n) else -1 for n in inst_ids] + [-1] * len(cls_ids)

    # relations
    preds = sorted({p for _, p, _ in triples} | {kg.RDF_TYPE, kg.SUBCLASS})
    pid = {p: i for i, p in enumerate(preds)}
    src, dst, rel = [], [], []

    def add(s, p, o):
        src.append(idx[s]); dst.append(idx[o]); rel.append(2 * pid[p])

    for s, p, o in triples:
        if s in idx and o in idx:
            add(s, p, o)
    for s, ts in types.items():
        if s in idx and s not in classes:
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

    # texts
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
    for c in cls_ids:
        name = tlabels.get(c) or split_name(kg.local_name(c))
        d = tdefs.get(c, "")
        texts.append(f"{prefix_of(c)} class {name}. {d}"[:300])
    rel_names = []
    for p in preds:
        rel_names += [p, "^" + p]
    rel_texts = []
    for p in preds:
        rel_texts += [rel_text(p), rel_text(p, inverse=True)]

    # ULTRA relation graph: meta-relations h2h, h2t, t2h, t2t over 2R relation types
    R2 = 2 * len(preds)
    heads = collections.defaultdict(set)
    tails = collections.defaultdict(set)
    s_all, d_all, r_all = ei[0].tolist(), ei[1].tolist(), et.tolist()
    for s, d, r in zip(s_all, d_all, r_all):
        heads[s].add(r)
        tails[d].add(r)
    meta = {k: set() for k in range(4)}  # 0 h2h, 1 h2t, 2 t2h, 3 t2t
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
        for a, b in pairs:
            rs.append(a); rd.append(b); rt.append(k)

    x = embed(texts).half()
    rx = embed(rel_texts).half()
    return {
        "building": building, "stratum": stratum,
        "node_ids": node_ids, "n_inst": n_inst, "is_class": torch.tensor([0] * n_inst + [1] * len(cls_ids), dtype=torch.bool),
        "primary_type": torch.tensor(prim, dtype=torch.long),
        "texts": texts, "x": x,
        "edge_index": ei, "edge_type": et, "n_fwd_edges": E,
        "rel_names": rel_names, "rel_texts": rel_texts, "rel_x": rx,
        "rel_graph": torch.tensor([rs, rd], dtype=torch.long), "rel_graph_type": torch.tensor(rt, dtype=torch.long),
    }


def nt_for(building, ttl, variant):
    if variant == "compiled":
        return kg.nt_cache(building + "@compiled", Path(str(ttl).replace(".ttl", ".compiled.ttl")))
    return kg.nt_cache(building, ttl)


def main(buildings=None, force=False):
    OUT.mkdir(parents=True, exist_ok=True)
    man = kg.manifest()
    for b, r in sorted(man.items()):
        if buildings and b not in buildings:
            continue
        out = OUT / f"{b}.pt"
        if out.exists() and not force:
            continue
        lab = ROOT / "runs" / "labels" / f"{b}.json"
        variant = json.loads(lab.read_text())["graph_variant"] if lab.exists() else "raw"
        g = build(b, nt_for(b, ROOT / r["path"], variant), r["stratum"])
        g["graph_variant"] = variant
        torch.save(g, out)
        print(f"{b}: nodes={len(g['node_ids'])} (cls {int(g['is_class'].sum())}) edges={g['edge_index'].shape[1]} rels={len(g['rel_names'])}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.buildings, a.force)
