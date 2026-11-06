"""Shared KG utilities: loading ABox + TBox into pyoxigraph, skolemized N-Triples cache."""
import csv
import json
from pathlib import Path

import pyoxigraph as ox

ROOT = Path(__file__).resolve().parent.parent
ONTO = ROOT / "data" / "BuildingQA" / "bschema" / "ontologies"
GDIR = ROOT / "runs" / "data" / "graphs"
NTDIR = ROOT / "runs" / "data" / "nt"
TBOX_BRICK = ONTO / "Brick.ttl"      # Brick 1.4.3
TBOX_223P = ONTO / "223p.ttl"

ABOX_G = ox.NamedNode("urn:bridge:abox")
TBOX_G = ox.NamedNode("urn:bridge:tbox")

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
SUBCLASS = "http://www.w3.org/2000/01/rdf-schema#subClassOf"
LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
SKOS_DEF = "http://www.w3.org/2004/02/skos/core#definition"
RDFS_COMMENT = "http://www.w3.org/2000/01/rdf-schema#comment"
OWL = "http://www.w3.org/2002/07/owl#"
S223 = "http://data.ashrae.org/standard223#"
BRICK = "https://brickschema.org/schema/Brick#"


def manifest():
    with open(ROOT / "runs" / "data" / "manifest.csv") as f:
        return {r["building"]: r for r in csv.DictReader(f)}


def split():
    return json.loads((ROOT / "data" / "splits" / "brickqa_g_split.json").read_text())


def is_223p(stratum_or_building):
    return stratum_or_building == "s223"


def tbox_path(stratum):
    return TBOX_223P if stratum == "s223" else TBOX_BRICK


def _skolem(term, prefix):
    if isinstance(term, ox.BlankNode):
        return ox.NamedNode(f"urn:bnode:{prefix}:{term.value}")
    return term


def load_triples(path, prefix):
    """Parse a TTL file, skolemizing blank nodes so ids stay stable across tools."""
    out = []
    for q in ox.parse(path=str(path), format=ox.RdfFormat.TURTLE):
        s, p, o = _skolem(q.subject, prefix), q.predicate, _skolem(q.object, prefix)
        out.append(ox.Triple(s, p, o))
    return out


def nt_cache(building, ttl_path):
    """Skolemized N-Triples file for a building (cached)."""
    NTDIR.mkdir(parents=True, exist_ok=True)
    nt = NTDIR / f"{building}.nt"
    if not nt.exists():
        triples = load_triples(ttl_path, building)
        ox.serialize(triples, output=str(nt), format=ox.RdfFormat.N_TRIPLES)
    return nt


_TBOX_CACHE = {}


def tbox_triples(path):
    path = Path(path)
    if path not in _TBOX_CACHE:
        _TBOX_CACHE[path] = load_triples(path, "tbox")
    return _TBOX_CACHE[path]


def make_store(abox_path, tbox, abox_format=ox.RdfFormat.N_TRIPLES, skolem_prefix=None):
    """Store with ABox in ABOX_G and TBox in TBOX_G; query with union default graph."""
    st = ox.Store()
    if skolem_prefix is not None:
        st.bulk_extend(ox.Quad(t.subject, t.predicate, t.object, ABOX_G)
                       for t in load_triples(abox_path, skolem_prefix))
    else:
        st.bulk_load(path=str(abox_path), format=abox_format, to_graph=ABOX_G)
    st.bulk_extend(ox.Quad(t.subject, t.predicate, t.object, TBOX_G) for t in tbox_triples(tbox))
    return st


def term_str(t):
    """Canonical string for a pyoxigraph term (IRI value, or literal lexical form)."""
    if isinstance(t, ox.NamedNode):
        return t.value
    if isinstance(t, ox.BlankNode):
        return "_:" + t.value
    if isinstance(t, ox.Literal):
        return t.value
    return str(t)


def local_name(iri):
    for sep in ("#", "/", ":"):
        if sep in iri:
            iri = iri.rsplit(sep, 1)[-1]
    return iri


def subclass_closure(triples):
    """Map class -> set of ancestors (including itself) from rdfs:subClassOf triples."""
    parents = {}
    for t in triples:
        if t.predicate.value == SUBCLASS and isinstance(t.object, ox.NamedNode):
            parents.setdefault(t.subject.value, set()).add(t.object.value)
    memo = {}

    def anc(c, stack=()):
        if c in memo:
            return memo[c]
        res = {c}
        for p in parents.get(c, ()):
            if p not in stack:
                res |= anc(p, stack + (c,))
        memo[c] = res
        return res

    for c in list(parents):
        anc(c)
    return memo, parents
