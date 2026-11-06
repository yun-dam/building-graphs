"""
Offline tests for BKGQA (no LLM calls). Run:
    .venv/Scripts/python BKGQA/tests/test_offline.py
"""

import io
import os
import re
import sys
import tempfile
import time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (AttributeError, ValueError):
        pass

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "offline-test-no-calls")  # decomposer init only; never called

from rdflib import Graph, Literal, Namespace, URIRef  # noqa: E402
from rdflib.namespace import RDF, XSD  # noqa: E402

from BKGQA.agent import AgentState, KGAgent  # noqa: E402
from BKGQA.decomposer import (AggregationRequirement, EntityMention, QueryDecomposition,  # noqa: E402
                              TemporalConstraint)
from BKGQA.graph_utils import (KGGraph, execute_sparql, format_entity_info, format_property_examples,  # noqa: E402
                               format_search_results, get_entity, get_property_examples, search_graph)
from BKGQA.temporal import KGTemporalHandler  # noqa: E402
from BKGQA.profile import build_profile  # noqa: E402

RESULTS = []
LOG_DIR = tempfile.mkdtemp(prefix="bkgqa_logs_")


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail and not cond else ""))


def quiet(fn, *a, **kw):
    """Run fn while swallowing its stdout chatter."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn(*a, **kw)
    finally:
        sys.stdout = old


GRAPHS = {
    "b59": (REPO / "data/BuildingQA/eval_buildings/b59.ttl", "s223"),
    "TUC": (REPO / "data/BuildingQA/eval_buildings/TUC_building.ttl", "brick"),
    "nist": (REPO / "runs/data/graphs/nist-bdg1-1.ttl", "s223"),
}

kgs = {}
for key, (path, expected) in GRAPHS.items():
    t = time.time()
    kg = quiet(KGGraph, ttl_file=str(path))
    kgs[key] = kg
    p = kg.profile
    print(f"\n=== {key}: {len(kg)} triples, loaded+profiled in {time.time() - t:.2f}s, ontology={p.ontology} {p.ns_scores}")
    check(f"{key}: ontology detected as {expected}", p.ontology == expected, p.ontology)
    declared = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:60]:
        m = re.match(r"\s*(?:@prefix|PREFIX)\s+(\w[\w\-.]*):\s*<([^>]*)>", line, re.IGNORECASE)
        if m:
            declared[m.group(1)] = m.group(2)
    check(f"{key}: prefix map contains all {len(declared)} declared prefixes with file URIs",
          declared and all(p.prefixes.get(k) == v for k, v in declared.items()),
          str({k: (v, p.prefixes.get(k)) for k, v in declared.items() if p.prefixes.get(k) != v}))
    check(f"{key}: no invented bldg-59# namespace", not any("bldg-59" in v for v in p.prefixes.values()))
    vb = kg.profile.vocabulary_block()
    print(f"    vocabulary block: {len(vb)} chars (~{len(vb) // 4} tokens)")
    check(f"{key}: vocabulary block renders and is bounded (<= 6000 chars)",
          "Vocabulary actually present in this graph" in vb and len(vb) <= 6000)
    check(f"{key}: literal attributes section present", "Attributes (literal-valued predicates" in vb)

b59v = kgs["b59"].profile.vocabulary_block()
check("b59: external-reference attributes listed (ref:endpoint, ref:writable on s223:ExternalReference)",
      "ref:endpoint" in b59v and "ref:writable" in b59v and re.search(r"s223:ExternalReference.*has attributes.*ref:endpoint", b59v))
check("b59: ref: resolves to the file's namespace (Brick/), not Brick/ref#",
      kgs["b59"].profile.prefixes["ref"] == "https://brickschema.org/schema/Brick/")
tucv = kgs["TUC"].profile.vocabulary_block()
check("TUC: ref:hasTimeseriesId attribute on external-reference blank nodes", "ref:hasTimeseriesId" in tucv and "blank node via ref:hasExternalReference" in tucv)
print("\n--- b59 vocabulary block (first 1500 chars) ---\n" + b59v[:1500] + "\n---")

# ---------------------------------------------------------------- compaction
pn = kgs["nist"].profile
check("compact: '/' namespace (quantitykind) -> qudtqk:Temperature (file prefix preferred)",
      pn.compact(URIRef("http://qudt.org/vocab/quantitykind/Temperature")) == "qudtqk:Temperature",
      pn.compact(URIRef("http://qudt.org/vocab/quantitykind/Temperature")))
check("compact: '#' namespace -> s223:Sensor", pn.compact(URIRef("http://data.ashrae.org/standard223#Sensor")) == "s223:Sensor")
check("compact: unknown namespace -> <full-iri>", pn.compact(URIRef("http://example.org/x/y")) == "<http://example.org/x/y>")
check("compact: digit-leading local name ex1:00061", kgs["b59"].profile.compact(
    URIRef("http://data.ashrae.org/standard223/data/lbnl-example-2#00061")) == "ex1:00061")
check("compact: ref:APIReference on '/' namespace", kgs["b59"].profile.compact(
    URIRef("https://brickschema.org/schema/Brick/APIReference")) == "ref:APIReference")

# ---------------------------------------------------------------- tools
print()
r = quiet(search_graph, kgs["b59"], "temperature sensor")
out = format_search_results(r)
print("search_graph(b59, 'temperature sensor'):\n" + out[:900])
check("search b59 'temperature sensor' finds class s223:TemperatureSensor", any(x["id"] == "s223:TemperatureSensor" for x in r))
check("search output compacted (no raw standard223# IRIs)", "http://data.ashrae.org/standard223#" not in out)
r = search_graph(kgs["b59"], "endpoint")
check("search b59 'endpoint' finds literal predicate ref:endpoint",
      any(x["id"] == "ref:endpoint" and x.get("literal") for x in r), format_search_results(r)[:300])
r = search_graph(kgs["b59"], '"cerc_templogger_1"')
check("search b59 by label (quoted arg) finds instance with attributes",
      any(x["category"] == "instance" for x in r), format_search_results(r)[:300])
r = search_graph(kgs["TUC"], "WC_I3")
out = format_search_results(r)
print("search_graph(TUC, 'WC_I3'):\n" + out)
check("search TUC 'WC_I3' (ifcName on blank node) returns parent om:Space_2217", any(x["id"] == "om:Space_2217" for x in r))
r = search_graph(kgs["nist"], "damper")
check("search nist 'damper' finds s223:Damper class and instances",
      any(x["id"] == "s223:Damper" for x in r) and any(x["category"] == "instance" for x in r))

info = get_entity(kgs["TUC"], "Space_2217")
out = format_entity_info(info)
print("\nget_entity(TUC, 'Space_2217'):\n" + out)
check("get_entity bare local name resolves to om:Space_2217", info["found"] and info["entity"] == "om:Space_2217")
check("get_entity inlines blank-node external reference literals", 'ref:ifcName "WC_I3"' in out)

first_ref = execute_sparql(kgs["b59"], "SELECT ?r WHERE { ?r a s223:ExternalReference ; ref:endpoint ?e } LIMIT 1")
ref_iri = first_ref[0]["r"] if first_ref else ""
info = get_entity(kgs["b59"], kgs["b59"].compact(URIRef(ref_iri)))
out = format_entity_info(info)
print(f"\nget_entity(b59, {info['entity']}):\n" + out)
check("get_entity b59 external reference shows literal attrs + incoming hasExternalReference",
      "ref:endpoint =" in out and "Incoming:" in out)
info = get_entity(kgs["nist"], f"<http://data.ashrae.org/standard223#Damper>")
out = format_entity_info(info)
check("get_entity <full-iri> of a class lists instances", info["found"] and "Instances (rdf:type this" in out, out[:300])
info = get_entity(kgs["nist"], "qudtqk:Temperature")
check("get_entity object-only node (qudtqk:Temperature) shows incoming", info["found"] and info["incoming"])

ex = get_property_examples(kgs["TUC"], "hasTimeseriesId")
out = format_property_examples(ex)
print("\nget_property_examples(TUC, 'hasTimeseriesId'):\n" + out)
check("get_property_examples bare name -> ref:hasTimeseriesId with examples", ex["property"] == "ref:hasTimeseriesId" and ex["examples"])
ex = get_property_examples(kgs["nist"], "s223:observes")
out = format_property_examples(ex)
print("\nget_property_examples(nist, 's223:observes'):\n" + out)
check("get_property_examples compacted ('/' and '#' namespaces)", "psm:" in out and "s223:" in out and "http" not in out)

# ---------------------------------------------------------------- SPARQL
print()
rows, st = quiet(execute_sparql, kgs["b59"], "SELECT (COUNT(?s) AS ?n) WHERE { ?s a s223:TemperatureSensor }", return_status=True)
check("execute_sparql b59 with undeclared s223: (auto-PREFIX) returns rows", rows and int(rows[0]["n"]) > 0, f"{rows} {st}")
rows = quiet(execute_sparql, kgs["TUC"], "SELECT ?p ?id WHERE { ?p brick:isPointOf ?e ; ref:hasExternalReference/ref:hasTimeseriesId ?id } LIMIT 5")
check("execute_sparql TUC brick query returns rows", len(rows) == 5, str(rows[:1]))
rows = quiet(execute_sparql, kgs["nist"], "PREFIX s223: <http://data.ashrae.org/standard223#>\nSELECT ?s ?prop WHERE { ?s s223:observes ?prop } LIMIT 3")
check("execute_sparql nist returns rows", len(rows) == 3)
rows, st = quiet(execute_sparql, kgs["nist"], "SELECT ?s WHERE { ?s a s223:Damper ", return_status=True)
check("execute_sparql syntax error -> SYNTAX_ERROR status + last_error message",
      st.value == "syntax_error" and kgs["nist"].last_error, f"{st} {kgs['nist'].last_error}")
kg_ox = quiet(KGGraph, ttl_file=str(GRAPHS["nist"][0]), store="Oxigraph")
rows = quiet(execute_sparql, kg_ox, "SELECT ?s WHERE { ?s a s223:Damper }")
check("Oxigraph store: load + query works", len(rows) == 6, str(len(rows)))

# ---------------------------------------------------------------- agent: fixer
agents = {k: quiet(KGAgent, kg=v, log_dir=LOG_DIR, pre_call_sleep=0, post_call_sleep=0) for k, v in kgs.items()}
a = agents["b59"]
q = "PREFIX s223: <http://data.ashrae.org/standard223#>\nSELECT ?s WHERE { ?s a s223:Sensor } ORDER BY DESC(?s)"
fixed = quiet(a._validate_and_fix_sparql, q)
check("fixer keeps a correct s223 PREFIX unchanged", "PREFIX s223: <http://data.ashrae.org/standard223#>" in fixed and fixed.count("PREFIX s223:") == 1)
check("fixer does not add LIMIT 1 to ORDER BY DESC on a metadata graph", "LIMIT 1" not in fixed)
q = "PREFIX ref: <https://brickschema.org/schema/Brick/ref#>\nSELECT ?e WHERE { ?r ref:endpoint ?e }"
fixed = quiet(a._validate_and_fix_sparql, q)
check("fixer maps ref: to the graph's own namespace (b59)", "PREFIX ref: <https://brickschema.org/schema/Brick/>" in fixed, fixed)
q = "PREFIX brick: <https://brickschema.org/schema/1.1/Brick#>\nSELECT ?p WHERE { ?p brick:isPointOf ?e }"
fixed = quiet(agents["TUC"]._validate_and_fix_sparql, q)
check("fixer fixes a wrong brick: URI (TUC)", "PREFIX brick: <https://brickschema.org/schema/Brick#>" in fixed and "1.1" not in fixed, fixed)
q = "PREFIX s223: http://data.ashrae.org/standard223#\nSELECT ?s WHERE { ?s s223:observes ?p ; rdfs:label ?l }"
fixed = quiet(agents["nist"]._validate_and_fix_sparql, q)
check("fixer adds angle brackets + missing rdfs PREFIX, no brick: forced",
      "PREFIX s223: <http://data.ashrae.org/standard223#>" in fixed and "PREFIX rdfs:" in fixed and "brick" not in fixed, fixed)
q = "PREFIX ex: <http://example.org/custom#>\nSELECT ?s WHERE { ?s a ex:Thing }"
fixed = quiet(a._validate_and_fix_sparql, q)
check("fixer leaves unknown custom prefixes alone", "PREFIX ex: <http://example.org/custom#>" in fixed)

# ---------------------------------------------------------------- prompt rendering
decomp = QueryDecomposition(
    entities=[EntityMention("temperature sensors", "s223:TemperatureSensor", "sensor class"),
              EntityMention("endpoint", "ref:endpoint", "endpoint attribute", kind="attribute")],
    entity_count=2, temporal=TemporalConstraint(False, "none", {}, ""),
    aggregation=AggregationRequirement(False, [], []), query_intent="List sensors and endpoints",
    raw_json={}, output_attributes=["rdfs:label", "ref:endpoint"])
for key in ("b59", "nist", "TUC"):
    ag = agents[key]
    st = AgentState(question="Which temperature sensors exist and what are their endpoints?")
    st.decomposition = decomp
    prompt = quiet(ag.build_controller_prompt, st)
    p = ag.profile
    print(f"\nRendered controller prompt for {key}: {len(prompt)} chars (~{len(prompt) // 4} tokens)")
    check(f"{key}: prompt has no unrendered template tags", "{{" not in prompt and "{%" not in prompt)
    if p.ontology == "s223":
        rest = prompt.replace(p.prefix_block(), "").replace(ag.vocabulary_block(), "")
        hits = [m.start() for m in re.finditer(r"brick", rest, re.IGNORECASE)]
        check(f"{key}: no 'Brick' instructions in 223P prompt (outside prefix/vocab blocks)", not hits,
              str([rest[max(0, h - 60):h + 40] for h in hits[:3]]))
        check(f"{key}: prompt contains 223P guidance", "s223:hasProperty" in prompt and "s223:observes" in prompt)
    else:
        check(f"{key}: Brick prompt contains Brick guidance", "brick:hasPoint" in prompt)

st = AgentState(question="Which temperature sensors exist and what are their endpoints?")
st.decomposition = decomp
ctx_agent = quiet(KGAgent, kg=kgs["b59"], log_dir=LOG_DIR, context_block="DUMMY RETRIEVED CONTEXT: sensors link to refs.",
                  use_decomposer=False)
prompt = quiet(ctx_agent.build_controller_prompt, st)
check("context_block injected in BUILDING CONTEXT GUIDE slot after the question",
      "BUILDING CONTEXT GUIDE" in prompt and prompt.index("DUMMY RETRIEVED CONTEXT") > prompt.index("User Question:"))
print("\n--- b59 prompt tail (last 1200 chars) ---\n" + prompt[-1200:] + "\n---")

# ---------------------------------------------------------------- parsing
pa = agents["nist"]
cases = [
    ("Thought: find sensors\nAction: search_brick(\"temperature sensor\")", "search_graph", "temperature sensor"),
    ("Thought: inspect\nAction: get_brick_entity(psm:Damper1)", "get_entity", "psm:Damper1"),
    ("Thought: check property\nAction: get_property_examples(s223:observes)", "get_property_examples", "s223:observes"),
    ("Thought: run it\nAction: execute_sparql(```sparql\nPREFIX s223: <http://data.ashrae.org/standard223#>\n"
     "SELECT ?s (COUNT(?p) AS ?n) WHERE { ?s s223:observes ?p . FILTER(CONTAINS(STR(?s), \"(x\")) } GROUP BY ?s\n```)",
     "execute_sparql", None),
    ("Thought: run it\nAction: execute_sparql\n```sparql\nSELECT ?s WHERE { ?s a s223:Damper }\n```", "execute_sparql",
     "SELECT ?s WHERE { ?s a s223:Damper }"),
    ("Thought: done\nAction: stop()", "stop", ""),
    ("Thought: done\nAction: finish(the answer)", "stop", ""),
]
for text, exp_name, exp_arg in cases:
    try:
        act = pa.parse_controller_output(text)
        ok = act.action_name == exp_name and (exp_arg is None or act.action_argument == exp_arg)
        if exp_name == "execute_sparql" and exp_arg is None:
            ok = ok and act.action_argument.startswith("PREFIX") and act.action_argument.endswith("GROUP BY ?s") and "```" not in act.action_argument
        check(f"parse: {text.splitlines()[1][:45]!r} -> {exp_name}", ok, f"{act.action_name} | {act.action_argument!r}")
    except Exception as e:  # noqa: BLE001
        check(f"parse: {text[:45]!r}", False, repr(e))
try:
    pa.parse_controller_output("I think we should look at the graph.")
    check("parse: garbage raises 'Could not parse LLM output'", False)
except ValueError as e:
    check("parse: garbage raises 'Could not parse LLM output'", "Could not parse LLM output" in str(e))
try:
    pa.parse_controller_output("Thought: x\nAction: teleport(here)")
    check("parse: unknown action raises parse error (retry path)", False)
except ValueError as e:
    check("parse: unknown action raises parse error (retry path)", "Could not parse LLM output" in str(e))

# ---------------------------------------------------------------- full loop with a scripted fake LLM (no network)
script = iter([
    "Thought: Find dampers.\nAction: search_brick(damper)",
    "Thought: Query them.\nAction: execute_sparql(SELECT ?d ?l WHERE { ?d a s223:Damper ; rdfs:label ?l })",
    "Thought: Done.\nAction: stop()",
])
loop_agent = quiet(KGAgent, kg=kgs["nist"], log_dir=LOG_DIR, use_decomposer=False, pre_call_sleep=0, post_call_sleep=0)
loop_agent._call_llm = lambda prompt, state: next(script)
state, final = quiet(loop_agent.run, "List the dampers and their labels.")
check("scripted loop: 3 actions, final SPARQL with 6 rows, auto-PREFIX applied",
      len(state.actions) == 3 and final is not None and len(final.execution_result) == 6 and "PREFIX s223:" in final.sparql,
      f"{[a.action_name for a in state.actions]} {final.execution_result if final else None}")
script2 = iter(["Thought: a\nAction: search_graph(x)"] * 3)
loop_agent._call_llm = lambda prompt, state: next(script2)
state, final = quiet(loop_agent.run, "dup test")
check("scripted loop: duplicate action forces stop", state.actions[-1].action_name == "stop" and len(state.actions) == 2)
script3 = iter(["garbage", "more garbage"])
loop_agent._call_llm = lambda prompt, state: next(script3)
state, final = quiet(loop_agent.run, "parse retry test")
check("scripted loop: parse retry once then stop", [a.action_name for a in state.actions] == ["error", "stop"])

# ---------------------------------------------------------------- temporal (inactive on metadata graphs; active with observations)
check("temporal handler inactive on b59/TUC/nist (no observations)",
      not any(ag.temporal_handler.active for ag in agents.values()))
REF = Namespace("https://brickschema.org/schema/Brick/ref#")
EX = Namespace("http://example.org/bldg#")
g = Graph()
g.bind("ex", EX)
g.bind("brick", "https://brickschema.org/schema/Brick#")
g.bind("ref", REF)
g.add((EX.T1, RDF.type, URIRef("https://brickschema.org/schema/Brick#Zone_Air_Temperature_Sensor")))
for i in range(5):
    o = EX[f"o{i}"]
    g.add((EX.T1, REF.hasObservation, o))
    g.add((o, REF.hasTimestamp, Literal(f"2018-01-0{i + 1}T00:00:00", datatype=XSD.dateTime)))
    g.add((o, REF.hasValue, Literal(20.0 + i, datatype=XSD.float)))
kg_ts = KGGraph(graph=g)
th = KGTemporalHandler(kg_ts.profile, default_year="2018")
tq = "SELECT ?timestamp ?value WHERE { ex:T1 ref:hasObservation ?obs . ?obs ref:hasTimestamp ?timestamp ; ref:hasValue ?value }"
res = th.apply_temporal_constraint(tq, temporal_type="latest", details={})
check("temporal handler active with observation triples; 'latest' adds ORDER BY DESC LIMIT 1",
      th.active and res.was_modified and "ORDER BY DESC(?timestamp)" in res.modified_sparql and "LIMIT 1" in res.modified_sparql)
rows = quiet(execute_sparql, kg_ts, res.modified_sparql)
check("temporal query executes and returns the latest value", rows and rows[0]["value"].startswith("24"), str(rows))
guide = th.render_guide(kg_ts.get_date_range())
check("temporal guide rendered with graph's observation predicates", "ref:hasObservation" in guide and "2018-01-01" in guide)

# ---------------------------------------------------------------- hybrid detection (synthetic graph)
BR = Namespace("https://brickschema.org/schema/Brick#")
S2 = Namespace("http://data.ashrae.org/standard223#")
gh = Graph()
for i in range(60):
    e, s, prop = EX[f"vav{i}"], EX[f"sen{i}"], EX[f"prop{i}"]
    gh.add((e, RDF.type, BR.VAV)); gh.add((e, BR.hasPoint, s)); gh.add((s, RDF.type, BR.Temperature_Sensor))
    gh.add((s, RDF.type, S2.TemperatureSensor)); gh.add((s, S2.observes, prop)); gh.add((prop, RDF.type, S2.QuantifiableObservableProperty))
ph = build_profile(gh)
check("synthetic Brick+223P graph detected as hybrid with both guidances",
      ph.ontology == "hybrid" and "brick:hasPoint" in ph.pattern_guidance() and "s223:observes" in ph.pattern_guidance(),
      f"{ph.ontology} {ph.ns_scores}")

# ---------------------------------------------------------------- summary
n_pass = sum(1 for _, ok in RESULTS if ok)
print(f"\n{'=' * 60}\n{n_pass}/{len(RESULTS)} checks passed")
for name, ok in RESULTS:
    if not ok:
        print(f"  FAILED: {name}")
sys.exit(0 if n_pass == len(RESULTS) else 1)
