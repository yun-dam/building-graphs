# BKGQA: building knowledge graph QA

BKGQA is an ontology-agnostic port of `BrickQA/`. It keeps BrickQA's pipeline:
decomposer, then a ReAct loop (graph tools plus SPARQL execution feedback), then the SPARQL fixer and validators, then the final SPARQL and its results.
It removes every Brick-specific assumption, so it works on Brick, ASHRAE 223P, hybrid, or other RDF building graphs.

## What changed vs BrickQA

| Component | BrickQA | BKGQA |
|---|---|---|
| Graph | `BrickGraph` process-wide singleton | `KGGraph` instance owned by the agent (`graph_utils.py`); rdflib default store or `store="Oxigraph"` |
| Prefixes | fixed `BRICK_PREFIXES` incl. invented `bldg: <bldg-59#>` | `OntologyProfile` (`profile.py`): file `@prefix`/`PREFIX` declarations > standard (rdf, rdfs, owl, xsd, brick, ref, s223, qudt, unit, quantitykind) > rdflib bindings |
| URI display | `split('#')[-1]` (breaks on `/` namespaces) | `profile.compact()` with longest-namespace match; falls back to `<full-iri>` |
| Ontology knowledge in prompt | Brick-only patterns + hard-coded Brick class table + LBNL FCU domain notes | detected ontology (brick/s223/hybrid/generic) -> short standards-based pattern guidance + **vocabulary computed from the graph** (classes, relationships with subject->object class pairs, literal attributes per class), bounded to about 1500 tokens |
| Tools | `search_brick` (entity/type/label substring, module `lru_cache`), `get_brick_entity` (outgoing only, `bldg:` assumed), `get_property_examples` (`ref:` assumed) | `search_graph` (classes, predicates incl. literal ones, instances, labels, name literals on blank nodes), `get_entity` (outgoing + incoming + inlined blank nodes, accepts `prefix:local` / `<iri>` / bare name), `get_property_examples` (with subject/object types); per-instance caches |
| execute_sparql | prepend all Brick prefixes if no `PREFIX` | adds only used-but-undeclared known prefixes; same timeout thread + status enum; error text is returned to the LLM |
| SPARQL fixer | forces Brick/bldg-59 prefix URIs, always adds `LIMIT 1` after `ORDER BY DESC` | corrects only known prefixes with a wrong URI (from the profile), adds missing ones; timestamp `Z` / `LIMIT 1` fixes only for graphs with observations |
| Decomposer | maps to hard-coded Brick classes (`sensors`) | maps only to terms in the vocabulary block (`entities`, `output_attributes`; `sensors` / `sensor_count` kept as aliases) |
| Temporal handler | hard-coded `ref:`/`bldg:`, dataset year 2018 | observation predicates and date range read from the graph; no-op when the graph has no observations |
| Actions | `search_brick`, `get_brick_entity`, ... | `search_graph`, `get_entity`, `get_property_examples`, `execute_sparql`, `stop` (old names and `finish` accepted as aliases) |
| External context | `schema_description` "BUILDING SCHEMA GUIDE" slot | `context_block=` (same slot, "BUILDING CONTEXT GUIDE"); `schema_description_file=` still accepted |
| Paths | prompts/logs relative to CWD | relative to the `BKGQA/` package; logs go to `log_dir` (default `BKGQA/logs/`) |

Defaults are the same as BrickQA:
- decomposer on, with the same heuristic gate
- `max_iterations=15`
- temporal handler available
- 6 s sleep before each call and 60 s after (`pre_call_sleep` / `post_call_sleep`)
- Vertex AI `gemini-2.5-flash`, stateless `generate_content` calls

## Run

```bash
export GOOGLE_CLOUD_PROJECT=<project>      # optional: GOOGLE_CLOUD_LOCATION (default us-central1)
.venv/Scripts/python BKGQA/main.py --ttl data/BuildingQA/eval_buildings/b59.ttl -q "Which sensors observe temperature?"
# options: --context-file ctx.txt  --schema-in-prompt sample.ttl  --no-decomposer  --no-temporal  --no-vocab
#          --store Oxigraph  --pre-call-sleep 6 --post-call-sleep 60  --log-dir runs/logs
```

```python
from BKGQA.agent import KGAgent
agent = KGAgent(context_block=None, log_dir="runs/bkgqa_logs")
agent.initialize_graph(ttl_file="building.ttl")
state, final = agent.run("List all AHUs and the zones they feed.")
final.sparql, final.execution_result, state.total_tokens
```

Offline tests (no LLM calls): `.venv/Scripts/python BKGQA/tests/test_offline.py`
