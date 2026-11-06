"""
KGAgent: ontology-agnostic port of BrickQA's BrickAgent.

Pipeline (unchanged from BrickQA):
  decomposition (optional, heuristic-gated) ->
  ReAct loop: controller LLM call -> parse Thought/Action -> execute action -> observation ->
  SPARQL fixer + temporal handler + validators on execute_sparql ->
  final SPARQL = last executed query.

What changed: the graph is an owned KGGraph instance (no singleton); prompts are
assembled from an OntologyProfile computed from the graph (prefixes, pattern guidance,
vocabulary summary) instead of hard-coded Brick content.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .decomposer import KGQueryDecomposer, QueryDecomposition
from .graph_utils import (
    KGGraph, SparqlExecutionStatus, add_missing_prefixes, declared_prefixes, execute_sparql,
    format_entity_info, format_property_examples, format_results_table, format_search_results,
    get_entity, get_property_examples, search_graph,
)
from .temporal import (
    KGTemporalHandler, should_use_temporal_guide, validate_aggregation, validate_entity_coverage,
    validate_multi_sensor_joins,
)

PACKAGE_DIR = Path(__file__).resolve().parent
PROMPT_DIR = PACKAGE_DIR / "prompts"

ONTOLOGY_NAMES = {
    "brick": "Brick Schema",
    "s223": "ASHRAE 223P",
    "hybrid": "hybrid (Brick Schema + ASHRAE 223P)",
    "generic": "generic RDF (no Brick or 223P terms detected)",
}

# Old BrickQA action names (and 'finish') accepted when parsing LLM output
ACTION_ALIASES = {
    "search_brick": "search_graph",
    "get_brick_entity": "get_entity",
    "finish": "stop",
}


@dataclass
class KGAction:
    """One Thought-Action-Observation step."""
    thought: str
    action_name: str
    action_argument: str
    observation: Optional[str] = None
    observation_raw: Optional[Any] = None

    POSSIBLE_ACTIONS = [
        "search_graph",           # search classes / predicates / instances
        "get_entity",             # inspect a node
        "get_property_examples",  # see how a predicate is used
        "execute_sparql",         # run SPARQL
        "stop",                   # finalize
        "error",                  # internal (parse-retry)
    ]

    def __post_init__(self):
        self.action_name = ACTION_ALIASES.get(self.action_name, self.action_name)
        if self.action_name not in self.POSSIBLE_ACTIONS:
            raise ValueError(f"Invalid action: {self.action_name}. Must be one of {self.POSSIBLE_ACTIONS}")

    def to_string(self, include_observation: bool = True) -> str:
        result = f"Thought: {self.thought}\n"
        result += f"Action: {self.action_name}({self.action_argument})\n"
        if include_observation and self.observation:
            result += f"Observation: {self.observation}\n"
        return result

    def __eq__(self, other):
        if not isinstance(other, KGAction):
            return False
        return self.action_name == other.action_name and self.action_argument == other.action_argument

    def __hash__(self):
        return hash((self.action_name, self.action_argument))


@dataclass
class KGSparqlQuery:
    """A SPARQL query with its execution results."""
    sparql: str
    execution_result: Optional[List[Dict]] = None
    execution_status: Optional[SparqlExecutionStatus] = None
    result_limit: Optional[int] = None
    error: Optional[str] = None
    timeout: int = 180
    kg: Optional[KGGraph] = field(default=None, repr=False, compare=False)

    def execute(self):
        self.execution_result, self.execution_status = execute_sparql(
            self.kg, self.sparql, return_status=True, timeout=self.timeout, result_limit=self.result_limit)
        self.error = self.kg.last_error

    def has_results(self) -> bool:
        return bool(self.execution_result)

    def results_as_table(self, max_rows: int = 10) -> str:
        return format_results_table(self.execution_result or [], self.kg.profile if self.kg else None, max_rows)


@dataclass
class AgentState:
    question: str
    actions: List[KGAction] = field(default_factory=list)
    generated_sparqls: List[KGSparqlQuery] = field(default_factory=list)
    final_sparql: Optional[KGSparqlQuery] = None
    max_iterations: int = 15
    decomposition: Optional[QueryDecomposition] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cooldown_seconds: float = 0
    # decomposer share of the token totals above (BrickQA did not count decomposer tokens)
    decomposer_prompt_tokens: int = 0
    decomposer_completion_tokens: int = 0
    decomposer_total_tokens: int = 0

    def get_action_history(self, last_n: int = 10, include_observation: bool = True) -> str:
        recent = self.actions[-last_n:] if len(self.actions) > last_n else self.actions
        out = []
        for i, a in enumerate(recent):
            inc = include_observation
            if i < len(recent) - 2 and a.action_name in ["search_graph", "get_entity"]:
                inc = False
            out.append(a.to_string(include_observation=inc))
        return "\n".join(out)

    def is_duplicate_action(self, action: KGAction) -> bool:
        return action in self.actions[-5:]

    def should_stop(self) -> bool:
        if len(self.actions) >= self.max_iterations:
            return True
        if self.actions and self.actions[-1].action_name == "stop":
            return True
        return False


class KGAgent:
    """Iterative NL -> SPARQL agent for Brick / 223P / hybrid / generic building graphs."""

    def __init__(self, kg: Optional[KGGraph] = None, engine: Optional[str] = None,
                 model: str = "gemini-2.5-flash", use_decomposer: bool = True,
                 use_temporal_handler: bool = True, use_aggregation: bool = True,
                 max_iterations: int = 15, pre_call_sleep: float = 6, post_call_sleep: float = 60,
                 decomposer_sleep: float = 6.5, log_dir: Optional[str] = None,
                 context_block: Optional[str] = None, ttl_schema_file: Optional[str] = None,
                 schema_description_file: Optional[str] = None, vocab_in_prompt: bool = True,
                 vocab_max_chars: int = 6000, result_limit: Optional[int] = None,
                 sparql_timeout: int = 180, context_in_decomposer: bool = False,
                 log_full_prompt: bool = False, project: Optional[str] = None,
                 projection_refiner: bool = False):
        self.model = engine or model
        self.engine = self.model
        self.use_decomposer = use_decomposer
        self.use_temporal_handler = use_temporal_handler
        self.use_aggregation = use_aggregation
        self.max_iterations = max_iterations
        self.pre_call_sleep = pre_call_sleep
        self.post_call_sleep = post_call_sleep
        self.vocab_in_prompt = vocab_in_prompt
        self.vocab_max_chars = vocab_max_chars
        self.result_limit = result_limit
        self.sparql_timeout = sparql_timeout
        self.context_in_decomposer = context_in_decomposer
        self.log_full_prompt = log_full_prompt
        self.project = project
        self.context_block = context_block
        self.projection_refiner = projection_refiner
        self.ttl_schema_file = ttl_schema_file
        self.ttl_schema_content = None
        self._vocab_cache: Optional[str] = None
        self._using_temporal_guide = False
        self._current_state: Optional[AgentState] = None

        if schema_description_file:
            try:
                desc = Path(schema_description_file).read_text(encoding="utf-8")
                self.context_block = (self.context_block + "\n\n" + desc) if self.context_block else desc
                print(f"[INFO] Schema description loaded from {schema_description_file} ({len(desc)} chars)")
            except OSError as e:
                print(f"[WARNING] Failed to load schema description: {e}")

        if ttl_schema_file:
            try:
                self.ttl_schema_content = Path(ttl_schema_file).read_text(encoding="utf-8")
                print(f"[INFO] TTL schema sample loaded from {ttl_schema_file} ({len(self.ttl_schema_content)} chars)")
            except OSError as e:
                print(f"[WARNING] Failed to load TTL schema sample: {e}")

        self.decomposer = None
        if use_decomposer:
            self.decomposer = KGQueryDecomposer(model=self.model, project=project, pre_call_sleep=decomposer_sleep)
            print("[INFO] Query decomposer initialized")
        else:
            print("[INFO] Running without query decomposer")

        self.temporal_handler = KGTemporalHandler() if use_temporal_handler else None
        print("[INFO] Temporal handler " + ("available (activates only if the graph has observations)"
                                            if use_temporal_handler else "disabled"))

        self.log_dir = Path(log_dir) if log_dir else PACKAGE_DIR / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = str(self.log_dir / f"prompt_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
        print(f"[INFO] LLM logs will be saved to: {self.log_file}")

        self.dataset_date_range: Tuple[Optional[str], Optional[str]] = (None, None)
        self.dataset_year: Optional[str] = None

        self.kg: Optional[KGGraph] = None
        if kg is not None:
            self.set_graph(kg)

    # ------------------------------------------------------------------ graph
    def initialize_graph(self, ttl_file: str = None, csv_file: str = None, max_csv_rows: int = 100,
                         start_date: str = None, end_date: str = None, store: str = "default",
                         **csv_kwargs):
        """Load a TTL file (and optionally CSV timeseries) into a new KGGraph owned by this agent."""
        kg = KGGraph(ttl_file=ttl_file, store=store)
        if csv_file:
            kg.add_timeseries_data(csv_file, max_rows=max_csv_rows, start_date=start_date,
                                   end_date=end_date, **csv_kwargs)
        self.set_graph(kg)
        return kg

    def set_graph(self, kg: KGGraph):
        if kg.profile is None:
            kg.refresh_profile()
        self.kg = kg
        self._vocab_cache = None
        p = kg.profile
        print(f"[INFO] Ontology detected: {p.ontology} (namespace scores {p.ns_scores}); "
              f"observations: {'yes' if p.has_observations else 'no'}")
        if self.temporal_handler:
            self.temporal_handler.profile = p
        self._extract_date_metadata()

    def _extract_date_metadata(self):
        """Date range from the data itself (BrickQA hard-coded 2018 for the LBNL FCU dataset)."""
        self.dataset_date_range = self.kg.get_date_range() if self.kg else (None, None)
        self.dataset_year = self.dataset_date_range[0][:4] if self.dataset_date_range[0] else None
        if self.temporal_handler:
            self.temporal_handler.default_year = self.dataset_year
        if self.dataset_year:
            print(f"[INFO] Dataset date range: {self.dataset_date_range}")

    @property
    def profile(self):
        return self.kg.profile if self.kg else None

    def vocabulary_block(self) -> str:
        if self._vocab_cache is None:
            self._vocab_cache = self.profile.vocabulary_block(self.vocab_max_chars) if self.profile else ""
        return self._vocab_cache

    # ------------------------------------------------------------------ prompt
    def build_controller_prompt(self, state: AgentState) -> str:
        """Assemble the controller prompt (no LLM call)."""
        if self.kg is None:
            raise RuntimeError("No graph loaded. Call initialize_graph() or pass kg=.")
        p = self.profile
        template = (PROMPT_DIR / "controller.prompt").read_text(encoding="utf-8")
        history = "\n\n".join(a.to_string(include_observation=True) for a in state.actions)

        inserted = ""
        if self.context_block:
            inserted += ("\nBUILDING CONTEXT GUIDE:\n------------------------------------\n"
                         "The following context describes this building's graph (structure, naming conventions, "
                         "useful query patterns). Verify it against the graph with the tools.\n\n"
                         f"{self.context_block.strip()}\n------------------------------------\n")
        if self.ttl_schema_content:
            is_sample = len(self.ttl_schema_content) < 1_000_000
            label = "GRAPH SAMPLE" if is_sample else "COMPLETE GRAPH"
            note = ("This is a representative sample of the building graph. Use search_graph() and get_entity() "
                    "to explore additional entities." if is_sample else "This is the complete building graph.")
            inserted += (f"\n{label} (Turtle/TTL Format):\n------------------------------------\n{note}\n"
                         f"Use this to understand the structure, entities, relationships, and available data.\n\n"
                         f"{self.ttl_schema_content}\n------------------------------------\n")

        temporal_active = bool(self.use_temporal_handler and self.temporal_handler and self.temporal_handler.active)
        self._using_temporal_guide = False
        if temporal_active and self.dataset_year:
            inserted += (f"\nDataset information: timeseries data covers {self.dataset_date_range[0]} to "
                         f"{self.dataset_date_range[1]}. If a date is mentioned WITHOUT a year, assume year "
                         f"{self.dataset_year}.\n")
        decomp = state.decomposition
        if temporal_active and decomp and decomp.temporal.has_constraint:
            if should_use_temporal_guide(decomp):
                inserted += "\n" + self.temporal_handler.render_guide(self.dataset_date_range) + "\n"
                self._using_temporal_guide = True
                print(f"[INFO] Using temporal patterns guide for '{decomp.temporal.type}'")
            else:
                print(f"[INFO] Skipping temporal guide for '{decomp.temporal.type}' (handler will apply)")

        if decomp:
            ents = "; ".join(f"{e.mention} -> {e.entity_id}" for e in decomp.entities) or "none identified"
            tnote = ""
            if decomp.temporal.has_constraint and self._using_temporal_guide:
                tnote = "\n  Follow the temporal patterns guide above to generate correct temporal SPARQL."
            agg = (", ".join(decomp.aggregation.operations)
                   if self.use_aggregation and decomp.aggregation.required else "None")
            outs = ", ".join(decomp.output_attributes) if decomp.output_attributes else "not specified"
            inserted += (f"\nQuery Decomposition (use this to guide your actions; verify terms with the tools):\n"
                         f"- Entities needed ({decomp.entity_count}): {ents}\n"
                         f"- Output attributes: {outs}\n"
                         f"- Temporal constraint: {decomp.temporal.type}{tnote}\n"
                         f"- Aggregation: {agg}\n"
                         f"- Intent: {decomp.query_intent}\n")

        vocab = self.vocabulary_block() if self.vocab_in_prompt else ""
        return (template
                .replace("{{ ontology_name }}", ONTOLOGY_NAMES.get(p.ontology, p.ontology))
                .replace("{{ prefix_block }}", p.prefix_block())
                .replace("{{ pattern_guidance }}", p.pattern_guidance())
                .replace("{{ vocabulary }}", vocab)
                .replace("{{ inserted_context }}", inserted)
                .replace("{{ action_history }}", history)
                .replace("{{ question }}", state.question))

    # ------------------------------------------------------------------ LLM
    def _call_llm(self, prompt: str, state: AgentState) -> str:
        from vertexai.generative_models import GenerativeModel
        import vertexai

        if self.pre_call_sleep:
            state.cooldown_seconds += self.pre_call_sleep
            time.sleep(self.pre_call_sleep)

        project_id = self.project or os.getenv("GOOGLE_CLOUD_PROJECT")
        if not project_id:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable is not set. "
                               "Set it to your GCP project ID before running.")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        vertexai.init(project=project_id, location=location)

        model = GenerativeModel(self.model)
        response = model.generate_content(prompt)

        um = getattr(response, "usage_metadata", None)
        if um:
            state.prompt_tokens += getattr(um, "prompt_token_count", 0) or 0
            state.completion_tokens += getattr(um, "candidates_token_count", 0) or 0
            state.total_tokens += getattr(um, "total_token_count", 0) or 0

        if self.post_call_sleep:
            state.cooldown_seconds += self.post_call_sleep
            time.sleep(self.post_call_sleep)

        try:
            return response.text
        except ValueError as e:
            if "Multiple content parts" in str(e):
                print("[WARNING] LLM returned multiple content parts, using first part only")
                return response.candidates[0].content.parts[0].text
            # Empty / blocked response (no parts): return "" so the caller's parse-error retry handles it
            # instead of killing the whole question (BrickQA raised here).
            print(f"[WARNING] Empty LLM response ({e}); treating as unparseable output")
            return ""

    def controller(self, state: AgentState) -> KGAction:
        """Decide the next action: build prompt -> call LLM -> parse."""
        try:
            prompt = self.build_controller_prompt(state)
            output = self._call_llm(prompt, state)
            self._log_to_jsonl(self.log_file, {
                "timestamp": datetime.now().isoformat(),
                "model": self.model,
                "question": state.question,
                "iteration": len(state.actions) + 1,
                "prompt_chars": len(prompt),
                "prompt": prompt if self.log_full_prompt else (prompt[:500] + "..." if len(prompt) > 500 else prompt),
                "response": output,
            })
            return self.parse_controller_output(output)
        except Exception as e:
            print(f"[ERROR] LLM controller failed: {e}")
            raise

    # ------------------------------------------------------------------ parsing
    def parse_controller_output(self, controller_output: str) -> KGAction:
        """Parse 'Thought: ... Action: name(arg)' (same strategy as BrickAgent).

        All failures raise ValueError starting with 'Could not parse LLM output' so that
        the parse-retry logic in run() handles them.
        """
        def fail(msg: str):
            raise ValueError(f"Could not parse LLM output ({msg}). Output: {controller_output[:500]}...")

        cleaned = controller_output or ""
        cleaned = re.sub(r"```[A-Za-z]*[ \t]*", "", cleaned)  # strip code fences (```sparql, ```)
        cleaned = cleaned.replace("**Thought:**", "Thought:").replace("**Action:**", "Action:")
        for marker in ["Observation:", 'Output one "Thought"', "Output one Thought"]:
            if marker in cleaned:
                cleaned = cleaned.split(marker)[0]

        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", cleaned, re.DOTALL)
        if thought_match:
            thought = thought_match.group(1).strip()
        elif cleaned.strip().startswith("Action:"):
            thought = "Proceeding with action"
        else:
            pos = cleaned.find("Action:")
            if pos > 0:
                thought = cleaned[:pos].strip() or "Proceeding with action"
            else:
                rest = cleaned.strip()
                if re.match(r"^(PREFIX|SELECT|ASK)\b", rest, re.IGNORECASE):
                    thought = "Executing query"
                else:
                    fail("could not find Thought or Action")

        names = "|".join(KGAction.POSSIBLE_ACTIONS[:-1] + list(ACTION_ALIASES))
        # Pattern 1: Action: name(argument)
        m = re.search(r"Action:\s*(\w+)\s*\(", cleaned)
        if m:
            name = m.group(1).strip()
            arg = self._extract_balanced_parens(cleaned[m.end() - 1:])
            if arg is None:
                fail("could not find matching parentheses for action argument")
            name = ACTION_ALIASES.get(name, name)
            if name not in KGAction.POSSIBLE_ACTIONS or name == "error":
                fail(f"unknown action '{name}'")
            return KGAction(thought=thought, action_name=name, action_argument=self._clean_argument(name, arg))

        # Pattern 1b: Action: execute_sparql <newline> query   (no parentheses)
        m = re.search(rf"Action:\s*({names})\s*[:\n]\s*(.+)", cleaned, re.DOTALL)
        if m:
            name = ACTION_ALIASES.get(m.group(1), m.group(1))
            arg = m.group(2).strip()
            if name == "stop":
                return KGAction(thought=thought, action_name="stop", action_argument="")
            return KGAction(thought=thought, action_name=name, action_argument=self._clean_argument(name, arg))

        # Pattern 2: Action: <SPARQL query>
        m = re.search(r"Action:\s*(.+?)(?=\n\n|$)", cleaned, re.DOTALL)
        if m:
            content = m.group(1).strip()
            if re.match(r"^(PREFIX|SELECT|ASK)\b", content, re.IGNORECASE):
                full = cleaned[cleaned.find(content):].strip()
                return KGAction(thought=thought, action_name="execute_sparql", action_argument=full)

        # Fallback 1: the remaining text is a SPARQL query (missing 'Action:' label)
        remaining = cleaned[thought_match.end():].strip() if thought_match else cleaned.strip()
        if re.match(r"^(PREFIX|SELECT|ASK)\b", remaining, re.IGNORECASE):
            print("[WARNING] LLM forgot 'Action:' label, but output looks like SPARQL - using it anyway")
            return KGAction(thought=thought, action_name="execute_sparql", action_argument=remaining)

        # Fallback 2: mentions finish/stop -> stop
        if re.search(r"\b(finish|stop)\s*\(", cleaned, re.IGNORECASE):
            print("[WARNING] LLM output mentions finish/stop but format unclear - stopping")
            return KGAction(thought=thought, action_name="stop", action_argument="")

        print(f"[ERROR] Failed to parse LLM output. Cleaned output:\n{cleaned[:800]}")
        fail("no recognizable action")

    @staticmethod
    def _clean_argument(name: str, arg: str) -> str:
        a = arg.strip()
        if name == "stop":
            return ""
        if name == "execute_sparql":
            for q in ('"""', "'''"):
                if a.startswith(q) and a.endswith(q) and len(a) >= 6:
                    a = a[3:-3].strip()
            if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'" and \
                    re.match(r"^(PREFIX|SELECT|ASK|CONSTRUCT)\b", a[1:].lstrip(), re.IGNORECASE):
                a = a[1:-1].strip()
            return a
        # tool arguments: strip quotes / backticks / keyword prefixes
        a = re.sub(r"^\w+\s*=\s*", "", a) if re.match(r"^(search_term|term|entity_id|uri|property|property_uri)\s*=", a) else a
        for _ in range(2):
            if len(a) >= 2 and a[0] == a[-1] and a[0] in "\"'`":
                a = a[1:-1].strip()
        return a

    @staticmethod
    def _extract_balanced_parens(text: str) -> Optional[str]:
        """Content between balanced parentheses; ignores parens inside string literals and IRIs."""
        if not text or text[0] != "(":
            return None
        depth = 0
        i = 0
        quote = None
        while i < len(text):
            ch = text[i]
            if quote:
                if ch == "\\":
                    i += 2
                    continue
                if text.startswith(quote, i):
                    i += len(quote)
                    quote = None
                    continue
            else:
                if text.startswith('"""', i) or text.startswith("'''", i):
                    quote = text[i:i + 3]
                    i += 3
                    continue
                if ch in "\"'":
                    # an apostrophe inside plain words (e.g. zone's) is not a string delimiter
                    if ch == "'" and i > 0 and text[i - 1].isalnum():
                        i += 1
                        continue
                    quote = ch
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        return text[1:i]
            i += 1
        return None

    # ------------------------------------------------------------------ SPARQL fixer
    def _validate_and_fix_sparql(self, sparql: str) -> str:
        """Generic auto-corrections. PREFIX fixes use the graph's profile, never a fixed ontology."""
        original = sparql
        fixed = sparql.strip()
        p = self.profile

        # Fix 0c (run first): missing angle brackets in PREFIX declarations
        fixed = re.sub(r"PREFIX\s+(\w[\w\-.]*)?:\s+(https?://[^\s<>]+)", r"PREFIX \1: <\2>", fixed, flags=re.IGNORECASE)

        if p is not None:
            # Fix 0b: wrong namespace IRI for a prefix known to this graph
            known_ns = set(p.prefixes.values())
            for name, uri in declared_prefixes(fixed).items():
                correct = p.prefixes.get(name)
                if correct and uri != correct and uri not in known_ns:
                    pat = rf"PREFIX\s+{re.escape(name)}:\s*<{re.escape(uri)}>"
                    print(f"[INFO] Correcting PREFIX {name}: <{uri}> -> <{correct}>")
                    fixed = re.sub(pat, f"PREFIX {name}: <{correct}>", fixed, flags=re.IGNORECASE)
            # Fix 0a: add PREFIX declarations that are used but missing
            fixed, added = add_missing_prefixes(fixed, p)
            if added:
                print(f"[INFO] Added missing PREFIX declarations: {', '.join(added)}")

        has_obs = bool(p is not None and p.has_observations)
        # Fix 0d: remove timezone 'Z' from dateTime literals (only for graphs with naive timestamps)
        if has_obs:
            before = fixed
            fixed = re.sub(r'"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})Z"', r'"\1"', fixed)
            if fixed != before:
                print("[INFO] Removed timezone 'Z' from timestamp literals")

        # Fix 2: 'latest' timeseries queries ordered by DESC(?timestamp) without LIMIT -> LIMIT 1
        if has_obs and re.search(r"ORDER\s+BY\s+DESC\s*\(\s*\?timestamp", fixed, re.IGNORECASE) and \
                not re.search(r"LIMIT\s+\d+", fixed, re.IGNORECASE):
            fixed += " LIMIT 1"
            print("[INFO] Auto-added 'LIMIT 1' to timeseries query with ORDER BY DESC(?timestamp)")

        if fixed != original.strip():
            print("[INFO] SPARQL auto-corrected:")
            print(f"  Before: {original[:100]}...")
            print(f"  After:  {fixed[:100]}...")
        return fixed

    # ------------------------------------------------------------------ actions
    def execute_action(self, action: KGAction):
        try:
            if action.action_name == "search_graph":
                results = search_graph(self.kg, action.action_argument)
                action.observation_raw = results
                action.observation = format_search_results(results)

            elif action.action_name == "get_entity":
                info = get_entity(self.kg, action.action_argument)
                action.observation_raw = info
                action.observation = format_entity_info(info)

            elif action.action_name == "get_property_examples":
                ex = get_property_examples(self.kg, action.action_argument)
                action.observation_raw = ex
                action.observation = format_property_examples(ex)

            elif action.action_name == "execute_sparql":
                print("\n[EXECUTE_SPARQL] Starting SPARQL query execution...")
                corrected = self._validate_and_fix_sparql(action.action_argument)
                state = self._current_state
                decomp = state.decomposition if state else None

                th = self.temporal_handler
                if self.use_temporal_handler and th and th.active and decomp and decomp.temporal.has_constraint:
                    ttype = decomp.temporal.type
                    ok, msg = th.validate_temporal_query(corrected, decomposition=decomp)
                    if self._using_temporal_guide:
                        if ok:
                            print(f"[INFO] Temporal constraint '{ttype}' correctly applied by LLM")
                        else:
                            print(f"[WARNING] Temporal validation failed: {msg}")
                            print(f"[INFO] Applying temporal handler as fallback for '{ttype}'")
                            res = th.apply_temporal_constraint(corrected, decomposition=decomp)
                            if res.was_modified:
                                print(f"[INFO] SPARQL corrected with pattern: {res.pattern_applied}")
                                corrected = res.modified_sparql
                    else:
                        print(f"[INFO] Applying temporal constraint: {ttype}")
                        res = th.apply_temporal_constraint(corrected, decomposition=decomp)
                        if res.was_modified:
                            print(f"[INFO] SPARQL modified with temporal pattern: {res.pattern_applied}")
                            corrected = res.modified_sparql

                if decomp:
                    if decomp.entities:
                        ok, msg = validate_entity_coverage(corrected, decomp, self.profile)
                        print(f"[INFO] {msg}" if ok else f"[WARNING] Entity coverage issue: {msg}")
                    if self.use_aggregation and decomp.aggregation.required:
                        ok, msg = validate_aggregation(corrected, decomp)
                        print(f"[INFO] {msg}" if ok else f"[WARNING] Aggregation issue: {msg}")
                    if len(decomp.entities) > 1 and th and th.active:
                        ok, msg = validate_multi_sensor_joins(corrected, decomp, self.profile)
                        print(f"[INFO] {msg}" if ok else f"[WARNING] Multi-entity join issue: {msg}")

                q = KGSparqlQuery(sparql=corrected, result_limit=self.result_limit,
                                  timeout=self.sparql_timeout, kg=self.kg)
                try:
                    q.execute()
                except Exception as exec_error:  # noqa: BLE001
                    em = str(exec_error).lower()
                    if "prefix" in em or "namespace" in em:
                        action.observation = (f"PREFIX ERROR: {exec_error}\n\nMake sure all PREFIX declarations are "
                                              f"properly formatted with angle brackets. Known prefixes: "
                                              f"{', '.join(self.profile.prompt_prefix_names())}")
                    else:
                        action.observation = f"SPARQL Execution Error: {exec_error}"
                    return action, None

                action.observation_raw = q
                if q.has_results():
                    action.observation = q.results_as_table()
                else:
                    obs = f"Query returned no results. Status: {q.execution_status.value}"
                    if q.error:
                        obs += f"\nError: {q.error}"
                        if "prefix" in q.error.lower():
                            obs += f"\nKnown prefixes: {', '.join(self.profile.prompt_prefix_names())}"
                    action.observation = obs
                return action, q

            elif action.action_name == "stop":
                action.observation = "Stopping execution"
            elif action.action_name == "error":
                pass
            else:
                action.observation = f"Unknown action: {action.action_name}"
        except Exception as e:  # noqa: BLE001
            action.observation = f"Error executing action: {e}"
        return action, None

    # ------------------------------------------------------------------ run
    def run(self, question: str, verbose: bool = True) -> Tuple[AgentState, Optional[KGSparqlQuery]]:
        if self.kg is None:
            raise RuntimeError("No graph loaded. Call initialize_graph() or pass kg=.")
        state = AgentState(question=question, max_iterations=self.max_iterations)
        self._current_state = state

        if verbose:
            print(f"\n{'=' * 80}\nQuestion: {question}\n{'=' * 80}\n")

        if self.use_decomposer and self.decomposer:
            if self.decomposer.should_decompose(question):
                if verbose:
                    print("Step 1: Decomposing query (complex query detected)...\n")
                d = self.decomposer
                before = (d.prompt_tokens, d.completion_tokens, d.total_tokens)
                state.decomposition = d.decompose(
                    question, verbose=verbose, vocabulary_block=self.vocabulary_block(),
                    has_timeseries=bool(self.profile and self.profile.has_observations),
                    context_block=self.context_block if self.context_in_decomposer else None)
                state.decomposer_prompt_tokens = d.prompt_tokens - before[0]
                state.decomposer_completion_tokens = d.completion_tokens - before[1]
                state.decomposer_total_tokens = d.total_tokens - before[2]
                state.prompt_tokens += state.decomposer_prompt_tokens
                state.completion_tokens += state.decomposer_completion_tokens
                state.total_tokens += state.decomposer_total_tokens
                state.cooldown_seconds += d.cooldown_seconds
                d.cooldown_seconds = 0
            elif verbose:
                print("Step 1: Skipping decomposition (simple query)...\n")

        if verbose:
            print("Iterative SPARQL generation" + (" (using decomposition)" if state.decomposition else "") + "...\n")

        max_parse_retries = 1
        parse_retry_count = 0
        while not state.should_stop():
            try:
                action = self.controller(state)
                parse_retry_count = 0
            except ValueError as e:
                if "Could not parse LLM output" not in str(e):
                    raise
                parse_retry_count += 1
                if parse_retry_count <= max_parse_retries:
                    print(f"[WARNING] Parse error (attempt {parse_retry_count}/{max_parse_retries + 1}): {e}")
                    err = KGAction(thought="Previous output format was invalid", action_name="error",
                                   action_argument="Parse error")
                    err.observation = ("ERROR: Your previous response could not be parsed. "
                                       "You MUST use this exact format:\n"
                                       "Thought: [your reasoning here]\n"
                                       "Action: execute_sparql(query) OR search_graph(term) OR get_entity(URI) "
                                       "OR get_property_examples(property) OR stop()\n\n"
                                       "Do NOT add extra text. Do NOT skip the Action: label.")
                    state.actions.append(err)
                    if verbose:
                        print(err.to_string(include_observation=True) + "\n")
                    continue
                print("[ERROR] Max parse retries exceeded. Stopping agent.")
                action = KGAction(thought="Unable to generate valid output format", action_name="stop",
                                  action_argument="")
                action.observation = "Agent stopped due to repeated parse errors"

            if state.is_duplicate_action(action):
                if verbose:
                    print(f"[WARNING] Duplicate action detected: {action.action_name}({action.action_argument[:80]})")
                action = KGAction(thought="Detected duplicate action, stopping", action_name="stop", action_argument="")

            action, sparql_query = self.execute_action(action)
            state.actions.append(action)
            if sparql_query:
                state.generated_sparqls.append(sparql_query)
            if verbose:
                print(action.to_string(include_observation=True) + "\n")

        if state.generated_sparqls:
            state.final_sparql = state.generated_sparqls[-1]

        if self.projection_refiner and state.final_sparql is not None and state.final_sparql.has_results():
            try:
                self.refine_projection(state, verbose=verbose)
            except Exception as e:  # noqa: BLE001 - refinement is best-effort
                print(f"[WARNING] projection refiner failed: {e}")

        if verbose:
            print(f"{'=' * 80}\nCompleted in {len(state.actions)} actions")
            if state.decomposition:
                print(f"  - Entities: {', '.join(e.entity_id for e in state.decomposition.entities)}")
                print(f"  - Temporal: {state.decomposition.temporal.type}")
            if state.final_sparql:
                print(f"\nFinal SPARQL:\n{state.final_sparql.sparql}")
                print(f"\nFinal Results:\n{state.final_sparql.results_as_table()}")
            print(f"Tokens: prompt={state.prompt_tokens} completion={state.completion_tokens} total={state.total_tokens}")
            print(f"{'=' * 80}\n")
        return state, state.final_sparql

    # ------------------------------------------------------------------ projection refiner
    def _entity_attributes(self, rows: List[Dict], max_values: int = 15) -> str:
        """Literal attributes (direct and via one object hop) of IRIs bound in each result column."""
        from rdflib import Literal, URIRef
        g = self.kg.graph
        comp = self.kg.profile.compact if self.kg.profile else str
        lines = []
        for col in rows[0].keys():
            vals = [str(r[col]) for r in rows if r.get(col)]
            vals = list(dict.fromkeys(v for v in vals if v.startswith(("http", "urn"))))[:max_values]
            direct, hop = {}, {}
            for v in vals:
                for p, o in g.predicate_objects(URIRef(v)):
                    if isinstance(o, Literal):
                        direct.setdefault(comp(str(p)), str(o)[:40])
                    else:
                        for p2, o2 in g.predicate_objects(o):
                            if isinstance(o2, Literal):
                                hop.setdefault(f"{comp(str(p))}/{comp(str(p2))}", str(o2)[:40])
            if direct or hop:
                ex = "; ".join(f'{k} (e.g. "{v}")' for k, v in list(direct.items())[:8])
                ex2 = "; ".join(f'{k} (e.g. "{v}")' for k, v in list(hop.items())[:10])
                lines.append(f"?{col}: direct: {ex or '-'}\n   via one hop: {ex2 or '-'}")
        return "\n".join(lines) or "(none found)"

    def refine_projection(self, state: AgentState, verbose: bool = True):
        """One extra LLM call: add columns the question asks for but the final query does not return."""
        final = state.final_sparql
        rows = final.execution_result or []
        cols = list(rows[0].keys())
        sample = "columns: " + ", ".join("?" + c for c in cols) + "\n" + final.results_as_table(max_rows=3)
        tmpl = (PACKAGE_DIR / "prompts" / "projection_refiner.prompt").read_text(encoding="utf-8")
        prompt = (tmpl.replace("{{ question }}", state.question).replace("{{ sparql }}", final.sparql)
                  .replace("{{ sample }}", sample).replace("{{ attributes }}", self._entity_attributes(rows)))
        out = self._call_llm(prompt, state)
        m = re.search(r"```(?:sparql)?\s*(.+?)```", out, re.S | re.I)
        if not m or out.strip().upper().startswith("KEEP"):
            if verbose:
                print("[REFINER] KEEP")
            return
        cand = KGSparqlQuery(sparql=self._validate_and_fix_sparql(m.group(1).strip()), kg=self.kg,
                             timeout=self.sparql_timeout, result_limit=self.result_limit)
        cand.execute()
        new_cols = set(cand.execution_result[0].keys()) if cand.execution_result else set()
        if cand.has_results() and set(cols) <= new_cols and len(new_cols) > len(cols):
            state.generated_sparqls.append(cand)
            state.final_sparql = cand
            if verbose:
                print(f"[REFINER] accepted: {len(cols)} -> {len(new_cols)} columns, {len(cand.execution_result)} rows")
        elif verbose:
            print(f"[REFINER] rejected revision (rows={len(cand.execution_result or [])}, cols={sorted(new_cols)})")

    # ------------------------------------------------------------------ logging
    def _log_to_jsonl(self, filename: str, data: dict):
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


# Backwards-friendly aliases
BrickAction = KGAction
AgentAction = KGAction
