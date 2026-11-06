"""Wrap BrickQA's BrickAgent without editing BrickQA/.

- injects a per-question context block into the controller prompt (schema slot)
- logs every Gemini call (full prompt, response, tokens) to a per-question JSONL file
- retries 429 / ResourceExhausted with exponential backoff
- caps BrickQA's built-in rate-limit sleeps (see runs/DECISIONS.md)
"""
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRICKQA = ROOT / "BrickQA"
CWD = ROOT / "runs" / "e2e" / "_cwd"
SLEEP_CAP = 2.0
_real_sleep = time.sleep
sys.dont_write_bytecode = True  # never write into BrickQA/ (read-only)


def prepare_cwd():
    """Working dir with a copy of BrickQA/prompts so BrickQA's relative paths resolve outside BrickQA/."""
    if not (CWD / "prompts").exists():
        CWD.mkdir(parents=True, exist_ok=True)
        shutil.copytree(BRICKQA / "prompts", CWD / "prompts")
    os.chdir(CWD)
    if str(BRICKQA) not in sys.path:
        sys.path.insert(0, str(BRICKQA))


class CallLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.path, "w", encoding="utf-8")
        self.tokens_in = self.tokens_out = self.calls = self.retries = 0

    def write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.f.flush()


def install_patches(log: CallLog):
    time.sleep = lambda s: _real_sleep(min(float(s), SLEEP_CAP))
    from vertexai.generative_models import GenerativeModel
    orig = GenerativeModel.generate_content
    if getattr(orig, "_bridge_patched", False):
        orig = orig._orig

    def patched(self, contents, *a, **kw):
        delay = 5.0
        for attempt in range(7):
            try:
                t0 = time.time()
                r = orig(self, contents, *a, **kw)
                um = getattr(r, "usage_metadata", None)
                pin = getattr(um, "prompt_token_count", 0) or 0
                tot = getattr(um, "total_token_count", 0) or 0
                pout = max(tot - pin, getattr(um, "candidates_token_count", 0) or 0)
                log.tokens_in += pin
                log.tokens_out += pout
                log.calls += 1
                try:
                    text = r.text
                except Exception:  # noqa: BLE001 - empty / blocked / multi-part responses
                    try:
                        text = str(r.candidates[0].content.parts[0].text)
                    except Exception:  # noqa: BLE001
                        text = ""  # logged as empty; the agent's own parse/retry logic handles it
                log.write({"ts": datetime.now().isoformat(), "model": getattr(self, "_model_name", ""),
                           "prompt": contents if isinstance(contents, str) else str(contents),
                           "response": text, "tokens_in": pin, "tokens_out": pout,
                           "latency_s": round(time.time() - t0, 2), "attempt": attempt})
                return r
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if any(k in msg for k in ("429", "ResourceExhausted", "Resource exhausted", "503", "Unavailable")) and attempt < 6:
                    log.retries += 1
                    log.write({"ts": datetime.now().isoformat(), "retry_after_error": msg[:300], "sleep_s": delay})
                    _real_sleep(delay)
                    delay = min(delay * 2, 160)
                    continue
                raise

    patched._bridge_patched = True
    patched._orig = orig
    GenerativeModel.generate_content = patched


def make_agent(ttl_path, context_block=None):
    """Fresh BrickAgent with BrickQA defaults (decomposer on, max 15 iterations); graph loaded fresh."""
    import brick_utils
    from brick_agent import BrickAgent
    brick_utils.BrickGraph.reset_singleton()
    for fn in (brick_utils.search_brick, brick_utils.get_brick_entity):
        fn.cache_clear()
    agent = BrickAgent(engine="gemini-flash", use_decomposer=True, use_temporal_handler=True)
    agent.initialize_graph(ttl_file=str(ttl_path))
    if context_block:
        # same prompt slot as --schema-in-prompt content (BrickAgent.controller builds it from these)
        agent.schema_description = context_block
    return agent


def make_bkgqa_agent(ttl_path, context_block=None, log_dir=None, store="default", projection_refiner=False):
    """Fresh BKGQA KGAgent (ontology-agnostic BrickQA port) with BrickQA defaults; graph loaded fresh.
    Sleeps are passed explicitly at the same capped value used for BrickQA runs."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from BKGQA.agent import KGAgent
    agent = KGAgent(model="gemini-2.5-flash", use_decomposer=True, use_temporal_handler=True, max_iterations=15,
                    pre_call_sleep=SLEEP_CAP, post_call_sleep=SLEEP_CAP, decomposer_sleep=SLEEP_CAP,
                    log_dir=str(log_dir) if log_dir else None, context_block=context_block,
                    projection_refiner=projection_refiner)
    agent.initialize_graph(ttl_file=str(ttl_path), store=store)
    return agent
