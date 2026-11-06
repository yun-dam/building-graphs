"""
BKGQA - Natural language to SPARQL for building knowledge graphs
(Brick, ASHRAE 223P, hybrid, or other RDF building graphs).

Usage (from any directory):
    python BKGQA/main.py --ttl building.ttl
    python BKGQA/main.py --ttl building.ttl --question "Which sensors observe temperature?"
    python BKGQA/main.py --ttl building.ttl --context-file retrieved_context.txt -q "..."

Requires GOOGLE_CLOUD_PROJECT (and optionally GOOGLE_CLOUD_LOCATION) for Vertex AI Gemini.
"""

import argparse
import io
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        if getattr(sys.stdout, "encoding", "").lower() != "utf-8":
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except (AttributeError, ValueError):
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from BKGQA.agent import KGAgent  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="BKGQA: NL -> SPARQL for building knowledge graphs")
    ap.add_argument("--ttl", required=True, help="Path to the building graph (Turtle)")
    ap.add_argument("--question", "-q", default=None, help="Single question (otherwise interactive mode)")
    ap.add_argument("--context-file", default=None, help="Text file with external/retrieved context for the prompt")
    ap.add_argument("--schema-in-prompt", default=None, help="TTL sample file to include in the prompt")
    ap.add_argument("--no-decomposer", action="store_true", help="Disable query decomposition")
    ap.add_argument("--no-temporal", action="store_true", help="Disable temporal handler")
    ap.add_argument("--no-vocab", action="store_true", help="Do not put the vocabulary summary in the prompt")
    ap.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name")
    ap.add_argument("--max-iterations", type=int, default=15)
    ap.add_argument("--pre-call-sleep", type=float, default=6.0)
    ap.add_argument("--post-call-sleep", type=float, default=60.0)
    ap.add_argument("--store", default="default", choices=["default", "Oxigraph"], help="rdflib store")
    ap.add_argument("--log-dir", default=None, help="Directory for JSONL prompt logs (default BKGQA/logs)")
    ap.add_argument("--project", default=None, help="Google Cloud project (defaults to GOOGLE_CLOUD_PROJECT)")
    ap.add_argument("--quiet", action="store_true", help="Minimal output")
    args = ap.parse_args()

    if args.project:
        os.environ["GOOGLE_CLOUD_PROJECT"] = args.project
    if not os.path.exists(args.ttl):
        print(f"Error: TTL file not found: {args.ttl}")
        sys.exit(1)
    context = None
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")

    agent = KGAgent(
        model=args.model,
        use_decomposer=not args.no_decomposer,
        use_temporal_handler=not args.no_temporal,
        max_iterations=args.max_iterations,
        pre_call_sleep=args.pre_call_sleep,
        post_call_sleep=args.post_call_sleep,
        log_dir=args.log_dir,
        context_block=context,
        ttl_schema_file=args.schema_in_prompt,
        vocab_in_prompt=not args.no_vocab,
    )
    agent.initialize_graph(ttl_file=args.ttl, store=args.store)
    verbose = not args.quiet

    def answer(q):
        state, final = agent.run(q, verbose=verbose)
        if final and final.has_results():
            print("\n" + "=" * 60 + "\nFINAL RESULTS\n" + "=" * 60)
            print(f"\nGenerated SPARQL:\n{final.sparql}\n\nResults:\n{final.results_as_table()}")
        else:
            print("\nNo results found or query generation failed.")

    if args.question:
        answer(args.question)
        return

    print("Interactive mode. Type 'exit', 'quit' or 'q' to end.")
    while True:
        try:
            q = input("\nYour question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", "q", ""):
            break
        answer(q)


if __name__ == "__main__":
    main()
