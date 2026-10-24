"""Interactive REPL showing every stage of the GraphRAG pipeline."""

from __future__ import annotations

import json
import sys

from .engine import GraphQueryEngine

BAR = "=" * 72


def print_result(r) -> None:
    print(f"\n{BAR}")
    print(f"[1] Human question : {r.question}")
    print(f"[2] Local small model ({r.query_source}) -> structured query:")
    print(json.dumps(r.structured_query, ensure_ascii=False, indent=2))
    if r.fallback_used:
        print("    (structured query was empty/invalid -> keyword fallback used)")
    assert r.result is not None
    print(f"[3] Graph execution: matched={len(r.result.matched)} nodes={len(r.result.nodes)} edges={len(r.result.edges)}" + (f" error={r.result.error}" if r.result.error else ""))
    print(f"[4] Cloud LLM ({r.answer_source}) -> answer:")
    print(r.answer)
    print(BAR)


def main() -> None:
    # --task code: ground code generation (e.g. LoRA variants) in the retrieved subgraph
    task = "code" if "code" in sys.argv[1:] else "answer"
    engine = GraphQueryEngine()
    print(f"Local GraphRAG engine | graph: {engine.store.meta.get('project', '?')} "
          f"({len(engine.store.nodes)} nodes, {len(engine.store.edges)} edges) | task: {task}")
    print(f"local model: {engine.local.model} @ {engine.local.base_url} | "
          f"cloud: {engine.cloud.model} ({'key set' if engine.cloud.available else 'offline template'})")
    print("Type a question (or 'quit').\n")
    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        print_result(engine.ask(question, task=task))


if __name__ == "__main__":
    sys.exit(main())
