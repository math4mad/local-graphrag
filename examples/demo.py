"""End-to-end demo of the GraphRAG query engine.

Runs without Ollama / cloud key (falls back to heuristic query planning and
template answers). With `OLLAMA_BASE_URL` / `CLOUD_LLM_API_KEY` configured,
the same script exercises the real small-model and cloud-LLM stages.
"""

from __future__ import annotations

import json

from graphrag_engine.cli import print_result
from graphrag_engine.engine import GraphQueryEngine
from graphrag_engine.query_dsl import execute, to_context_text

QUESTIONS = [
    "支付回调 handle_callback 是怎么处理的? 它调用了哪些函数?",
    "如果修改 TransactionManager, 哪些模块会受影响?",
    "用户积分兑换商品的实现流程是什么?",
]


def main() -> None:
    engine = GraphQueryEngine()
    print(f"Graph schema derived from local data:\n{engine.store.schema_text()}\n")
    for q in QUESTIONS:
        print_result(engine.ask(q))

    # show a hand-written structured query executed directly (bypassing the small model)
    manual = {
        "match": {"type": None, "where": {"name": {"$contains": "charge"}}},
        "traverse": {"relation": ["calls"], "direction": "both", "hops": 2},
        "aggregate": {"group_by": "community"},
        "include_community_summaries": False,
        "limit": 20,
    }
    print("\nManual structured query (bypassing the small model):\n" + json.dumps(manual, ensure_ascii=False))
    result = execute(engine.store, manual)
    print(f"matched={len(result.matched)} nodes={len(result.nodes)} edges={len(result.edges)} aggregates={result.aggregates}")
    print(to_context_text(engine.store, result))


if __name__ == "__main__":
    main()
