"""Local small model (GraphRAG stage): natural-language question -> structured graph query.

Primary path: a small instruct model served by Ollama (e.g. qwen2.5-coder:7b),
constrained to JSON output, prompted with the live graph schema + few-shot pairs.
Fallback path: a pure-Python heuristic anchor finder, so the pipeline always works.
"""

from __future__ import annotations

import json

import requests

from .graph_store import GraphStore
from .query_dsl import DSL_DOC, keyword_query

FEW_SHOT = [
    (
        "支付回调 handle_callback 都调用了哪些函数?",
        {"match": {"type": "Function", "where": {"name": {"$eq": "handle_callback"}}},
         "traverse": {"relation": ["calls"], "direction": "out", "hops": 1},
         "aggregate": None, "include_community_summaries": False, "limit": 30},
    ),
    (
        "如果修改 TransactionManager, 哪些地方会受影响?",
        {"match": {"type": None, "where": {"name": {"$contains": "TransactionManager"}}},
         "traverse": {"relation": ["inherits", "calls", "imports", "depends"], "direction": "in", "hops": 2},
         "aggregate": {"group_by": "community"}, "include_community_summaries": True, "limit": 30},
    ),
    (
        "订单模块的整体结构是什么样的?",
        {"match": {"type": None, "where": {"community": {"$eq": "order"}}},
         "traverse": None, "aggregate": {"group_by": "type"},
         "include_community_summaries": True, "limit": 30},
    ),
    (
        "下单 create_order 的实现流程是什么?",
        {"match": {"type": None, "where": {"name": {"$contains": "create_order"}}},
         "traverse": {"relation": ["calls"], "direction": "out", "hops": 2},
         "aggregate": None, "include_community_summaries": True, "limit": 30},
    ),
]


class LocalSmallModel:
    def __init__(self, store: GraphStore, base_url: str = "http://localhost:11434", model: str = "qwen2.5-coder:7b", timeout: int = 120):
        self.store = store
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    # ------------------------------------------------------------------ prompt
    def system_prompt(self) -> str:
        examples = "\n".join(
            f"Q: {q}\nA: {json.dumps(a, ensure_ascii=False)}" for q, a in FEW_SHOT
        )
        return (
            "You are the query-planning stage of a local GraphRAG engine. "
            "Convert the user's question into ONE structured graph query (JSON only, no prose, no markdown fences).\n\n"
            + DSL_DOC
            + "\n\nCURRENT GRAPH SCHEMA:\n"
            + self.store.schema_text()
            + "\n\nFEW-SHOT EXAMPLES:\n"
            + examples
            + "\n\nRules: anchor entities via match.where on name/description; keep hops <= 2; "
            "use direction 'in' ONLY for 'who calls X / what is affected by changing X'; "
            "for 'what does X depend on / call / use / affect' (X as subject) use direction 'out'; "
            "when unsure which relation exists, prefer \"relation\": \"*\"; "
            "for 'how does X work / 实现流程' questions use 'out'; "
            "set include_community_summaries=true for architecture/overview questions."
        )

    # ------------------------------------------------------------------ convert
    def to_query(self, question: str) -> tuple[dict, str]:
        """Return (structured_query, source) where source is 'ollama' or 'heuristic'."""
        query = self._ollama_query(question)
        if query is not None:
            return query, "ollama"
        return keyword_query(self.store, question), "heuristic"

    def _ollama_query(self, question: str) -> dict | None:
        payload = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {"role": "system", "content": self.system_prompt()},
                {"role": "user", "content": question},
            ],
        }
        try:
            resp = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            query = json.loads(content)
        except Exception:
            return None
        if not isinstance(query, dict) or "match" not in query:
            return None
        return query
