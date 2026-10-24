"""Pipeline orchestrator: Human -> Local Small Model -> Structured Query -> Graph -> Cloud LLM -> Results."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .cloud_llm import CloudLLM
from .graph_store import GraphStore
from .local_model import LocalSmallModel
from .query_dsl import QueryResult, execute, keyword_query, to_context_text

DEFAULT_GRAPH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "code_graph.json")
# GRAPH_PATH env overrides the demo graph (e.g. data/lora_graph.json built by tools/build_graph.py)
DEFAULT_GRAPH = os.getenv("GRAPH_PATH", DEFAULT_GRAPH)


@dataclass
class EngineResult:
    question: str
    structured_query: dict = field(default_factory=dict)
    query_source: str = ""          # ollama | heuristic
    fallback_used: bool = False     # keyword fallback after empty/failed structured query
    result: QueryResult | None = None
    context_text: str = ""
    answer: str = ""
    answer_source: str = ""         # cloud | template


class GraphQueryEngine:
    def __init__(
        self,
        graph_path: str = DEFAULT_GRAPH,
        ollama_base_url: str | None = None,
        local_model: str | None = None,
        cloud_base_url: str | None = None,
        cloud_api_key: str | None = None,
        cloud_model: str | None = None,
    ):
        self.store = GraphStore(graph_path)
        self.local = LocalSmallModel(
            self.store,
            base_url=ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=local_model or os.getenv("LOCAL_MODEL", "qwen2.5-coder:7b"),
        )
        # cloud key: CLOUD_LLM_API_KEY first, then DEEPSEEK_KEY (with DeepSeek defaults)
        deepseek_key = os.getenv("DEEPSEEK_KEY", "")
        key = cloud_api_key or os.getenv("CLOUD_LLM_API_KEY") or deepseek_key
        default_base = "https://api.deepseek.com/v1" if (deepseek_key and not os.getenv("CLOUD_LLM_API_KEY")) else "https://api.openai.com/v1"
        default_model = "deepseek-chat" if (deepseek_key and not os.getenv("CLOUD_LLM_API_KEY")) else "gpt-4o-mini"
        self.cloud = CloudLLM(
            base_url=cloud_base_url or os.getenv("CLOUD_LLM_BASE_URL", default_base),
            api_key=key,
            model=cloud_model or os.getenv("CLOUD_LLM_MODEL", default_model),
        )

    def ask(self, question: str, task: str = "answer") -> EngineResult:
        out = EngineResult(question=question)

        # 1) local small model: question -> structured query
        query, source = self.local.to_query(question)
        out.structured_query, out.query_source = query, source

        # 2) execute the structured query against the local graph
        result = execute(self.store, query)

        # 3) fallback: keyword search when the query errored, matched nothing,
        #    or returned an isolated node (bad direction/relation choice by the small model)
        degraded = result.error or result.is_empty or (len(result.nodes) <= 3 and len(result.edges) <= 3)
        if degraded:
            fallback_query = keyword_query(self.store, question)
            fb_result = execute(self.store, fallback_query)
            if not fb_result.is_empty:
                query, result = fallback_query, fb_result
                out.fallback_used = True
        out.result = result

        # 4) cloud LLM: subgraph context + question -> final answer
        out.context_text = to_context_text(self.store, result)
        out.answer, out.answer_source = self.cloud.synthesize(question, self.store, result, out.context_text, task=task)
        return out
