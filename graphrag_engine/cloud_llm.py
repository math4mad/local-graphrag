"""Cloud LLM stage: subgraph context + original question -> final natural-language answer.

Primary path: any OpenAI-compatible chat-completions endpoint (configured via env).
Fallback path: a deterministic template summarizer, so the demo runs fully offline.
"""

from __future__ import annotations

import requests

from .graph_store import GraphStore
from .query_dsl import QueryResult

SYSTEM = (
    "你是本地 GraphRAG 查询引擎的最终推理层。用户问题对应的代码知识图谱子图已在下方给出, "
    "请仅依据该子图与社区摘要作答, 用中文简洁回答, 指出关键实体、调用/依赖链路与模块边界。"
    "若子图信息不足以回答, 明确说明缺口, 不要编造。"
)


class CloudLLM:
    def __init__(self, base_url: str = "https://api.openai.com/v1", api_key: str = "", model: str = "gpt-4o-mini", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def synthesize(self, question: str, store: GraphStore, result: QueryResult, context_text: str) -> tuple[str, str]:
        """Return (answer, source) where source is 'cloud' or 'template'."""
        if self.available:
            answer = self._chat(question, context_text)
            if answer is not None:
                return answer, "cloud"
        return self._template(question, result), "template"

    def _chat(self, question: str, context_text: str) -> str | None:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"问题: {question}\n\n{context_text}"},
            ],
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception:
            return None

    @staticmethod
    def _template(question: str, result: QueryResult) -> str:
        """Offline fallback: render the retrieved subgraph as a readable answer."""
        lines = [f"针对问题「{question}」, 本地图谱检索结果如下 (离线模板渲染, 未调用云端 LLM):"]
        if result.matched:
            names = ", ".join(n.name for n in result.matched)
            lines.append(f"- 命中实体: {names}")
        if result.edges:
            lines.append(f"- 关联边 ({len(result.edges)} 条):")
            for e in result.edges[:12]:
                s = GraphStore._short(e.source)
                t = GraphStore._short(e.target)
                lines.append(f"    {s} -[{e.relation}]-> {t}")
        if result.aggregates:
            agg = ", ".join(f"{k}={v}" for k, v in result.aggregates.items())
            lines.append(f"- 聚合: {agg}")
        for sm in result.summaries:
            lines.append(f"- 社区摘要: {sm}")
        if result.is_empty:
            lines.append("- 子图为空: 图谱中没有与该问题相关的实体。")
        return "\n".join(lines)
