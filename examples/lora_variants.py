"""End-to-end demo on the REAL LoRA repo + literature graph.

Pipeline (same as AGENTS.MD, task='code'):
    Human question (implement a LoRA variant from the papers)
    -> local small model (Ollama) plans a structured graph query
    -> deterministic execution over data/lora_graph.json
    -> cloud LLM (DeepSeek) returns executable PyTorch code grounded in
       real loralib signatures + paper abstracts.

Build the graph first:
    python3 tools/build_graph.py --repo /path/to/LoRA --papers /path/to/LORA-Tech \
        --out data/lora_graph.json --summarize
Run:
    GRAPH_PATH=data/lora_graph.json python3 examples/lora_variants.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag_engine.engine import GraphQueryEngine  # noqa: E402

QUESTIONS = [
    "基于 loralib 的 Linear 类实现 QLoRA: 基座权重 NF4 4-bit 量化冻结, LoRA 旁路以 bf16 训练并反传梯度, 给出可执行代码",
    "实现 AdaLoRA: 用 SVD 重参数化替换 loralib Linear 的 lora_A/lora_B 初始化, 并按 spectral movement 在固定预算下分配秩, 给出可执行代码",
    "实现 Bayesian-LoRA: 对 LoRALayer 的低秩因子 A、B 施加高斯后验并训练变分参数, 给出可执行代码",
    "基于 lora_state_dict 实现 HiLoRA 的分层联邦聚合: 全局共享 LoRA 适配器与客户端私有适配器按数据量加权合并, 给出可执行代码",
]


def main() -> None:
    graph = os.getenv("GRAPH_PATH", "data/lora_graph.json")
    engine = GraphQueryEngine(graph)
    print(f"graph: {graph} ({len(engine.store.nodes)} nodes, {len(engine.store.edges)} edges)\n")
    for i, q in enumerate(QUESTIONS, 1):
        r = engine.ask(q, task="code")
        print("=" * 80)
        print(f"[{i}] Human question: {q}")
        print(f"[2] structured query ({r.query_source}): {json.dumps(r.structured_query, ensure_ascii=False)}")
        res = r.result
        print(f"[3] graph execution: matched={len(res.matched)} nodes={len(res.nodes)} edges={len(res.edges)}"
              f"{' [fallback keyword search]' if r.fallback_used else ''}")
        print(f"    anchors: {[n.name for n in res.matched][:6]}")
        print(f"[4] cloud LLM ({r.answer_source}):")
        print(r.answer)
        print()


if __name__ == "__main__":
    main()
