"""End-to-end demo on the REAL Bayesian stack: NumPyro + rethinking (R) + literature.

Pipeline (same as AGENTS.MD, task='code'):
    Human question (implement a Bayesian method/variant from the books & papers)
    -> local small model (Ollama) plans a structured graph query
    -> deterministic execution over data/bayes_graph.json
       (NumPyro AST + rethinking R sources + 6 books/papers + concept layer)
    -> cloud LLM (DeepSeek) returns executable NumPyro code grounded in
       real class signatures + concept/paper descriptions.

Build the graph first:
    python3 tools/build_graph.py --domain bayes --repo <site-packages/numpyro> \
        --rrepo <rethinking pkg root> --papers <Bayes_Reading_List> --papers data/bayes_papers \
        --out data/bayes_graph.json --summarize
Run:
    GRAPH_PATH=data/bayes_graph.json python3 examples/bayes_variants.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graphrag_engine.engine import GraphQueryEngine  # noqa: E402

QUESTIONS = [
    "基于 Conjugate-Prior 概念和 numpyro 的 BetaBinomial, 实现 Beta-Binomial 共轭模型: 闭式后验与 NUTS 采样对比验证, 给出可执行代码",
    "按照 Statistical Rethinking 的 Hierarchical-Model, 用 NumPyro 实现部分池化的多层模型 (plate + 非中心参数化), 用 NUTS 采样并用 Predictive 做后验预测检查, 给出可执行代码",
    "用 numpyro 的 SVI 与 AutoNormal 实现变分推断近似后验, 并对比 rethinking quap() 的 MAP+高斯近似思路, 给出可执行代码",
    "基于 Model-Comparison 概念, 在 NumPyro 中实现逐观测 log 似然计算 WAIC 与 LOO 比较两个模型 (对应 rethinking compare()/WAIC()/LOO()), 给出可执行代码",
]


def main() -> None:
    graph = os.getenv("GRAPH_PATH", "data/bayes_graph.json")
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
