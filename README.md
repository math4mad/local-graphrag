# Local GraphRAG Query Engine

> 展示 Agent 处理图数据的能力：**本地小模型做结构化图查询规划，云端大模型做最终推理**。
> 设计依据见 [AGENTS.MD](AGENTS.MD)。

## 核心流水线

```
Human ──> Local Small Model ──> Structured Query ──> Local Graph ──> Cloud LLM ──> Results
(自然语言)   (Ollama qwen2.5-coder)   (JSON DSL)        (NetworkX)      (DeepSeek 等)   (中文答案)
   本地                本地                 本地             本地             云端
```

隐私模型：**图谱数据与查询计划全程留在本地**，只有「精简后的子图上下文 + 原始问题」发送到云端 LLM，Token 消耗远低于传统 RAG 全量塞上下文的做法。

## 为什么是这个架构

| 环节 | 职责 | 选型 |
|---|---|---|
| 本地图数据 | 代码知识图谱（File/Class/Function 节点，defines/imports/inherits/calls/depends 边，社区摘要） | JSON + NetworkX，零运维 |
| 本地小模型 | 自然语言 → 结构化图查询（意图理解 + Schema 对齐），**不需要强生成能力，只需精准的结构化输出** | Ollama `qwen2.5-coder:7b`（1.5b 可用） |
| 结构化查询 | 确定性执行，不经过任何 LLM，可审计、可复现 | 自研 JSON DSL（见下） |
| 云端大模型 | 基于子图上下文做深度推理与答案合成 | 任意 OpenAI 兼容端点（已验证 DeepSeek `deepseek-chat`） |
| 回退机制 | 小模型不可用 / 查询失败 / 结果退化时，自动降级到关键词检索，保证可用性 | 字符 bigram 评分（对中文友好，无需分词器） |

## 目录结构

```
graphrag_engine/
├── graph_store.py    # 图数据加载、Schema 推导、关键词回退检索、子图序列化
├── query_dsl.py      # 结构化查询 DSL 定义 + 确定性执行器
├── local_model.py    # Ollama 小模型客户端（Schema + few-shot 约束 JSON 输出）
├── cloud_llm.py      # OpenAI 兼容云端客户端 + 离线模板回退
├── engine.py         # 流水线编排与回退策略
└── cli.py            # 交互式 REPL（逐阶段透明展示）
data/code_graph.json  # 示例：shop-service 代码知识图谱（30 节点 / 57 边 / 5 社区）
examples/demo.py      # 端到端演示脚本
```

## 结构化查询 DSL

小模型被约束为只输出如下 JSON，由执行器确定性地跑在本地图上：

```json
{
  "match":    {"type": "Function", "where": {"name": {"$contains": "callback"}}},
  "traverse": {"relation": ["calls"], "direction": "out", "hops": 2},
  "aggregate": {"group_by": "community"},
  "include_community_summaries": true,
  "limit": 30
}
```

- `where` 支持 `$eq` / `$contains` / `$in` / `$regex`；可按 `name`、`description`、`community`、`type` 过滤
- `traverse.direction`：`out` = 它调用了谁（下游/依赖）；`in` = 谁调用它（上游/影响面）；`both` = 邻域
- `relation` 填 `"*"` 表示不限关系类型

## 快速开始

### 1. 安装依赖

```bash
pip install networkx requests   # Python >= 3.10
```

### 2. 本地小模型（可选，不装也能跑——自动降级到启发式查询规划）

```bash
brew install --cask ollama
ollama pull qwen2.5-coder:7b    # 4.7GB；低配机器可用 qwen2.5-coder:1.5b
```

### 3. 配置云端 LLM（可选，不配则用离线模板渲染答案）

支持任意 OpenAI 兼容端点。若只设置了 `DEEPSEEK_KEY`，引擎会自动使用 DeepSeek 默认端点与模型：

```bash
export DEEPSEEK_KEY=sk-xxxx                       # 最简单
# 或通用配置：
export CLOUD_LLM_API_KEY=sk-xxxx
export CLOUD_LLM_BASE_URL=https://api.openai.com/v1
export CLOUD_LLM_MODEL=gpt-4o-mini
```

完整模板见 [.env.example](.env.example)。

### 4. 运行

```bash
python3 -m graphrag_engine          # 交互式 REPL，逐阶段展示流水线
python3 examples/demo.py            # 端到端演示（3 类问题 + 手写查询）
```

REPL 输出示例（每个阶段可独立审计）：

```
[1] Human question : 支付回调 handle_callback 是怎么处理的? 它调用了哪些函数?
[2] Local small model (ollama) -> structured query: {...}
[3] Graph execution: matched=1 nodes=4 edges=3
[4] Cloud LLM (cloud) -> answer: ...
```

## 已验证效果（M1 Pro / 16GB，qwen2.5-coder:7b + deepseek-chat）

| 问题类型 | 小模型生成的查询 | 图执行结果 |
|---|---|---|
| 调用链：「handle_callback 调用了哪些函数」 | `calls` 出向 1 跳 | 精确命中 3 条调用边（验签→改单→发积分） |
| 影响分析：「修改 TransactionManager 哪些受影响」 | 多关系入向 2 跳 + 社区聚合 | 5 个服务类 + 全部社区摘要 + 按模块聚合 |
| 实现流程：「积分兑换的流程」 | `calls` 出向 2 跳 | 11 节点 / 14 边，跨 user/inventory/common 模块 |
| 图外问题：「如何配置 K8s」 | — | 正确拒答，不编造 |

云端答案能正确重建跨模块调用链、标注模块边界，并**主动声明子图未覆盖的信息缺口**（提示词强约束，见 `cloud_llm.py` SYSTEM）。

## 注意事项

1. **`OLLAMA_MODELS` 环境变量陷阱**：若曾把它指向自定义目录（如 MLX 模型目录），`ollama list` 会显示空、重启服务后"丢模型"。留空即使用默认 `~/.ollama/models`。
2. **小模型的方向混淆**：7b 会把「X 依赖哪些模块」（应 `out`）误判为影响分析（`in`）。已在 system prompt 规则中显式约束；即使仍出错，引擎的回退机制会检测到「孤立节点 / 空结果」并自动降级关键词检索兜底。
3. **few-shot 过拟合**：示例问法与测试问法完全一致会导致小模型照抄而非泛化。few-shot 请使用与目标问题**不同实体**的问法（见 `local_model.py` FEW_SHOT）。
4. **社区摘要必须与图边一致**：摘要声称的依赖关系（如"各服务依赖 TransactionManager"）若图中没有对应边，检索会落空。本示例已补 `depends` 边对齐。
5. **换模型只改环境变量**：`LOCAL_MODEL=qwen2.5-coder:1.5b python3 -m graphrag_engine`，无需改代码。7b 查询准确率显著高于 1.5b。
6. **成本**：发往云端的是裁剪后的子图文本（通常 < 1K token），一次问答成本极低。
7. **扩展为真实项目**：把 `data/code_graph.json` 换成你的代码解析产物（tree-sitter / Jedi / semgrep 均可生成同构 JSON），Schema 与 few-shot 自动从图中推导，无需改引擎。
8. **生产化建议**（本 demo 未实现）：增量索引（监听 git 变更只更新受影响子图）、向量检索作为二级回退、社区摘要在索引阶段预生成。

## License

MIT
