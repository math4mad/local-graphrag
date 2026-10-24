# Local GraphRAG Query Engine

> 展示 Agent 处理图数据的能力：**本地小模型做结构化图查询规划，云端大模型做最终推理**。
> 设计依据见 [AGENTS.MD](AGENTS.MD)。

## 核心流水线

```
Human ──> Local Small Model ──> Structured Query ──> Local Graph ──> Cloud LLM ──> Results
(自然语言)   (Ollama qwen2.5-coder)   (JSON DSL)        (NetworkX)      (DeepSeek 等)   (中文答案 / 可执行代码)
   本地                本地                 本地             本地             云端
```

云端 LLM 支持两种任务模式：`answer`（问答，默认）与 `code`（代码生成——基于检索到的真实类/函数签名 + 文献要点，产出可运行实现，见下文 LoRA 变体案例）。

隐私模型：**图谱数据与查询计划全程留在本地**，只有「精简后的子图上下文 + 原始问题」发送到云端 LLM，Token 消耗远低于传统 RAG 全量塞上下文的做法。

## 为什么是这个架构

| 环节 | 职责 | 选型 |
|---|---|---|
| 本地图数据 | 代码知识图谱（File/Class/Method/Function + Paper/Concept 节点，defines/inherits/calls/imports/extends/modifies/implemented_by 边，社区摘要） | JSON + NetworkX，零运维；`tools/build_graph.py` 从真实仓库 AST + PDF 自动构建 |
| 本地小模型 | 自然语言 → 结构化图查询（意图理解 + Schema 对齐），**不需要强生成能力，只需精准的结构化输出** | Ollama `qwen2.5-coder:7b`（1.5b 可用） |
| 结构化查询 | 确定性执行，不经过任何 LLM，可审计、可复现 | 自研 JSON DSL（见下） |
| 云端大模型 | 基于子图上下文做深度推理：问答合成（`answer`）或接地代码生成（`code`，强制与图中真实 API 签名兼容、缺口显式标注假设） | 任意 OpenAI 兼容端点（已验证 DeepSeek `deepseek-chat`） |
| 回退机制 | 小模型不可用 / 查询失败 / 结果退化（空或 ≤3 节点 ≤3 边的薄子图）时，自动降级到关键词检索，保证可用性 | 字符 bigram 评分（对中文友好，无需分词器） |

## 目录结构

```
graphrag_engine/
├── graph_store.py    # 图数据加载、Schema 推导、关键词回退检索、子图序列化
├── query_dsl.py      # 结构化查询 DSL 定义 + 确定性执行器
├── local_model.py    # Ollama 小模型客户端（Schema + few-shot 约束 JSON 输出）
├── cloud_llm.py      # OpenAI 兼容云端客户端（answer/code 双模式）+ 离线模板回退
├── engine.py         # 流水线编排与回退策略
└── cli.py            # 交互式 REPL（逐阶段透明展示）
data/code_graph.json  # 示例：shop-service 代码知识图谱（30 节点 / 57 边 / 5 社区）
data/lora_graph.json  # 真实图谱：Microsoft LoRA 仓库 + 8 篇 LoRA 文献（179 节点 / 241 边）
tools/build_graph.py  # 索引器：AST 解析代码 + pypdf 提取文献摘要 + 增量索引 + 索引期社区摘要
examples/demo.py      # 端到端演示脚本（demo 图谱）
examples/lora_variants.py  # 真实场景演示：基于文献生成 LoRA 变体的可执行代码
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

## 真实仓库扩展（已落地）

`tools/build_graph.py` 把本引擎接到了**真实代码库 + 真实文献**上——Microsoft [LoRA](https://github.com/microsoft/LoRA) 官方仓库（`loralib/` + NLG 示例，17 个 Python 文件经 `ast` 自动解析）与 8 篇 LoRA 论文（pypdf 自动提取摘要），产出 `data/lora_graph.json`：

```bash
pip install pypdf
python3 tools/build_graph.py \
    --repo /path/to/LoRA \
    --papers "/path/to/LORA-Tech" \
    --out data/lora_graph.json \
    --summarize --incremental
```

- **全自动部分**：代码图谱（File/Class/Method/Function + defines/inherits/calls/imports 边）、论文摘要抽取、社区划分与摘要（`--summarize` 在**索引阶段**调用云端 LLM 预计算，查询期零开销）
- **增量索引**（`--incremental`）：按文件 sha256 判断变更，未变文件复用缓存节点/边；已验证增量输出与全量完全一致
- **策展层**：变体概念节点（LoRA/QLoRA/AdaLoRA/Bayesian-LoRA/HiLoRA/Intrinsic-Dimension）与 `extends`/`modifies`/`implemented_by` 边是脚本内显式种子（`SEED_CONCEPTS`/`SEED_EDGES`），换领域时替换即可

### 基于文献生成 LoRA 变体的可执行代码

`cloud_llm.py` 新增 `task="code"` 模式：云端提示词要求「与子图中的真实 API 签名兼容、文献要点必须落实到代码、未覆盖细节显式标注假设」：

```bash
GRAPH_PATH=data/lora_graph.json python3 examples/lora_variants.py        # 4 个变体实现问题
GRAPH_PATH=data/lora_graph.json python3 -m graphrag_engine code          # REPL 代码生成模式
```

已验证（qwen2.5-coder:7b + deepseek-chat）：QLoRA（NF4 双重量化 + bf16 LoRA 旁路）、AdaLoRA（SVD 重参数化 + spectral-movement 预算分配）、Bayesian-LoRA（低秩因子变分后验）、HiLoRA（分层联邦聚合，正确按 `ΔW=B@A` 增量加权合并再 SVD 截断回低秩）——四份输出均为继承 loralib 真实签名、含假设标注的完整可运行代码。

## 注意事项

1. **`OLLAMA_MODELS` 环境变量陷阱**：若曾把它指向自定义目录（如 MLX 模型目录），`ollama list` 会显示空、重启服务后"丢模型"。留空即使用默认 `~/.ollama/models`。
2. **小模型的方向混淆**：7b 会把「X 依赖哪些模块」（应 `out`）误判为影响分析（`in`）。已在 system prompt 规则中显式约束；即使仍出错，引擎的回退机制会检测到「孤立节点 / 空结果」并自动降级关键词检索兜底。
3. **实现类问题的"薄子图"陷阱**（LoRA 变体验证时发现）：小模型会把「实现 X 变体」规划成只走 `proposed_in` 边命中论文节点，子图仅 3 节点 2 边、不含要修改的代码实体——答案质量骤降。对策：prompt 规则要求实现类问题锚定 Concept/Class 后用 `"relation": ["*"], "direction": "both", hops 2`；同时回退阈值收紧为「≤3 节点且≤3 边即降级关键词检索」，AdaLoRA 子图从 3 节点提升到 19 节点。
4. **概念映射不要用朴素子串匹配**：论文文件名/标题里 "AdaLoRA"、"QLoRA" 都包含子串 "LoRA"，先匹配短名会把变体论文全挂到基座概念上。`build_graph.py` 用显式的最具体优先映射（`paper_concept()`），并保证 `proposed_in` 边从 Paper 节点属性派生——全量与增量构建输出逐边一致（已验证）。
5. **索引范围要排除 vendored 代码**：LoRA 仓库的 `examples/NLU/` 是 transformers 完整 fork，纳入会引入数千噪声节点。`CODE_SCOPE_DIRS` 按项目实际核心裁剪（本例仅 `loralib/` + `examples/NLG/src/`）。
6. **few-shot 过拟合**：示例问法与测试问法完全一致会导致小模型照抄而非泛化。few-shot 请使用与目标问题**不同实体**的问法（见 `local_model.py` FEW_SHOT）。
7. **社区摘要必须与图边一致**：摘要声称的依赖关系（如"各服务依赖 TransactionManager"）若图中没有对应边，检索会落空。策展 `SEED_EDGES` 时每条边都要指向真实存在的节点 id（构建器会静默丢弃悬空边）。
8. **换模型只改环境变量**：`LOCAL_MODEL=qwen2.5-coder:1.5b python3 -m graphrag_engine`，无需改代码。7b 查询准确率显著高于 1.5b。
9. **成本**：发往云端的是裁剪后的子图文本（问答通常 < 1K token），一次问答成本极低。`code` 模式输出完整实现代码，回答较长，但输入侧成本不变。
10. **增量索引依赖已有输出文件**：`--incremental` 以 `--out` 文件自身为缓存，首次构建必须先全量跑；解析器逻辑变更后也应先全量重建（缓存的旧节点不会自动重解析）。
11. **接入其他真实仓库**：运行 `tools/build_graph.py`（AST 解析 + pypdf 摘要，见上节）生成同构 JSON，`GRAPH_PATH` 指向新图谱即可；概念种子层 `SEED_CONCEPTS`/`SEED_EDGES` 与 `paper_concept()` 需按领域替换。
12. **生产化建议**（本 demo 未实现）：监听 git 事件的自动增量索引（当前为手动 `--incremental`）、向量检索作为二级回退。索引期社区摘要与增量索引已在 `tools/build_graph.py` 落地。

## License

MIT
