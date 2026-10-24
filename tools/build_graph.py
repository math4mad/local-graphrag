"""Indexer: build a code+literature knowledge graph from a real repo and PDF papers.

Domains:
  --domain lora   Microsoft LoRA repo + LoRA papers (default, backward compatible)
  --domain bayes  NumPyro (installed package) + rethinking R package + Bayes books/papers
                  --repo <site-packages/numpyro> --rrepo <rethinking pkg root> --papers <dir>

Implements README 注意事项7 (real-project extension: AST -> same JSON schema)
and 步骤8 (incremental indexing by file hash; community summaries precomputed
at index time by the cloud LLM instead of query time).

Usage:
    python3 tools/build_graph.py \
        --repo /path/to/LoRA \
        --papers "/path/to/LORA-Tech" \
        --out data/lora_graph.json [--summarize] [--incremental]

Code graph is fully automatic (Python ast; regex for R). Paper abstracts are
extracted automatically (pypdf). The concept layer is curated seed knowledge
(DOMAINS[domain]["seeds"]) — swap it when indexing another domain.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys

# ----------------------------------------------------------------- domain config
# which repo files to index (relative to --repo); examples/NLU is a vendored
# transformers fork and would add thousands of noise nodes -> excluded
DOMAINS: dict[str, dict] = {}

DOMAINS["lora"] = {
    "package": "loralib",
    "code_scope_dirs": ["loralib", os.path.join("examples", "NLG", "src")],
    "code_scope_files": ["setup.py"],
    "paper_files": None,  # all PDFs under --papers
    "community": lambda rel: "nlg-examples" if rel.startswith("examples") else "loralib-core",
    "paper_community": lambda name: "lora-theory" if ("intrinsic" in name.lower() or "measuring" in name.lower()) else "lora-variants",
}

DOMAINS["bayes"] = {
    # --repo points at the installed numpyro package dir; only these subpackages
    "package": "numpyro",
    "code_scope_dirs": ["distributions", "infer"],
    "code_scope_files": [],
    "r_scope": ["R"],  # rethinking package source (regex-parsed)
    # curated MVP reading set from Bayes_Reading_List (3-5 core texts)
    "paper_files": [
        "Statistical Rethinking A Bayesian Course with Examples in R and STAN",
        "Andrew Gelman-Bayesian Data Analysis 3rd Edition.pdf",
        "Think Bayes_ Bayesian Statistics in Python-O'Reilly Media (2021).pdf",
        "Bayesian Statistics The Fun Way(2019).pdf",
        "5-How to become a Bayesian in eight easy steps- An annotated reading list.pdf",
        "A Short Introduction to Bayesian Neural Networks • David Stutz.pdf",
    ],
    "community": lambda rel: "numpyro-infer" if rel.startswith("infer") else "numpyro-distributions",
    "paper_community": lambda name: "bayes-theory",
}


def current_domain() -> dict:
    return DOMAINS[_domain_name[0]]


_domain_name = ["lora"]


def collect_code_files(repo: str) -> list[str]:
    cfg = current_domain()
    files = []
    for d in cfg["code_scope_dirs"]:
        root = os.path.join(repo, d)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.endswith(".py") and "__pycache__" not in name:
                files.append(os.path.relpath(os.path.join(root, name), repo))
    for f in cfg["code_scope_files"]:
        if os.path.isfile(os.path.join(repo, f)):
            files.append(f)
    return files


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ----------------------------------------------------------------- code AST
def docstring_first_line(node) -> str:
    doc = ast.get_docstring(node)
    return doc.strip().splitlines()[0][:160] if doc else ""


def signature(fn: ast.FunctionDef) -> str:
    try:
        return "(" + ast.unparse(fn.args) + ")"
    except Exception:
        return "()"


def community_for(rel_path: str) -> str:
    return current_domain()["community"](rel_path)


def parse_code_file(repo: str, rel_path: str) -> tuple[list[dict], list[dict], list[tuple[str, str]]]:
    """Return (nodes, local_edges, raw_imports) for one file."""
    path = os.path.join(repo, rel_path)
    src = open(path, encoding="utf-8").read()
    tree = ast.parse(src, filename=path)
    community = community_for(rel_path)
    file_id = f"file:{rel_path}"
    nodes = [{
        "id": file_id, "type": "File", "name": rel_path, "community": community,
        "properties": {"path": rel_path, "description": f"{len(src.splitlines())} lines", "loc": len(src.splitlines())},
    }]
    edges = [{"source": file_id, "target": file_id, "relation": "self"}]  # placeholder removed below
    edges = []
    imports: list[tuple[str, str]] = []  # (module string, is_relative)

    def add(node_dict):
        nodes.append(node_dict)
        edges.append({"source": file_id, "target": node_dict["id"], "relation": "defines"})

    def collect_calls(body, owner_name: str, src_id: str):
        names = set()
        for n in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
        edges.extend({"source": src_id, "target": ("@name:" + x), "relation": "calls"} for x in names if x != owner_name)

    for item in tree.body:
        if isinstance(item, (ast.Import, ast.ImportFrom)):
            if isinstance(item, ast.Import):
                for a in item.names:
                    imports.append((a.name, False))
            else:
                mod = item.module or ""
                pkg = current_domain()["package"]
                if item.level:  # relative import inside the package
                    mod = f"{pkg}." + mod if mod else pkg
                imports.append((mod, True))
                for a in item.names:
                    imports.append((f"{mod}.{a.name}", True))
        elif isinstance(item, ast.ClassDef):
            cid = f"class:{item.name}"
            add({"id": cid, "type": "Class", "name": item.name, "community": community,
                 "properties": {"description": docstring_first_line(item) or f"class {item.name}",
                                "bases": [ast.unparse(b) for b in item.bases]}})
            for b in item.bases:
                # only unqualified names (e.g. LoRALayer) resolve in-scope;
                # qualified bases (nn.Linear) are external -> skip
                if isinstance(b, ast.Name):
                    edges.append({"source": cid, "target": f"@class:{b.id}", "relation": "inherits"})
            for m in item.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    mid = f"method:{item.name}.{m.name}"
                    nodes.append({"id": mid, "type": "Method", "name": f"{item.name}.{m.name}",
                                  "community": community,
                                  "properties": {"owner": item.name, "signature": signature(m),
                                                "description": docstring_first_line(m) or f"method {m.name} of {item.name}"}})
                    edges.append({"source": file_id, "target": mid, "relation": "defines"})
                    collect_calls(m.body, m.name, mid)
            for fn in [x for x in item.body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                pass  # methods already handled
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fid = f"func:{item.name}"
            add({"id": fid, "type": "Function", "name": item.name, "community": community,
                 "properties": {"signature": signature(item),
                                "description": docstring_first_line(item) or f"function {item.name}"}})
            collect_calls(item.body, item.name, fid)
    return nodes, edges, imports


def resolve_code(repo: str, files: list[str]) -> tuple[list[dict], list[dict]]:
    """Parse all files, then resolve @name:/@class: placeholders and imports into real edges."""
    all_nodes, raw_edges, imports = [], [], []
    file_imports = {}
    for rel in files:
        nodes, edges, imps = parse_code_file(repo, rel)
        all_nodes.extend(nodes)
        raw_edges.extend(edges)
        file_imports[rel] = imps
        imports.extend((rel, m[0]) for m in imps)

    by_name = {}
    for n in all_nodes:
        if n["type"] in ("Class",):
            by_name.setdefault("@class:" + n["name"], n["id"])
        if n["type"] in ("Function", "Method"):
            short = n["name"].split(".")[-1]
            by_name.setdefault("@name:" + short, n["id"])
            # methods also register Class.method exact
        if n["type"] == "Method":
            by_name.setdefault("@name:" + n["name"], n["id"])

    path_by_id = {f"file:{rel}": rel for rel in files}
    module_to_file = {}
    for rel in files:
        mod = rel[:-3].replace("/", ".").replace(".__init__", "")
        module_to_file[mod] = rel
        module_to_file[mod.split(".")[0]] = rel  # package import -> its __init__

    edges = []
    for e in raw_edges:
        t = e["target"]
        if t.startswith("@class:") or t.startswith("@name:"):
            real = by_name.get(t)
            if real:
                edges.append({**e, "target": real})
        elif t != e["source"]:
            edges.append(e)
    for src_rel, mod in imports:
        tgt_rel = module_to_file.get(mod) or module_to_file.get(mod.split(".")[0])
        if tgt_rel and tgt_rel != src_rel:
            edges.append({"source": f"file:{src_rel}", "target": f"file:{tgt_rel}", "relation": "imports"})
    return all_nodes, edges


# ----------------------------------------------------------------- R sources (rethinking)
R_FUNC_RE = re.compile(r'^([A-Za-z._][\w.]*|`[^`]+`)\s*(?:<-|=)\s*function\s*\(', re.M)
R_SETCLASS_RE = re.compile(r'setClass\(\s*["\'](\w+)["\']')
R_SETMETHOD_RE = re.compile(r'set(?:Generic|Method)\(\s*["\']([\w.]+)["\']')
R_ASSIGN_RE = re.compile(r'^([A-Za-z._][\w.]*)\s*(?:<-|=)\s*(?!function)', re.M)


def collect_r_files(rrepo: str) -> list[str]:
    root = os.path.join(rrepo, "R")
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        if name.lower().endswith(".r") and not name.startswith("."):
            out.append(os.path.relpath(os.path.join(root, name), rrepo))
    return out


def parse_r_file(rrepo: str, rel_path: str) -> tuple[list[dict], list[dict]]:
    """Regex-level parse: top-level `name <- function(...)` -> Function node,
    setClass("Name") -> Class node, name <- non-function -> Variable node.
    Call edges resolved later against known rfunc names (same trick as @name:)."""
    path = os.path.join(rrepo, rel_path)
    src = open(path, encoding="utf-8", errors="replace").read()
    community = "rethinking-r"
    file_id = f"rfile:{rel_path}"
    nodes = [{"id": file_id, "type": "File", "name": rel_path, "community": community,
              "properties": {"path": rel_path, "description": f"R source, {len(src.splitlines())} lines", "lang": "R"}}]
    edges: list[dict] = []
    defined: list[str] = []

    def add(node_dict):
        nodes.append(node_dict)
        edges.append({"source": file_id, "target": node_dict["id"], "relation": "defines"})

    # function spans: from match start to the matching close of the body (approx: next top-level def)
    func_matches = list(R_FUNC_RE.finditer(src))
    for i, m in enumerate(func_matches):
        name = m.group(1).strip("`")
        if name in defined:
            continue
        defined.append(name)
        span_end = func_matches[i + 1].start() if i + 1 < len(func_matches) else len(src)
        span = src[m.start():span_end]
        # signature = text between 'function(' and its first closing paren on the def line region
        sig_m = re.search(r'function\s*\(((?:[^()]|\([^()]*\))*)\)', src[m.end() - 1:m.end() + 1500])
        sig = "(" + re.sub(r"\s+", " ", sig_m.group(1)).strip() + ")" if sig_m else "()"
        # preceding comment block as description
        pre = src[:m.start()].rstrip().splitlines()[-3:]
        desc = " ".join(l.lstrip("#' ").strip() for l in pre if l.strip().startswith("#"))[:160] or f"R function {name}"
        fid = f"rfunc:{name}"
        add({"id": fid, "type": "Function", "name": name, "community": community,
             "properties": {"signature": sig, "description": desc, "lang": "R", "calls": sorted(set(re.findall(r'([A-Za-z._][\w.]*)\s*\(', span)))}})
    for m in R_SETMETHOD_RE.finditer(src):
        name = m.group(1)
        if name in defined:
            continue
        defined.append(name)
        fid = f"rfunc:{name}"
        if not any(n["id"] == fid for n in nodes):
            add({"id": fid, "type": "Function", "name": name, "community": community,
                 "properties": {"signature": "()", "description": f"R S4 generic/method {name}", "lang": "R", "calls": []}})
    for m in R_SETCLASS_RE.finditer(src):
        name = m.group(1)
        cid = f"class:{name}"
        if not any(n["id"] == cid for n in nodes):
            add({"id": cid, "type": "Class", "name": name, "community": community,
                 "properties": {"description": f"R S4 class {name}", "lang": "R"}})
    for m in R_ASSIGN_RE.finditer(src):
        name = m.group(1)
        if name in defined or name.startswith("."):
            continue
        vid = f"rvar:{name}"
        if not any(n["id"] == vid for n in nodes):
            add({"id": vid, "type": "Variable", "name": name, "community": community,
                 "properties": {"description": f"R top-level value {name}", "lang": "R"}})
    return nodes, edges


def resolve_r(rrepo: str) -> tuple[list[dict], list[dict]]:
    files = collect_r_files(rrepo)
    all_nodes, raw_edges = [], []
    for rel in files:
        nodes, edges = parse_r_file(rrepo, rel)
        all_nodes.extend(nodes)
        raw_edges.extend(edges)
    rfunc_by_name = {n["name"]: n["id"] for n in all_nodes if n["type"] == "Function" and n["id"].startswith("rfunc:")}
    class_by_name = {n["name"]: n["id"] for n in all_nodes if n["type"] == "Class"}
    edges = []
    for n in all_nodes:
        if n["type"] == "Function" and n["id"].startswith("rfunc:"):
            for callee in n["properties"].pop("calls", []):
                tgt = rfunc_by_name.get(callee) or class_by_name.get(callee)
                if tgt and tgt != n["id"]:
                    edges.append({"source": n["id"], "target": tgt, "relation": "calls"})
    edges.extend(raw_edges)
    return all_nodes, edges


# ----------------------------------------------------------------- papers
def extract_abstract(pdf_path: str, max_chars: int = 520) -> str:
    try:
        from pypdf import PdfReader
        text = ""
        for page in PdfReader(pdf_path).pages[:2]:
            text += (page.extract_text() or "") + " "
            if len(text) > 2500:
                break
    except Exception:
        return ""
    m = re.search(r"(?:Abstract|ABSTRACT)(.{40,%d})" % max_chars, text, re.S)
    frag = m.group(1) if m else text[:max_chars]
    return re.sub(r"\s+", " ", frag).strip()[:max_chars]


def slug(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "", name.split("-")[0])
    return s or "Paper"


# curated variant/concept layer — the only hand-written part of the graph
SEED_CONCEPTS = [
    {"id": "concept:LoRA", "name": "LoRA", "community": "lora-variants", "properties": {
        "description": "冻结预训练权重 W, 增量 ΔW=BA (B: d×r, A: r×k, r<<d); 训练 ΔW=B@A*scaling, scaling=alpha/r; 推理可合并零延迟; 初始化 A~kaiming, B=0"}},
    {"id": "concept:QLoRA", "name": "QLoRA", "community": "lora-variants", "properties": {
        "description": "基座权重 4-bit NF4 量化冻结, LoRA 旁路以 bf16 训练并反传梯度穿过量化层; 配合双重量化与分页优化器, 65B 模型单卡 48GB 微调"}},
    {"id": "concept:AdaLoRA", "name": "AdaLoRA", "community": "lora-variants", "properties": {
        "description": "将 ΔW 重参数化为 A(t)B(t) 的 SVD 形式 Λ 对角加权; 按奇异值的 spectral movement 在固定总预算下自适应分配/剪枝秩, 替代 LoRA 均匀秩与固定初始化"}},
    {"id": "concept:BayesianLoRA", "name": "Bayesian-LoRA", "community": "lora-variants", "properties": {
        "description": "对低秩因子 A,B 施加高斯先验并用变分推断/采样近似后验, 提供不确定性估计; 初始化与训练目标从点估计改为分布参数 (均值+方差)"}},
    {"id": "concept:HiLoRA", "name": "HiLoRA", "community": "lora-variants", "properties": {
        "description": "个性化联邦学习下的分层 LoRA: 全局共享低秩适配器 + 客户端私有适配器分层聚合, 按客户端数据分布加权聚合 B@A 增量"}},
    {"id": "concept:IntrinsicDim", "name": "Intrinsic-Dimension", "community": "lora-theory", "properties": {
        "description": "LM 微调的有效参数更新集中在极低内在维度 (ID1 假说: 微调增益可用 <1000 维子空间达到 90%+; ID2: 损失路径内在维度低), 解释了 LoRA 低秩约束为何有效, 指导秩 r 的选择"}},
]

# (concept/paper id, relation, target id) — targets must exist in code/paper nodes
SEED_EDGES = [
    ("concept:LoRA", "implemented_by", "class:Linear"),
    ("concept:LoRA", "implemented_by", "class:Embedding"),
    ("concept:LoRA", "implemented_by", "class:MergedLinear"),
    ("concept:LoRA", "implemented_by", "class:ConvLoRA"),
    ("concept:LoRA", "implemented_by", "class:LoRALayer"),
    ("concept:LoRA", "implemented_by", "func:mark_only_lora_as_trainable"),
    ("concept:LoRA", "implemented_by", "func:lora_state_dict"),
    ("concept:QLoRA", "extends", "concept:LoRA"),
    ("concept:AdaLoRA", "extends", "concept:LoRA"),
    ("concept:BayesianLoRA", "extends", "concept:LoRA"),
    ("concept:HiLoRA", "extends", "concept:LoRA"),
    ("concept:IntrinsicDim", "motivates", "concept:LoRA"),
    ("concept:QLoRA", "modifies", "method:Linear.forward"),
    ("concept:AdaLoRA", "modifies", "method:Linear.reset_parameters"),
    ("concept:AdaLoRA", "modifies", "method:Linear.__init__"),
    ("concept:BayesianLoRA", "modifies", "class:LoRALayer"),
    ("concept:HiLoRA", "applies_to", "func:lora_state_dict"),
    ("concept:HiLoRA", "applies_to", "func:mark_only_lora_as_trainable"),
]


def paper_concept(filename: str) -> str | None:
    """Explicit filename -> concept mapping (most specific first; substring "lora"
    must not match before variant names like adalora/qlora)."""
    f = filename.lower().replace("-", "").replace(" ", "")
    fn = DOMAINS[_domain_name[0]].get("paper_concept")
    if fn:
        return fn(f)
    if "adalora" in f:
        return "concept:AdaLoRA"
    if "bayesian" in f:
        return "concept:BayesianLoRA"
    if "hilora" in f:
        return "concept:HiLoRA"
    if "qlora" in f:
        return "concept:QLoRA"
    if "intrinsic" in f or "measuring" in f:
        return "concept:IntrinsicDim"
    if "lowrankadaptation" in f:
        return "concept:LoRA"
    return None


# ---- Bayes domain seed layer (NumPyro + rethinking + Statistical Rethinking MVP)
BAYES_SEED_CONCEPTS = [
    {"id": "concept:ConjugatePrior", "name": "Conjugate-Prior", "community": "bayes-theory", "properties": {
        "description": "先验与似然同族使后验闭式可解: Beta-Binomial, Gamma-Poisson, Dirichlet-Multinomial; numpyro.distributions.conjugate 内置复合分布, 无需 MCMC 即可精确后验"}},
    {"id": "concept:Reparameterization", "name": "Reparameterization-Trick", "community": "bayes-theory", "properties": {
        "description": "z = mu + sigma*eps (eps~N(0,1)) 把随机节点变确定函数, 使 ELBO 梯度可低方差反传; numpyro Distribution.rsample 即此接口, TransformedDistribution 链式支持"}},
    {"id": "concept:HMC", "name": "HMC", "community": "bayes-theory", "properties": {
        "description": "引入动量 p, 用 Hamilton 动力学沿等势面 Leapfrog 积分提案, 抑制随机游走; 需手调 stepsize/L; numpyro.infer.HMC"}},
    {"id": "concept:NUTS", "name": "NUTS", "community": "bayes-theory", "properties": {
        "description": "HMC 自适应版: 二叉树倍增轨迹 + 双平均自适应 stepsize + 无转向终止判据, Stan/rethinking ulam() 的默认采样器; numpyro.infer.NUTS"}},
    {"id": "concept:MCMC", "name": "MCMC", "community": "bayes-theory", "properties": {
        "description": "马尔可夫链蒙特卡洛后验采样框架 (Gibbs/HMC/NUTS/ensemble); numpyro.infer.MCMC 驱动 kernel, 提供 warmup/多链/收敛诊断接口"}},
    {"id": "concept:VariationalInference", "name": "Variational-Inference", "community": "bayes-theory", "properties": {
        "description": "以参数族 q(z;phi) 最小化 KL(q||p)=ELBO 近似后验, 把推断变优化; ADVI/自动导引 (AutoNormal/MVN); rethinking quap() 为 MAP+高斯拉普拉斯近似的 VI 实现"}},
    {"id": "concept:HierarchicalModel", "name": "Hierarchical-Model", "community": "bayes-theory", "properties": {
        "description": "多层/部分池化模型: 组级参数共享超先验, 小样本组自动收缩 (pooling); Statistical Rethinking 核心主题, ulam()/numpyro plate 表达, 非中心参数化改善 NUTS 几何"}},
    {"id": "concept:PosteriorPredictive", "name": "Posterior-Predictive-Check", "community": "bayes-theory", "properties": {
        "description": "从后验抽参数再模拟 replicate 数据与观测对比, 检验模型失配; rethinking link()/sim() 与 numpyro Predictive(posterior_samples) 对应实现"}},
    {"id": "concept:ModelComparison", "name": "Model-Comparison", "community": "bayes-theory", "properties": {
        "description": "WAIC/LOO 基于逐观测点 log 似然 (pointwise) 的渐近信息准则比较预测性能, 含惩罚项与 SE; rethinking compare()/WAIC()/LOO(); numpyro 需自行计算 point_log_prob"}},
]

BAYES_SEED_EDGES = [
    ("concept:ConjugatePrior", "implemented_by", "class:BetaBinomial"),
    ("concept:ConjugatePrior", "implemented_by", "class:GammaPoisson"),
    ("concept:ConjugatePrior", "implemented_by", "class:DirichletMultinomial"),
    ("concept:ConjugatePrior", "corresponds_to", "class:Beta"),
    ("concept:ConjugatePrior", "corresponds_to", "class:Gamma"),
    ("concept:Reparameterization", "modifies", "method:Distribution.rsample"),
    ("concept:Reparameterization", "applies_to", "class:TransformedDistribution"),
    ("concept:Reparameterization", "motivates", "concept:VariationalInference"),
    ("concept:NUTS", "extends", "concept:HMC"),
    ("concept:NUTS", "implemented_by", "class:NUTS"),
    ("concept:NUTS", "corresponds_to", "rfunc:ulam"),
    ("concept:HMC", "implemented_by", "class:HMC"),
    ("concept:MCMC", "implemented_by", "class:MCMC"),
    ("concept:MCMC", "implemented_by", "class:Gibbs"),
    ("concept:MCMC", "covers", "concept:NUTS"),
    ("concept:MCMC", "covers", "concept:HMC"),
    ("concept:VariationalInference", "implemented_by", "class:SVI"),
    ("concept:VariationalInference", "implemented_by", "class:AutoNormal"),
    ("concept:VariationalInference", "implemented_by", "class:ELBO"),
    ("concept:VariationalInference", "corresponds_to", "rfunc:quap"),
    ("concept:HierarchicalModel", "applies_to", "rfunc:ulam"),
    ("concept:HierarchicalModel", "applies_to", "class:Independent"),
    ("concept:HierarchicalModel", "applies_to", "class:Normal"),
    ("concept:PosteriorPredictive", "implemented_by", "rfunc:link"),
    ("concept:PosteriorPredictive", "implemented_by", "rfunc:sim"),
    ("concept:PosteriorPredictive", "implemented_by", "class:Predictive"),
    ("concept:ModelComparison", "implemented_by", "rfunc:compare"),
    ("concept:ModelComparison", "implemented_by", "rfunc:WAIC"),
    ("concept:ModelComparison", "implemented_by", "rfunc:LOO"),
    ("concept:ModelComparison", "proposed_in", "paper:AndrewGelman"),
]

DOMAINS["bayes"]["seeds"] = (BAYES_SEED_CONCEPTS, BAYES_SEED_EDGES)
DOMAINS["bayes"]["paper_concept"] = lambda f: (
    "concept:HierarchicalModel" if "rethinking" in f else
    "concept:ModelComparison" if "gelman" in f or "bayesiandataanalysis" in f else
    "concept:ConjugatePrior" if "thinkbayes" in f else
    "concept:MCMC" if "funway" in f else
    "concept:VariationalInference" if "bayesianneuralnetworks" in f or "stutz" in f else None)


def parse_papers(papers_dirs: list[str]) -> tuple[list[dict], list[dict]]:
    cfg = current_domain()
    wanted = cfg.get("paper_files")
    nodes, edges = [], []
    seen_pids = set()
    for papers_dir in papers_dirs:
        names = wanted if wanted else sorted(os.listdir(papers_dir))
        for name in names:
            path = os.path.join(papers_dir, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1] and not name.lower().endswith(".pdf"):
                continue
            pid = f"paper:{slug(name)}"
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            title = (name[:-4] if name.lower().endswith(".pdf") else name).replace("-", " ").replace(".pdf", "")
            abstract = extract_abstract(path)
            cid = paper_concept(name)
            nodes.append({"id": pid, "type": "Paper", "name": title[:70], "community": cfg["paper_community"](name),
                          "properties": {"description": abstract or title, "file": name, "abstract": abstract,
                                         "concept": cid or ""}})
    if wanted:
        found = {n for n in sum((sorted(os.listdir(d)) for d in papers_dirs), [])}
        for name in wanted:
            if name not in found:
                print(f"  [warn] paper missing, skipped: {name}", file=sys.stderr)
    return nodes, edges


# ----------------------------------------------------------------- summaries
def resolve_cloud_config() -> dict:
    deepseek = os.getenv("DEEPSEEK_KEY", "")
    key = os.getenv("CLOUD_LLM_API_KEY") or deepseek
    base = os.getenv("CLOUD_LLM_BASE_URL") or ("https://api.deepseek.com/v1" if deepseek else "https://api.openai.com/v1")
    model = os.getenv("CLOUD_LLM_MODEL") or ("deepseek-chat" if deepseek else "gpt-4o-mini")
    return {"api_key": key, "base_url": base.rstrip("/"), "model": model}


def summarize_communities(nodes: list[dict], communities: list[str]) -> dict[str, str]:
    """Precompute community summaries at index time (步骤8). Cloud LLM if key set, else heuristic."""
    cfg = resolve_cloud_config()
    out = {}
    for cid in communities:
        members = [n for n in nodes if n["community"] == cid]
        listing = "\n".join(f"- {n['type']} {n['name']}: {str(n['properties'].get('description', ''))[:120]}" for n in members)
        if cfg["api_key"]:
            try:
                import requests
                resp = requests.post(
                    f"{cfg['base_url']}/chat/completions",
                    headers={"Authorization": f"Bearer {cfg['api_key']}"},
                    json={"model": cfg["model"], "temperature": 0.2, "messages": [
                        {"role": "system", "content": "为知识图谱社区写一段 80 字以内的中文摘要, 概括成员实体的职责与相互关系。只输出摘要文本。"},
                        {"role": "user", "content": f"社区: {cid}\n成员:\n{listing}"}]},
                    timeout=90)
                resp.raise_for_status()
                out[cid] = resp.json()["choices"][0]["message"]["content"].strip()
                continue
            except Exception as e:
                print(f"  [warn] cloud summary failed for {cid}: {e}", file=sys.stderr)
        names = ", ".join(n["name"] for n in members[:8])
        out[cid] = f"社区 {cid}: 包含 {len(members)} 个实体 ({names} 等)。"
    return out


# ----------------------------------------------------------------- incremental
def load_cache(out_path: str) -> dict:
    if not os.path.isfile(out_path):
        return {}
    with open(out_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="lora", choices=sorted(DOMAINS))
    ap.add_argument("--repo", required=True, help="root of the code repository")
    ap.add_argument("--rrepo", default="", help="root of an R package source (regex-parsed, bayes domain)")
    ap.add_argument("--papers", default=None, action="append", help="directory containing PDF papers (repeatable)")
    ap.add_argument("--out", default="data/lora_graph.json")
    ap.add_argument("--summarize", action="store_true", help="precompute community summaries (cloud LLM)")
    ap.add_argument("--incremental", action="store_true", help="reuse cached parse results for unchanged files")
    args = ap.parse_args()
    _domain_name[0] = args.domain

    repo = os.path.abspath(args.repo)
    cache = load_cache(args.out) if args.incremental else {}
    cached_hashes = cache.get("meta", {}).get("file_hashes", {})
    cached_nodes = {n["id"]: n for n in cache.get("nodes", [])}
    cached_edges = cache.get("edges", [])

    new_hashes, nodes, edges = {}, [], []
    code_files = collect_code_files(repo)
    print(f"code scope: {len(code_files)} files")
    changed_files = []
    for rel in code_files:
        h = sha256(os.path.join(repo, rel))
        new_hashes[f"code:{rel}"] = h
        if h != cached_hashes.get(f"code:{rel}") or not cached_nodes.get(f"file:{rel}"):
            changed_files.append(rel)
    if changed_files:
        f_nodes, f_edges = resolve_code(repo, code_files)  # cross-file resolution needs full set
        nodes.extend(f_nodes)
        edges.extend(f_edges)
        print(f"re-parsed {len(changed_files)} changed files: {changed_files[:6]}{'...' if len(changed_files) > 6 else ''}")
    else:
        for n in cached_nodes.values():
            if n["type"] != "Paper" and not n["id"].startswith(("rfile:", "rfunc:", "rvar:")):
                nodes.append(n)
        edges.extend([e for e in cached_edges if not e["source"].startswith(("paper:", "concept:", "rfile:", "rfunc:"))])
        print("code unchanged: reused cached nodes/edges")

    if args.rrepo:
        r_nodes, r_edges = resolve_r(os.path.abspath(args.rrepo))
        nodes.extend(r_nodes)
        edges.extend(r_edges)
        print(f"R sources: {len(r_nodes)} nodes, {len(r_edges)} edges")

    papers_dirs = [os.path.abspath(p) for p in (args.papers or []) if p]
    if papers_dirs:
        p_nodes, p_edges = [], []
        pdf_changed = False
        wanted = current_domain().get("paper_files")
        names = []
        for d in papers_dirs:
            names += [n for n in (wanted if wanted else sorted(os.listdir(d))) if os.path.isfile(os.path.join(d, n))]
        for name in names:
            for d in papers_dirs:
                path = os.path.join(d, name)
                if os.path.isfile(path):
                    h = sha256(path)
                    new_hashes[f"paper:{name}"] = h
                    if h != cached_hashes.get(f"paper:{name}"):
                        pdf_changed = True
                    break
        # first run has no paper hashes -> pdf_changed is True anyway
        if pdf_changed:
            p_nodes, p_edges = parse_papers(papers_dirs)
            print(f"papers re-extracted: {len(p_nodes)}")
        else:
            p_nodes = [n for n in cached_nodes.values() if n["type"] == "Paper"]
            p_edges = [e for e in cached_edges if e["source"].startswith("paper:")]
            print("papers unchanged: reused cache")
        nodes.extend(p_nodes)
        edges.extend(p_edges)

    # curated concept layer (always rebuilt, cheap)
    seed_concepts, seed_edges = current_domain().get("seeds", (SEED_CONCEPTS, SEED_EDGES))
    for c in seed_concepts:
        c.setdefault("type", "Concept")
    existing_ids = {n["id"] for n in nodes} | {c["id"] for c in seed_concepts}
    for c in seed_concepts:
        if c["id"] not in {n["id"] for n in nodes}:
            nodes.append(c)
    for src, rel, tgt in seed_edges:
        if src in existing_ids and tgt in existing_ids:
            edges.append({"source": src, "target": tgt, "relation": rel})

    # dedupe
    seen_ids = set()
    uniq_nodes = [n for n in nodes if not (n["id"] in seen_ids or seen_ids.add(n["id"]))]
    # derive concept->paper edges from Paper node properties (stable across full/incremental)
    for n in uniq_nodes:
        cid = n.get("properties", {}).get("concept")
        if n["type"] == "Paper" and cid:
            edges.append({"source": cid, "target": n["id"], "relation": "proposed_in"})
    seen_edges = set()
    uniq_edges = []
    for e in edges:
        k = (e["source"], e["target"], e["relation"])
        if e["source"] in {n["id"] for n in uniq_nodes} and e["target"] in {n["id"] for n in uniq_nodes} and k not in seen_edges:
            seen_edges.add(k)
            uniq_edges.append(e)

    communities = sorted({n["community"] for n in uniq_nodes if n.get("community")})
    if args.summarize:
        print("precomputing community summaries at index time...")
        summaries = summarize_communities(uniq_nodes, communities)
    else:
        old_summaries = {c["id"]: c["summary"] for c in cache.get("communities", [])}
        summaries = {cid: old_summaries.get(cid, f"社区 {cid}") for cid in communities}

    graph = {
        "meta": {"project": os.path.basename(repo), "built_from": "ast+pypdf indexer (tools/build_graph.py)",
                 "file_hashes": new_hashes},
        "communities": [{"id": cid, "name": cid, "summary": summaries[cid]} for cid in communities],
        "nodes": uniq_nodes,
        "edges": uniq_edges,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}: {len(uniq_nodes)} nodes, {len(uniq_edges)} edges, communities={communities}")


if __name__ == "__main__":
    main()
