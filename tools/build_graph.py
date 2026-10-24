"""Indexer: build a code+literature knowledge graph from a real repo and PDF papers.

Implements README 注意事项7 (real-project extension: AST -> same JSON schema)
and 步骤8 (incremental indexing by file hash; community summaries precomputed
at index time by the cloud LLM instead of query time).

Usage:
    python3 tools/build_graph.py \
        --repo /path/to/LoRA \
        --papers "/path/to/LORA-Tech" \
        --out data/lora_graph.json [--summarize] [--incremental]

Code graph is fully automatic (Python ast). Paper abstracts are extracted
automatically (pypdf). The variant/concept layer is curated seed knowledge
(SEED_CONCEPTS / SEED_EDGES below) — swap it when indexing another domain.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys

# ----------------------------------------------------------------- scope
# which repo files to index (relative to --repo); examples/NLU is a vendored
# transformers fork and would add thousands of noise nodes -> excluded
CODE_SCOPE_DIRS = ["loralib", os.path.join("examples", "NLG", "src")]
CODE_SCOPE_FILES = ["setup.py"]


def collect_code_files(repo: str) -> list[str]:
    files = []
    for d in CODE_SCOPE_DIRS:
        root = os.path.join(repo, d)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            if name.endswith(".py"):
                files.append(os.path.relpath(os.path.join(root, name), repo))
    for f in CODE_SCOPE_FILES:
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
    return "nlg-examples" if rel_path.startswith("examples") else "loralib-core"


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
                if item.level:  # relative import inside the package
                    mod = "loralib." + mod if mod else "loralib"
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


def parse_papers(papers_dir: str) -> tuple[list[dict], list[dict]]:
    nodes, edges = [], []
    for name in sorted(os.listdir(papers_dir)):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(papers_dir, name)
        pid = f"paper:{slug(name)}"
        title = name[:-4].replace("-", " ").replace(".pdf", "")
        abstract = extract_abstract(path)
        theory = "intrinsic" in name.lower() or "measuring" in name.lower()
        cid = paper_concept(name)
        nodes.append({"id": pid, "type": "Paper", "name": title[:70], "community": "lora-theory" if theory else "lora-variants",
                      "properties": {"description": abstract or title, "file": name, "abstract": abstract,
                                     "concept": cid or ""}})
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
    ap.add_argument("--repo", required=True, help="root of the code repository")
    ap.add_argument("--papers", default="", help="directory containing PDF papers")
    ap.add_argument("--out", default="data/lora_graph.json")
    ap.add_argument("--summarize", action="store_true", help="precompute community summaries (cloud LLM)")
    ap.add_argument("--incremental", action="store_true", help="reuse cached parse results for unchanged files")
    args = ap.parse_args()

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
            if n["type"] != "Paper":
                nodes.append(n)
        edges.extend([e for e in cached_edges if not e["source"].startswith(("paper:", "concept:"))])
        print("code unchanged: reused cached nodes/edges")

    if args.papers:
        papers_dir = os.path.abspath(args.papers)
        p_nodes, p_edges = [], []
        pdf_changed = False
        for name in sorted(os.listdir(papers_dir)):
            if name.lower().endswith(".pdf"):
                h = sha256(os.path.join(papers_dir, name))
                new_hashes[f"paper:{name}"] = h
                if h != cached_hashes.get(f"paper:{name}"):
                    pdf_changed = True
        # first run has no paper hashes -> pdf_changed is True anyway
        if pdf_changed:
            p_nodes, p_edges = parse_papers(papers_dir)
            print(f"papers re-extracted: {len(p_nodes)}")
        else:
            p_nodes = [n for n in cached_nodes.values() if n["type"] == "Paper"]
            p_edges = [e for e in cached_edges if e["source"].startswith("paper:")]
            print("papers unchanged: reused cache")
        nodes.extend(p_nodes)
        edges.extend(p_edges)

    # curated concept layer (always rebuilt, cheap)
    for c in SEED_CONCEPTS:
        c.setdefault("type", "Concept")
    existing_ids = {n["id"] for n in nodes} | {c["id"] for c in SEED_CONCEPTS}
    for c in SEED_CONCEPTS:
        if c["id"] not in {n["id"] for n in nodes}:
            nodes.append(c)
    for src, rel, tgt in SEED_EDGES:
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
