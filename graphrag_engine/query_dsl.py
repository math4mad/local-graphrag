"""Structured query DSL and its executor over the local graph.

The small model emits a JSON query of this shape; the engine executes it
deterministically against the GraphStore (no LLM involved in execution).

{
  "match":    {"type": "Function", "where": {"name": {"$contains": "callback"}}},
  "traverse": {"relation": ["calls"], "direction": "both", "hops": 2},
  "aggregate": {"group_by": "type"},
  "include_community_summaries": true,
  "limit": 30
}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .graph_store import Edge, GraphStore, Node

DSL_DOC = """Structured graph query (JSON only, no prose):
{
  "match": {"type": "<NodeType or null for any>", "where": {"<property>": {"$eq|$contains|$in|$regex": <value>}}},
  "traverse": {"relation": ["<rel>", ...] or "*", "direction": "out"|"in"|"both", "hops": 1..3},
  "aggregate": {"group_by": "type"|"community"} or null,
  "include_community_summaries": true|false,
  "limit": <max nodes, default 30>
}
Notes: "in" direction finds callers/upstream; "out" finds callees/downstream; "both" finds neighbors.
Use "where" on property "name" or "description" to anchor on entities mentioned in the question."""


@dataclass
class QueryResult:
    matched: list[Node] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)
    aggregates: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.nodes


# ------------------------------------------------------------------ filters
def _match_op(value, spec) -> bool:
    """spec: literal (equality) or {"$op": operand}."""
    if isinstance(spec, dict):
        for op, operand in spec.items():
            if op == "$eq" and value != operand:
                return False
            elif op == "$contains":
                if operand is None or str(operand).lower() not in str(value or "").lower():
                    return False
            elif op == "$in" and value not in operand:
                return False
            elif op == "$regex" and not re.search(str(operand), str(value or "")):
                return False
        return True
    return value == spec


def _node_matches(store: GraphStore, node: Node, match: dict) -> bool:
    mtype = match.get("type")
    if mtype and node.type != mtype:
        return False
    for key, spec in (match.get("where") or {}).items():
        if key == "name":
            value = node.name
        elif key in ("community", "id", "type"):
            value = getattr(node, key)
        else:
            value = node.properties.get(key)
        if not _match_op(value, spec):
            return False
    return True


# ------------------------------------------------------------------ traverse
def _neighbors(store: GraphStore, node_id: str, direction: str, relations: set[str] | None) -> set[str]:
    out: set[str] = set()
    if direction in ("out", "both"):
        for _, dst, data in store.g.out_edges(node_id, data=True):
            if relations is None or data["relation"] in relations:
                out.add(dst)
    if direction in ("in", "both"):
        for src, _, data in store.g.in_edges(node_id, data=True):
            if relations is None or data["relation"] in relations:
                out.add(src)
    return out


# ------------------------------------------------------------------ execute
def execute(store: GraphStore, query: dict) -> QueryResult:
    """Run a validated structured query against the graph."""
    result = QueryResult()
    try:
        match = query.get("match") or {}
        matched_ids = [n.id for n in store.nodes.values() if _node_matches(store, n, match)]
        result.matched = [store.nodes[i] for i in matched_ids]

        node_ids = set(matched_ids)
        traverse = query.get("traverse")
        if traverse:
            rel = traverse.get("relation")
            relations = None if rel in (None, "*", "any") else set(rel if isinstance(rel, list) else [rel])
            direction = traverse.get("direction", "both")
            hops = max(1, min(int(traverse.get("hops", 1)), 3))
            frontier = set(matched_ids)
            for _ in range(hops):
                nxt: set[str] = set()
                for nid in frontier:
                    nxt |= _neighbors(store, nid, direction, relations)
                nxt -= node_ids
                node_ids |= nxt
                frontier = nxt
                if not frontier:
                    break

        limit = int(query.get("limit", 30))
        # keep matched nodes first, then traversal neighbors
        ordered = list(dict.fromkeys(matched_ids)) + sorted(node_ids - set(matched_ids))
        ordered = ordered[:limit]
        result.nodes = [store.nodes[i] for i in ordered]

        kept = set(ordered)
        result.edges = [e for e in store.edges if e.source in kept and e.target in kept]

        agg = query.get("aggregate")
        if agg:
            key = agg.get("group_by", "type")
            counts: dict[str, int] = {}
            for n in result.nodes:
                k = getattr(n, key, None) or "-"
                counts[k] = counts.get(k, 0) + 1
            result.aggregates = counts

        if query.get("include_community_summaries"):
            seen: set[str] = set()
            for n in result.nodes:
                if n.community and n.community not in seen:
                    seen.add(n.community)
                    s = store.community_summary(n.community)
                    if s:
                        result.summaries.append(s)
    except Exception as exc:  # malformed query from the small model
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def keyword_query(store: GraphStore, question: str) -> dict:
    """Fallback query when the small model is unavailable or its query failed:
    anchor on keyword-matched nodes and pull 1-hop neighbors."""
    anchors = store.keyword_search(question, top_k=3)
    return {
        "match": {"type": None, "where": {"name": {"$in": [store.nodes[a].name for a in anchors]}}},
        "traverse": {"relation": "*", "direction": "both", "hops": 1},
        "aggregate": None,
        "include_community_summaries": True,
        "limit": 30,
    }


def to_context_text(store: GraphStore, result: QueryResult) -> str:
    """Serialize the query result subgraph into compact text for the cloud LLM."""
    parts = []
    parts.append("MATCHED ENTITIES:\n" + (store.nodes_text(result.matched) or "  (none)"))
    parts.append("SUBGRAPH NODES:\n" + (store.nodes_text(result.nodes) or "  (none)"))
    parts.append("SUBGRAPH EDGES:\n" + (store.edges_text({n.id for n in result.nodes}, result.edges) or "  (none)"))
    if result.aggregates:
        parts.append("AGGREGATE: " + ", ".join(f"{k}={v}" for k, v in result.aggregates.items()))
    if result.summaries:
        parts.append("COMMUNITY SUMMARIES:\n" + "\n".join("  " + s for s in result.summaries))
    return "\n\n".join(parts)
