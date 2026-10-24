"""Local graph data: load a JSON knowledge graph, expose schema / search / context helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import networkx as nx


@dataclass
class Node:
    id: str
    type: str
    name: str
    community: str | None
    properties: dict = field(default_factory=dict)

    def as_text(self) -> str:
        desc = self.properties.get("description", "")
        sig = self.properties.get("signature", "")
        extra = f" ({desc})" if desc else ""
        sig_text = f" sig={sig}" if sig else ""
        return f"{self.type} {self.name}{extra}{sig_text} [community={self.community or '-'}]"


@dataclass
class Edge:
    source: str
    target: str
    relation: str
    properties: dict = field(default_factory=dict)


class GraphStore:
    """In-memory directed graph (NetworkX) loaded from a local JSON file."""

    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        self.meta = raw.get("meta", {})
        self.communities = {c["id"]: c for c in raw.get("communities", [])}
        self.nodes: dict[str, Node] = {}
        for n in raw["nodes"]:
            node = Node(
                id=n["id"],
                type=n["type"],
                name=n["name"],
                community=n.get("community"),
                properties=n.get("properties", {}),
            )
            self.nodes[node.id] = node
        self.edges: list[Edge] = []
        self.g = nx.DiGraph()
        for node in self.nodes.values():
            self.g.add_node(node.id, type=node.type, name=node.name, community=node.community)
        for e in raw["edges"]:
            if e["source"] in self.nodes and e["target"] in self.nodes:
                edge = Edge(source=e["source"], target=e["target"], relation=e["relation"], properties=e.get("properties", {}))
                self.edges.append(edge)
                self.g.add_edge(edge.source, edge.target, relation=edge.relation)

    # ------------------------------------------------------------------ schema
    def schema(self) -> dict:
        """Derive the graph schema (node types, relation types, property keys) for the small model prompt."""
        node_types: dict[str, dict] = {}
        for n in self.nodes.values():
            info = node_types.setdefault(n.type, {"count": 0, "property_keys": set(), "sample_names": []})
            info["count"] += 1
            info["property_keys"].update(n.properties.keys())
            if len(info["sample_names"]) < 5:
                info["sample_names"].append(n.name)
        relations: dict[str, int] = {}
        for e in self.edges:
            relations[e.relation] = relations.get(e.relation, 0) + 1
        return {
            "node_types": {t: {"count": i["count"], "property_keys": sorted(i["property_keys"]), "sample_names": i["sample_names"]} for t, i in node_types.items()},
            "relations": relations,
            "communities": sorted(self.communities.keys()),
        }

    def schema_text(self) -> str:
        s = self.schema()
        lines = ["NODE TYPES:"]
        for t, info in s["node_types"].items():
            lines.append(f"  - {t} (x{info['count']}) props={info['property_keys']} e.g. {info['sample_names']}")
        lines.append("RELATIONS (source -> target counts):")
        for r, c in s["relations"].items():
            lines.append(f"  - {r}: {c}")
        lines.append(f"COMMUNITIES: {s['communities']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ lookup
    def get_node(self, node_id: str) -> Node | None:
        return self.nodes.get(node_id)

    def community_summary(self, community_id: str | None) -> str | None:
        c = self.communities.get(community_id or "")
        return f"[{c['name']}] {c['summary']}" if c else None

    # ------------------------------------------------------- keyword fallback
    @staticmethod
    def _bigrams(text: str) -> set[str]:
        """Character bigrams work for CJK without a tokenizer; also split latin words."""
        text = text.lower()
        words = set(re.findall(r"[a-z0-9_]{2,}", text))
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        bigrams = {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
        return words | bigrams

    def keyword_search(self, question: str, top_k: int = 3) -> list[str]:
        """Score every node by bigram overlap with the question; return top node ids."""
        q_grams = self._bigrams(question)
        if not q_grams:
            return []
        scored = []
        for n in self.nodes.values():
            hay = " ".join([n.name, n.type, n.community or "", str(n.properties.get("description", "")), str(n.properties.get("owner", ""))])
            grams = self._bigrams(hay)
            if not grams:
                continue
            overlap = len(q_grams & grams) / max(len(grams), 1)
            # exact name mention in the question is a strong signal
            if n.name.lower() in question.lower():
                overlap += 1.0
            if overlap > 0:
                scored.append((overlap, n.id))
        scored.sort(reverse=True)
        return [nid for _, nid in scored[:top_k]]

    # ----------------------------------------------------------- serialization
    @staticmethod
    def nodes_text(nodes: list[Node]) -> str:
        return "\n".join("  " + n.as_text() for n in nodes)

    @staticmethod
    def edges_text(node_ids: set[str], edges: list[Edge]) -> str:
        out = []
        for e in edges:
            if e.source in node_ids and e.target in node_ids:
                s = GraphStore._short(e.source)
                t = GraphStore._short(e.target)
                out.append(f"  {s} -[{e.relation}]-> {t}")
        return "\n".join(out)

    @staticmethod
    def _short(node_id: str) -> str:
        return node_id.split(":", 1)[-1]
