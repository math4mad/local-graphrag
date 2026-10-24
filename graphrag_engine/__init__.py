"""Local GraphRAG query engine.

Pipeline: Human -> Local Small Model (GraphRAG) -> Structured Query -> Cloud LLM -> Results
"""

from .engine import GraphQueryEngine

__all__ = ["GraphQueryEngine"]
