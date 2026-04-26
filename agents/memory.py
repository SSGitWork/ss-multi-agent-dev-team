"""
Dual-layer memory system for the Coder Agent.

Layer 1 – Sliding-window conversation buffer:
    Keeps the last N turns (user + assistant pairs) in a simple list.
    Provides immediate conversational context to the LLM.

Layer 2 – ChromaDB semantic memory:
    Every turn is embedded and persisted in a Chroma collection.
    Before each LLM call the agent queries this store for the top-k
    most relevant past interactions, enabling long-term recall even
    after the sliding window has evicted older turns.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SLIDING_WINDOW_SIZE: int = int(os.getenv("SLIDING_WINDOW_SIZE", "10"))
CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION_NAME", "coder_agent_memory")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class MemoryTurn:
    """A single conversational turn."""

    role: str          # "user" | "assistant" | "system" | "tool"
    content: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer 1 – Sliding-window buffer
# ---------------------------------------------------------------------------
class SlidingWindowBuffer:
    """Fixed-size FIFO buffer of the most recent conversation turns."""

    def __init__(self, max_size: int = SLIDING_WINDOW_SIZE) -> None:
        self._max_size = max_size
        self._buffer: List[MemoryTurn] = []

    # -- public API --------------------------------------------------------
    def add(self, turn: MemoryTurn) -> None:
        """Append a turn; evict the oldest if the window is full."""
        self._buffer.append(turn)
        if len(self._buffer) > self._max_size:
            self._buffer.pop(0)

    def get_recent(self, n: Optional[int] = None) -> List[MemoryTurn]:
        """Return the last *n* turns (defaults to entire window)."""
        if n is None:
            return list(self._buffer)
        return list(self._buffer[-n:])

    def clear(self) -> None:
        self._buffer.clear()

    @property
    def size(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Layer 2 – ChromaDB semantic memory
# ---------------------------------------------------------------------------
class SemanticMemory:
    """Chroma-backed vector store for long-term semantic retrieval.

    Uses Chroma's *default* embedding function (all-MiniLM-L6-v2) so
    there is no dependency on an external embedding API for local dev.
    """

    def __init__(
        self,
        collection_name: str = CHROMA_COLLECTION,
        persist_dir: str = CHROMA_PERSIST_DIR,
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- public API --------------------------------------------------------
    def store(self, turn: MemoryTurn) -> None:
        """Embed and persist a single turn."""
        self._collection.add(
            ids=[turn.turn_id],
            documents=[turn.content],
            metadatas=[{"role": turn.role, **turn.metadata}],
        )

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """Return the *top_k* most semantically similar past turns."""
        results = self._collection.query(
            query_texts=[query],
            n_results=top_k,
        )
        hits: List[dict] = []
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                hits.append(
                    {"content": doc, "metadata": meta, "distance": dist}
                )
        return hits

    def clear(self) -> None:
        """Drop and recreate the collection (useful in tests)."""
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Unified memory façade
# ---------------------------------------------------------------------------
class AgentMemory:
    """Combines both memory layers behind a single interface.

    Usage:
        memory = AgentMemory()
        memory.add_turn("user", "Write a fibonacci function")
        context = memory.build_context("fibonacci")
    """

    def __init__(self) -> None:
        self.buffer = SlidingWindowBuffer()
        self.semantic = SemanticMemory()

    def add_turn(self, role: str, content: str, **metadata: str) -> None:
        """Record a turn in *both* memory layers."""
        turn = MemoryTurn(role=role, content=content, metadata=metadata)
        self.buffer.add(turn)
        self.semantic.store(turn)

    def build_context(self, current_query: str, top_k: int = 5) -> str:
        """Assemble a context string for the next LLM call.

        1. Retrieve semantically relevant past turns.
        2. Append the sliding-window recent turns.
        3. De-duplicate and return as a formatted string.
        """
        # -- semantic hits --------------------------------------------------
        semantic_hits = self.semantic.retrieve(current_query, top_k=top_k)
        seen_contents: set[str] = set()
        parts: List[str] = []

        if semantic_hits:
            parts.append("=== Relevant Past Context (Semantic Memory) ===")
            for hit in semantic_hits:
                if hit["content"] not in seen_contents:
                    role = hit["metadata"].get("role", "unknown")
                    parts.append(f"[{role}] {hit['content']}")
                    seen_contents.add(hit["content"])

        # -- sliding window -------------------------------------------------
        recent = self.buffer.get_recent()
        if recent:
            parts.append("\n=== Recent Conversation (Sliding Window) ===")
            for turn in recent:
                if turn.content not in seen_contents:
                    parts.append(f"[{turn.role}] {turn.content}")
                    seen_contents.add(turn.content)

        return "\n".join(parts) if parts else ""

    def clear(self) -> None:
        self.buffer.clear()
        self.semantic.clear()
