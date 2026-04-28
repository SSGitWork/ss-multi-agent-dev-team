"""
Dual-layer memory system with resilience.
"""

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import chromadb
from dotenv import load_dotenv

from agents.config import get_settings

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class MemoryTurn:
    """Represents a single conversational memory entry.

    Stores the role, content, metadata, and unique identifier for
    a conversation turn that can be stored in short-term or semantic memory.
    """
    role: str
    content: str
    turn_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    metadata: dict = field(default_factory=dict)


class SlidingWindowBuffer:
    """Short-term memory buffer for recent conversation turns.

    Maintains a fixed-size sliding window of the most recent interactions
    used to provide immediate conversational context.
    """

    def __init__(self, max_size: int = 10) -> None:
        settings = get_settings()
        self._max_size = max_size or settings.agent.sliding_window_size
        self._buffer: List[MemoryTurn] = []

    def add(self, turn: MemoryTurn) -> None:
        """Add a conversation turn to the sliding window buffer."""
        self._buffer.append(turn)
        if len(self._buffer) > self._max_size:
            self._buffer.pop(0)

    def get_recent(self, n: Optional[int] = None) -> List[MemoryTurn]:
        """Return the most recent conversation turns from the buffer."""
        if n is None:
            return list(self._buffer)
        return list(self._buffer[-n:])

    def clear(self) -> None:
        """Remove all stored turns from the sliding window buffer."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Return the current number of conversation turns stored in the sliding window buffer."""
        return len(self._buffer)


class SemanticMemory:
    """Long-term semantic memory backed by a vector database.

    Stores conversation turns as embeddings and allows retrieval of
    relevant past context using semantic similarity search.
    """

    def __init__(
        self,
        collection_name: str = "",
        persist_dir: str = "",
    ) -> None:
        settings = get_settings()
        self._collection_name = collection_name or settings.chroma.collection_name
        self._persist_dir = persist_dir or settings.chroma.persist_dir

        try:
            if settings.chroma.use_http:
                self._client = chromadb.HttpClient(
                    host=settings.chroma.host,
                    port=settings.chroma.port,
                )
            else:
                self._client = chromadb.PersistentClient(path=self._persist_dir)

            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info("ChromaDB initialized: collection=%s", self._collection_name)
        except Exception as exc:
            logger.warning("ChromaDB initialization failed: %s. Using fallback.", exc)
            self._client = None
            self._collection = None

    def store(self, turn: MemoryTurn) -> None:
        """Persist a conversation turn in semantic memory.

        Stores the turn content and metadata in the configured vector database
        for later semantic retrieval. If the vector store is unavailable,
        the operation is silently skipped.
        """
        if self._collection is None:
            return
        try:
            self._collection.add(
                ids=[turn.turn_id],
                documents=[turn.content],
                metadatas=[{"role": turn.role, **turn.metadata}],
            )
        except Exception as exc:
            logger.warning("Failed to store in ChromaDB: %s", exc)

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """Retrieve semantically similar memory entries.

        Args:
            query: Natural language query used for similarity search.
            top_k: Maximum number of results to return.

        Returns:
            A list of dictionaries containing content, metadata, and similarity distance.
        """
        if self._collection is None:
            return []
        try:
            if self._collection.count() == 0:
                return []
            n = min(top_k, self._collection.count())
            results = self._collection.query(query_texts=[query], n_results=n)
            hits: List[dict] = []
            if results and results["documents"]:
                for doc, meta, dist in zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                ):
                    hits.append({"content": doc, "metadata": meta, "distance": dist})
            return hits
        except Exception as exc:
            logger.warning("ChromaDB query failed: %s", exc)
            return []

    def clear(self) -> None:
        """Remove all stored semantic memory entries and recreate the collection."""
        if self._client is None or self._collection is None:
            return
        try:
            self._client.delete_collection(self._collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            logger.warning("ChromaDB clear failed: %s", exc)

    @property
    def count(self) -> int:
        """Return the total number of stored semantic memory records."""
        if self._collection is None:
            return 0
        try:
            return self._collection.count()
        except Exception:
            return 0


class AgentMemory:
    """Combined memory system used by agents.

    Provides both short-term sliding window memory and long-term
    semantic retrieval for building rich LLM context.
    """

    def __init__(self) -> None:
        self.buffer = SlidingWindowBuffer()
        self.semantic = SemanticMemory()

    def add_turn(self, role: str, content: str, **metadata: str) -> None:
        """Add a new conversation turn to both short-term and semantic memory.

        The turn is appended to the sliding window buffer and stored in the
        vector database for semantic retrieval.
        """
        turn = MemoryTurn(role=role, content=content, metadata=metadata)
        self.buffer.add(turn)
        self.semantic.store(turn)

    def build_context(self, current_query: str, top_k: int = 5) -> str:
        """Construct a context string combining semantic and recent memory.

        Retrieves relevant past messages from semantic memory and combines them
        with the recent sliding-window conversation to produce a prompt context
        for the LLM.
        """
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

        recent = self.buffer.get_recent()
        if recent:
            parts.append("\n=== Recent Conversation (Sliding Window) ===")
            for turn in recent:
                if turn.content not in seen_contents:
                    parts.append(f"[{turn.role}] {turn.content}")
                    seen_contents.add(turn.content)

        return "\n".join(parts) if parts else ""

    def clear(self) -> None:
        """Clear both sliding window and semantic memory stores."""
        self.buffer.clear()
        self.semantic.clear()
