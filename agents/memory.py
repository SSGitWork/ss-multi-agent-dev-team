from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any

from dotenv import load_dotenv
import chromadb

load_dotenv()


@dataclass
class Message:
    role: str  # "user" or "assistant" or "system"
    content: str


@dataclass
class MemoryManager:
    window_size: int = 6
    collection_name: str = "coder_memory"
    _conversation: List[Message] = field(default_factory=list, init=False)
    _client: chromadb.Client | None = field(default=None, init=False)
    _collection: Any = field(default=None, init=False)

    def __post_init__(self):
        db_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
        self._client = chromadb.PersistentClient(path=db_dir)
        self._collection = self._client.get_or_create_collection(self.collection_name)

    # Sliding window memory
    def add_message(self, role: str, content: str):
        self._conversation.append(Message(role=role, content=content))
        if len(self._conversation) > self.window_size:
            self._conversation = self._conversation[-self.window_size :]

    def get_recent_messages(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self._conversation]

    # Semantic memory
    def add_semantic_memory(self, text: str, metadata: Dict[str, Any] | None = None):
        doc_id = str(uuid.uuid4())
        self._collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )

    def search_semantic_memory(self, query: str, k: int = 3) -> List[str]:
        if not query.strip():
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
        )
        docs = results.get("documents", [[]])[0]
        return docs
