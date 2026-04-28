"""
A2A Message Bus.

A lightweight in-process message bus that routes A2A messages between
agents.  All messages are logged for observability and can be replayed.

This is NOT a network transport - both agents run in the same process.
The bus provides:
  - Intent validation on send.
  - Correlation-based message retrieval.
  - Full message history for debugging.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from agents.schemas_a2a import (
    A2AIntent,
    A2AMessage,
    AgentRole,
    validate_message_route,
)


class A2AMessageBus:
    """In-process message bus for A2A protocol communication.

    Usage:
        bus = A2AMessageBus()
        bus.send(message)
        replies = bus.get_messages_for(AgentRole.QA, correlation_id="abc")
    """

    def __init__(self) -> None:
        self._messages: List[A2AMessage] = []
        self._by_correlation: Dict[str, List[A2AMessage]] = {}
        self._by_receiver: Dict[AgentRole, List[A2AMessage]] = {}

    # -- Send --------------------------------------------------------------
    def send(self, message: A2AMessage) -> None:
        """Validate and enqueue a message.

        Raises ValueError if the intent is not valid for the route.
        """
        if not validate_message_route(message):
            raise ValueError(
                f"Invalid A2A route: {message.sender.value} , "
                f"{message.receiver.value} with intent {message.intent.value}"
            )

        self._messages.append(message)

        # Index by correlation_id
        self._by_correlation.setdefault(message.correlation_id, []).append(message)

        # Index by receiver
        self._by_receiver.setdefault(message.receiver, []).append(message)

    # -- Receive -----------------------------------------------------------
    def get_messages_for(
        self,
        receiver: AgentRole,
        correlation_id: Optional[str] = None,
        intent: Optional[A2AIntent] = None,
    ) -> List[A2AMessage]:
        """Retrieve messages for a specific receiver, optionally filtered."""
        msgs = self._by_receiver.get(receiver, [])

        if correlation_id:
            msgs = [m for m in msgs if m.correlation_id == correlation_id]

        if intent:
            msgs = [m for m in msgs if m.intent == intent]

        return msgs

    def get_latest_for(
        self,
        receiver: AgentRole,
        correlation_id: str,
        intent: Optional[A2AIntent] = None,
    ) -> Optional[A2AMessage]:
        """Get the most recent message matching the filters."""
        msgs = self.get_messages_for(receiver, correlation_id, intent)
        return msgs[-1] if msgs else None

    def get_conversation(self, correlation_id: str) -> List[A2AMessage]:
        """Get all messages in a correlation chain, ordered by time."""
        return sorted(
            self._by_correlation.get(correlation_id, []),
            key=lambda m: m.timestamp,
        )

    # -- Inspection --------------------------------------------------------
    @property
    def all_messages(self) -> List[A2AMessage]:
        """Return a copy of all messages that have passed through the bus."""
        return list(self._messages)

    @property
    def message_count(self) -> int:
        """Return the total number of messages currently stored in the bus."""
        return len(self._messages)

    def clear(self) -> None:
        """Remove all stored messages and reset all internal indexes."""
        self._messages.clear()
        self._by_correlation.clear()
        self._by_receiver.clear()

    def summary(self) -> str:
        """Human-readable summary of all messages."""
        lines = [f"A2A Message Bus — {self.message_count} messages:"]
        for m in self._messages:
            lines.append(
                f"  [{m.timestamp}] {m.sender.value} , {m.receiver.value} "
                f"| {m.intent.value} | corr={m.correlation_id[:8]}..."
            )
        return "\n".join(lines)
