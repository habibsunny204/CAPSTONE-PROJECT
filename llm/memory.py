"""Conversational context: keeps the last 5 (question, sql, answer) tuples for
follow-up questions (Task B4).

Storage-agnostic on purpose: ConversationMemory just wraps a plain list, so Task C
can hold an instance in st.session_state (which persists across Streamlit reruns
within a session and can hold arbitrary Python objects, not just JSON) without this
module importing Streamlit itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

MAX_TURNS = 5


@dataclass
class ConversationMemory:
    """Keeps the last MAX_TURNS (question, sql, answer) turns, newest-last."""

    turns: list[dict[str, Any]] = field(default_factory=list)

    def add(self, question: str, sql: str, answer: str) -> None:
        """Record one turn, evicting the oldest once over MAX_TURNS."""
        self.turns.append({"question": question, "sql": sql, "answer": answer})
        if len(self.turns) > MAX_TURNS:
            self.turns = self.turns[-MAX_TURNS:]

    def get_history(self) -> list[dict[str, Any]]:
        """The current turns, newest-last -- the exact shape
        llm.prompts.build_conversation_context() and llm.pipeline.answer_question()
        expect for their `history` argument.
        """
        return list(self.turns)

    def reset(self) -> None:
        """Clear all history (backs Task C's reset button)."""
        self.turns = []
