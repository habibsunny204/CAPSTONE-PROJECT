"""Conversational context: keeps the last 5 (question, sql, answer) tuples for
follow-up questions (Task B4).

Not yet implemented. Design: backed by st.session_state, injected into the prompt for
follow-ups, with a reset button in the UI.
"""
