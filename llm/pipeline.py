"""Phase 1/2/3 orchestration: LLM generates SQL -> sandbox validates and executes ->
LLM generates a narrative from the result (Task B2).

Not yet implemented. Design: on Phase 2 failure, sends the error back to the LLM once
for a single retry, then surfaces a clean failure message. The LLM never executes
anything directly; it only ever produces text.
"""
