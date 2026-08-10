"""LLM client with Gemini->Groq failover (Task B5).

Not yet implemented. Design: generate() tries Gemini first, falls back to Groq
transparently on error/timeout/rate-limit only, and logs which provider actually
served each request. No fake/hardcoded responses — all calls are live.
"""
