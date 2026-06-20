"""API-level behavior-scenario test suite for the requirements agent.

Drives full Q&A conversations through the real HTTP API + real LangGraph (with a
deterministic scripted LLM), records every raw message/payload/tool-call to a JSON
transcript, and scores the produced artifacts with the eval judge.
"""
