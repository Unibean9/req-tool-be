"""Pure modules behind analyze_node: context loading, prompt assembly, tool gating, turn audit.

Extracted mechanically from app/graphs/nodes.py (plan 260702-agent-behavior-quality Phase 1);
behavior is byte-identical, guarded by the golden-transcript regression in
tests/eval/test_golden_prompts.py. None of these modules may import app.graphs.nodes.
"""
