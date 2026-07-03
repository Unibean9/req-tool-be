"""Structural guards for the analyze_node decomposition.

The nodes ↔ agent_tools import cycle is broken via app.graphs.interrupts; nothing in app/graphs
may re-introduce it with a function-local import of nodes/agent_tools/interrupts, and the
analysis modules must never import nodes (they are its dependencies, not its peers).
"""

import ast
from pathlib import Path

APP_GRAPHS = Path(__file__).parents[2] / "app" / "graphs"
_CYCLE_MODULES = {"app.graphs.nodes", "app.graphs.agent_tools", "app.graphs.interrupts"}


def _function_local_imports(tree: ast.AST) -> list[str]:
    offenders: list[str] = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.ImportFrom) and node.module in _CYCLE_MODULES:
                offenders.append(f"{func.name}: from {node.module} import ...")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in _CYCLE_MODULES:
                        offenders.append(f"{func.name}: import {alias.name}")
    return offenders


def test_no_function_local_imports_of_cycle_modules():
    offenders: list[str] = []
    for path in APP_GRAPHS.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        offenders.extend(f"{path.name} :: {item}" for item in _function_local_imports(tree))
    assert offenders == [], f"function-local imports of nodes/agent_tools/interrupts remain: {offenders}"


def test_analysis_modules_do_not_import_nodes():
    for path in (APP_GRAPHS / "analysis").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module != "app.graphs.nodes", f"{path.name} imports app.graphs.nodes"
            if isinstance(node, ast.Import):
                assert all(a.name != "app.graphs.nodes" for a in node.names), f"{path.name} imports app.graphs.nodes"


def test_interrupts_is_a_leaf_module():
    tree = ast.parse((APP_GRAPHS / "interrupts.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("app.graphs."):
            assert node.module == "app.graphs.state", f"interrupts.py may only import state, got {node.module}"
