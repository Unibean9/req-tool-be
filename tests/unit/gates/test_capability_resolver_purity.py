"""Import-boundary test: `CapabilityResolver` must stay a pure policy evaluator.

Locks the invariant described in phase-03-capability-resolver-shadow-enforcement.md — no DB
session, no `logging`, no tool-handler import — with an AST check rather than a one-time code
review, so a future edit that reintroduces a side-effect import fails CI immediately.
"""

import ast
import importlib.util
from pathlib import Path

_RESOLVER_PATH = Path(importlib.util.find_spec("app.graphs.gating.capability_resolver").origin)

_FORBIDDEN_MODULE_PREFIXES = (
    "logging",
    "sqlalchemy",
    "app.db",
    "app.graphs.agent_tools",
    "app.graphs.gate_logging",
    "app.graphs.gating.decision_projection",
)


def _imported_module_names(tree: ast.AST) -> list[str]:
    """Collect every dotted path a forbidden-prefix check should consider.

    For `from app.graphs import gate_logging`, `node.module` alone is only "app.graphs" — the
    forbidden module is actually `node.module + "." + alias.name`. Emitting both the bare module
    and each module+alias combination catches `import x.y.z`, `from x.y import z`, and
    `from x.y import z as w` alike (the bound name `w` is irrelevant; what matters is what was
    imported, not what it's called locally).
    """
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _forbidden_offenders(imported: list[str]) -> list[str]:
    return [
        name
        for name in imported
        if any(name == prefix or name.startswith(prefix + ".") for prefix in _FORBIDDEN_MODULE_PREFIXES)
    ]


def test_capability_resolver_has_no_forbidden_imports():
    tree = ast.parse(_RESOLVER_PATH.read_text(encoding="utf-8"))
    imported = _imported_module_names(tree)
    offenders = _forbidden_offenders(imported)
    assert offenders == [], f"capability_resolver.py imports forbidden modules: {offenders}"


def test_ast_check_catches_from_package_import_alias_form():
    """Regression: `from app.graphs import gate_logging` must be caught even though
    `node.module` alone ("app.graphs") doesn't start with a forbidden prefix — only the
    module+alias combination ("app.graphs.gate_logging") does."""
    tree = ast.parse("from app.graphs import gate_logging as gl\n")
    imported = _imported_module_names(tree)
    offenders = _forbidden_offenders(imported)
    assert offenders == ["app.graphs.gate_logging"]


def test_capability_resolver_module_has_no_db_session_or_logger_symbol():
    """Belt-and-suspenders runtime check: the imported module object itself carries no `logging`
    logger or DB-session-shaped attribute at the module level."""
    import app.graphs.gating.capability_resolver as resolver_module

    assert not hasattr(resolver_module, "logger")
    assert not hasattr(resolver_module, "logging")
    assert not hasattr(resolver_module, "log_gate_decision")
