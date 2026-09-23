"""Deterministic PlantUML source generation for the completed use-case table.

Follows ``docs/rule-diagram-usecase/use-case-generation-plantuml-spec-v2.md``: the LLM only
determines use-case and relationship semantics; this module alone decides arrow syntax,
aliasing, module grouping, left/right actor balancing, and layout hints for the single
system-wide use-case diagram. ``part-of`` hierarchy is rendered as comments because it is
table hierarchy rather than a UML relationship.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from app.use_cases.models import UseCaseActor, UseCaseEntry, UseCaseModel

_ASSOCIATION = "association"
_INCLUDE = "include"
_EXTEND = "extend"
_GENERALIZATION = "generalization"

_RULE = "' " + "=" * 25


def render_plantuml(model: UseCaseModel) -> str:
    """Render one complete, left/right-balanced use-case diagram for the whole system."""

    actor_aliases = _aliases("ACT", (actor.id for actor in model.actors))
    use_case_aliases = _aliases("UC", (item.id for item in model.use_cases))
    actors_by_id = {actor.id: actor for actor in model.actors}

    module_order = sorted({item.subsystem_id for item in model.use_cases})
    module_index = {subsystem_id: index for index, subsystem_id in enumerate(module_order)}
    use_case_order = _use_case_order(model.use_cases, module_index)
    left_ids, right_ids = _balance_actor_sides(model, module_index)

    nodesep, ranksep = _spacing_for(len(model.use_cases))
    lines: list[str] = [
        "@startuml",
        "' Generated from the ReqTool use-case table. Edit this source directly when needed.",
        "",
        f'title "{_quote(model.system_name)} \u2014 Complete Use Case Model"',
        "",
        "left to right direction",
        "skinparam shadowing false",
        "skinparam linetype ortho",
        "skinparam packageStyle rectangle",
        f"skinparam nodesep {nodesep}",
        f"skinparam ranksep {ranksep}",
        "",
        "skinparam usecase {",
        "    BackgroundColor #FFF7ED",
        "    BorderColor #F97316",
        "    FontColor #2B2118",
        "}",
        "",
        "skinparam actor {",
        "    BorderColor #0F766E",
        "    FontColor #2B2118",
        "}",
        "",
        "skinparam ArrowColor #57534E",
        "",
    ]

    lines += _actor_block("LEFT ACTORS", left_ids, actors_by_id, actor_aliases)
    lines += _system_block(model, module_order, use_case_order, use_case_aliases)
    lines += _actor_block("RIGHT ACTORS", right_ids, actors_by_id, actor_aliases)
    lines += _association_lines(model, actor_aliases, use_case_aliases, use_case_order)
    lines += _relationship_lines(model, actor_aliases, use_case_aliases, use_case_order)
    lines += _layout_hint_lines(left_ids, right_ids, actor_aliases)
    lines += ["", "@enduml", ""]
    return "\n".join(lines)


def _use_case_order(use_cases: Iterable[UseCaseEntry], module_index: dict[str, int]) -> dict[str, int]:
    ordered = sorted(
        use_cases,
        key=lambda item: (module_index.get(item.subsystem_id, len(module_index)), item.id),
    )
    return {item.id: position for position, item in enumerate(ordered)}


def _spacing_for(use_case_count: int) -> tuple[int, int]:
    if use_case_count <= 15:
        return 60, 70
    if use_case_count <= 30:
        return 80, 90
    return 100, 110


def _actor_block(
    heading: str,
    actor_ids: list[str],
    actors_by_id: dict[str, UseCaseActor],
    actor_aliases: dict[str, str],
) -> list[str]:
    if not actor_ids:
        return []
    lines = [_RULE, f"' {heading}", _RULE, ""]
    for actor_id in actor_ids:
        actor = actors_by_id[actor_id]
        stereotype = " <<system>>" if actor.kind == "external_system" else ""
        lines.append(f'actor "{_quote(actor.name)}" as {actor_aliases[actor_id]}{stereotype}')
    lines.append("")
    return lines


def _system_block(
    model: UseCaseModel,
    module_order: list[str],
    use_case_order: dict[str, int],
    use_case_aliases: dict[str, str],
) -> list[str]:
    subsystem_names = {item.id: item.name for item in model.subsystems}
    use_cases_by_subsystem: dict[str, list[UseCaseEntry]] = defaultdict(list)
    for item in model.use_cases:
        use_cases_by_subsystem[item.subsystem_id].append(item)

    lines = [_RULE, "' SYSTEM", _RULE, "", f'rectangle "{_quote(model.system_name)}" as SYSTEM {{']
    for subsystem_id in module_order:
        items = use_cases_by_subsystem.get(subsystem_id, [])
        if not items:
            continue
        items_sorted = sorted(items, key=lambda item: use_case_order[item.id])
        package_name = subsystem_names.get(subsystem_id, subsystem_id)
        lines.append(f'  package "{_quote(package_name)}" {{')
        for item in items_sorted:
            label = f"{_quote(item.id)}\\n{_quote(item.name)}"
            lines.append(f'    usecase "{label}" as {use_case_aliases[item.id]}')
        lines.append("  }")
    lines.append("}")

    hierarchy_comments = [
        f"' hierarchy: {item.parent_use_case_id} -> {item.id} ({item.name})"
        for item in sorted(model.use_cases, key=lambda entry: use_case_order[entry.id])
        if item.parent_use_case_id and item.parent_use_case_id in use_case_aliases
    ]
    if hierarchy_comments:
        lines.append("")
        lines.extend(hierarchy_comments)
    lines.append("")
    return lines


def _actor_use_case_connections(model: UseCaseModel) -> dict[str, set[str]]:
    connections: dict[str, set[str]] = defaultdict(set)
    for item in model.use_cases:
        connections[item.primary_actor_id].add(item.id)
        for actor_id in item.secondary_actor_ids:
            connections[actor_id].add(item.id)

    use_case_ids = {item.id for item in model.use_cases}
    actor_ids = {actor.id for actor in model.actors}
    for relation in model.relations:
        if relation.kind != _ASSOCIATION:
            continue
        if relation.source_id in actor_ids and relation.target_id in use_case_ids:
            connections[relation.source_id].add(relation.target_id)
        elif relation.target_id in actor_ids and relation.source_id in use_case_ids:
            connections[relation.target_id].add(relation.source_id)
    return connections


def _actor_generalization_clusters(
    model: UseCaseModel, actor_ids: list[str]
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Union-find over actor-to-actor generalization edges so a parent/child pair stays together."""

    parent = {actor_id: actor_id for actor_id in actor_ids}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a: str, b: str) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    actor_id_set = set(actor_ids)
    for relation in model.relations:
        if relation.kind != _GENERALIZATION:
            continue
        if relation.source_id in actor_id_set and relation.target_id in actor_id_set:
            union(relation.source_id, relation.target_id)

    cluster_of = {actor_id: find(actor_id) for actor_id in actor_ids}
    members: dict[str, list[str]] = defaultdict(list)
    for actor_id, root in cluster_of.items():
        members[root].append(actor_id)
    return cluster_of, members


def _actor_generalization_rank(model: UseCaseModel, actor_ids: list[str]) -> dict[str, int]:
    """0 for a generalization root/parent, increasing with distance from the root."""

    actor_id_set = set(actor_ids)
    parent_of: dict[str, str] = {}
    for relation in model.relations:
        if relation.kind != _GENERALIZATION:
            continue
        if relation.source_id in actor_id_set and relation.target_id in actor_id_set:
            parent_of[relation.source_id] = relation.target_id

    def depth(actor_id: str, seen: frozenset[str]) -> int:
        parent_id = parent_of.get(actor_id)
        if parent_id is None or parent_id in seen:
            return 0
        return 1 + depth(parent_id, seen | {actor_id})

    return {actor_id: depth(actor_id, frozenset()) for actor_id in actor_ids}


def _balance_actor_sides(
    model: UseCaseModel,
    module_index: dict[str, int],
) -> tuple[list[str], list[str]]:
    actor_ids = [actor.id for actor in model.actors]
    if not actor_ids:
        return [], []

    use_case_module = {item.id: module_index.get(item.subsystem_id, 0) for item in model.use_cases}
    connections = _actor_use_case_connections(model)
    cluster_of, cluster_members = _actor_generalization_clusters(model, actor_ids)
    parent_rank = _actor_generalization_rank(model, actor_ids)

    median_index = (max(len(module_index), 1) - 1) / 2
    cluster_stats: dict[str, dict[str, float]] = {}
    for root, members in cluster_members.items():
        total = 0
        weighted_sum = 0.0
        for member in members:
            for use_case_id in connections.get(member, ()):
                total += 1
                weighted_sum += use_case_module.get(use_case_id, 0)
        avg_module = weighted_sum / total if total else median_index
        cluster_stats[root] = {"connections": total, "avg_module": avg_module}

    side_of_cluster = {
        root: ("left" if stats["avg_module"] <= median_index else "right") for root, stats in cluster_stats.items()
    }

    def counts() -> tuple[int, int]:
        left = sum(len(cluster_members[root]) for root, side in side_of_cluster.items() if side == "left")
        right = sum(len(cluster_members[root]) for root, side in side_of_cluster.items() if side == "right")
        return left, right

    flipped: set[str] = set()
    left_count, right_count = counts()
    while abs(left_count - right_count) > 1:
        majority_side = "left" if left_count > right_count else "right"
        candidates = [root for root, side in side_of_cluster.items() if side == majority_side and root not in flipped]
        if not candidates:
            break
        candidates.sort(key=lambda root: abs(cluster_stats[root]["avg_module"] - median_index))
        chosen = candidates[0]
        side_of_cluster[chosen] = "right" if side_of_cluster[chosen] == "left" else "left"
        flipped.add(chosen)
        left_count, right_count = counts()

    def sort_key(actor_id: str) -> tuple:
        root = cluster_of[actor_id]
        stats = cluster_stats[root]
        return (parent_rank.get(actor_id, 0), -stats["connections"], round(stats["avg_module"], 3), actor_id)

    left_ids = sorted((a for a in actor_ids if side_of_cluster[cluster_of[a]] == "left"), key=sort_key)
    right_ids = sorted((a for a in actor_ids if side_of_cluster[cluster_of[a]] == "right"), key=sort_key)
    return left_ids, right_ids


def _association_lines(
    model: UseCaseModel,
    actor_aliases: dict[str, str],
    use_case_aliases: dict[str, str],
    use_case_order: dict[str, int],
) -> list[str]:
    seen: set[tuple[str, str]] = set()
    entries: list[tuple[int, str, str]] = []

    def add(actor_id: str, use_case_id: str) -> None:
        if actor_id not in actor_aliases or use_case_id not in use_case_aliases:
            return
        key = (actor_id, use_case_id)
        if key in seen:
            return
        seen.add(key)
        entries.append((use_case_order.get(use_case_id, 0), actor_aliases[actor_id], use_case_aliases[use_case_id]))

    for relation in model.relations:
        if relation.kind != _ASSOCIATION:
            continue
        if relation.source_id in actor_aliases and relation.target_id in use_case_aliases:
            add(relation.source_id, relation.target_id)
        elif relation.target_id in actor_aliases and relation.source_id in use_case_aliases:
            add(relation.target_id, relation.source_id)

    # Fall back to the table's declared primary/secondary actors so every use case stays
    # connected even when the relationship list omits an explicit association row.
    for item in model.use_cases:
        add(item.primary_actor_id, item.id)
        for actor_id in item.secondary_actor_ids:
            add(actor_id, item.id)

    if not entries:
        return []
    entries.sort(key=lambda entry: (entry[0], entry[1]))
    lines = [_RULE, "' ASSOCIATIONS", _RULE, ""]
    lines.extend(f"{actor_alias} -- {use_case_alias}" for _, actor_alias, use_case_alias in entries)
    lines.append("")
    return lines


def _relationship_lines(
    model: UseCaseModel,
    actor_aliases: dict[str, str],
    use_case_aliases: dict[str, str],
    use_case_order: dict[str, int],
) -> list[str]:
    include_entries: list[tuple[int, str]] = []
    extend_entries: list[tuple[int, str]] = []
    generalization_entries: list[tuple[str, str]] = []

    for relation in model.relations:
        source = actor_aliases.get(relation.source_id) or use_case_aliases.get(relation.source_id)
        target = actor_aliases.get(relation.target_id) or use_case_aliases.get(relation.target_id)
        if source is None or target is None:
            continue

        if relation.kind == _INCLUDE:
            if relation.source_id not in use_case_aliases or relation.target_id not in use_case_aliases:
                continue
            order = use_case_order.get(relation.source_id, 0)
            include_entries.append((order, f"{source} ..> {target} : <<include>>"))
        elif relation.kind == _EXTEND:
            if relation.source_id not in use_case_aliases or relation.target_id not in use_case_aliases:
                continue
            order = use_case_order.get(relation.source_id, 0)
            condition = f"\\n[{_quote(_comment(relation.condition))}]" if relation.condition else ""
            extend_entries.append((order, f"{source} ..> {target} : <<extend>>{condition}"))
            if relation.condition:
                extend_entries.append((order, f"' extend condition ({relation.id}): {_comment(relation.condition)}"))
        elif relation.kind == _GENERALIZATION:
            generalization_entries.append((relation.source_id, f"{source} -|> {target}"))

    include_entries.sort(key=lambda entry: entry[0])
    extend_entries.sort(key=lambda entry: entry[0])
    generalization_entries.sort(key=lambda entry: entry[0])

    body = (
        [text for _, text in include_entries]
        + [text for _, text in extend_entries]
        + [text for _, text in generalization_entries]
    )
    if not body:
        return []
    lines = [_RULE, "' RELATIONSHIPS", _RULE, ""]
    lines.extend(body)
    lines.append("")
    return lines


def _layout_hint_lines(left_ids: list[str], right_ids: list[str], actor_aliases: dict[str, str]) -> list[str]:
    hints: list[str] = []
    for ids in (left_ids, right_ids):
        for a, b in zip(ids, ids[1:], strict=False):
            hints.append(f"{actor_aliases[a]} -[hidden]down- {actor_aliases[b]}")
    if not hints:
        return []
    lines = [_RULE, "' LAYOUT HINTS", _RULE, ""]
    lines.extend(hints)
    return lines


def _aliases(prefix: str, ids: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for value in ids:
        item_id = str(value)
        base = f"{prefix}_{re.sub(r'[^A-Za-z0-9]+', '_', item_id).strip('_')}" or prefix
        alias = base
        suffix = 2
        while alias in used:
            alias = f"{base}_{suffix}"
            suffix += 1
        used.add(alias)
        result[item_id] = alias
    return result


def _quote(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")


def _comment(value: str) -> str:
    return " ".join(str(value).split()).replace("\n", " ").replace("\r", " ")
