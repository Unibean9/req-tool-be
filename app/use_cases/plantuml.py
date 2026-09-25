"""Deterministic PlantUML renderer for the complete canonical use-case model."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from app.use_cases.models import UseCaseActor, UseCaseEntry, UseCaseModel

_RULE = "' " + "=" * 25


def render_plantuml(model: UseCaseModel) -> str:
    actor_aliases = _aliases("ACT", (item.id for item in model.actors))
    use_case_aliases = _aliases("UC", (item.id for item in model.use_cases))
    actors = {item.id: item for item in model.actors}
    modules = {item.id: item for item in model.modules}
    module_order = sorted(modules.values(), key=lambda item: (item.name.casefold(), item.id))
    module_index = {item.id: index for index, item in enumerate(module_order)}
    ordered_use_cases = sorted(model.use_cases, key=lambda item: (module_index.get(item.module_id, 999), item.id))
    use_case_order = {item.id: index for index, item in enumerate(ordered_use_cases)}
    left_ids, right_ids = _balance_actor_sides(model, module_index)
    for actor_id in left_ids:
        actors[actor_id].side = "left"
    for actor_id in right_ids:
        actors[actor_id].side = "right"

    nodesep, ranksep = _spacing_for(len(ordered_use_cases))
    lines = [
        "@startuml",
        "' Generated from the canonical BRD/PRD-backed Use Case Table. Edit this source directly when needed.",
        "",
        f'title "{_quote(model.system_name)} — Complete Use Case Model"',
        "",
        "left to right direction",
        "skinparam shadowing false",
        "skinparam linetype ortho",
        f"skinparam nodesep {nodesep}",
        f"skinparam ranksep {ranksep}",
        "skinparam packageStyle rectangle",
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
    lines += _actor_block("LEFT ACTORS", left_ids, actors, actor_aliases)
    lines += _system_block(model, module_order, ordered_use_cases, use_case_aliases)
    lines += _actor_block("RIGHT ACTORS", right_ids, actors, actor_aliases)
    lines += _side_anchor_lines(left_ids, right_ids, actor_aliases)
    lines += _association_lines(model, actor_aliases, use_case_aliases, use_case_order)
    lines += _relationship_lines(model, actor_aliases, use_case_aliases, use_case_order)
    lines += _actor_stack_hints(left_ids, right_ids, actor_aliases)
    lines += ["", "@enduml", ""]
    return "\n".join(lines)


def _actor_block(
    heading: str, actor_ids: list[str], actors: dict[str, UseCaseActor], aliases: dict[str, str]
) -> list[str]:
    if not actor_ids:
        return []
    lines = [_RULE, f"' {heading}", _RULE, ""]
    for actor_id in actor_ids:
        actor = actors[actor_id]
        stereotype = (
            " <<external system>>"
            if actor.kind == "external_system"
            else " <<scheduler>>"
            if actor.kind == "scheduler"
            else ""
        )
        lines.append(f'actor "{_quote(actor.name)}" as {aliases[actor_id]}{stereotype}')
    lines.append("")
    return lines


def _system_block(
    model: UseCaseModel, module_order: list, ordered_use_cases: list[UseCaseEntry], aliases: dict[str, str]
) -> list[str]:
    by_module: dict[str, list[UseCaseEntry]] = defaultdict(list)
    for item in ordered_use_cases:
        by_module[item.module_id].append(item)
    lines = [_RULE, "' SYSTEM", _RULE, "", f'rectangle "{_quote(model.system_name)}" as SYSTEM {{']
    for module in module_order:
        items = by_module.get(module.id, [])
        if not items:
            continue
        lines.append(f'  package "{_quote(module.name)}" {{')
        for item in items:
            lines.append(f'    usecase "{_quote(item.id)}\\n{_quote(item.name)}" as {aliases[item.id]}')
        lines.append("  }")
    lines.append("}")
    lines.append("")
    return lines


def _side_anchor_lines(left_ids: list[str], right_ids: list[str], aliases: dict[str, str]) -> list[str]:
    lines = [_RULE, "' SIDE ANCHORS", _RULE, ""]
    if left_ids:
        lines.extend(f"{aliases[item]} -[hidden]right- SYSTEM" for item in left_ids)
    if right_ids:
        lines.extend(f"SYSTEM -[hidden]right- {aliases[item]}" for item in right_ids)
    lines.append("")
    return lines


def _association_lines(
    model: UseCaseModel, actor_aliases: dict[str, str], uc_aliases: dict[str, str], order: dict[str, int]
) -> list[str]:
    pairs: set[tuple[str, str]] = set()
    for item in model.use_cases:
        for actor_id in [item.primary_actor_id, *item.secondary_actor_ids]:
            if actor_id in actor_aliases:
                pairs.add((actor_id, item.id))
    for relation in model.relationships:
        if relation.kind != "association":
            continue
        if relation.source_id in actor_aliases and relation.target_id in uc_aliases:
            pairs.add((relation.source_id, relation.target_id))
        elif relation.target_id in actor_aliases and relation.source_id in uc_aliases:
            pairs.add((relation.target_id, relation.source_id))
    if not pairs:
        return []
    lines = [_RULE, "' ASSOCIATIONS", _RULE, ""]
    lines.extend(
        f"{actor_aliases[a]} -- {uc_aliases[u]}"
        for a, u in sorted(pairs, key=lambda pair: (order.get(pair[1], 999), pair[0], pair[1]))
    )
    lines.append("")
    return lines


def _relationship_lines(
    model: UseCaseModel, actor_aliases: dict[str, str], uc_aliases: dict[str, str], order: dict[str, int]
) -> list[str]:
    rows: list[tuple[int, str, str]] = []
    for relation in model.relationships:
        if relation.review_state == "rejected":
            continue
        source = actor_aliases.get(relation.source_id) or uc_aliases.get(relation.source_id)
        target = actor_aliases.get(relation.target_id) or uc_aliases.get(relation.target_id)
        if not source or not target or relation.kind == "association":
            continue
        if relation.kind == "include":
            text = f"{source} ..> {target} : <<include>>"
        elif relation.kind == "extend":
            text = f"{source} ..> {target} : <<extend>>"
        elif relation.kind == "generalization":
            text = f"{source} -|> {target}"
        else:
            continue
        rows.append((order.get(relation.source_id, 999), relation.id, text))
    if not rows:
        return []
    lines = [_RULE, "' USE CASE RELATIONSHIPS", _RULE, ""]
    lines.extend(text for _, _, text in sorted(rows, key=lambda row: (row[0], row[1])))
    lines.append("")
    return lines


def _actor_stack_hints(left_ids: list[str], right_ids: list[str], aliases: dict[str, str]) -> list[str]:
    hints: list[str] = []
    for ids in (left_ids, right_ids):
        hints.extend(f"{aliases[a]} -[hidden]down- {aliases[b]}" for a, b in zip(ids, ids[1:], strict=False))
    if not hints:
        return []
    return [_RULE, "' ACTOR STACK HINTS", _RULE, "", *hints, ""]


def _balance_actor_sides(model: UseCaseModel, module_index: dict[str, int]) -> tuple[list[str], list[str]]:
    actor_ids = [item.id for item in model.actors]
    if not actor_ids:
        return [], []
    connections: dict[str, list[str]] = defaultdict(list)
    for item in model.use_cases:
        for actor_id in [item.primary_actor_id, *item.secondary_actor_ids]:
            connections[actor_id].append(item.id)
    # Keep actor inheritance trees on one side.
    parent: dict[str, str] = {actor_id: actor_id for actor_id in actor_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra if ra < rb else rb

    for relation in model.relationships:
        if relation.kind == "generalization" and relation.source_id in parent and relation.target_id in parent:
            union(relation.source_id, relation.target_id)
    clusters: dict[str, list[str]] = defaultdict(list)
    for actor_id in actor_ids:
        clusters[find(actor_id)].append(actor_id)
    count = max(len(module_index), 1)
    median = (count - 1) / 2
    stats: dict[str, tuple[float, int]] = {}
    for root, members in clusters.items():
        values = [
            module_index.get(item.module_id, 0)
            for item in model.use_cases
            for actor in [item.primary_actor_id, *item.secondary_actor_ids]
            if actor in members
        ]
        stats[root] = (sum(values) / len(values) if values else median, len(members))
    side = {root: ("left" if value[0] <= median else "right") for root, value in stats.items()}

    def totals() -> tuple[int, int]:
        return sum(stats[root][1] for root in side if side[root] == "left"), sum(
            stats[root][1] for root in side if side[root] == "right"
        )

    left, right = totals()
    flipped: set[str] = set()
    while abs(left - right) > 1:
        majority = "left" if left > right else "right"
        candidates = [root for root in side if side[root] == majority and root not in flipped]
        if not candidates:
            break
        candidates.sort(key=lambda root: (abs(stats[root][0] - median), root))
        chosen = candidates[0]
        side[chosen] = "right" if majority == "left" else "left"
        flipped.add(chosen)
        left, right = totals()

    def actor_key(actor_id: str) -> tuple:
        return (
            -len(connections.get(actor_id, [])),
            min(
                (
                    module_index.get(item.module_id, 999)
                    for item in model.use_cases
                    if actor_id in [item.primary_actor_id, *item.secondary_actor_ids]
                ),
                default=999,
            ),
            actor_id,
        )

    left_ids = sorted((actor for actor in actor_ids if side[find(actor)] == "left"), key=actor_key)
    right_ids = sorted((actor for actor in actor_ids if side[find(actor)] == "right"), key=actor_key)
    return left_ids, right_ids


def _spacing_for(count: int) -> tuple[int, int]:
    if count <= 15:
        return 70, 80
    if count <= 30:
        return 85, 95
    return 105, 115


def _aliases(prefix: str, ids: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    used: set[str] = set()
    for value in ids:
        base = f"{prefix}_{re.sub(r'[^A-Za-z0-9]+', '_', str(value)).strip('_')}" or prefix
        alias = base
        suffix = 2
        while alias in used:
            alias = f"{base}_{suffix}"
            suffix += 1
        used.add(alias)
        result[str(value)] = alias
    return result


def _quote(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\r", " ").replace("\n", " ")
