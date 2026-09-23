"""Convert semantic use-case relations into renderer-neutral UML notation."""

from __future__ import annotations

from app.use_cases.models import (
    DiagramRenderEdge,
    DiagramRenderNode,
    DiagramRenderPlan,
    UseCaseModel,
)


def build_diagram_render_plan(
    model: UseCaseModel,
    diagram_id: str,
    *,
    require_confirmed: bool = True,
) -> DiagramRenderPlan:
    """Build the exact node/edge semantics a React Flow/UML renderer should draw.

    The renderer receives no freedom to reinterpret relationship notation:

    * association = solid, undirected actor-to-use-case line;
    * include/extend = dashed, directed open arrow with a stereotype label;
    * generalization = solid, directed open triangle from child to parent;
    * actors stay outside the rectangular system boundary and use cases stay inside ellipses.
    """

    diagram = next((item for item in model.diagrams if item.id == diagram_id), None)
    if diagram is None:
        raise ValueError(f"Unknown diagram {diagram_id}")
    use_cases = {item.id: item for item in model.use_cases}
    actors = {item.id: item for item in model.actors}
    relations = {item.id: item for item in model.relations}
    if require_confirmed:
        unconfirmed = [
            item_id
            for item_id in diagram.use_case_ids
            if use_cases.get(item_id) and use_cases[item_id].status != "confirmed"
        ]
        if unconfirmed:
            raise ValueError(f"Diagram {diagram_id} contains unconfirmed use cases: {', '.join(unconfirmed)}")

    boundary_id = f"BOUNDARY-{diagram.id}"
    nodes = [
        DiagramRenderNode(
            id=boundary_id,
            kind="system_boundary",
            label=diagram.system_boundary,
            shape="rectangle",
            side="inside",
        )
    ]
    primary_actor_ids = {
        item.primary_actor_id
        for item in model.use_cases
        if item.id in diagram.use_case_ids
    }
    secondary_actor_ids = {
        actor_id
        for item in model.use_cases
        if item.id in diagram.use_case_ids
        for actor_id in item.secondary_actor_ids
    }
    for actor_id in diagram.actor_ids:
        actor = actors.get(actor_id)
        if actor is None:
            raise ValueError(f"Diagram {diagram.id} references unknown actor {actor_id}")
        side = "left" if actor_id in primary_actor_ids or actor_id not in secondary_actor_ids else "right"
        nodes.append(
            DiagramRenderNode(
                id=actor.id,
                kind="actor",
                label=actor.name,
                shape="actor",
                side=side,
            )
        )
    for use_case_id in diagram.use_case_ids:
        item = use_cases.get(use_case_id)
        if item is None:
            raise ValueError(f"Diagram {diagram.id} references unknown use case {use_case_id}")
        nodes.append(
            DiagramRenderNode(
                id=item.id,
                kind="use_case",
                label=f"{item.id} {item.name}",
                shape="ellipse",
                side="inside",
            )
        )

    edges: list[DiagramRenderEdge] = []
    diagram_node_ids = {node.id for node in nodes}
    for relation_id in diagram.relation_ids:
        relation = relations.get(relation_id)
        if relation is None:
            raise ValueError(f"Diagram {diagram.id} references unknown relation {relation_id}")
        if relation.source_id not in diagram_node_ids or relation.target_id not in diagram_node_ids:
            raise ValueError(f"Relation {relation.id} crosses the boundary of diagram {diagram.id}")
        source_is_actor = relation.source_id in actors
        target_is_actor = relation.target_id in actors
        source_is_use_case = relation.source_id in use_cases
        target_is_use_case = relation.target_id in use_cases
        if relation.kind == "association" and not (
            (source_is_actor and target_is_use_case) or (source_is_use_case and target_is_actor)
        ):
            raise ValueError(f"Association {relation.id} must connect one actor and one use case")
        if relation.kind in {"include", "extend"} and not (source_is_use_case and target_is_use_case):
            raise ValueError(f"{relation.kind} {relation.id} must connect two use cases")
        if relation.kind == "generalization" and not (
            (source_is_actor and target_is_actor) or (source_is_use_case and target_is_use_case)
        ):
            raise ValueError(f"Generalization {relation.id} must connect two actors or two use cases")
        if relation.kind == "extend" and not relation.condition:
            raise ValueError(f"Extend relation {relation.id} needs a condition or extension point")
        if relation.kind == "association":
            line_style = "solid"
            directed = False
            marker = "none"
            label = None
        elif relation.kind == "generalization":
            line_style = "solid"
            directed = True
            marker = "open_triangle"
            label = None
        else:
            line_style = "dashed"
            directed = True
            marker = "open_arrow"
            label = f"«{relation.kind}»"
        edges.append(
            DiagramRenderEdge(
                id=relation.id,
                source_id=relation.source_id,
                target_id=relation.target_id,
                kind=relation.kind,
                line_style=line_style,
                directed=directed,
                marker=marker,
                label=label,
                condition=relation.condition,
            )
        )
    return DiagramRenderPlan(
        diagram_id=diagram.id,
        level=diagram.level,
        system_boundary=diagram.system_boundary,
        nodes=nodes,
        edges=edges,
    )
