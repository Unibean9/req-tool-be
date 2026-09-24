"""Legacy renderer-neutral adapter.

New API output is the single editable PlantUML document. This helper remains import-compatible for
older callers but derives one complete system plan from the canonical model; it never creates
module-specific React Flow diagrams.
"""

from __future__ import annotations

from app.use_cases.models import DiagramRenderEdge, DiagramRenderNode, DiagramRenderPlan, UseCaseModel


def build_diagram_render_plan(
    model: UseCaseModel, diagram_id: str = "SYSTEM", *, require_confirmed: bool = False
) -> DiagramRenderPlan:
    _ = require_confirmed
    nodes = [
        DiagramRenderNode(
            id="BOUNDARY-SYSTEM", kind="system_boundary", label=model.system_name, shape="rectangle", side="inside"
        )
    ]
    nodes.extend(
        DiagramRenderNode(id=actor.id, kind="actor", label=actor.name, shape="actor", side=actor.side or "left")
        for actor in model.actors
    )
    nodes.extend(
        DiagramRenderNode(id=item.id, kind="use_case", label=f"{item.id} {item.name}", shape="ellipse", side="inside")
        for item in model.use_cases
    )
    edges: list[DiagramRenderEdge] = []
    for item in model.use_cases:
        for actor_id in [item.primary_actor_id, *item.secondary_actor_ids]:
            edges.append(
                DiagramRenderEdge(
                    id=f"REL-ASSOC-{actor_id}-{item.id}",
                    source_id=actor_id,
                    target_id=item.id,
                    kind="association",
                    line_style="solid",
                    directed=False,
                    marker="none",
                )
            )
    for relation in model.relationships:
        if relation.kind == "association":
            continue
        if relation.kind == "generalization":
            style, directed, marker, label = "solid", True, "open_triangle", None
        else:
            style, directed, marker, label = "dashed", True, "open_arrow", f"«{relation.kind}»"
        edges.append(
            DiagramRenderEdge(
                id=relation.id,
                source_id=relation.source_id,
                target_id=relation.target_id,
                kind=relation.kind,
                line_style=style,
                directed=directed,
                marker=marker,
                label=label,
                condition=relation.condition,
            )
        )
    return DiagramRenderPlan(diagram_id=diagram_id, system_boundary=model.system_name, nodes=nodes, edges=edges)
