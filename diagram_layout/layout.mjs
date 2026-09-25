import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const ELK = require("elkjs/lib/elk.bundled.js");

const elk = new ELK();
const USE_CASE_WIDTH = 224;
const USE_CASE_HEIGHT = 72;
const ACTOR_WIDTH = 160;
const ACTOR_HEIGHT = 110;
const SYSTEM_X = 280;
const SYSTEM_Y = 24;
const SYSTEM_PADDING_X = 48;
const SYSTEM_PADDING_Y = 92;

function asString(value, fallback = "") {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function asList(value) {
  return Array.isArray(value) ? value : [];
}

function moduleGroups(input) {
  const modules = asList(input.modules).map((module, index) => ({
    id: asString(module?.id, `module-${index + 1}`),
    name: asString(module?.name, "General"),
  }));
  const seen = new Set(modules.map((module) => module.id));
  for (const item of asList(input.useCases)) {
    const moduleId = asString(item?.moduleId, "module-general");
    if (seen.has(moduleId)) continue;
    seen.add(moduleId);
    modules.push({ id: moduleId, name: "General" });
  }
  return modules.length ? modules : [{ id: "module-general", name: "General" }];
}

function buildElkGraph(input) {
  const groups = moduleGroups(input);
  const useCases = asList(input.useCases);
  const childrenByModule = new Map(groups.map((module) => [module.id, []]));
  for (const item of useCases) {
    const moduleId = asString(item?.moduleId, "module-general");
    if (!childrenByModule.has(moduleId)) childrenByModule.set(moduleId, []);
    childrenByModule.get(moduleId).push({
      id: asString(item?.id),
      width: USE_CASE_WIDTH,
      height: USE_CASE_HEIGHT,
      layoutOptions: {
        "elk.portConstraints": "FIXED_SIDE",
      },
    });
  }

  const children = groups.map((module) => ({
    id: `module-${module.id}`,
    layoutOptions: {
      "elk.padding": "[top=52,left=36,bottom=36,right=36]",
      "elk.portConstraints": "FIXED_SIDE",
    },
    children: childrenByModule.get(module.id) ?? [],
  }));

  // include/extend get a rendered "«include»"/"«extend»" label at the edge's midpoint (see the
  // `label` field built below); generalization has none. Declaring an ELK `labels` box for the
  // first two -- not just a bare source/target edge -- is what makes the layered algorithm treat
  // the label's width as real space it must route around, instead of routing two use cases as
  // close together as the plain node spacing allows and leaving the label to overlap whatever
  // ends up in that gap.
  const EDGE_LABEL_SIZE = { width: 74, height: 16 };
  const edges = asList(input.relationships)
    .filter((relation) => {
      const type = asString(relation?.type);
      return ["include", "extend", "generalization"].includes(type) && relation?.sourceId && relation?.targetId;
    })
    .map((relation) => {
      const type = asString(relation?.type);
      const hasLabel = type === "include" || type === "extend";
      return {
        id: asString(relation.id, `relation-${relation.sourceId}-${relation.targetId}`),
        sources: [asString(relation.sourceId)],
        targets: [asString(relation.targetId)],
        labels: hasLabel ? [{ text: type === "include" ? "«include»" : "«extend»", ...EDGE_LABEL_SIZE }] : [],
      };
    });

  return {
    id: "root",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "RIGHT",
      "elk.edgeRouting": "POLYLINE",
      "elk.hierarchyHandling": "INCLUDE_CHILDREN",
      "elk.spacing.nodeNode": "50",
      "elk.spacing.edgeNode": "30",
      "elk.spacing.edgeEdge": "20",
      "elk.spacing.edgeLabel": "12",
      // A labeled edge between two use cases in adjacent layers needs enough of a gap for the
      // label text to sit without touching either ellipse; the plain node-to-node default (80)
      // was sized for unlabeled spacing only.
      "elk.layered.spacing.nodeNodeBetweenLayers": "80",
      "elk.layered.spacing.edgeNodeBetweenLayers": "40",
      "elk.layered.edgeLabels.sideSelection": "ALWAYS_UP",
      "elk.layered.crossingMinimization.strategy": "LAYER_SWEEP",
      "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
      "elk.layered.nodePlacement.favorStraightEdges": "true",
      "elk.padding": "[top=24,left=24,bottom=24,right=24]",
    },
    children,
    edges,
  };
}

function flattenUseCases(node, offsetX = 0, offsetY = 0, result = new Map()) {
  const x = offsetX + Number(node?.x ?? 0);
  const y = offsetY + Number(node?.y ?? 0);
  if (typeof node?.id === "string" && !node.id.startsWith("module-") && node.id !== "root") {
    result.set(node.id, {
      x,
      y,
      width: USE_CASE_WIDTH,
      height: USE_CASE_HEIGHT,
    });
  }
  for (const child of asList(node?.children)) flattenUseCases(child, x, y, result);
  return result;
}

function shiftPoint(point, dx, dy) {
  return { x: Number(point?.x ?? 0) + dx, y: Number(point?.y ?? 0) + dy };
}

function collectElkEdges(node, offsetX = 0, offsetY = 0, result = new Map(), containerOffsets = new Map()) {
  const x = offsetX + Number(node?.x ?? 0);
  const y = offsetY + Number(node?.y ?? 0);
  if (node?.id) containerOffsets.set(node.id, { x, y });
  for (const child of asList(node?.children)) collectElkEdges(child, x, y, result, containerOffsets);
  for (const edge of asList(node?.edges)) {
    const container = containerOffsets.get(edge.container) ?? { x, y };
    result.set(edge.id, { edge, offsetX: container.x, offsetY: container.y });
  }
  return result;
}

function sectionPoints(edgeInfo, dx, dy) {
  const edge = edgeInfo?.edge;
  const section = asList(edge?.sections)[0];
  if (!section) return [];
  const points = [];
  const edgeOffsetX = Number(edgeInfo?.offsetX ?? 0);
  const edgeOffsetY = Number(edgeInfo?.offsetY ?? 0);
  if (section.startPoint) points.push(shiftPoint(section.startPoint, dx + edgeOffsetX, dy + edgeOffsetY));
  for (const point of asList(section.bendPoints)) points.push(shiftPoint(point, dx + edgeOffsetX, dy + edgeOffsetY));
  if (section.endPoint) points.push(shiftPoint(section.endPoint, dx + edgeOffsetX, dy + edgeOffsetY));
  return points;
}

function center(node) {
  return { x: node.x + node.width / 2, y: node.y + node.height / 2 };
}

function median(values) {
  if (!values.length) return null;
  const ordered = [...values].sort((left, right) => left - right);
  return ordered[Math.floor(ordered.length / 2)];
}

// A node the user dragged and saved carries its persisted position back in on every later call
// (the backend round-trips `diagramLayout.nodes[].manual`/x/y into `manualPosition` for any
// actor/use case id that still exists) -- distinct from the old, since-removed "side" stickiness
// bug: that one silently froze a position no one asked to freeze. This is an explicit, per-node
// override the user asked for, so it takes precedence over anything this file would otherwise
// compute for that one node, while every node without one is still recomputed fresh from the
// current table, exactly as before.
function manualPositionOf(item) {
  const manual = item?.manualPosition;
  if (!manual || typeof manual !== "object") return null;
  const x = Number(manual.x);
  const y = Number(manual.y);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y };
}

function actorSides(input, boundary) {
  const actors = asList(input.actors).map((actor, index) => {
    const manual = manualPositionOf(actor);
    return {
      id: asString(actor?.id, `actor-${index + 1}`),
      name: asString(actor?.name, "Actor"),
      kind: asString(actor?.kind, "human"),
      manual,
      // A manually placed actor's side is read off which half of the canvas it was actually
      // dropped on, not decided by the balance algorithm below -- its weight is then counted
      // toward that side up front so the remaining (auto) actors balance against where this one
      // really is, not wherever the algorithm would have put it if it were free to choose.
      side: manual ? (manual.x < boundary.x + boundary.width / 2 ? "left" : "right") : null,
    };
  });
  const linked = new Map(actors.map((actor) => [actor.id, []]));
  for (const item of asList(input.useCases)) {
    const actorIds = [item?.primaryActorId, ...asList(item?.secondaryActorIds)].filter(Boolean);
    for (const actorId of actorIds) {
      if (!linked.has(actorId)) linked.set(actorId, []);
      linked.get(actorId).push(item.id);
    }
  }
  let leftWeight = actors.filter((actor) => actor.side === "left").reduce((sum, actor) => sum + linked.get(actor.id).length, 0);
  let rightWeight = actors.filter((actor) => actor.side === "right").reduce((sum, actor) => sum + linked.get(actor.id).length, 0);
  const unresolved = actors
    .filter((actor) => !actor.manual)
    .sort((left, right) => linked.get(right.id).length - linked.get(left.id).length || left.name.localeCompare(right.name));
  for (const actor of unresolved) {
    actor.side = leftWeight <= rightWeight ? "left" : "right";
    if (actor.side === "left") leftWeight += linked.get(actor.id).length;
    else rightWeight += linked.get(actor.id).length;
  }
  return { actors, linked };
}

function boundaryPoint(node, towardX) {
  // Exits from whichever side (left or right) faces the other endpoint, at vertical mid-height --
  // the same rule the association edges already use for their actor/use-case connection point.
  const exitLeft = towardX < node.x + node.width / 2;
  return { x: exitLeft ? node.x : node.x + node.width, y: node.y + node.height / 2 };
}

function fallbackPoints(source, target) {
  // Used whenever a relationship/generalization edge can no longer trust ELK's own routing (see
  // the manual-position override above, and the rare case where sectionPoints comes back empty).
  // A straight line has to still stop at each ellipse's boundary, not cut through its center and
  // the label text inside it -- plain center() would draw exactly that.
  const sourceCenter = center(source);
  const targetCenter = center(target);
  return [boundaryPoint(source, targetCenter.x), boundaryPoint(target, sourceCenter.x)];
}

function countOverlaps(placedNodes) {
  let count = 0;
  for (let i = 0; i < placedNodes.length; i += 1) {
    for (let j = i + 1; j < placedNodes.length; j += 1) {
      const a = placedNodes[i];
      const b = placedNodes[j];
      const overlapsX = a.x < b.x + b.width && b.x < a.x + a.width;
      const overlapsY = a.y < b.y + b.height && b.y < a.y + a.height;
      if (overlapsX && overlapsY) count += 1;
    }
  }
  return count;
}

function buildLayout(input, elkGraph, elkResult) {
  const useCasePositions = flattenUseCases(elkResult);
  const positionValues = [...useCasePositions.values()];
  const minX = positionValues.length ? Math.min(...positionValues.map((item) => item.x)) : 0;
  const minY = positionValues.length ? Math.min(...positionValues.map((item) => item.y)) : 0;
  const maxX = positionValues.length ? Math.max(...positionValues.map((item) => item.x + item.width)) : USE_CASE_WIDTH;
  const maxY = positionValues.length ? Math.max(...positionValues.map((item) => item.y + item.height)) : USE_CASE_HEIGHT;
  const dx = SYSTEM_X + SYSTEM_PADDING_X - minX;
  const dy = SYSTEM_Y + SYSTEM_PADDING_Y - minY;
  const boundary = {
    x: SYSTEM_X,
    y: SYSTEM_Y,
    width: Math.max(800, maxX - minX + SYSTEM_PADDING_X * 2),
    height: Math.max(470, maxY - minY + SYSTEM_PADDING_Y + 28),
  };

  const modules = new Map(asList(input.modules).map((module) => [module.id, module.name]));
  const nodes = [];
  for (const item of asList(input.useCases)) {
    const manual = manualPositionOf(item);
    const position = useCasePositions.get(item.id) ?? { x: minX, y: minY, width: USE_CASE_WIDTH, height: USE_CASE_HEIGHT };
    nodes.push({
      id: item.id,
      kind: "use_case",
      name: asString(item.name, item.id),
      moduleId: asString(item.moduleId),
      moduleName: asString(modules.get(item.moduleId), "General"),
      priority: asString(item.priority),
      // The ELK pass above still computes a position for a manually placed use case like any
      // other -- it has no notion of "manual" -- so the rest of the layout (other use cases in
      // its module, the include/extend edges among them) is arranged as if this one were free to
      // move too. Only here, at the very end, is that computed position discarded in favor of
      // the saved one. That trade-off (an auto node can end up visually overlapping a pinned one
      // instead of routing around its real position) is deliberate: a true node-locking layout
      // pass would need ELK's interactive/constraint mode, which is far harder to keep
      // predictable and verify than a plain, deterministic full recompute plus a final override.
      x: manual ? manual.x : position.x + dx,
      y: manual ? manual.y : position.y + dy,
      width: position.width,
      height: position.height,
      manual: Boolean(manual),
    });
  }
  const useCasesById = new Map(nodes.map((node) => [node.id, node]));
  const { actors, linked } = actorSides(input, boundary);
  // An actor with no linked use case at all (one that was only ever a secondary/supporting
  // actor back when that existed, or simply never got assigned anything) has nothing to draw an
  // association line to and nothing to say about its position on either side -- drawing it
  // anyway is a dangling node with no information in it, and app/use_cases/rules.py already surfaces
  // this in the table as an ACTOR_WITHOUT_USE_CASE validation warning, so it is not silently
  // lost, just not given diagram real estate for nothing.
  const connectedActors = actors.filter((actor) => linked.get(actor.id).length > 0);
  const desiredY = (actor) => {
    const ys = linked.get(actor.id).map((useCaseId) => useCasesById.get(useCaseId)?.y + USE_CASE_HEIGHT / 2).filter(Number.isFinite);
    return median(ys) ?? SYSTEM_Y + 120;
  };
  // Placed in desired-Y order (top to bottom), not alphabetically: an actor is pushed down only
  // as far as the one immediately above it in that same order requires, so it stays as close as
  // possible to the use cases it actually connects to. Sorting by name instead meant an actor
  // whose connections sit near the bottom could land alphabetically first and get placed at the
  // top on its own, while the actors that belonged near the top got cascade-pushed down below
  // it -- stranding it far from its own edges with everyone else bunched together elsewhere. A
  // manual actor sorts by its own saved position (still a real Y, just not linkage-derived) and
  // is placed there exactly; only non-manual actors get pushed to clear the previous actor.
  const placeColumn = (side, x) => {
    const effectiveDesiredY = (actor) => (actor.manual ? actor.manual.y : desiredY(actor));
    const ordered = connectedActors
      .filter((actor) => actor.side === side)
      .sort((left, right) => effectiveDesiredY(left) - effectiveDesiredY(right) || left.name.localeCompare(right.name));
    let previousY = null;
    for (const actor of ordered) {
      let y;
      if (actor.manual) {
        y = actor.manual.y;
      } else {
        const desired = desiredY(actor);
        y = Math.max(
          SYSTEM_Y + 34,
          previousY == null ? desired - ACTOR_HEIGHT / 2 : Math.max(desired - ACTOR_HEIGHT / 2, previousY + ACTOR_HEIGHT + 24),
        );
      }
      previousY = y;
      nodes.push({
        id: actor.id,
        kind: "actor",
        name: actor.name,
        actorKind: actor.kind,
        side,
        x: actor.manual ? actor.manual.x : x,
        y,
        width: ACTOR_WIDTH,
        height: ACTOR_HEIGHT,
        manual: Boolean(actor.manual),
      });
    }
  };
  placeColumn("left", 0);
  placeColumn("right", SYSTEM_X + boundary.width + 160);

  const elkEdges = collectElkEdges(elkResult);
  const edges = [];
  const associationKeys = new Set();
  for (const item of asList(input.useCases)) {
    const actorIds = [item?.primaryActorId, ...asList(item?.secondaryActorIds)].filter(Boolean);
    for (const actorId of actorIds) {
      const actor = nodes.find((node) => node.id === actorId && node.kind === "actor");
      const useCase = useCasesById.get(item.id);
      if (!actor || !useCase) continue;
      const key = `${actor.id}:${useCase.id}`;
      if (associationKeys.has(key)) continue;
      associationKeys.add(key);
      const from = { x: actor.side === "left" ? actor.x + actor.width : actor.x, y: actor.y + actor.height / 2 };
      const to = { x: actor.side === "left" ? useCase.x : useCase.x + useCase.width, y: useCase.y + useCase.height / 2 };
      edges.push({ id: `association-${actor.id}-${useCase.id}`, source: actor.id, target: useCase.id, kind: "association", lineStyle: "solid", directed: false, sourceHandle: "actor-source", targetHandle: actor.side === "left" ? "target-left-1" : "target-right-1", points: [from, to] });
    }
  }
  for (const relation of asList(input.relationships)) {
    const source = useCasesById.get(relation.sourceId);
    const target = useCasesById.get(relation.targetId);
    if (!source || !target) continue;
    // ELK's bend-point routing for this edge was computed before either endpoint's manual
    // override (if any) was applied, so it can no longer be trusted to still avoid other nodes
    // once one end has moved -- fall back to a direct line between the two use cases instead,
    // the same as association edges already use.
    let actualPoints;
    if (source.manual || target.manual) {
      actualPoints = fallbackPoints(source, target);
    } else {
      const elkEdge = elkEdges.get(relation.id);
      const points = sectionPoints(elkEdge, dx, dy);
      actualPoints = points.length >= 2 ? points : fallbackPoints(source, target);
    }
    const sourcePoint = actualPoints[0];
    const targetPoint = actualPoints[actualPoints.length - 1];
    const toRight = targetPoint.x >= sourcePoint.x;
    const type = asString(relation.type);
    edges.push({
      id: asString(relation.id, `relation-${relation.sourceId}-${relation.targetId}`),
      source: source.id,
      target: target.id,
      kind: type,
      label: type === "include" ? "«include»" : type === "extend" ? "«extend»" : null,
      lineStyle: type === "generalization" ? "solid" : "dashed",
      directed: true,
      sourceHandle: `${toRight ? "source-right" : "source-left"}-1`,
      targetHandle: `${toRight ? "target-left" : "target-right"}-1`,
      points: actualPoints,
    });
  }
  return {
    engine: "elk",
    version: "0.11",
    system: { id: "SYSTEM", name: asString(input.systemName, "Requirements System"), ...boundary },
    nodes: [
      { id: "SYSTEM", kind: "system_boundary", name: asString(input.systemName, "Requirements System"), ...boundary },
      ...nodes,
    ],
    edges,
    // A manual override can legitimately overlap an auto-placed node (see the note above on
    // why this file does not attempt to route around a pinned node's real position), so this is
    // no longer always 0 -- it is a real count now that that trade-off exists.
    diagnostics: { warnings: [], overlapCount: countOverlaps(nodes) },
  };
}

async function main(input) {
  const graph = buildElkGraph(input ?? {});
  const result = await elk.layout(graph);
  return buildLayout(input ?? {}, graph, result);
}

let source = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { source += chunk; });
process.stdin.on("end", async () => {
  try {
    const input = JSON.parse(source || "{}");
    const output = await main(input);
    process.stdout.write(JSON.stringify(output));
  } catch (error) {
    process.stderr.write(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  }
});
