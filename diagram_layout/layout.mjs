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

  const edges = asList(input.relationships)
    .filter((relation) => {
      const type = asString(relation?.type);
      return ["include", "extend", "generalization"].includes(type) && relation?.sourceId && relation?.targetId;
    })
    .map((relation) => ({
      id: asString(relation.id, `relation-${relation.sourceId}-${relation.targetId}`),
      sources: [asString(relation.sourceId)],
      targets: [asString(relation.targetId)],
    }));

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
      "elk.layered.spacing.nodeNodeBetweenLayers": "80",
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

function actorSides(input, useCasePositions) {
  // `side` is always recomputed here, never taken from `input.actors[].side` -- the backend
  // round-trips whatever this function last returned back into the stored actor list, and an
  // earlier version of this function treated an actor's existing side as fixed and only
  // balanced actors that had none yet. Once every actor had been assigned a side once (often
  // early on, while the table was still small and every actor's weight was 0 or tied, which
  // this same tie-break sends to "left"), no actor was ever rebalanced again on a later run,
  // no matter how lopsided the real weights had become. Recomputing from scratch every time
  // keeps this a pure function of the current table (same input -> same output, via the
  // weight-then-name sort below), not of whatever an earlier, possibly much smaller table
  // happened to produce.
  const actors = asList(input.actors).map((actor, index) => ({
    id: asString(actor?.id, `actor-${index + 1}`),
    name: asString(actor?.name, "Actor"),
    kind: asString(actor?.kind, "human"),
    side: null,
  }));
  const linked = new Map(actors.map((actor) => [actor.id, []]));
  for (const item of asList(input.useCases)) {
    const actorIds = [item?.primaryActorId, ...asList(item?.secondaryActorIds)].filter(Boolean);
    for (const actorId of actorIds) {
      if (!linked.has(actorId)) linked.set(actorId, []);
      linked.get(actorId).push(item.id);
    }
  }
  let leftWeight = 0;
  let rightWeight = 0;
  const ordered = [...actors].sort(
    (left, right) => linked.get(right.id).length - linked.get(left.id).length || left.name.localeCompare(right.name),
  );
  for (const actor of ordered) {
    actor.side = leftWeight <= rightWeight ? "left" : "right";
    if (actor.side === "left") leftWeight += linked.get(actor.id).length;
    else rightWeight += linked.get(actor.id).length;
  }
  return { actors, linked };
}

function fallbackPoints(source, target) {
  return [center(source), center(target)];
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
    const position = useCasePositions.get(item.id) ?? { x: minX, y: minY, width: USE_CASE_WIDTH, height: USE_CASE_HEIGHT };
    nodes.push({
      id: item.id,
      kind: "use_case",
      name: asString(item.name, item.id),
      moduleId: asString(item.moduleId),
      moduleName: asString(modules.get(item.moduleId), "General"),
      priority: asString(item.priority),
      x: position.x + dx,
      y: position.y + dy,
      width: position.width,
      height: position.height,
    });
  }
  const useCasesById = new Map(nodes.map((node) => [node.id, node]));
  const { actors, linked } = actorSides(input, useCasesById);
  const desiredY = (actor) => {
    const ys = linked.get(actor.id).map((useCaseId) => useCasesById.get(useCaseId)?.y + USE_CASE_HEIGHT / 2).filter(Number.isFinite);
    return median(ys) ?? SYSTEM_Y + 120;
  };
  // Placed in desired-Y order (top to bottom), not alphabetically: an actor is pushed down only
  // as far as the one immediately above it in that same order requires, so it stays as close as
  // possible to the use cases it actually connects to. Sorting by name instead meant an actor
  // whose connections sit near the bottom could land alphabetically first and get placed at the
  // top on its own, while the actors that belonged near the top got cascade-pushed down below
  // it -- stranding it far from its own edges with everyone else bunched together elsewhere.
  const placeColumn = (side, x) => {
    const ordered = actors
      .filter((actor) => actor.side === side)
      .sort((left, right) => desiredY(left) - desiredY(right) || left.name.localeCompare(right.name));
    let previousY = null;
    for (const actor of ordered) {
      const desired = desiredY(actor);
      const y = Math.max(
        SYSTEM_Y + 34,
        previousY == null ? desired - ACTOR_HEIGHT / 2 : Math.max(desired - ACTOR_HEIGHT / 2, previousY + ACTOR_HEIGHT + 24),
      );
      previousY = y;
      nodes.push({ id: actor.id, kind: "actor", name: actor.name, actorKind: actor.kind, side, x, y, width: ACTOR_WIDTH, height: ACTOR_HEIGHT });
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
    const elkEdge = elkEdges.get(relation.id);
    const points = sectionPoints(elkEdge, dx, dy);
    const actualPoints = points.length >= 2 ? points : fallbackPoints(source, target);
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
    diagnostics: { warnings: [], overlapCount: 0 },
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
