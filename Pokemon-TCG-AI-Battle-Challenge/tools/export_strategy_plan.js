const fs = require("fs");
const path = require("path");

const root = path.resolve(__dirname, "..");
const roadmapPath = path.join(root, "dashboard", "roadmap.html");
const html = fs.readFileSync(roadmapPath, "utf8");

function readLiteral(pattern, label) {
  const match = html.match(pattern);
  if (!match) throw new Error(`Could not read ${label} from dashboard/roadmap.html`);
  return Function(`"use strict"; return (${match[1]});`)();
}

const strategies = readLiteral(/const strategies = (\[[\s\S]*?\n    \]);/, "strategies");
const progress = readLiteral(/const strategyProgress = (\{[\s\S]*?\n    \});/, "strategy progress");

const profiles = {
  Low: { build: 1, compute: 2 },
  Med: { build: 3, compute: 6 },
  High: { build: 8, compute: 18 },
};
const multipliers = {
  policy: { build: 1.0, compute: 1.0 },
  deck: { build: 0.8, compute: 1.15 },
  search: { build: 1.2, compute: 1.35 },
  learning: { build: 1.35, compute: 1.5 },
  evaluation: { build: 0.7, compute: 1.2 },
  safety: { build: 0.8, compute: 0.7 },
};
const overrides = {
  "Live competition replay mining": { build: 0.5, compute: 1, hardware: "CPU + network" },
  "Leaderboard score trajectory logger": { build: 0.25, compute: 0.25, hardware: "Network" },
  "Two track specialist portfolio": { build: 0.5, compute: 4, hardware: "CPU" },
  "Clean room policy model scale sweep": { build: 8, compute: 36, hardware: "GPU" },
  "Frozen champion curriculum": { build: 8, compute: 48, hardware: "GPU" },
  "Root sampling flat Monte Carlo": { build: 5, compute: 24, hardware: "CPU" },
  "Information set MCTS": { build: 12, compute: 28, hardware: "CPU" },
  "Shadow package canary": { build: 1, compute: 2, hardware: "CPU" },
};

function statusFor(item) {
  const current = progress[item.title];
  if (current?.state === "complete") return "completed";
  if (current?.state === "active" && /paused/i.test(current.label || "")) return "paused";
  if (current?.state === "active") return "running";
  return "pending";
}

function estimateFor(item, status) {
  const base = profiles[item.effort] || profiles.Med;
  const factor = multipliers[item.category] || { build: 1, compute: 1 };
  const custom = overrides[item.title];
  const build = custom?.build ?? Math.round(base.build * factor.build);
  const compute = custom?.compute ?? Math.round(base.compute * factor.compute);
  const hardware = custom?.hardware
    ?? (item.category === "learning" ? "GPU" : item.category === "evaluation" ? "CPU + network" : "CPU");
  const remainingFactor = status === "completed" ? 0 : status === "running" ? 0.5 : 1;
  const remainingBuild = Math.round(build * remainingFactor * 10) / 10;
  const remainingCompute = Math.round(compute * remainingFactor * 10) / 10;
  const remaining = remainingBuild + remainingCompute;
  const low = remaining === 0 ? 0 : Math.max(1, Math.round(remaining * 0.7));
  const high = remaining === 0 ? 0 : Math.max(low, Math.round(remaining * 1.4));
  return {
    build_hours: build,
    validation_hours: compute,
    nominal_hours: build + compute,
    remaining_build_hours: remainingBuild,
    remaining_validation_hours: remainingCompute,
    remaining_hours: remaining,
    remaining_range_hours: [low, high],
    hardware,
    parallel_notebook: hardware.includes("CPU") || hardware.includes("GPU"),
  };
}

const entries = strategies.map((item, index) => {
  const status = statusFor(item);
  return {
    rank: index + 1,
    title: item.title,
    priority_band: index < 12 ? "now" : index < 30 ? "next" : index < 52 ? "research" : "frontier",
    area: item.category,
    effort: item.effort,
    status,
    result: progress[item.title]?.result || "",
    estimate: estimateFor(item, status),
  };
});

const counts = {
  total: entries.length,
  completed: entries.filter(item => item.status === "completed").length,
  running: entries.filter(item => item.status === "running").length,
  paused: entries.filter(item => item.status === "paused").length,
  pending: entries.filter(item => item.status === "pending").length,
};
const hours = entries.reduce(
  (acc, item) => {
    acc.build += item.estimate.remaining_build_hours;
    if (item.estimate.hardware.includes("GPU")) acc.gpu += item.estimate.remaining_validation_hours;
    else acc.cpu += item.estimate.remaining_validation_hours;
    acc.total += item.estimate.remaining_hours;
    return acc;
  },
  { build: 0, cpu: 0, gpu: 0, total: 0 },
);
for (const key of Object.keys(hours)) hours[key] = Math.round(hours[key] * 10) / 10;

const payload = {
  generated_at: new Date().toISOString(),
  estimate_basis: "Implementation plus first screen and one fresh confirmation for strategies that survive.",
  estimate_warning: "Ranges are planning estimates. Dependencies and rejected screens will reduce actual work; full promotion testing for every survivor increases it.",
  capacities: {
    kaggle_cpu_lanes: 10,
    kaggle_gpu_lanes: 1,
    remote_cpu_lanes: 0,
    remote_gpu_lanes: 0,
    local_experiment_lanes: 0,
  },
  remote_backend: {
    configured: false,
    reason: "No remote connection configuration is present in the repository.",
  },
  counts,
  remaining_hours: hours,
  strategies: entries,
};

if (payload.counts.total !== 68) {
  throw new Error(`Expected 68 strategies, found ${payload.counts.total}`);
}

const jsonPath = path.join(root, "automation", "strategy_execution_plan.json");
const jsPath = path.join(root, "dashboard", "strategy-plan.js");
fs.writeFileSync(jsonPath, `${JSON.stringify(payload, null, 2)}\n`);
fs.writeFileSync(jsPath, `window.PTCG_STRATEGY_PLAN = ${JSON.stringify(payload)};\n`);
console.log(JSON.stringify({ jsonPath, jsPath, counts, hours }, null, 2));
