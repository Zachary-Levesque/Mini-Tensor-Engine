const state = {
  reference: null,
  benchmarks: null,
};

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || payload.command?.stderr || "Request failed");
  }
  return payload;
}

function formatNumber(value) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "n/a";
  }
  if (Math.abs(value) >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(2)} ms`;
  }
  if (Math.abs(value) >= 1_000) {
    return `${(value / 1_000).toFixed(2)} μs`;
  }
  return `${value.toFixed(2)} ns`;
}

function renderMetrics() {
  const container = document.getElementById("hero-metrics");
  const validation = state.reference?.validation;
  const modelResults = state.benchmarks?.model_results || [];
  const bestModel = modelResults.reduce((best, current) => {
    if (!best || current.avg_ns < best.avg_ns) return current;
    return best;
  }, null);

  const cards = [
    {
      label: "Reference Match",
      value: validation?.matches_reference ? "Verified" : "Needs review",
    },
    {
      label: "Max Output Diff",
      value: validation ? validation.max_abs_diff.toExponential(2) : "n/a",
    },
    {
      label: "Best Model Run",
      value: bestModel ? `${bestModel.case} · ${formatNumber(bestModel.avg_ns)}` : "No benchmark yet",
    },
  ];

  container.innerHTML = cards
    .map(
      (card) => `
        <article class="metric-card">
          <span class="metric-label">${card.label}</span>
          <span class="metric-value">${card.value}</span>
        </article>
      `
    )
    .join("");
}

function renderArchitecture() {
  const flow = state.reference?.architecture?.flow || [];
  document.getElementById("architecture-flow").innerHTML = flow
    .map((step) => `<div class="flow-chip">${step}</div>`)
    .join("");

  const validation = state.reference?.validation;
  const className = validation?.matches_reference ? "validation-good" : "validation-bad";
  document.getElementById("validation-card").innerHTML = `
    <h3 class="${className}">${validation?.matches_reference ? "C++ output matches Python reference" : "Reference mismatch detected"}</h3>
    <p class="muted">Max absolute difference: ${validation ? validation.max_abs_diff.toExponential(3) : "n/a"}</p>
    <p class="muted">The dashboard computes the same layer-by-layer path as the exported Python reference and compares it against the stored expected output.</p>
  `;
}

function renderTensors() {
  const tensors = state.reference?.tensors || {};
  const interesting = [
    ["input", "Input"],
    ["hidden_linear", "Hidden Pre-ReLU"],
    ["hidden_relu", "Hidden Post-ReLU"],
    ["logits", "Logits"],
    ["output", "Output"],
    ["expected_output", "Expected Output"],
    ["w1", "Weights 1"],
    ["w2", "Weights 2"],
  ];

  document.getElementById("tensor-grid").innerHTML = interesting
    .filter(([key]) => tensors[key])
    .map(([key, label]) => {
      const tensor = tensors[key];
      const rows = tensor.values
        .map(
          (row) =>
            `<tr>${row.map((value) => `<td>${Number(value).toFixed(4)}</td>`).join("")}</tr>`
        )
        .join("");
      return `
        <article class="tensor-card">
          <h3>${label}</h3>
          <p class="tensor-shape">shape: [${tensor.shape.join(", ")}]</p>
          <table class="tensor-table"><tbody>${rows}</tbody></table>
        </article>
      `;
    })
    .join("");
}

function renderGroupedBars(containerId, titleField, results) {
  const container = document.getElementById(containerId);
  if (!results.length) {
    container.innerHTML = `<p class="muted">No benchmark data available yet. Run the benchmark from the panel above.</p>`;
    return;
  }

  const grouped = new Map();
  for (const result of results) {
    const key = result[titleField];
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key).push(result);
  }

  const maxValue = Math.max(...results.map((item) => item.avg_ns));
  container.innerHTML = Array.from(grouped.entries())
    .map(([groupName, groupResults]) => {
      const rows = groupResults
        .map((result) => {
          const percent = maxValue === 0 ? 0 : (result.avg_ns / maxValue) * 100;
          return `
            <div class="bar-row">
              <div class="bar-label">${result.backend}${result.threads > 1 ? ` · ${result.threads}t` : ""}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${percent}%"></div></div>
              <div class="bar-value">${formatNumber(result.avg_ns)}</div>
            </div>
          `;
        })
        .join("");
      return `<article class="chart-card"><h3>${groupName}</h3><div class="chart-bars">${rows}</div></article>`;
    })
    .join("");
}

function renderBenchmarks() {
  const benchmarks = state.benchmarks;
  document.getElementById("benchmark-source").textContent = benchmarks?.source_path
    ? `Loaded from ${benchmarks.source_path}`
    : "No benchmark file loaded";

  renderGroupedBars("matmul-chart", "case", benchmarks?.matmul_results || []);
  renderGroupedBars("model-chart", "case", benchmarks?.model_results || []);
}

function setConsole(id, text) {
  document.getElementById(id).textContent = text || "";
}

async function loadState() {
  const payload = await fetchJson("/api/state");
  state.reference = payload.reference;
  state.benchmarks = payload.benchmarks;
  renderMetrics();
  renderArchitecture();
  renderTensors();
  renderBenchmarks();
}

document.getElementById("refresh-reference").addEventListener("click", async () => {
  const payload = await fetchJson("/api/refresh-reference", { method: "POST", body: "{}" });
  state.reference = payload.reference;
  renderMetrics();
  renderArchitecture();
  renderTensors();
  setConsole("inference-output", payload.command.stdout || payload.command.stderr);
});

document.getElementById("inference-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = await fetchJson("/api/run-inference", {
    method: "POST",
    body: JSON.stringify({
      backend: form.get("backend"),
      threads: Number(form.get("threads")),
    }),
  });
  setConsole("inference-output", `${payload.command.stdout}${payload.command.stderr}`);
});

document.getElementById("benchmark-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = await fetchJson("/api/run-benchmark", {
    method: "POST",
    body: JSON.stringify({
      iterations: Number(form.get("iterations")),
      warmup: Number(form.get("warmup")),
      threads: String(form.get("threads"))
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean)
        .map(Number),
    }),
  });
  state.benchmarks = payload.benchmarks;
  renderMetrics();
  renderBenchmarks();
  setConsole("benchmark-output", `${payload.command.stdout}${payload.command.stderr}`);
});

loadState().catch((error) => {
  setConsole("benchmark-output", error.message);
});
