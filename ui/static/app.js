const state = {
  reference: null,
  benchmarks: null,
};

const tensorHelp = {
  input: "This is the data that goes into the model.",
  hidden_linear: "This is the first linear layer output before ReLU is applied.",
  hidden_relu: "This is the hidden layer after all negative values are clamped to zero.",
  logits: "These are the raw final scores before Softmax normalization.",
  output: "This is the final prediction after Softmax. The values sum to 1.",
  expected_output: "This is the stored Python reference output used for correctness checking.",
  w1: "These are the first layer weights used by the model.",
  w2: "These are the second layer weights used by the model.",
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
  if (typeof value !== "number" || Number.isNaN(value)) return "n/a";
  if (Math.abs(value) >= 1_000_000) return `${(value / 1_000_000).toFixed(2)} ms`;
  if (Math.abs(value) >= 1_000) return `${(value / 1_000).toFixed(2)} μs`;
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
      label: "What It Runs",
      value: "Linear → ReLU → Linear → Softmax",
    },
    {
      label: "Fastest Current Model Run",
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
    <h3 class="${className}">
      ${validation?.matches_reference ? "The C++ engine matches the Python reference output." : "The current output does not match the Python reference."}
    </h3>
    <p class="muted">Max absolute difference: ${validation ? validation.max_abs_diff.toExponential(3) : "n/a"}</p>
    <p class="muted">
      This matters because correctness comes first. Performance improvements are only useful if the
      engine still produces the same answer as the trusted Python path.
    </p>
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
          <p class="tensor-help">${tensorHelp[key] || ""}</p>
          <table class="tensor-table"><tbody>${rows}</tbody></table>
        </article>
      `;
    })
    .join("");
}

function renderGroupedBars(containerId, titleField, results, descriptionBuilder) {
  const container = document.getElementById(containerId);
  if (!results.length) {
    container.innerHTML = `<p class="muted">No benchmark data is loaded yet. Run the benchmark to populate this section.</p>`;
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
              <div class="bar-label">${result.backend}${result.threads > 1 ? ` · ${result.threads} threads` : " · 1 thread"}</div>
              <div class="bar-track"><div class="bar-fill" style="width:${percent}%"></div></div>
              <div class="bar-value">${formatNumber(result.avg_ns)}</div>
            </div>
          `;
        })
        .join("");
      return `
        <article class="chart-card">
          <h3>${groupName}</h3>
          <p class="chart-help">${descriptionBuilder(groupName, groupResults)}</p>
          <div class="chart-bars">${rows}</div>
        </article>
      `;
    })
    .join("");
}

function renderBenchmarks() {
  const benchmarks = state.benchmarks;
  document.getElementById("benchmark-source").textContent = benchmarks?.source_path
    ? `Loaded from ${benchmarks.source_path}`
    : "No benchmark file loaded";

  renderGroupedBars(
    "matmul-chart",
    "case",
    benchmarks?.matmul_results || [],
    (groupName) =>
      `This compares raw matrix multiplication for shape ${groupName}. Lower time means the kernel is faster for that problem size.`
  );

  renderGroupedBars(
    "model-chart",
    "case",
    benchmarks?.model_results || [],
    (groupName, groupResults) => {
      const sample = groupResults[0];
      return `This measures full inference for ${groupName}. Batch size ${sample.batch}, input width ${sample.input}, hidden width ${sample.hidden}, output width ${sample.output}. Lower time means the entire model runs faster.`;
    }
  );
}

function setConsole(id, text) {
  document.getElementById(id).textContent = text || "";
}

function showPlayground() {
  document.getElementById("intro-shell").classList.add("hidden");
  document.getElementById("playground").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function showSummary() {
  document.getElementById("playground").classList.add("hidden");
  document.getElementById("intro-shell").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
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

document.getElementById("open-playground").addEventListener("click", showPlayground);
document.getElementById("back-to-summary").addEventListener("click", showSummary);

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
