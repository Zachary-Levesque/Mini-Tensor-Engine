const state = {
  examples: [],
  selectedExample: null,
  reference: null,
  benchmarks: null,
};

const walkthroughSteps = [
  {
    title: "Choose a sample model",
    detail: "Start with the model selector so you know which manifest, tensors, and outputs the page is showing.",
  },
  {
    title: "Read the inference flow",
    detail: "Look at the layer sequence before inspecting numbers. This tells you what mathematical path the input follows.",
  },
  {
    title: "Confirm correctness",
    detail: "Check that the C++ engine matches the Python reference output. This is the quality gate for the entire project.",
  },
  {
    title: "Inspect tensors",
    detail: "Move through the activation tensors in order and see how the data changes after each layer.",
  },
  {
    title: "Compare performance",
    detail: "Use the benchmark results last to explain which backend is fastest and when threading helps.",
  },
];

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

function getCurrentExample() {
  return state.reference?.example || state.examples.find((example) => example.id === state.selectedExample) || null;
}

function renderExampleSelectors() {
  const selects = [
    document.getElementById("summary-example-select"),
    document.getElementById("playground-example-select"),
  ];
  const options = state.examples
    .map(
      (example) =>
        `<option value="${example.id}" ${example.id === state.selectedExample ? "selected" : ""}>${example.title}</option>`
    )
    .join("");

  for (const select of selects) {
    if (!select) continue;
    select.innerHTML = options;
    select.value = state.selectedExample;
  }

  const example = getCurrentExample();
  if (!example) return;

  document.getElementById("summary-example-title").textContent = example.title;
  document.getElementById("summary-example-text").textContent = example.summary;
  document.getElementById("summary-example-note").textContent = example.interview_note;

  document.getElementById("playground-example-title").textContent = example.title;
  document.getElementById("playground-example-text").textContent = example.summary;
}

function renderWalkthrough() {
  const markup = walkthroughSteps
    .map(
      (step, index) => `
        <article class="walkthrough-step">
          <span class="step-number">${index + 1}</span>
          <h3>${step.title}</h3>
          <p>${step.detail}</p>
        </article>
      `
    )
    .join("");

  document.getElementById("playground-walkthrough").innerHTML = markup;
}

function renderMetrics() {
  const container = document.getElementById("hero-metrics");
  const validation = state.reference?.validation;
  const flow = state.reference?.architecture?.flow || [];
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
      value: flow.join(" -> ") || "No model loaded",
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

  const example = getCurrentExample();
  document.getElementById("architecture-context").innerHTML = example
    ? `<strong>${example.title}</strong>: ${example.summary}`
    : "";

  const validation = state.reference?.validation;
  const className = validation?.matches_reference ? "validation-good" : "validation-bad";
  document.getElementById("validation-card").innerHTML = `
    <h3 class="${className}">
      ${validation?.matches_reference ? "The C++ engine matches the Python reference output." : "The current output does not match the Python reference."}
    </h3>
    <p class="muted">Max absolute difference: ${validation ? validation.max_abs_diff.toExponential(3) : "n/a"}</p>
    <p class="muted">
      This check is the foundation of the project. If the C++ engine does not match the trusted
      Python output, then performance numbers are not meaningful yet.
    </p>
  `;
}

function renderTensors() {
  const tensors = state.reference?.tensors || [];
  const activationCards = tensors.filter((tensor) => tensor.group === "activations" || tensor.group === "reference");
  const parameterCards = tensors.filter((tensor) => tensor.group === "parameters");

  const renderCards = (cards) =>
    cards
      .map((tensor) => {
        const rows = tensor.values
          .map(
            (row) =>
              `<tr>${row.map((value) => `<td>${Number(value).toFixed(4)}</td>`).join("")}</tr>`
          )
          .join("");
        return `
          <article class="tensor-card">
            <h3>${tensor.label}</h3>
            <p class="tensor-shape">shape: [${tensor.shape.join(", ")}]</p>
            <p class="tensor-help">${tensor.description}</p>
            <table class="tensor-table"><tbody>${rows}</tbody></table>
          </article>
        `;
      })
      .join("");

  document.getElementById("activation-grid").innerHTML = renderCards(activationCards);
  document.getElementById("parameter-grid").innerHTML = renderCards(parameterCards);
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
      `This compares raw matrix multiplication for shape ${groupName}. The three numbers mean rows x inner dimension x columns. Lower time means the kernel is faster for that problem size.`
  );

  renderGroupedBars(
    "model-chart",
    "case",
    benchmarks?.model_results || [],
    (groupName, groupResults) => {
      const sample = groupResults[0];
      return `This measures full inference for ${groupName}. Batch size ${sample.batch}, input width ${sample.input}, hidden width ${sample.hidden}, output width ${sample.output}. This is usually the most important performance view because it times the whole model, not just one isolated kernel.`;
    }
  );

  renderBenchmarkInsights();
}

function bestResultByCase(results) {
  const grouped = new Map();
  for (const result of results) {
    const best = grouped.get(result.case);
    if (!best || result.avg_ns < best.avg_ns) {
      grouped.set(result.case, result);
    }
  }
  return Array.from(grouped.values());
}

function renderInsightCards(containerId, items) {
  const container = document.getElementById(containerId);
  if (!items.length) {
    container.innerHTML = "";
    return;
  }

  container.innerHTML = items
    .map(
      (item) => `
        <article class="insight-card">
          <p class="insight-label">${item.label}</p>
          <h3>${item.title}</h3>
          <p>${item.text}</p>
        </article>
      `
    )
    .join("");
}

function renderBenchmarkInsights() {
  const matmulResults = state.benchmarks?.matmul_results || [];
  const modelResults = state.benchmarks?.model_results || [];

  const bestMatmul = bestResultByCase(matmulResults);
  const bestModel = bestResultByCase(modelResults);

  const matmulInsights = [];
  if (bestMatmul.length) {
    const largestCase = [...bestMatmul].sort((lhs, rhs) => lhs.case.localeCompare(rhs.case)).at(-1);
    if (largestCase) {
      matmulInsights.push({
        label: "Main takeaway",
        title: `${largestCase.case} is fastest with ${largestCase.backend}${largestCase.threads > 1 ? ` at ${largestCase.threads} threads` : ""}`,
        text: "On the largest raw matrix multiply, the winning backend shows which implementation handles heavy numerical work best on this machine.",
      });
    }
    const threadedWins = bestMatmul.filter((result) => result.backend === "threaded_transpose_rhs").length;
    matmulInsights.push({
      label: "Pattern",
      title: `${threadedWins} of ${bestMatmul.length} raw matmul cases are won by the threaded backend`,
      text: "This helps you explain whether parallel work is paying off in the low-level kernel, before looking at full model inference.",
    });
  }

  const modelInsights = [];
  if (bestModel.length) {
    const tinyCase = bestModel.find((result) => result.case === "tiny_demo");
    if (tinyCase) {
      modelInsights.push({
        label: "Small workload lesson",
        title: `${tinyCase.backend}${tinyCase.threads > 1 ? ` at ${tinyCase.threads} threads` : ""} is best for tiny_demo`,
        text: "Small models often gain little from extra threads because thread-management overhead can cancel out the benefit.",
      });
    }
    const largestModelCase = [...bestModel].sort((lhs, rhs) => lhs.batch - rhs.batch).at(-1);
    if (largestModelCase) {
      modelInsights.push({
        label: "Large workload lesson",
        title: `${largestModelCase.case} is fastest with ${largestModelCase.backend}${largestModelCase.threads > 1 ? ` at ${largestModelCase.threads} threads` : ""}`,
        text: "This is the clearest end-to-end proof of which implementation helps real inference workloads the most.",
      });
    }
  }

  renderInsightCards("matmul-insights", matmulInsights);
  renderInsightCards("model-insights", modelInsights);
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

function renderAll() {
  renderExampleSelectors();
  renderWalkthrough();
  renderMetrics();
  renderArchitecture();
  renderTensors();
  renderBenchmarks();
}

async function loadState(exampleId = state.selectedExample) {
  const query = exampleId ? `?example=${encodeURIComponent(exampleId)}` : "";
  const payload = await fetchJson(`/api/state${query}`);
  state.examples = payload.examples || [];
  state.reference = payload.reference;
  state.benchmarks = payload.benchmarks;
  state.selectedExample = payload.reference?.example?.id || state.examples[0]?.id || null;
  renderAll();
}

async function switchExample(exampleId) {
  state.selectedExample = exampleId;
  await loadState(exampleId);
  setConsole("inference-output", "");
}

document.getElementById("open-playground").addEventListener("click", showPlayground);
document.getElementById("back-to-summary").addEventListener("click", showSummary);

document.getElementById("summary-example-select").addEventListener("change", async (event) => {
  await switchExample(event.target.value);
});

document.getElementById("playground-example-select").addEventListener("change", async (event) => {
  await switchExample(event.target.value);
});

document.getElementById("refresh-reference").addEventListener("click", async () => {
  const payload = await fetchJson("/api/refresh-reference", {
    method: "POST",
    body: JSON.stringify({ example: state.selectedExample }),
  });
  state.examples = payload.examples || state.examples;
  state.reference = payload.reference;
  renderAll();
  setConsole("inference-output", payload.command.stdout || payload.command.stderr);
});

document.getElementById("inference-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = await fetchJson("/api/run-inference", {
    method: "POST",
    body: JSON.stringify({
      example: state.selectedExample,
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
