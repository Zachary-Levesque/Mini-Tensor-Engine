from __future__ import annotations

import json
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
REFERENCE_DIR = ROOT / "data" / "reference"
EXAMPLES_DIR = ROOT / "data" / "examples"
DEFAULT_BENCHMARK_JSON = ROOT / "build" / "benchmark_results.json"
DEFAULT_BENCHMARK_CSV = ROOT / "build" / "benchmark_results.csv"
INFER_BINARY = ROOT / "build" / "mte_infer"
BENCHMARK_BINARY = ROOT / "build" / "mte_benchmark"
EXPORT_SCRIPT = ROOT / "python" / "export_reference.py"
DEFAULT_EXAMPLE_ID = "relu_classifier"

ALLOWED_BACKENDS = {
    "naive",
    "transpose_rhs",
    "threaded_transpose_rhs",
}

LAYER_LABELS = {
    "linear": "Linear",
    "relu": "ReLU",
    "sigmoid": "Sigmoid",
    "tanh": "Tanh",
    "softmax": "Softmax",
}


def read_tensor(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        rows, cols = map(int, handle.readline().split())
        values = [float(value) for value in handle.read().split()]
    return np.array(values, dtype=np.float32).reshape(rows, cols)


def tensor_payload(tensor: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "values": [[float(value) for value in row] for row in tensor.tolist()],
    }


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def parse_manifest(directory: Path) -> list[dict[str, str]]:
    manifest_path = directory / "model.txt"
    if not manifest_path.exists():
        return [
            {"type": "linear", "weights": "w1.txt", "bias": "b1.txt"},
            {"type": "relu"},
            {"type": "linear", "weights": "w2.txt", "bias": "b2.txt"},
            {"type": "softmax"},
        ]

    layers: list[dict[str, str]] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        layer_type = tokens[0].lower()
        if layer_type == "linear":
            layers.append({"type": layer_type, "weights": tokens[1], "bias": tokens[2]})
        else:
            layers.append({"type": layer_type})
    return layers


def discover_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    if EXAMPLES_DIR.exists():
        for directory in sorted(path for path in EXAMPLES_DIR.iterdir() if path.is_dir()):
            meta_path = directory / "meta.json"
            metadata: dict[str, Any] = {}
            if meta_path.exists():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            example_id = metadata.get("id", directory.name)
            examples.append(
                {
                    "id": example_id,
                    "title": metadata.get("title", directory.name.replace("_", " ").title()),
                    "summary": metadata.get("summary", "Example model bundle."),
                    "interview_note": metadata.get("interview_note", ""),
                    "path": str(directory.relative_to(ROOT)),
                }
            )

    if not examples:
        examples.append(
            {
                "id": DEFAULT_EXAMPLE_ID,
                "title": "Reference Model",
                "summary": "Fallback reference bundle.",
                "interview_note": "",
                "path": str(REFERENCE_DIR.relative_to(ROOT)),
            }
        )
    return examples


def resolve_example_directory(example_id: str | None) -> tuple[dict[str, Any], Path]:
    examples = discover_examples()
    example_map = {example["id"]: example for example in examples}
    selected = example_map.get(example_id or DEFAULT_EXAMPLE_ID, examples[0])
    return selected, ROOT / selected["path"]


def describe_linear_output(index: int) -> str:
    return (
        f"This is the output immediately after linear layer {index}. "
        "A linear layer multiplies the input by a weight matrix and then adds a bias."
    )


def describe_activation_output(layer_type: str, index: int) -> str:
    if layer_type == "relu":
        return (
            f"This is the output after ReLU {index}. ReLU keeps positive values and clamps "
            "negative values to zero."
        )
    if layer_type == "sigmoid":
        return (
            f"This is the output after Sigmoid {index}. Sigmoid squashes each value into the "
            "range from 0 to 1."
        )
    if layer_type == "tanh":
        return (
            f"This is the output after Tanh {index}. Tanh squashes each value into the range "
            "from -1 to 1."
        )
    return (
        "This is the output after Softmax. Softmax turns the final scores into normalized "
        "probabilities that sum to 1."
    )


def build_trace_tensors(directory: Path, input_tensor: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    layers = parse_manifest(directory)
    activations = input_tensor
    tensors: list[dict[str, Any]] = [
        {
            "key": "input",
            "label": "Input",
            "group": "activations",
            "description": "This is the data that goes into the model.",
            **tensor_payload(input_tensor),
        }
    ]

    linear_count = 0
    relu_count = 0
    sigmoid_count = 0
    tanh_count = 0

    for layer in layers:
        layer_type = layer["type"]
        if layer_type == "linear":
            linear_count += 1
            weights = read_tensor(directory / layer["weights"])
            bias = read_tensor(directory / layer["bias"])
            tensors.append(
                {
                    "key": f"weights_{linear_count}",
                    "label": f"Weights {linear_count}",
                    "group": "parameters",
                    "description": f"These are the learned weights for linear layer {linear_count}.",
                    **tensor_payload(weights),
                }
            )
            tensors.append(
                {
                    "key": f"bias_{linear_count}",
                    "label": f"Bias {linear_count}",
                    "group": "parameters",
                    "description": f"This is the bias vector added after linear layer {linear_count}.",
                    **tensor_payload(bias),
                }
            )
            activations = activations @ weights + bias
            tensors.append(
                {
                    "key": f"linear_{linear_count}_output",
                    "label": f"After Linear {linear_count}",
                    "group": "activations",
                    "description": describe_linear_output(linear_count),
                    **tensor_payload(activations),
                }
            )
            continue

        if layer_type == "relu":
            relu_count += 1
            activations = relu(activations)
            label = "Output" if layer == layers[-1] else f"After ReLU {relu_count}"
            tensors.append(
                {
                    "key": f"relu_{relu_count}_output",
                    "label": label,
                    "group": "activations",
                    "description": describe_activation_output("relu", relu_count),
                    **tensor_payload(activations),
                }
            )
            continue

        if layer_type == "sigmoid":
            sigmoid_count += 1
            activations = sigmoid(activations)
            label = "Output" if layer == layers[-1] else f"After Sigmoid {sigmoid_count}"
            tensors.append(
                {
                    "key": f"sigmoid_{sigmoid_count}_output",
                    "label": label,
                    "group": "activations",
                    "description": describe_activation_output("sigmoid", sigmoid_count),
                    **tensor_payload(activations),
                }
            )
            continue

        if layer_type == "tanh":
            tanh_count += 1
            activations = tanh(activations)
            label = "Output" if layer == layers[-1] else f"After Tanh {tanh_count}"
            tensors.append(
                {
                    "key": f"tanh_{tanh_count}_output",
                    "label": label,
                    "group": "activations",
                    "description": describe_activation_output("tanh", tanh_count),
                    **tensor_payload(activations),
                }
            )
            continue

        if layer_type == "softmax":
            activations = softmax(activations)
            tensors.append(
                {
                    "key": "output",
                    "label": "Output",
                    "group": "activations",
                    "description": describe_activation_output("softmax", 1),
                    **tensor_payload(activations),
                }
            )
            continue

        raise ValueError(f"Unsupported layer type in UI trace: {layer_type}")

    return tensors, activations


def load_reference_payload(example_id: str | None = None) -> dict[str, Any]:
    example, directory = resolve_example_directory(example_id)
    input_tensor = read_tensor(directory / "input.txt")
    expected_output = read_tensor(directory / "output.txt")
    trace_tensors, computed_output = build_trace_tensors(directory, input_tensor)
    max_abs_diff = float(np.max(np.abs(computed_output - expected_output)))

    tensors = trace_tensors + [
        {
            "key": "expected_output",
            "label": "Expected Output",
            "group": "reference",
            "description": "This is the stored Python reference output used for correctness checking.",
            **tensor_payload(expected_output),
        }
    ]

    return {
        "example": example,
        "architecture": {
            "name": "FeedForwardModel",
            "flow": ["Input"] + [LAYER_LABELS.get(layer["type"], layer["type"].title()) for layer in parse_manifest(directory)],
        },
        "tensors": tensors,
        "validation": {
            "matches_reference": bool(max_abs_diff <= 1e-5),
            "max_abs_diff": max_abs_diff,
        },
    }


def load_benchmark_payload() -> dict[str, Any] | None:
    candidate_paths = [
        DEFAULT_BENCHMARK_JSON,
        ROOT / "results.json",
    ]
    for path in candidate_paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            data["source_path"] = str(path.relative_to(ROOT))
            return data
    return None


def run_command(args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "MiniTensorEngineUI/0.2"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        selected_example = params.get("example", [None])[0]

        if parsed.path == "/":
            self._serve_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/app.js":
            self._serve_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")
            return
        if parsed.path == "/styles.css":
            self._serve_file(STATIC_DIR / "styles.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._send_json(
                {
                    "project": {
                        "title": "Mini Tensor Engine",
                        "summary": (
                            "Custom C++ inference engine with tensor storage, a manifest-driven "
                            "feed-forward model, multiple matmul backends, validation against "
                            "Python, and benchmark-driven optimization."
                        ),
                    },
                    "examples": discover_examples(),
                    "reference": load_reference_payload(selected_example),
                    "benchmarks": load_benchmark_payload(),
                }
            )
            return
        if parsed.path == "/api/benchmarks":
            self._send_json({"benchmarks": load_benchmark_payload()})
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self._read_json_body()
        selected_example = payload.get("example")

        if parsed.path == "/api/refresh-reference":
            command = run_command(["python3", str(EXPORT_SCRIPT)])
            self._send_json(
                {
                    "command": command,
                    "examples": discover_examples(),
                    "reference": load_reference_payload(selected_example),
                },
                HTTPStatus.OK if command["returncode"] == 0 else HTTPStatus.BAD_REQUEST,
            )
            return

        if parsed.path == "/api/run-inference":
            backend = payload.get("backend", "threaded_transpose_rhs")
            threads = int(payload.get("threads", 1))
            if backend not in ALLOWED_BACKENDS:
                self._send_json({"error": "Invalid backend"}, HTTPStatus.BAD_REQUEST)
                return
            if threads <= 0:
                self._send_json({"error": "Threads must be greater than 0"}, HTTPStatus.BAD_REQUEST)
                return

            _, directory = resolve_example_directory(selected_example)
            command = run_command(
                [
                    str(INFER_BINARY),
                    "--data-dir",
                    str(directory),
                    "--backend",
                    backend,
                    "--threads",
                    str(threads),
                ]
            )
            self._send_json(
                {"command": command},
                HTTPStatus.OK if command["returncode"] == 0 else HTTPStatus.BAD_REQUEST,
            )
            return

        if parsed.path == "/api/run-benchmark":
            iterations = int(payload.get("iterations", 20))
            warmup = int(payload.get("warmup", 5))
            thread_values = payload.get("threads", [1, 2, 4])
            threads = [int(value) for value in thread_values]

            if iterations <= 0:
                self._send_json({"error": "Iterations must be greater than 0"}, HTTPStatus.BAD_REQUEST)
                return
            if any(value <= 0 for value in threads):
                self._send_json({"error": "Thread counts must be greater than 0"}, HTTPStatus.BAD_REQUEST)
                return

            command = run_command(
                [
                    str(BENCHMARK_BINARY),
                    "--iterations",
                    str(iterations),
                    "--warmup",
                    str(warmup),
                    "--threads",
                    ",".join(str(value) for value in threads),
                    "--csv-out",
                    str(DEFAULT_BENCHMARK_CSV),
                    "--json-out",
                    str(DEFAULT_BENCHMARK_JSON),
                ]
            )
            self._send_json(
                {
                    "command": command,
                    "benchmarks": load_benchmark_payload(),
                },
                HTTPStatus.OK if command["returncode"] == 0 else HTTPStatus.BAD_REQUEST,
            )
            return

        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _serve_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length == 0:
            return {}
        body = self.rfile.read(content_length)
        return json.loads(body.decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler)
    print("Mini Tensor Engine UI running at http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down UI server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
