from __future__ import annotations

import json
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"
REFERENCE_DIR = ROOT / "data" / "reference"
DEFAULT_BENCHMARK_JSON = ROOT / "build" / "benchmark_results.json"
DEFAULT_BENCHMARK_CSV = ROOT / "build" / "benchmark_results.csv"
INFER_BINARY = ROOT / "build" / "mte_infer"
BENCHMARK_BINARY = ROOT / "build" / "mte_benchmark"
EXPORT_SCRIPT = ROOT / "python" / "export_reference.py"

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


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def load_model_architecture() -> dict[str, Any]:
    manifest_path = REFERENCE_DIR / "model.txt"
    if not manifest_path.exists():
        return {
            "name": "FeedForwardModel",
            "flow": ["Input", "Linear", "ReLU", "Linear", "Softmax"],
        }

    flow = ["Input"]
    with manifest_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            layer_name = line.split()[0].lower()
            flow.append(LAYER_LABELS.get(layer_name, layer_name.title()))

    return {
        "name": "FeedForwardModel",
        "flow": flow,
    }


def load_reference_payload() -> dict[str, Any]:
    input_tensor = read_tensor(REFERENCE_DIR / "input.txt")
    w1 = read_tensor(REFERENCE_DIR / "w1.txt")
    b1 = read_tensor(REFERENCE_DIR / "b1.txt")
    w2 = read_tensor(REFERENCE_DIR / "w2.txt")
    b2 = read_tensor(REFERENCE_DIR / "b2.txt")
    expected_output = read_tensor(REFERENCE_DIR / "output.txt")

    hidden_linear = input_tensor @ w1 + b1
    hidden_relu = np.maximum(hidden_linear, 0.0)
    logits = hidden_relu @ w2 + b2
    output = softmax(logits)
    max_abs_diff = float(np.max(np.abs(output - expected_output)))

    return {
        "architecture": load_model_architecture(),
        "tensors": {
            "input": tensor_payload(input_tensor),
            "w1": tensor_payload(w1),
            "b1": tensor_payload(b1),
            "w2": tensor_payload(w2),
            "b2": tensor_payload(b2),
            "hidden_linear": tensor_payload(hidden_linear),
            "hidden_relu": tensor_payload(hidden_relu),
            "logits": tensor_payload(logits),
            "output": tensor_payload(output),
            "expected_output": tensor_payload(expected_output),
        },
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
    server_version = "MiniTensorEngineUI/0.1"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
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
                    "reference": load_reference_payload(),
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

        if parsed.path == "/api/refresh-reference":
            command = run_command(["python3", str(EXPORT_SCRIPT)])
            self._send_json(
                {
                    "command": command,
                    "reference": load_reference_payload(),
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

            command = run_command(
                [
                    str(INFER_BINARY),
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
    server.serve_forever()


if __name__ == "__main__":
    main()
