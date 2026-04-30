from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"
EXAMPLES_DIR = ROOT / "data" / "examples"


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


def write_tensor(path: Path, tensor: np.ndarray) -> None:
    rows, cols = tensor.shape
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{rows} {cols}\n")
        flat = tensor.reshape(-1)
        handle.write(" ".join(f"{value:.8f}" for value in flat))
        handle.write("\n")


def run_manifest(directory: Path, input_tensor: np.ndarray) -> np.ndarray:
    activations = input_tensor
    for raw_line in (directory / "model.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        layer_type = tokens[0]

        if layer_type == "linear":
            weights = read_tensor(directory / tokens[1])
            bias = read_tensor(directory / tokens[2])
            activations = activations @ weights + bias
        elif layer_type == "relu":
            activations = relu(activations)
        elif layer_type == "sigmoid":
            activations = sigmoid(activations)
        elif layer_type == "tanh":
            activations = tanh(activations)
        elif layer_type == "softmax":
            activations = softmax(activations)
        else:
            raise ValueError(f"Unknown layer type in manifest: {layer_type}")
    return activations


def read_tensor(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        rows, cols = map(int, handle.readline().split())
        values = [float(value) for value in handle.read().split()]
    return np.array(values, dtype=np.float32).reshape(rows, cols)


def write_example(
    example_id: str,
    title: str,
    summary: str,
    interview_note: str,
    input_tensor: np.ndarray,
    tensors: dict[str, np.ndarray],
    manifest_lines: list[str],
) -> None:
    directory = EXAMPLES_DIR / example_id
    directory.mkdir(parents=True, exist_ok=True)

    write_tensor(directory / "input.txt", input_tensor)
    for filename, tensor in tensors.items():
        write_tensor(directory / filename, tensor)

    (directory / "model.txt").write_text(
        "# Sequential feedforward model definition\n" + "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )

    output = run_manifest(directory, input_tensor)
    write_tensor(directory / "output.txt", output)

    metadata = {
        "id": example_id,
        "title": title,
        "summary": summary,
        "interview_note": interview_note,
    }
    (directory / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_examples() -> None:
    if EXAMPLES_DIR.exists():
        shutil.rmtree(EXAMPLES_DIR)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    write_example(
        "relu_classifier",
        "ReLU Classifier",
        "A small classifier that uses the classic Linear -> ReLU -> Linear -> Softmax pipeline.",
        "This is the clean baseline example for correctness, cache-aware matmul, and threading.",
        np.array([[1.0, -2.0, 3.0, 0.5]], dtype=np.float32),
        {
            "w1.txt": np.array(
                [
                    [0.2, -0.4, 0.1, 0.5, -0.3],
                    [0.7, 0.6, -0.2, 0.1, 0.8],
                    [-0.5, 0.2, 0.3, -0.6, 0.4],
                    [0.9, -0.1, 0.5, 0.2, -0.7],
                ],
                dtype=np.float32,
            ),
            "b1.txt": np.array([[0.1, -0.2, 0.05, 0.3, -0.4]], dtype=np.float32),
            "w2.txt": np.array(
                [
                    [0.3, -0.1, 0.8],
                    [-0.6, 0.4, 0.2],
                    [0.5, 0.7, -0.3],
                    [0.1, -0.5, 0.9],
                    [-0.2, 0.6, 0.4],
                ],
                dtype=np.float32,
            ),
            "b2.txt": np.array([[0.05, -0.15, 0.25]], dtype=np.float32),
        },
        [
            "linear w1.txt b1.txt",
            "relu",
            "linear w2.txt b2.txt",
            "softmax",
        ],
    )

    write_example(
        "sigmoid_gate",
        "Sigmoid Gate",
        "A compact example that uses Sigmoid to squash hidden activations into the 0 to 1 range.",
        "This shows that the engine is no longer tied to ReLU-only manifests and can execute other nonlinearities.",
        np.array([[0.25, -1.5, 2.0]], dtype=np.float32),
        {
            "w1.txt": np.array(
                [
                    [0.8, -0.3, 0.4, 0.2],
                    [-0.5, 0.7, 0.1, -0.6],
                    [0.3, 0.5, -0.2, 0.9],
                ],
                dtype=np.float32,
            ),
            "b1.txt": np.array([[0.05, -0.1, 0.2, -0.25]], dtype=np.float32),
            "w2.txt": np.array(
                [
                    [0.4, -0.2],
                    [-0.7, 0.5],
                    [0.6, 0.1],
                    [0.2, 0.8],
                ],
                dtype=np.float32,
            ),
            "b2.txt": np.array([[0.15, -0.05]], dtype=np.float32),
        },
        [
            "linear w1.txt b1.txt",
            "sigmoid",
            "linear w2.txt b2.txt",
            "softmax",
        ],
    )

    write_example(
        "tanh_mixer",
        "Tanh Mixer",
        "A deeper example that mixes Tanh and ReLU to show a longer feed-forward manifest.",
        "This demonstrates that the model system can represent more than one hidden stage without hardcoding architecture logic.",
        np.array([[1.2, -0.7, 0.5]], dtype=np.float32),
        {
            "w1.txt": np.array(
                [
                    [0.6, -0.2, 0.1, 0.4],
                    [-0.3, 0.8, -0.5, 0.2],
                    [0.7, 0.1, 0.6, -0.4],
                ],
                dtype=np.float32,
            ),
            "b1.txt": np.array([[0.0, -0.15, 0.1, 0.05]], dtype=np.float32),
            "w2.txt": np.array(
                [
                    [0.5, -0.4, 0.2],
                    [0.1, 0.6, -0.3],
                    [-0.2, 0.4, 0.7],
                    [0.8, -0.1, 0.3],
                ],
                dtype=np.float32,
            ),
            "b2.txt": np.array([[0.1, -0.05, 0.2]], dtype=np.float32),
            "w3.txt": np.array(
                [
                    [0.3, -0.6],
                    [0.7, 0.2],
                    [-0.4, 0.5],
                ],
                dtype=np.float32,
            ),
            "b3.txt": np.array([[0.05, 0.12]], dtype=np.float32),
        },
        [
            "linear w1.txt b1.txt",
            "tanh",
            "linear w2.txt b2.txt",
            "relu",
            "linear w3.txt b3.txt",
            "softmax",
        ],
    )


def sync_default_reference() -> None:
    default_example = EXAMPLES_DIR / "relu_classifier"
    if REFERENCE_DIR.exists():
        shutil.rmtree(REFERENCE_DIR)
    shutil.copytree(default_example, REFERENCE_DIR)


def main() -> None:
    write_examples()
    sync_default_reference()
    print("Reference tensors written to", REFERENCE_DIR)
    print("Example model bundles written to", EXAMPLES_DIR)


if __name__ == "__main__":
    main()
