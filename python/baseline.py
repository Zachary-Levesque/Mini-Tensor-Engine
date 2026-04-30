from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"


def read_tensor(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        rows, cols = map(int, handle.readline().split())
        values = [float(value) for value in handle.read().split()]
    return np.array(values, dtype=np.float32).reshape(rows, cols)


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


def run_model_from_manifest(directory: Path, input_tensor: np.ndarray) -> np.ndarray:
    activations = input_tensor
    manifest_path = directory / "model.txt"

    if not manifest_path.exists():
        w1 = read_tensor(directory / "w1.txt")
        b1 = read_tensor(directory / "b1.txt")
        w2 = read_tensor(directory / "w2.txt")
        b2 = read_tensor(directory / "b2.txt")
        return softmax(relu(activations @ w1 + b1) @ w2 + b2)

    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Python reference model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REFERENCE_DIR,
        help="Directory containing input.txt, model.txt, and tensor files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_tensor = read_tensor(args.data_dir / "input.txt")
    output = run_model_from_manifest(args.data_dir, input_tensor)

    np.set_printoptions(precision=6, suppress=True)
    print(f"Python baseline output ({args.data_dir}):")
    print(output)


if __name__ == "__main__":
    main()
