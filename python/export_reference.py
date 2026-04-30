from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


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


def main() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)

    input_tensor = np.array([[1.0, -2.0, 3.0, 0.5]], dtype=np.float32)
    w1 = np.array(
        [
            [0.2, -0.4, 0.1, 0.5, -0.3],
            [0.7, 0.6, -0.2, 0.1, 0.8],
            [-0.5, 0.2, 0.3, -0.6, 0.4],
            [0.9, -0.1, 0.5, 0.2, -0.7],
        ],
        dtype=np.float32,
    )
    b1 = np.array([[0.1, -0.2, 0.05, 0.3, -0.4]], dtype=np.float32)

    w2 = np.array(
        [
            [0.3, -0.1, 0.8],
            [-0.6, 0.4, 0.2],
            [0.5, 0.7, -0.3],
            [0.1, -0.5, 0.9],
            [-0.2, 0.6, 0.4],
        ],
        dtype=np.float32,
    )
    b2 = np.array([[0.05, -0.15, 0.25]], dtype=np.float32)

    hidden = relu(input_tensor @ w1 + b1)
    output = softmax(hidden @ w2 + b2)

    write_tensor(REFERENCE_DIR / "input.txt", input_tensor)
    write_tensor(REFERENCE_DIR / "w1.txt", w1)
    write_tensor(REFERENCE_DIR / "b1.txt", b1)
    write_tensor(REFERENCE_DIR / "w2.txt", w2)
    write_tensor(REFERENCE_DIR / "b2.txt", b2)
    write_tensor(REFERENCE_DIR / "output.txt", output)

    print("Reference tensors written to", REFERENCE_DIR)


if __name__ == "__main__":
    main()
