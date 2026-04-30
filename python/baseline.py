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


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def main() -> None:
    input_tensor = read_tensor(REFERENCE_DIR / "input.txt")
    w1 = read_tensor(REFERENCE_DIR / "w1.txt")
    b1 = read_tensor(REFERENCE_DIR / "b1.txt")
    w2 = read_tensor(REFERENCE_DIR / "w2.txt")
    b2 = read_tensor(REFERENCE_DIR / "b2.txt")

    hidden = relu(input_tensor @ w1 + b1)
    output = softmax(hidden @ w2 + b2)

    np.set_printoptions(precision=6, suppress=True)
    print("Python baseline output:")
    print(output)


if __name__ == "__main__":
    main()
