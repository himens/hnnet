#!/usr/bin/env python3
"""Plot training epochs and mean squared error from a neural-network log."""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt


TRAINING_LINE = re.compile(
    r"^\s*NNet::train: epoch:\s*(\d+), mean squared error:\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
)


def read_points(input_path: Path) -> tuple[list[float], list[float]]:
    epochs = []
    errors = []
    for line in input_path.read_text().splitlines():
        match = TRAINING_LINE.match(line)
        if match is None:
            continue
        epochs.append(int(match.group(1)))
        errors.append(float(match.group(2)))
    if not epochs:
        raise SystemExit(f"No numeric data found in {input_path}")
    return epochs, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Path to the data file")
    args = parser.parse_args()

    input_path = args.input
    output_path = input_path.with_suffix(".png")
    epochs, errors = read_points(input_path)

    plt.figure(figsize=(12, 7))
    plt.plot(epochs, errors, color="#d4573d", linewidth=1.5, label="Errore")
    plt.title("Errore per epoca")
    plt.xlabel("Epoca")
    plt.ylabel("Errore")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Wrote {output_path} ({len(epochs)} points)")


if __name__ == "__main__":
    main()