#!/usr/bin/env python3
"""
Scarica il dataset MNIST (formato CSV: label,pixel0,...,pixel783) e lo salva in data/mnist.
Output:
 - salva `data/mnist/mnist_train.csv` con le prime N righe del training set (N configurabile)
 - salva `data/mnist/mnist_test.csv` con le prime N righe del test set (N configurabile)
"""

import argparse
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path("data/mnist")
TRAIN_URL = "https://data.pjreddie.com/files/mnist_train.csv"
TEST_URL  = "https://data.pjreddie.com/files/mnist_test.csv"
FULL_TRAIN_SAMPLES = 60000
FULL_TEST_SAMPLES = 10000

def download_samples(url: str, dest: Path, nb_samples: int) -> None:
    print(f"Downloading {nb_samples} samples from {url}...")
    request = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
    with urllib.request.urlopen(request) as response:
        with dest.open("w") as file:
            for idx, line in enumerate(response):
                if idx >= nb_samples:
                    break
                file.write(line.decode("utf-8"))
    print(f"Saved: {dest.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Download MNIST dataset (CSV format).")
    parser.add_argument("--train-samples", type=int, default=10000, help="number of training samples to download")
    parser.add_argument("--test-samples", type=int, default=1000, help="number of test samples to download")
    parser.add_argument("--all", action="store_true", help="download the full dataset (60000 train, 10000 test)")
    args = parser.parse_args()

    train_samples = FULL_TRAIN_SAMPLES if args.all else args.train_samples
    test_samples = FULL_TEST_SAMPLES if args.all else args.test_samples

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if train_samples > 0:
        download_samples(TRAIN_URL, OUTPUT_DIR / "mnist_train.csv", train_samples)
    if test_samples > 0:
        download_samples(TEST_URL, OUTPUT_DIR / "mnist_test.csv", test_samples)

if __name__ == '__main__':
    main()
