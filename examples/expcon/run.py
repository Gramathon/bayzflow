"""Run only the inference specified by an EXPCON snapshot."""

import argparse
import json
from pathlib import Path

from bayzflow.examples.guide_benchmark import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    snapshot = json.loads(args.contract.read_text())
    config = snapshot["scientific_parameters"]
    evaluate(
        args.output_root,
        [snapshot["reproducibility"]["seed"]],
        config["steps"],
        config["samples"],
        config["guides"],
    )


if __name__ == "__main__":
    main()
