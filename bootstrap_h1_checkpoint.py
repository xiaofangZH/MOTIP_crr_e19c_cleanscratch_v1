import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.input, map_location="cpu", weights_only=False)
    model = state["model"] if "model" in state else state
    key = "trajectory_modeling.causal_prototype_scale"
    if key in model:
        raise ValueError(f"Input already contains {key}")
    model[key] = torch.zeros(())
    output = {
        "model": model,
        "optimizer": None,
        "scheduler": None,
        "states": {"start_epoch": 0, "global_step": 0},
        "source_checkpoint": str(args.input),
        "added_parameter": key,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)


if __name__ == "__main__":
    main()

