import argparse
from pathlib import Path

import torch

from models.motip.entity_pointer import CausalEntityPointer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    state = torch.load(args.input, map_location="cpu", weights_only=False)
    model_state = dict(state["model"] if "model" in state else state)
    if any(key.startswith("entity_pointer.") for key in model_state):
        raise ValueError("Input already contains an entity pointer")
    torch.manual_seed(42)
    pointer = CausalEntityPointer(feature_dim=256, num_id_vocabulary=50)
    for key, value in pointer.state_dict().items():
        model_state[f"entity_pointer.{key}"] = value
    output = {
        "model": model_state,
        "optimizer": None,
        "scheduler": None,
        "states": {"start_epoch": 0, "global_step": 0},
        "source_checkpoint": str(args.input),
        "added_module": "CausalEntityPointer",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)


if __name__ == "__main__":
    main()
