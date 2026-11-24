import argparse
from pathlib import Path

from pcb_reference import ROOT
from pcb_reference.dataset import prepare
from pcb_reference.inputs import ARMS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert paired DeepPCB images to a YOLO HBB dataset.")
    parser.add_argument("--source", type=Path, default=ROOT / "data/DeepPCB/PCBData")
    parser.add_argument("--arm", choices=ARMS, default="single")
    parser.add_argument("--val-shift", type=int, nargs=2, default=(0, 0), metavar=("DX", "DY"))
    args = parser.parse_args()
    dx, dy = args.val_shift
    suffix = f"_dx{dx}_dy{dy}" if dx or dy else ""
    prepare(args.source, ROOT / "data" / f"{args.arm}{suffix}", args.arm, (dx, dy))
