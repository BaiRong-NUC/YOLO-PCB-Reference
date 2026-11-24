import argparse

from pcb_reference.inputs import ARMS
from pcb_reference.training import train


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DeepPCB YOLO26s comparison arm.")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.arm, args.seed)
