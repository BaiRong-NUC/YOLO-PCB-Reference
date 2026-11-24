from pcb_reference import ROOT
from pcb_reference.diagnostics import measure


if __name__ == "__main__":
    measure(ROOT / "data/DeepPCB/PCBData", ROOT / "outputs/probe")
