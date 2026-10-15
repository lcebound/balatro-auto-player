"""AirtestIDE entry point for the Balatro decision-tree runner."""

from pathlib import Path
import sys

from airtest.core.api import auto_setup

auto_setup(__file__)

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from runner import main


if __name__ == "__main__":
    # The IDE supplies Airtest arguments; runner.main expects its own CLI.
    sys.argv = [__file__, "batch", "--games", "10"]
    main()
