"""Entry point for running the Foveal CPT benchmark pipeline.

Re-exports foveal_cpt.run_eval_pipeline for convenience within the benchmarks suite.
"""

import sys
from foveal_cpt.run_eval_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
