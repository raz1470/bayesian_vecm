"""Run guide 08 on a CSV, the way a user runs it on their own data.

Writes simulated weekly data to a CSV, points a copy of the notebook's
configuration cell at it (DATA_PATH), cuts the sampler to 200 draws (this checks
that the notebook runs, not its numbers), and executes the copy. The default run of
the notebook uses simulated data in memory and skips the branch that reads a
file and saves the fit, so this is the only test of that path.

Usage: python tools/ci/guide08_on_csv.py <output_dir>
"""

import json
import sys
from pathlib import Path

import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import reads, write

from bayesian_vecm import simulate_brand_data

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "notebooks" / "guides" / "08_response_curves_on_your_data.ipynb"
out = Path(sys.argv[1] if len(sys.argv) > 1 else "guide08_csv").resolve()
out.mkdir(parents=True, exist_ok=True)

sim = simulate_brand_data(
    exog_unit_root=("interest_rate",),
    exog_adstock={"brand_spend": 0.5, "pr_reach": 0.3, "sponsorship_reach": 0.6},
    exog_saturation={"brand_spend": 5.0, "pr_reach": 3.0, "sponsorship_reach": 5.0},
    contemporaneous={"brand_consideration": {"brand_awareness": 0.2}},
    seed=7,
)
data = np.exp(sim.endog).join(sim.exog)
data.index.name = "week"
csv = out / "brand_data.csv"
data.to_csv(csv)

nb = json.loads(NOTEBOOK.read_text())
config = "".join(nb["cells"][1]["source"])
assert "DATA_PATH = None" in config, "configuration cell has changed"
nb["cells"][1]["source"] = config.replace("DATA_PATH = None", f"DATA_PATH = {str(csv)!r}")
# A smoke test of the code path, not of the answers: sample briefly.
setup = "".join(nb["cells"][2]["source"])
draws_line = "DRAWS, TUNE, CHAINS = (2000, 1000, 4) if FINAL_RUN else (1000, 1000, 2)"
assert draws_line in setup, "setup cell has changed"
nb["cells"][2]["source"] = setup.replace(draws_line, "DRAWS, TUNE, CHAINS = 200, 200, 2")

notebook = reads(json.dumps(nb), as_version=4)
ExecutePreprocessor(timeout=900, kernel_name="python3").preprocess(
    notebook, {"metadata": {"path": str(out)}}
)
write(notebook, out / "guide08_on_csv.ipynb")

for name in ("decision_log.md", "fit.zarr"):
    assert (out / name).exists(), f"guide 08 did not write {name}"
print("Guide 08 ran on a CSV and saved its decision log and fit.")
