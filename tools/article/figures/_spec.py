"""Write a chart spec as JSON for the article page to draw.

Each figure script builds a dict (title, legend, panels with layers) and passes it
to ``save``. ``build.py`` embeds every spec in the page and ``charts.js`` draws it
as inline SVG.
"""

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"


def r(x, digits: int = 4):
    """Round an array or number to ``digits`` significant figures for compact JSON."""
    a = np.asarray(x, dtype=float)
    out = np.vectorize(lambda v: float(f"{v:.{digits}g}"))(a) if a.size else a
    return out.tolist()


def save(name: str, spec: dict) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / f"{name}.json"
    path.write_text(json.dumps(spec, separators=(",", ":"), ensure_ascii=False) + "\n")
    print("wrote", path.relative_to(DATA.parents[2]))
