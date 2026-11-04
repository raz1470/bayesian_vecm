"""§14 table: the checks the plug-in notebook (guide 08) prints, as HTML.

Reads the check table from the saved output of
notebooks/guides/08_response_curves_on_your_data.ipynb (run it first) and
rewrites the rows of html/fig14_check_table.html. render_html.py then turns it
into fig14_check_table.png.
"""

import html
import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks" / "guides" / "08_response_curves_on_your_data.ipynb"
PAGE = Path(__file__).resolve().parent / "html" / "fig14_check_table.html"
TAGS = {
    "ok": "background:#ecfdf5;color:#047857",
    "decide": "background:#eff6ff;color:#1d4ed8",
    "look": "background:#fffbeb;color:#92400e",
}

cells = json.loads(NOTEBOOK.read_text())["cells"]
source = next(c for c in cells if "".join(c["source"]).startswith("checks = pd.DataFrame(CHECKS)"))
text = next("".join(o["text"]) for o in source["outputs"] if o.get("name") == "stdout")
lines = text.split("\n# Decision log")[0].rstrip("\n").split("\n")
header = lines[0]
# Columns are right-aligned by to_string: cut each line at the header's column ends.
ends = [m.end() for m in re.finditer(r"\S+", header)]
starts = [0, *ends[:-1]]
widths = [e - s for s, e in zip(starts, ends, strict=True)]
table = pd.read_fwf(StringIO("\n".join(lines[1:])), widths=widths, header=None)
table.columns = ["step", "check", "result", "status", "action"]

rows = []
for r in table.fillna("").itertuples():
    status = str(r.status).strip()
    rows.append(
        f'<tr><td><span class=tag style="{TAGS[status]}">{status}</span></td>'
        f"<td>{html.escape(str(r.check))}</td><td class=r>{html.escape(str(r.result))}</td>"
        f"<td class=a>{html.escape(str(r.action))}</td></tr>"
    )
page = PAGE.read_text()
page = re.sub(r"<tbody>.*</tbody>", "<tbody>" + "".join(rows) + "</tbody>", page, flags=re.S)
PAGE.write_text(page)
print(f"{len(rows)} checks written to {PAGE.name}")
