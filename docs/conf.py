"""Sphinx configuration for the bayesian_vecm docs site.

Build locally with::

    uv run --group docs sphinx-build -W -b html docs docs/_build/html
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import bayesian_vecm

DOCS = Path(__file__).parent
ROOT = DOCS.parent

project = "bayesian_vecm"
author = "Ryan O'Sullivan"
copyright = "2026, Ryan O'Sullivan"
release = bayesian_vecm.__version__
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
]

# Pages in docs/. walkthrough.html is a finished page: copied as-is so its
# URL (raz1470.github.io/bayesian_vecm/walkthrough.html) keeps working.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "walkthrough.html"]
html_extra_path = ["walkthrough.html"]

# The guide and build notebooks live in notebooks/. Copy them in at build
# time and render their saved outputs; executing them would take minutes.
_GUIDES = {
    "01_brand_vecm_walkthrough.ipynb": "practitioner_walkthrough.ipynb",
    "02_statsmodels_comparison.ipynb": "statsmodels_comparison.ipynb",
    "03_carryover_and_saturation.ipynb": "carryover_and_saturation.ipynb",
    "04_same_week_links.ipynb": "same_week_links.ipynb",
    "05_putting_it_together.ipynb": "putting_it_together.ipynb",
    "06_base_sales_from_total_sales.ipynb": "base_sales_from_total_sales.ipynb",
}
(DOCS / "guide").mkdir(exist_ok=True)
for _source, _target in _GUIDES.items():
    shutil.copyfile(ROOT / "notebooks" / "guides" / _source, DOCS / "guide" / _target)

# The build notebooks record how the package was built, one feature at a time.
# Copied under their own names so the links between them keep working.
(DOCS / "build_notebooks").mkdir(exist_ok=True)
for _notebook in sorted((ROOT / "notebooks" / "build").glob("*.ipynb")):
    shutil.copyfile(_notebook, DOCS / "build_notebooks" / _notebook.name)
nb_execution_mode = "off"

myst_enable_extensions = ["dollarmath", "colon_fence"]
myst_heading_anchors = 3

# API pages: signatures from the code, sections from the numpydoc docstrings.
autodoc_default_options = {"member-order": "bysource"}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_xref_param_type = False
add_module_names = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "pymc": ("https://www.pymc.io/projects/docs/en/stable/", None),
}
# Set DOCS_OFFLINE=1 to build without fetching the inventories above.
if os.environ.get("DOCS_OFFLINE"):
    intersphinx_mapping = {}

html_theme = "pydata_sphinx_theme"
html_title = "bayesian_vecm"
html_theme_options = {
    "github_url": "https://github.com/raz1470/bayesian_vecm",
    "navbar_align": "left",
    "show_toc_level": 2,
}
html_show_sourcelink = False
