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

# The practitioner notebook lives in notebooks/guides. Copy it in at build
# time and render its saved outputs; executing it would take minutes.
_GUIDE = ROOT / "notebooks" / "guides" / "01_brand_vecm_walkthrough.ipynb"
(DOCS / "guide").mkdir(exist_ok=True)
shutil.copyfile(_GUIDE, DOCS / "guide" / "practitioner_walkthrough.ipynb")
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
