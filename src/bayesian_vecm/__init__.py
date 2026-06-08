"""Bayesian Vector Error Correction Models in Python."""

from bayesian_vecm._model import BayesianVECM
from bayesian_vecm._rank import CointRankResult, select_coint_rank

__version__ = "0.1.0"
__all__ = ["BayesianVECM", "CointRankResult", "__version__", "select_coint_rank"]
