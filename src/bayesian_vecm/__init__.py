"""Bayesian Vector Error Correction Models in Python."""

from bayesian_vecm._causality import CausalityResult
from bayesian_vecm._model import BayesianVECM
from bayesian_vecm._rank import CointRankResult, select_coint_rank
from bayesian_vecm._simulate import BrandData, BrandTruth, simulate_brand_data

__version__ = "0.1.0"
__all__ = [
    "BayesianVECM",
    "BrandData",
    "BrandTruth",
    "CausalityResult",
    "CointRankResult",
    "__version__",
    "select_coint_rank",
    "simulate_brand_data",
]
