# bayesian_vecm

Bayesian Vector Error Correction Models (VECM) in Python, built on [PyMC](https://www.pymc.io/).

A VECM captures the long-run relationships between trending series and the short-run dynamics that pull them back when they drift apart. This package fits it in full Bayes: posterior uncertainty over every parameter, optional horseshoe shrinkage on the lags, and impulse responses with credible bands.

The main use is **brand marketing**. Brand metrics and sales move together over years, so spend that lifts awareness can lift sales for good. A media mix model whose effects fade within weeks misses that long-run tail.

**Built on Peter Cain's work.** The method follows Cain (2022): [Modelling short- and long-term marketing effects in the consumer purchase journey](https://doi.org/10.1016/j.ijresmar.2021.06.006) in the *International Journal of Research in Marketing* 39(1). Two ideas come from that paper. Brand metrics are tied to base sales by cointegration. Same-week links between them are estimated by instruments.

Cain (2025), [Long-term advertising effects: The Adstock illusion](https://ideas.repec.org/a/aza/ama000/y2025v11i1p23-42.html), explains why a long adstock cannot stand in for brand building. The [README](https://github.com/raz1470/bayesian_vecm#built-on-peter-cains-work) lists his other papers and what the package takes from each.

```bash
pip install bayesian_vecm
```

```python
from bayesian_vecm import BayesianVECM, simulate_brand_data

data = simulate_brand_data()
model = BayesianVECM(k_ar_diff=1, coint_rank=2, deterministic="co")
model.fit(data.endog, exog=data.exog)
irf = model.irf(steps=52, method="girf")
```

## Where to start

- **[The walkthrough](https://raz1470.github.io/bayesian_vecm/walkthrough.html)**: how long brand lasts. The method, the tests and the results on simulated data.
- **[The user guide](user_guide.md)**: the practitioner notebook and pages on choosing the rank and lags, the Granger pretest and future work.
- **[The practitioner guide](guide/practitioner_walkthrough.ipynb)**: the full workflow in a notebook, from data to return per channel.
- **[API reference](api/index.md)**: every public function and class.
- **[How it was built](how_it_was_built.md)**: one notebook per feature, in the order they were built.

```{toctree}
:hidden:

user_guide
api/index
model_details
how_it_was_built
```
