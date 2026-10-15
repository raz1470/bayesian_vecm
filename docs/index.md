# bayesian_vecm

Bayesian Vector Error Correction Models (VECM) in Python, built on [PyMC](https://www.pymc.io/).

A VECM captures the long-run relationships between trending series and the short-run dynamics that pull them back when they drift apart. This package fits it in full Bayes: posterior uncertainty over every parameter, optional horseshoe shrinkage on the lags, and impulse responses with credible bands.

The main use is **brand marketing**. Brand metrics and sales move together over years, so spend that lifts awareness can lift sales for good. A media mix model whose effects fade within weeks misses that long-run tail.

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

```{toctree}
:hidden:

user_guide
api/index
model_details
```
