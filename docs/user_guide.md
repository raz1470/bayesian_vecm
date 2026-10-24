# User guide

Start with the walkthrough for the idea and the results, then the practitioner guide to run the workflow on your own data. The other pages go deeper into single steps.

| Page | What it covers |
|---|---|
| [Walkthrough](https://raz1470.github.io/bayesian_vecm/walkthrough.html) | How long brand lasts: the method, the tests and the results on simulated brand data |
| [Practitioner guide](guide/practitioner_walkthrough.ipynb) | End-to-end notebook: data, tests, fit, impulse responses, counterfactual ROI, return per channel |
| [Comparison with statsmodels](guide/statsmodels_comparison.ipynb) | Same estimates as statsmodels' VECM, how often each model's intervals contain the truth, and the return per £ with intervals |
| [Carryover and saturation](guide/carryover_and_saturation.ipynb) | `adstock=` and `saturation=` against a linear same-week fit: the effect by week, response curves and big budget moves |
| [Same-week links](guide/same_week_links.ipynb) | Whether awareness moves consideration within the week: why regression fails, the instruments, the test and the model, and when it goes wrong |
| [Putting it together](guide/putting_it_together.ipynb) | One model with adstock, saturation and a same-week link: the response curve for brand spend in £ and where one more £ stops paying back |
| [Agent skill](https://github.com/raz1470/bayesian_vecm/tree/main/skills) | Markdown instructions for an AI agent to guide an analyst through the whole analysis, from data to write-up |
| [Choosing the rank and the lags](choosing_rank_and_lags.md) | Which tests to use and how often they find the truth on simulated data |
| [The Granger pretest](granger_pretest.md) | What helps predict what before fitting: drivers, delays, links and feedback |
| [Future work](future_work.md) | What was tested and left out of the first release, and what to check first |

```{toctree}
:hidden:

guide/practitioner_walkthrough
guide/statsmodels_comparison
guide/carryover_and_saturation
guide/same_week_links
guide/putting_it_together
choosing_rank_and_lags
granger_pretest
future_work
```
