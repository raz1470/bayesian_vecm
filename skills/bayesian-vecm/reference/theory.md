# Theory

Two versions. Give a new user the short one first. Offer the full one when they want to understand why.

## The short version (read this to the user, in your own words)

**The problem.** A media mix model says a TV burst lifts sales for a few weeks and then fades. Brand people say brand builds for years. Both can be right: the MMM measures the short-term sales spike and misses the slow climb.

**The idea.** Sales, consideration and awareness each grow over years. They do not grow independently. They move together like a dog on a lead: they wander but never far apart. If awareness rises for good, consideration and then sales follow it up to a new level and stay there.

**The model.** A VECM learns three things from the data:

1. **The ties (β).** Which series move together in the long run and in what proportion.
2. **Who follows whom (α).** When a tie is stretched, which series moves to close the gap and how fast.
3. **The short run (Γ and B).** How last week's changes and this week's spend move this week's growth.

Put together, these trace what happens after a campaign. Spend lifts awareness this week. The tie pulls consideration up and the next tie pulls sales up. Because the ties hold, the sales lift does not fade. That is the long-term effect.

**Why Bayesian.** Every answer (the return per £, the shape of the response) comes with a range that reflects how sure the data allows us to be. Analysts get "£2.20 per £1, very likely between £1.60 and £2.80" instead of a single number.

**What it cannot do.** It measures what helps predict what, under assumptions we write down. It does not prove cause on its own. It is only as good as the series fed into it.

## The full version

### Why not ordinary regression

Regressing one trending series on another finds strong relationships between series that have nothing to do with each other (spurious regression). Two random walks regressed on each other give a "significant" slope most of the time. Differencing removes the trend but also throws away the long-run information. Cointegration keeps both.

### Cointegration

Series are cointegrated when each trends but some combination of them does not. For brand data, the funnel suggests one underlying trend (demand for the brand) that all the metrics share. With K series sharing one trend there are K - 1 ties. The test is `select_coint_rank`. Five years of weekly data find the right number about 80% of the time. Three years find it under half the time.

### The equation

```
Δy_t = α β' y_{t-1} + Γ Δy_{t-1} + B x_t + constant + u_t
```

- `y` is the log levels of the brand metrics and sales. `Δy` is their weekly change.
- `β' y_{t-1}` is how far each tie was stretched last week.
- `α` turns that stretch into this week's correction.
- `Γ` carries short-run momentum. `B` is the same-week effect of the drivers `x`.
- `u` is the shock: the part nothing explains. The shocks of different series are correlated, because news, competitors and seasons hit them together.

### Who leads

If a series' row of `α` is zero it does not respond to the ties: it is **weakly exogenous** and leads. In a healthy brand funnel awareness leads, consideration follows awareness and sales follow consideration. `model.granger_causality()` reports this as the weak exogeneity table.

### The long-term effect

A one-off lift in a leading series moves the whole system to a new level. The impulse response (`model.irf()`) shows the path. For drivers, `model.exog_irf()` gives the path of every series after one more unit of spend for one week. Summed over a window and multiplied by recent sales, it is the return per £.

### Shocks and ordering

A shock to awareness usually comes with shocks to the others in the same week. To trace "an awareness shock", the model must decide how to split that shared movement. A Cholesky impulse response assumes an order (awareness first, say). A generalised impulse response (GIRF, the default) lets each shock carry its usual companions. With awareness first in a funnel, the two agree.

### Same-week links

Does awareness move consideration within the same week? The shared shocks make a regression say yes even when the answer is no (about 0.3 on simulated data with no link). The fix uses a driver that moves awareness but reaches consideration only through awareness, such as brand TV. The ratio of its effect on consideration to its effect on awareness is the link. This is instrumental variables. It rests on the exclusion, which is an assumption about the business. `contemporaneous=` and `excluded=` estimate it in the model.

### Carryover and diminishing returns

- **Adstock:** spend keeps working, with a decay per week. The model estimates one decay and one total effect per channel. Without it a same-week model misses about half of each channel's effect.
- **Saturation:** a logistic curve per channel. Without it a straight line is fine near current spend and badly wrong for big moves: on simulated data, switching brand off was understated by 60% and doubling overstated by 120%.

### What Bayesian adds

- Intervals for every derived quantity (returns, curves, forecasts) without bootstraps.
- Intervals that hold up on short samples. On 60 periods the bootstrap intervals for the return per £ contained the truth in 70% to 87% of data sets. The posterior intervals held 87% or more.
- On five years of clean data the point estimates match `statsmodels` (guide 02 in the repo).

### Further reading in the repo

- Guide 01: the full workflow on simulated data.
- Guide 02: comparison with `statsmodels`.
- Guide 03: carryover and saturation.
- Guide 04: same-week links step by step.
- Guide 05: one model with everything and the response curve in £.
- `docs/choosing_rank_and_lags.md` and `docs/granger_pretest.md`.
- Cain (2022), *Modelling short- and long-term marketing effects in the consumer purchase journey*, IJRM 39, 96-116.
- Binet and Field (2013), *The Long and the Short of It*, IPA.
