# Pitfalls

Check each headline result against this list before the write-up. The numbers come from the package's validation runs on simulated data with a known truth (`tools/` in the repo).

| Pitfall | What happens | How to spot it | What to do |
|---|---|---|---|
| **Budget follows performance** | Spend responds to sales or brand health, so its effect is biased. | Pretest feedback table. Ask how budgets are set. | Record it. Use only the periods where budgets were planned ahead, or treat the result as an upper or lower bound. |
| **A missing channel** | A channel that moves with the others is left out, and its effect lands on them. Monthly brand spend came out 40% too high with PR and sponsorship left out. | A channel the business runs that is not in the data. | Get the data, or name the bias in the write-up. |
| **Smoothed base sales** | Removes the brand effect and the noise. The model looks surer than it is. | `base-sales.md` smoothness check. | Rebuild with the UCM recipe or use total sales. |
| **Short-term media left in base sales** | Counted twice. | Spend predicts base sales in the same week. | Take the short-term contributions out consistently. |
| **Monthly data** | α no longer reads as weekly speed. Same-month links include lagged weekly effects (about 0.32 with no weekly link). Instruments weaken. | Frequency. | Prefer weekly. If monthly, use `exog_lags=1` and say what the link means. |
| **Too little data** | The rank test is right under half the time on three years of weekly data. | Fewer than 4 years. | Fit the neighbouring rank too and report both. |
| **Extrapolation** | Switch-off and doubling rest on the assumed curve shape. One simulated data set overstated the switch-off loss by 120%. | Spend levels outside the range in the data. | Report the range, show where the data ends, avoid single numbers. |
| **Wrong exclusion** | A same-week link of 0.2 came out at 0.33, and Sargan caught it 10% of the time. | Only the business argument catches most of these. | Exclude only what the user can defend. Drop the link if in doubt. |
| **Weak instruments** | The link repeats the prior. | First-stage F below 10, `fit()` warning. | Leave the link out. |
| **Reading prediction as cause** | "Granger causes" heard as "causes". | Wording. | Say "helps predict". State the assumptions for any causal claim. |
| **Evidence score as effect size** | An evidence of 1.00 on a tiny effect. | Wording. | Use the return per £ or the response for size. |
| **Picking choices for the ROI** | Rank, lags or exclusions chosen because they give the best number. | Choices made after seeing the ROI. | Decide them first, in the decision log. Report sensitivity. |
| **Measurement breaks** | A tracker supplier change looks like a brand effect. | Ask. Look for steps in the series. | Dummy, splice or shorten the sample. |
| **A saturation prior on linear data** | With no real saturation the default prior still bends the curve. Far-away effects came out up to 30% too large. | Little spend variation. | Compare with a linear fit before reporting far-away effects. |
| **Unhandled seasonality** | Residual autocorrelation and false links. | `test_whiteness`, a seasonal pattern in residuals. | Fourier terms as drivers, or deseasonalise every series the same way. |
