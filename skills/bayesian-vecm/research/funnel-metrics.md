# Is this funnel metric usable?

Use when the user asks whether a metric can go into the model. Examples are share of search, ad recall and branded search volume. Web visits, social mentions, NPS and tracker scores come up too. The output is a scorecard (end of this file) with a verdict and reasons.

Work through the five steps in order. Do the research with web search and cite every source. Run the data checks on the user's own data.

## 1. What is already known

Search for evidence that the metric moves with or ahead of sales or market share. Prefer the IPA and WARC, the Ehrenberg-Bass Institute and peer-reviewed marketing journals (IJRM, Journal of Marketing Research, Journal of Advertising Research). Named practitioners who publish their method are fine too. For each source record the claim and where it was tested (categories, countries, years). Also record whether it is correlation or something stronger.

Starting points:

- **Share of search.** Les Binet presented it at the IPA EffWorks Global 2020 conference. Share of search correlated with market share in cars, energy and mobile phone handsets. It also led market share, by up to a year for cars. The gap between share of search and market share was a useful early signal. He called it "by no means a silver bullet": price and other factors affect conversion and not all searches are positive. ([IPA](https://ipa.co.uk/news/binet-presents-fast-cheap-predictive-share-of-search-metric/))
  An IPA think tank led by James Hankins then covered 30 case studies in 12 categories and 7 countries. It found share of search was about 83% of share of market on average, with the ratio varying by category. Share of search moved with share of market, and mass-reach media such as TV moved share of search more than targeted channels. The findings are correlational. They computed it from Google Trends as a brand's searches over the searches for all brands in the competitor set, over a rolling 6 to 12 months. ([Marketing Week](https://www.marketingweek.com/share-of-search-market-share/))
- **Tracker metrics (awareness, consideration, preference).** Cain (2022) links awareness, consideration and base sales in a cointegrated model. *Modelling short- and long-term marketing effects in the consumer purchase journey*, IJRM 39, 96-116.
- **Long-term brand effects in general.** Binet and Field (2013), *The Long and the Short of It*, IPA.

Do not rely on a single vendor case study. Say how strong the evidence is: "several independent studies across categories", "one study, one category" or "no published evidence found".

## 2. What the metric measures

Answer in plain words:

- **Where in the funnel?** Salience (share of search, unprompted awareness), consideration, intent or behaviour (web visits, branded search clicks). It decides which metrics it should be tied to.
- **Share or level?** A share (share of search, share of voice) is relative to competitors. Sales are usually a level. A share can be flat while the whole category grows, so it may tie to **market share** rather than sales. If the target is sales, consider the brand's own search volume instead of its share.
- **What else moves it?** Search moves with news, PR crises and product recalls. A brand name that is also a common word ("Apple", "Shell", "Dove") adds noise. Paid search and SEO changes matter, and so do navigational searches by existing customers. Tracker scores move with question wording and with changes of panel supplier or sample.
- **Is it a driver or an outcome?** Almost every funnel metric responds to the same shocks as sales. Treat it as a brand metric (`endog`), not a driver.

## 3. Data quality

| Check | Question | Problem signs |
|---|---|---|
| Frequency | Weekly available? | Monthly only blurs links (see `reference/data.md`). |
| History | At least four years? | Shorter weakens the rank test. |
| Construction | How exactly is it computed? | Rolling averages smooth away the signal. Ask for the raw weekly value. |
| Breaks | Any change in method, supplier or definition? | Steps in the series. |
| Noise | Sample size per wave for surveys? | Small samples (a few hundred a week) make a noisy series. Ask for the base size. |

**Google Trends specifics** (for share of search and search volume):

- Each request is scaled 0 to 100 within itself, so brands must be compared in the same request or chained through a common anchor term.
- A request compares at most five terms.
- The granularity depends on the window. Long windows come back monthly, so pull several overlapping windows and splice them if weekly data over many years is needed.
- Values are sampled and can differ slightly between pulls. Pull twice and compare, and keep the raw pulls with their dates.
- Choose search terms or topics deliberately: the brand name alone, misspellings, the "topic" entity. Write the choice down.
- For the model, use the **weekly unsmoothed** share. The 6 to 12 month rolling window in the IPA work suits dashboards, not a time-series model.

## 4. Statistical checks on the user's data

Run these and explain each result in a sentence.

```python
from bayesian_vecm import check_stationarity, granger_pretest, select_coint_rank

candidate = np.log(raw["share_of_search"])
panel = pd.concat([endog, candidate.rename("share_of_search")], axis=1)

print(check_stationarity(panel))  # does it trend?
print(select_coint_rank(panel, k_ar_diff=1, det_order=0))  # does it add a tie?
print(granger_pretest(panel, exog))  # does it lead or follow?
```

| Question | Good sign | Bad sign |
|---|---|---|
| Does it trend? | Classed as trending, like sales. | Stationary: it cannot share a long-run tie. Common for shares. |
| Does it move with sales or the other metrics in the long run? | The rank rises by one when it is added. | No new tie: it wanders on its own. |
| Does it lead? | Past values help predict sales or consideration, not the other way round. | Only sales predict it: it follows. |
| Do the drivers move it? | Brand media shows up in the drivers table for this metric. | No driver moves it: it cannot carry a media effect. |
| Can it be instrumented? | Some drivers act on it and not on the next stage. | No candidate exclusions: no same-week link can be estimated. |

If the user already has the core model, also fit it with the candidate added and compare the return per £ and the hold-out forecast. A metric that adds nothing to either is not worth the complexity.

## 5. Verdict

One of:

- **Use as a brand metric** (`endog`): it trends and ties to the others. It leads or follows sensibly and the data quality holds.
- **Use as a check only:** plausible but fails a test (no tie, too short, too noisy). Compare its path with the model's story without putting it in the model.
- **Do not use:** it does not measure what it claims, or the data quality is too poor.
- **Use instead of another metric:** for example, share of search as a weekly stand-in for a monthly tracker. It needs a tie to the metric it replaces over the overlap.

## The scorecard

Write this as `funnel-metric-<name>.md` in the project folder.

```markdown
# <Metric>: is it usable?

**Verdict:** <use / check only / do not use / use instead of X>, because <one sentence>.

## Evidence from outside
- <claim>, <categories, years>, <correlational or stronger>. Source: <link>
- Strength of evidence: <several independent studies / one study / none found>

## What it measures
- Funnel stage: <...>. Share or level: <...>.
- Other things that move it: <...>.

## Data quality
| Check | Result |
|---|---|
| Frequency | |
| History | |
| Construction | |
| Breaks | |
| Noise | |

## Tests on our data
| Question | Result | Meaning |
|---|---|---|
| Trends? | | |
| Long-run tie? | | |
| Leads or follows? | | |
| Moved by drivers? | | |
| Instruments available? | | |

## Open questions for the data owner
- <...>
```
