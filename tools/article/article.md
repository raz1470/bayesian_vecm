<!-- Source text for docs/_extra/walkthrough.html. Build with tools/article/build.py. -->
<!-- Charts: a line [[chart NAME | ALT | CAPTION]] places the chart in tools/article/data/NAME.json. -->

# How to Measure Long-Term Brand Effects When Your MMM Can't

*A Bayesian VECM ties brand metrics to base sales. Simulations test how much data it needs, how big the effect must be, and what happens when lifts fade.*

## 1. Introduction

Brand teams and marketing mix models disagree about time. Brand teams say a campaign builds the brand for years. A marketing mix model (MMM) usually says its effect is gone within a couple of months.

Both can be right. An MMM measures the short-term lift in sales that follows a burst of spend. What it struggles to see is the slow climb: more people who know the brand, more who would consider it, and a higher level of sales that stays.

Les Binet and Peter Field made this case with the IPA databank in [*The Long and the Short of It*](https://ipa.co.uk/knowledge/effectiveness-research-analysis/les-binet-peter-field) (2013). Measuring the effect for one brand from its own data is harder, but it can be done. In [*Modelling short- and long-term marketing effects in the consumer purchase journey*](https://doi.org/10.1016/j.ijresmar.2021.06.006) (2022), Peter Cain ties brand metrics to base sales with cointegration, so a lasting lift in awareness carries through to sales. More recently, [PyMC-Marketing](https://www.pymc-marketing.io/en/stable/notebooks/mmm/mmm_brand_metrics_long_term.html) has taken a similar route: an MMM with a baseline that moves over time, and a Bayesian VAR that links that baseline to awareness and consideration. [Recast](https://getrecast.com/research-update-long-term-brand-effects/) does something similar on top of its MMM: brand spend moves awareness, awareness moves consideration, and consideration moves base sales, with the surveys treated as noisy readings.

The chart shows three ways a campaign can play out:

- **Adstock.** The lift fades back to zero within weeks. This is what an MMM assumes.
- **A lasting lift.** Awareness moves to a new level and sales follow it up for good. This is what the method in Part 1 assumes.
- **A fading lift.** Awareness rises and then drains slowly back towards where it was heading. For most brands this is probably the more realistic case.

[[chart fig01_three_ways | Three sketches of sales after a 13-week burst of spend: adstock fading to zero, a lasting lift, a fading lift | ]]

The gap between these is the whole question. Over five years a lasting lift earns many times what an MMM credits it with. A fading lift earns less, but still more than adstock suggests. Telling the two apart from the data is hard.

This article has two aims:

1. **Explain the method**, a vector error correction model (VECM) in which every lift lasts (Part 1).
2. **Test it on what a real brand brings** (Part 2): less data, base sales handed over from an MMM, and lifts that fade. A last section looks at brands with no tracker.

We illustrate each step on simulated data, so we always know the right answer.

## 2. The problem with adstock

An MMM explains weekly sales with spend, price, seasonality and a baseline. Spend enters through adstock: each week keeps a share of the previous week's effect, so a burst of TV fades out over a few weeks. That suits the question an MMM is built for: how did this quarter's spend move this quarter's sales?

It doesn't suit brand. The usual fix is a second, much longer adstock for brand spend: a short decay for activation and a long one for brand building. Cain (2025) refers to this practice as [the adstock illusion](https://ideas.repec.org/a/aza/ama000/y2025v11i1p23-42.html).

The problem is the trend. Sales drift over years. A slowly decaying spend series drifts too. Put one against the other and they line up whether or not brand drives sales. The long adstock ends up fitting the drift in the baseline, and it tells us nothing about how brand building works.

The chart shows it. The baseline is a slow trend we built with no link to spend at all. The long adstock of brand spend climbs alongside it, and a regression credits brand with the climb.

[[chart fig02_adstock_trend | Five years of flighted brand spend, a baseline trend built with no link to spend, and the long adstock of spend fitted to it | ]]

Cain (2022) uses an alternative, the VECM, which separates the two. Short-term activation is the week-to-week movement around base sales. Long-term brand building is the base itself: the slow level of sales, which moves with awareness and consideration.

That needs three things an MMM leaves out:

- **The brand metrics**, alongside sales.
- **The trends treated as information**, not something to remove.
- **A way for a change in one series to carry through to the others.**

## 3. The brand funnel

A brand works through a funnel. Marketing makes people aware of the brand. Some of those people come to consider it, and some of those buy. Each step feeds the next, so a lasting rise at the top should show up lower down.

Brand trackers measure the middle of the funnel:

- **Awareness** is the share of people who know the brand.
- **Consideration** is the share who would think about buying it.

Tracker data has its own problems. Consideration can rise because more people are in the market, not because of the brand; a measure of category demand as a driver helps separate the two. Panels often over-represent existing customers. A constant bias cancels out, but one that shifts as the customer base grows does not: growing sales would then lift measured awareness, and the model could read sales as leading. Small samples add noise on top.

Together with sales these form the system. Sales here means base sales: sales without the short-term lift from performance media and promotions.

Around the system sit the drivers. Brand spend, PR and sponsorship push on awareness. Interest rates and consumer confidence push on consideration from outside. The drivers move the brand, but the brand does not move them.

<figure class="fig diagram"><div class="fig-h">Marketing moves awareness. Consideration and sales follow.</div><div class="system"><div class="sys-col"><div class="node drv">Brand spend</div><div class="node drv">PR reach</div><div class="node drv">Sponsorship reach</div></div><div class="arrow">→</div><div class="sys-col"><div class="node aw">Awareness<small>leads the system</small></div></div><div class="arrow">→</div><div class="sys-col macro"><div class="node drv">Interest rate</div><div class="varrow">↓</div><div class="node co">Consideration<small>follows awareness</small></div><div class="varrow">↑</div><div class="node drv">Consumer confidence</div></div><div class="arrow">→</div><div class="sys-col"><div class="node sa">Base sales<small>follows consideration</small></div></div></div><figcaption class="fig-cap">The simulated system. Drivers act from outside. The three brand series trend and are tied together.</figcaption></figure>

To show each step of the method we simulated five years of weekly data for a brand like this. Because we built it, we know the right answer at every step. This is what we built in:

| What we built in | Setting |
|---|---|
| Funnel order | Drivers → awareness → consideration → sales |
| How far a lift carries | A lasting 1% rise in awareness ends up as about 0.9% in consideration and about 1.1% in sales |
| Who catches up | Sales close about a quarter of any gap to consideration each week. Consideration closes about a fifth of its gap to awareness. Awareness never catches up to anything: it leads |
| Weekly surprises | News and competitors hit all three series in the same week |
| Does a lift last? | Yes, for good |

The rest of the article shows how the method recovers each of these from the data alone.

## 4. Why trending data fools regression

Most analysts would start with a regression. Regression assumes each series is stationary: it moves around a stable average, and after a shock it comes back.

Brand series don't behave like that. Each week starts from where the last one ended, and nothing pulls the series back to a fixed level. Awareness that rises after a strong year stays risen until something else moves it. A series like this is non-stationary. Economists say it has a unit root.

Regress one trending series on another and the two will often look related when they aren't. This is a spurious regression, first shown by [Granger and Newbold (1974)](https://doi.org/10.1016/0304-4076%2874%2990034-7).

The chart shows two random walks that we built completely independently. Regress one on the other and the slope comes out highly significant. The leftover gap between them wanders as much as the series do.

[[chart fig03_spurious | Two unrelated random walks, and what is left over after regressing one on the other | ]]

This isn't bad luck. Across 1,000 pairs of unrelated random walks of five years each, 84% gave a slope significant at the 5% level.

The usual fix is to difference the data: model the weekly change instead of the level. Weekly changes are stationary, so the regression is safe. But differencing throws away the levels, and the levels hold the long-run story. A campaign that lifts awareness for good shows up only in the level.

A series that is stationary as it stands is I(0). One that becomes stationary once differenced is I(1). Trending brand series are usually I(1), and the method in this article is built for them.

The Augmented Dickey-Fuller (ADF) test tells the two apart. On the simulated data, sales, awareness, consideration and the interest rate are I(1). Brand spend, PR, sponsorship and consumer confidence are I(0).

So we need a method that keeps the levels without being fooled by them. That method rests on cointegration.

## 5. Cointegration: the drunk and her dog

Some trending series are tied together. Each one wanders, but a weighted gap between them does not: it keeps coming back to a stable level. Series like these are cointegrated. The gap is the cointegrating relationship, and its stable level is the long-run equilibrium. The idea comes from [Engle and Granger (1987)](https://doi.org/10.2307/1913236).

Michael Murray gave it its best-known picture in "A Drunk and Her Dog" (1994). A drunk leaves the pub and wanders home. Her path is a random walk. So is her dog's. Watch either alone and you can't say where it will be in ten minutes. But when they drift too far apart, she calls and the dog comes back towards her. Both paths wander, but the distance between them does not.

In the simulated data one such tie runs between consideration and sales. Each series trends, but the gap between log sales and about 1.25 times log consideration stays close to its average for all five years.

[[chart fig04_cointegration | Base sales and 1.25 times consideration over five years, and the gap between them | Left: both series as % from their average, in logs.]]

We can test for this with the ADF test from section 4: take the gap and ask whether it is stationary. For the two unrelated random walks in section 4 it is not. For sales and consideration it clearly is. This two-step check is the Engle-Granger test.

A second tie runs between awareness and consideration. Two ties among three series leave one shared trend. Here that trend belongs to awareness, and it is why a lasting lift in awareness carries through to sales.

The number of ties is the cointegration rank. With more than two series, Johansen's (1991) trace test finds it by asking in turn: none, at most one, at most two? It stops at the first answer it can't reject.

## 6. The VECM

A vector error correction model (VECM) describes the weekly change in each brand series and splits it into parts. Each part has a plain reading for the brand.

<figure class="fig eqfig"><div class="fig-h">The VECM, term by term</div><div class="eqn"><span class="term"><span class="sym">Δ<b>y</b><sub>t</sub></span><span class="lab">Weekly change</span></span><span class="op">=</span><span class="term"><span class="sym"><b>αβ</b>′<b>y</b><sub>t−1</sub></span><span class="lab">Error correction</span></span><span class="op">+</span><span class="term"><span class="sym"><b>Γ</b>Δ<b>y</b><sub>t−1</sub></span><span class="lab">Momentum</span></span><span class="op">+</span><span class="term"><span class="sym"><b>Bx</b><sub>t</sub></span><span class="lab">Drivers</span></span><span class="op">+</span><span class="term"><span class="sym"><b>c</b></span><span class="lab">Growth</span></span><span class="op">+</span><span class="term"><span class="sym"><b>ε</b><sub>t</sub></span><span class="lab">Shock</span></span></div><figcaption class="fig-cap">y<sub>t</sub> holds the logs of base sales, awareness and consideration. x<sub>t</sub> holds the drivers.</figcaption></figure>

The left side is this week's change in log sales, log awareness and log consideration. Changes in logs read as percentage changes. Bold letters stand for all three series at once, so this is really three equations fitted together.

The parts, in order:

- **Error correction (αβ′y).** β′y<sub>t−1</sub> measures how far each tie from section 5 was stretched last week, for example how far sales sat from 1.25 times consideration. α turns that stretch into this week's move. This is what keeps the series together.
- **Momentum (ΓΔy).** Last week's changes carry into this week. If awareness jumped last week, some of that jump shows up in consideration now. These effects fade within weeks.
- **Drivers (Bx).** This week's brand spend, PR and other drivers push on each series. Drivers that move around a stable level enter as they are. A trending driver, like the interest rate, enters as its weekly change, because the left side is a weekly change and doesn't trend.
- **Growth (c).** A constant that lets each series grow at a steady rate.
- **Shock (ε).** The part of this week that nothing in the model predicts. The three shocks are correlated, because news and competitors hit the series in the same week.

**Who closes the gap.** When the drunk and her dog drift apart, someone has to move, and α says who. A series with a large α follows: it moves to close a stretched tie. A series with α near zero on every tie leads.

In the simulated data, sales close about a quarter of their gap to consideration each week, and consideration about a fifth of its gap to awareness. Awareness closes neither, and the fitted model agrees.

So a lift in awareness is never pulled back. Consideration and sales rise to meet it and stay there, which is why a campaign that moves awareness keeps paying. A lift in sales alone is pulled back within weeks.

Hannan-Quinn picks the number of momentum lags and the rank test the number of ties: one week and two ties here. We fit the model by Bayesian sampling, which gives the intervals used from section 7 on. Intervals are 89% unless stated.

## 7. From a campaign to a response curve

The ties, momentum and drivers explain how the brand works. A budget decision needs what a campaign earns, so we follow one campaign through the system.

**Spend reaches awareness.** Take a 13-week burst of brand spend. Before it moves awareness, it goes through the two steps familiar from MMM:

- **Carryover.** Each week keeps a share of the last week's spend, so the push on awareness builds and then tails off.
- **Diminishing returns.** A saturation curve means each extra pound moves awareness a little less than the one before.

Both sit inside the drivers. The ties, momentum and growth are untouched.

**Awareness reaches consideration, partly in the same week.** Momentum carries last week's awareness into this week's consideration. A same-week effect is harder to measure. Awareness and consideration share their weekly shocks, so a plain regression of one on the other finds a same-week link even when there is none.

Following [Cain (2022)](https://doi.org/10.1016/j.ijresmar.2021.06.006), we use an instrumental variable. Brand spend moves awareness but reaches consideration only through awareness. So its effect on consideration, divided by its effect on awareness, is the same-week link. That assumption is a business judgement, not something the data can settle.

**Consideration reaches sales.** The ties pull consideration and sales up to meet the new level of awareness, and hold them there.

**From sales to a response curve.** Summing the extra sales each week gives what the campaign earned. The ties carry the lift forward, so the total keeps growing with the window, and we report it at one year and at five. Repeating this for campaigns from nothing up to half as much again as current spend gives a response curve.

[[chart fig08a_lasting_curve | Response curves for a 13-week campaign within one and five years, for a brand whose lifts last, fitted with the VECM | Extra sales against no campaign. 1 is the true first-year return at current spend.]]

Within a year a campaign at current spend earns 0.8 on the chart's scale, against a true 1.0. Within five years it earns 5.2, against a true 6.2. Both intervals hold the truth. Because every lift lasts, five years earn about six times the first year.

This is the method on clean data: five years of weekly figures, a true base sales series and drivers set from outside the brand. Real brands rarely have all three.

## 8. How much data is enough?

Part 1 ran on five years of weekly data. Most brands have less: a tracker that started three years ago, or one that reports monthly. So we cut the simulated data four ways: weekly or monthly, over five years or three. Monthly over three years leaves 36 points.

Marketing's effect matters too. In the simulated data, marketing drives about half the week-to-week movement in awareness. That is a large effect. So we also halved and quartered it, which leaves marketing driving about a fifth and under a tenth. On 20 versions of each brand, each with its own random shocks, we ran the rank test and measured the 5-year return on brand spend.

[[chart fig_data_rank | How often the rank test found two ties, for weekly and monthly data over three and five years | Pooled over the three effect sizes: 60 data sets per bar.]]

**Years find the ties.** A tie is a gap that keeps coming back, and that takes time to see. With five years the rank test found both ties about 85% of the time; with three, a third to two in five. Effect size made little difference, because the ties come from the shared trend.

[[chart fig_data_size | The 5-year return as a share of the truth for weekly and monthly data over three and five years, for large, medium and small marketing effects | The median across 20 data sets of each fit's median and 89% interval, divided by the true return.]]

**Data points set the precision.** The intervals held the truth at least nine times in ten throughout; what changed was their width. With the large effect, five years of weekly data put the return within about a quarter of the truth. Three years of monthly data gave anything from almost nothing to twice the truth.

**Smaller effects need more data.** Relative to the truth, the intervals roughly doubled each time the effect halved. With a small effect, even five years of weekly data gave an interval reaching zero in a quarter of fits. On the shortest data the estimate also shrank towards zero, as the prior pulled a weak signal back.

A Bayesian fit at least shows this with a wide interval. A classical fit on 60 weeks claimed 90% intervals for the ties that held the truth only about 60% of the time.

## 9. Base sales from an MMM

Nobody observes base sales. In practice they come from an MMM, and there is more than one way to take them. We built total sales as the base plus a short-term layer: paid search and social with carryover and diminishing returns, brand spend's same-week hit, promotions, seasonality and noise. Then we gave the VECM four versions of base sales, on 20 data sets each:

- **The true base**, which only a simulation has.
- **Minus MMM**: total sales minus the MMM's short-term contributions, week by week.
- **Total sales as they are**, with the short-term drivers in the VECM's sales equation instead.
- **The MMM's trend**: the smooth baseline many MMMs report.

[[chart fig_base_sales | The long-run lift from brand spend and the speed at which sales catch up, for four versions of base sales | Twenty simulated brands. The median across data sets of each fit's median and 90% interval, divided by the truth.]]

**The long run survives a rough base.** The first three versions put the long-run lift from brand spend within 2% of the truth. The long run is read from the level base sales settle at, and short-term noise around that level averages out.

**The speed doesn't.** How fast sales close their gap to consideration depends on every wiggle. Minus MMM made sales about 13% too fast. Total sales as they are made them 1.5 times too fast. The MMM's trend smooths away the wiggles altogether, so sales looked twenty times too slow, with a narrow interval. In 12 of 20 data sets it also made sales look like the leader, and it put the long run 17% high.

Of the versions a real team has, total sales minus the MMM's short-term contributions did best. Its long run held up; its speed ran about 13% fast.

## 10. Letting lifts fade

So far nothing pulls a lift back: once a campaign moves awareness, the ties carry the whole lift to sales and keep it there. That is a strong assumption. People forget brands, and rivals answer campaigns with their own.

So we added a term that lets a lift fade, which we call the drain. Each week part of the lift from past spend drains away, set by two numbers: the share of each lift that eventually drains, and the half-life. With a share of zero the model is the VECM from Part 1. Unlike a long adstock, the drain doesn't fit the trend in sales, because the ties handle the trends first.

We simulated two more versions of the brand, where half or all of each lift drains with a one-year half-life, and fitted all three with the drain.

[[chart fig08_response_curves | Response curves for a 13-week campaign within one and five years, for a brand whose lifts last, half fade or all fade | Extra sales against no campaign. 1 is the true first-year return at current spend if lifts last. Each brand fitted with the drain.]]

Over the first year the three look much alike, and the intervals hold the truth for all three. Over five years the true returns are 6.2, 4.3 and 2.1. The model gives 4.0, 3.7 and 3.2, pulling all three towards the middle.

Leaving the drain out is worse. On brands whose whole lift fades, with half-lives from six months to two years, the VECM without it overstated the 5-year return by 2 to 6 times, and its intervals missed the truth in all 15 fits.

## 11. Brands without a tracker: share of search

Everything so far assumes a brand tracker with years of weekly data. Many brands have nothing like it, and others have one they can't trust. Share of search is a cheap stand-in: a brand's searches as a share of all searches for brands in its category. Les Binet [introduced it at the IPA in 2020](https://ipa.co.uk/news/binet-presents-fast-cheap-predictive-share-of-search-metric/). In cars, energy and mobile phones it tracked share of market and moved ahead of it, by up to a year for cars.

It suits this method:

- **It has the history.** Google Trends gives weekly data for up to five years and monthly data back to 2004.
- **Google's scaling cancels out.** Trends rescales every request to run from 0 to 100. A share divides that out, as long as every brand comes from the same request.
- **It trends**, as awareness does, so it can be tied to base sales.

<figure class="fig diagram"><div class="fig-h">A brand system without a tracker</div><div class="system"><div class="sys-col"><div class="node drv">Brand spend</div><div class="node drv">PR reach</div><div class="node drv">Sponsorship reach</div></div><div class="arrow">→</div><div class="sys-col"><div class="node aw">Share of search<small>leads, if it works</small></div></div><div class="arrow">↔</div><div class="sys-col macro"><div class="node drv">Category searches</div><div class="varrow">↓</div><div class="node sa">Base sales<small>follows share of search</small></div></div></div><figcaption class="fig-cap">A sketch, not a fitted model. Two brand series with at most one tie. Category searches stand in for how many people are in the market.</figcaption></figure>

Two questions decide whether it works:

- **Does it lead?** Some branded searches come from existing customers looking for the website. That part follows sales. The VECM can test this: if share of search closes part of its gap to sales each week, it is a follower.
- **Brand or activation?** Binet found advertising moves share of search twice: a large short-term effect that fades fast, and a smaller one that builds up. The short-term part is activation inside a series meant to measure the brand, the base sales problem of section 9 in a new place.

## 12. Summary

A marketing mix model sees the short-term lift that follows a campaign and misses the slow climb in base sales. A longer adstock doesn't fix this: it fits the trend in sales whether or not brand caused it.

- **The VECM** ties sales to awareness and consideration, finds which series leads, and turns a campaign into a response curve. On simulated data it gave 0.8 within a year and 5.2 within five, against a true 1.0 and 6.2, with both intervals holding the truth.
- **Data.** Years find the ties; data points and effect size set the precision. Three years often missed the ties, and a small effect gave intervals that reached zero in a quarter of fits even on five years of weekly data.
- **Base sales.** Total sales minus the MMM's short-term contributions kept the long run within a few per cent of the truth, though its speed ran fast. The MMM's smoothed trend got the speed badly wrong.
- **Fading lifts.** The 1-year return held up whichever way the brand worked. Over five years the drain couldn't tell lasting and fading brands apart: it gave 4.0, 3.7 and 3.2 against a true 6.2, 4.3 and 2.1. Leaving it out was worse, overstating fading brands by 2 to 6 times.
- **Brands without a tracker** might use share of search in place of awareness, but whether it leads sales needs testing first.

Everything in this article was fitted with [bayesian_vecm](https://github.com/raz1470/bayesian_vecm), an open-source Python package. To run it on your own brand, start with [the notebook built for your own data](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/08_response_curves_on_your_data.ipynb): edit one cell and it runs every check in order and draws the response curves.

### References

- Binet, L. and Field, P. (2013). *The Long and the Short of It: Balancing Short and Long-Term Marketing Strategies.* IPA.
- Binet, L. (2020). Share of Search as a predictive measure. IPA EffWorks Global 2020.
- Cain, P. M. (2022). Modelling short- and long-term marketing effects in the consumer purchase journey. *International Journal of Research in Marketing*, 39(1), 96–116.
- Cain, P. M. (2025). Long-term advertising effects: The Adstock illusion. *Applied Marketing Analytics*, 11(1), 23–42.
- Engle, R. F. and Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- Granger, C. W. J. and Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics*, 2(2), 111–120.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551–1580.
- Murray, M. P. (1994). A drunk and her dog: An illustration of cointegration and error correction. *The American Statistician*, 48(1), 37–39.
