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

This article has three aims:

1. **Explain the method**, a vector error correction model (VECM) in which every lift lasts (Part 1).
2. **Ask whether the data a real brand has can recover it** (Part 2). Most brands have five years of data or less, base sales handed over from an MMM, and performance budgets that chase demand.
3. **Improve it** (Part 3). The method assumes every lift lasts, so I add a term that lets a lift fade.

We illustrate each step on simulated data, so we always know the right answer.

## 2. The problem with adstock

An MMM explains weekly sales with spend, price, seasonality and a baseline. Spend enters through adstock: each week keeps a share of the previous week's effect, so a burst of TV fades out over a few weeks. That suits the question an MMM is built for: how did this quarter's spend move this quarter's sales?

It doesn't suit brand. The usual fix is a second, much longer adstock for brand spend: a short decay for activation and a long one for brand building. Cain (2025) refers to this practice as [the adstock illusion](https://ideas.repec.org/a/aza/ama000/y2025v11i1p23-42.html).

The problem is the trend. Sales drift over years. A slowly decaying spend series drifts too. Put one against the other and they line up whether or not brand drives sales. The long adstock ends up fitting the drift in the baseline, and it tells us nothing about how brand building works.

The chart shows it. The baseline is a slow trend we built with no link to spend at all. The long adstock of brand spend climbs alongside it, and a regression credits brand with the climb.

[[chart fig02_adstock_trend | Five years of flighted brand spend, a baseline trend built with no link to spend, and the long adstock of spend fitted to it | ]]

Cain (2022) proposes an alternative, the VECM, which separates the two. Short-term activation is the week-to-week movement around base sales. Long-term brand building is the base itself: the slow level of sales, which moves with awareness and consideration.

That needs three things an MMM leaves out:

- **The brand metrics**, alongside sales.
- **The trends treated as information**, not something to remove.
- **A way for a change in one series to carry through to the others.**

## 3. The brand funnel

A brand works through a funnel. Marketing makes people aware of the brand. Some of those people come to consider it, and some of those buy. Each step feeds the next, so a lasting rise at the top should show up lower down.

Brand trackers measure the middle of the funnel:

- **Awareness** is the share of people who know the brand.
- **Consideration** is the share who would think about buying it.

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

This isn't bad luck. Across 1,000 pairs of unrelated random walks of five years each, 84% gave a significant slope.

The usual fix is to difference the data: model the weekly change instead of the level. Weekly changes are stationary, so the regression is safe. But differencing throws away the levels, and the levels hold the long-run story. A campaign that lifts awareness for good shows up only in the level.

A series that is stationary as it stands is I(0). One that becomes stationary once differenced is I(1). Trending brand series are usually I(1), and the method in this article is built for them.

The Augmented Dickey-Fuller (ADF) test tells the two apart. On the simulated data, sales, awareness, consideration and the interest rate are I(1). Brand spend, PR, sponsorship and consumer confidence are I(0).

So we need a method that keeps the levels without being fooled by them. That method rests on cointegration.

## 5. Cointegration: the drunk and her dog

Some trending series are tied together. Each one wanders, but a weighted gap between them does not: it keeps coming back to a stable level. Series like these are cointegrated. The gap is the cointegrating relationship, and its stable level is the long-run equilibrium. The idea comes from [Engle and Granger (1987)](https://doi.org/10.2307/1913236).

Michael Murray gave it its best-known picture in "A Drunk and Her Dog" (1994). A drunk leaves the pub and wanders home. Her path is a random walk. So is her dog's. Watch either alone and you can't say where it will be in ten minutes. But when they drift too far apart, she calls and the dog comes back towards her. Both paths wander, but the distance between them does not.

In the simulated data one such tie runs between consideration and sales. Each series trends, but the gap between log sales and about 1.25 times log consideration stays close to its average for all five years.

[[chart fig04_cointegration | Base sales and 1.27 times consideration over five years, and the gap between them | Left: both series as % from their average, in logs.]]

We can test for this with the ADF test from section 4: take the gap and ask whether it is stationary. For the two unrelated random walks in section 4 it is not. For sales and consideration it clearly is. This two-step check is the Engle-Granger test.

A second tie runs between awareness and consideration. Two ties among three series leave one shared trend. Here that trend belongs to awareness, and it is why a lasting lift in awareness carries through to sales.

The number of ties is the cointegration rank. With more than two series, Johansen's trace test finds it by asking in turn: none, at most one, at most two? It stops at the first answer it can't reject.

## 6. The VECM

A vector error correction model (VECM) describes the weekly change in each brand series and splits it into parts. Each part has a plain reading for the brand.

<figure class="fig eqfig"><div class="fig-h">The VECM, term by term</div><div class="eqn"><span class="term"><span class="sym">Δ<b>y</b><sub>t</sub></span><span class="lab">Weekly change</span></span><span class="op">=</span><span class="term"><span class="sym"><b>αβ</b>′<b>y</b><sub>t−1</sub></span><span class="lab">Error correction</span></span><span class="op">+</span><span class="term"><span class="sym"><b>Γ</b>Δ<b>y</b><sub>t−1</sub></span><span class="lab">Momentum</span></span><span class="op">+</span><span class="term"><span class="sym"><b>Bx</b><sub>t</sub></span><span class="lab">Drivers</span></span><span class="op">+</span><span class="term"><span class="sym"><b>c</b></span><span class="lab">Growth</span></span><span class="op">+</span><span class="term"><span class="sym"><b>ε</b><sub>t</sub></span><span class="lab">Shock</span></span></div><figcaption class="fig-cap">y<sub>t</sub> holds the logs of base sales, awareness and consideration. x<sub>t</sub> holds the drivers.</figcaption></figure>

The left side is this week's change in log sales, log awareness and log consideration. Changes in logs read as percentage changes. Bold letters stand for all three series at once, so this is really three equations fitted together.

The parts, in order:

- **Error correction (αβ′y).** β′y<sub>t−1</sub> measures how far each tie from section 5 was stretched last week, for example how far sales sat from 1.25 times consideration. α turns that stretch into this week's move. This is what keeps the series together.
- **Momentum (ΓΔy).** Last week's changes carry into this week. If awareness jumped last week, some of that jump shows up in consideration now. These effects fade within weeks.
- **Drivers (Bx).** This week's brand spend, PR and other drivers push on each series. Drivers that move around a stable level enter as they are. A trending driver, like the interest rate, enters as its weekly change: a trend can't explain movement in a gap that keeps coming back.
- **Growth (c).** A constant that lets each series grow at a steady rate.
- **Shock (ε).** The part of this week that nothing in the model predicts. The three shocks are correlated, because news and competitors hit the series in the same week.

**Who closes the gap.** When the drunk and her dog drift apart, someone has to move, and α says who. A series with a large α follows: it moves to close a stretched tie. A series with α near zero on every tie leads.

In the simulated data, sales close about a quarter of their gap to consideration each week, and consideration about a fifth of its gap to awareness. Awareness closes neither, and the fitted model agrees.

So a lift in awareness is never pulled back. Consideration and sales rise to meet it and stay there, which is why a campaign that moves awareness keeps paying. A lift in sales alone is pulled back within weeks.

Hannan-Quinn picks the number of momentum lags and the rank test the number of ties: one week and two ties here. We fit the model by Bayesian sampling, which gives the intervals used from section 7 on.

## 7. From a campaign to a response curve

The pieces so far explain how the brand works. A budget decision needs something else: what a campaign earns. To get it we follow one campaign through the whole system.

**Spend reaches awareness.** Take a 13-week burst of brand spend. Before it moves awareness, it goes through the two steps familiar from MMM:

- **Carryover.** Each week keeps a share of the last week's spend, so the push on awareness builds and then tails off.
- **Diminishing returns.** A saturation curve means each extra pound moves awareness a little less than the one before.

These sit inside the drivers. The ties, momentum and growth are untouched.

**Awareness reaches consideration, partly in the same week.** Momentum carries last week's awareness into this week's consideration. Some of the effect may land in the same week, and that is harder to measure. Awareness and consideration share their weekly shocks (section 6), so a plain regression of one on the other finds a same-week link even when there is none.

The fix, as in [Cain (2022)](https://doi.org/10.1016/j.ijresmar.2021.06.006), is an instrument. Brand spend moves awareness but reaches consideration only through awareness. So its effect on consideration, divided by its effect on awareness, is the same-week link. That assumption is a business judgement, not something the data can settle (section 11).

**Consideration reaches sales.** The ties then pull consideration and sales up to meet the new level of awareness, and hold them there.

**From sales to a response curve.** Add up the extra sales each week and we have what the campaign earned. Because the ties carry the lift forward, the total keeps growing with the window, so we report it at one year and at five. Repeat for campaigns of different sizes, from nothing up to half as much again as current spend, and we have a response curve.

The chart shows the response curve on the simulated data, fitted with the VECM. Every lift lasts, so every week after the campaign adds the same extra sales, and the 5-year curve sits far above the 1-year curve.

[[chart fig08a_lasting_curve | Response curves for a 13-week campaign within one and five years, for a brand whose lifts last, fitted with the VECM | Extra sales against no campaign, indexed so the true first-year return of a campaign at current spend (about £950 a week) is 1. Fitted without a drain; medians with 89% intervals; dashed lines show the truth.]]

This is the payoff of Part 1. Within a year a campaign at current spend earns 0.8 on the chart's scale, against a true 1.0. Within five years it earns 5.2, against a true 6.2. Both intervals hold the truth. Five years earn about six times the first year, because the lift never goes away.

That is the method on clean data: five years of weekly figures, a true base sales series and drivers set from outside the brand. Real brands rarely have all three. Part 2 asks what happens when they don't. Part 3 then relaxes the method's strongest assumption, that every lift lasts.

## 8. How much data is enough?

Part 1 ran on five years of weekly data. Most brands have less: a tracker that started three years ago, or one that reports monthly. So we cut the simulated data four ways: weekly or monthly, over five years or three. Monthly over three years leaves 36 points.

Marketing's effect matters too. In the simulated data, marketing drives about half the week-to-week movement in awareness. That is a large effect. So we also halved and quartered it, which leaves marketing driving about a fifth and under a tenth. On 20 versions of each brand, each with its own random shocks, we ran the rank test and measured the 5-year return on brand spend.

[[chart fig_data_size | The 5-year return as a share of the truth, and how often the rank test found two ties, for weekly and monthly data over three and five years and for large, medium and small marketing effects | Left: the median across 20 data sets of each fit's median and 89% interval, divided by the true return. Right: the share of data sets where the Johansen trace test found the two ties we built in.]]

**Years find the ties.** A tie is a gap that keeps coming back, and a gap needs time to wander off and return. With five years the rank test found both ties eight or nine times in ten, weekly or monthly. With three years it found them two to five times in ten. Most misses found too few. The size of the effect made no difference here, because the ties come from the shared trend, not from marketing.

**Data points set the precision.** The return stayed close to the truth throughout, and its intervals held the truth nine times in ten or more. What changed was their width. Five years of weekly data put it within about a quarter of the truth either way. Five years of monthly data, only 60 points, did worse than three years of weekly. Three years of monthly data gave an interval from almost nothing to nearly twice the truth.

**Smaller effects need more data.** Halving the effect didn't halve the noise. In sales terms the intervals stayed about as wide, so as a share of the truth they roughly doubled each time the effect halved. Five years of weekly data put a medium effect within about 40% of the truth either way, and a small one within about 80%. With a small effect the interval reached zero in a quarter of the five-year weekly fits, and in most fits on less data. The intervals still held the truth nine times in ten or more. The model wasn't wrong; it was saying it couldn't tell.

On the shortest data the middle of the interval also slipped towards zero: 0.6 of the truth for a medium effect on three years of monthly data, and 0.3 for a small one. That is the prior pulling a weak signal back.

There the priors do most of the work, and a Bayesian fit says so with a wide interval. A classical fit doesn't: on 60 periods its intervals for the ties held the truth only about 60% of the time when they claimed 90%. That is the main practical reason to fit by sampling.

What to do:

- **Use weekly data if it exists.**
- **With under four years, set the rank from the funnel**: three brand series sharing one trend have two ties.
- **With monthly data, expect wide intervals** and report them as they are.
- **If brand spend moves awareness only a little, expect the long run to stay unclear.** Even five years of weekly data may give an interval that reaches zero.

## 9. Base sales from an MMM

Nobody observes base sales. In practice they come from an MMM, and there is more than one way to take them. We built total sales as the base plus a short-term layer: paid search and social with carryover and diminishing returns, brand spend's same-week hit, promotions, seasonality and noise. Then we gave the VECM four versions of base sales, on 20 data sets each:

- **The true base**, which only a simulation has.
- **Minus MMM**: total sales minus the MMM's short-term contributions, week by week.
- **Total sales as they are**, with the short-term drivers in the VECM's sales equation instead.
- **The MMM's trend**: the smooth baseline many MMMs report.

[[chart fig_base_sales | The long-run lift from brand spend and the speed at which sales catch up, for four versions of base sales | Twenty simulated brands. The median across data sets of each fit's median and 90% interval, divided by the truth.]]

**The long run survives a rough base.** The first three versions put the long-run lift from brand spend within 2% of the truth. The long run is read from the level base sales settle at, and short-term noise around that level averages out.

**The speed doesn't.** How fast sales close their gap to consideration depends on every wiggle. Minus MMM made sales about 13% too fast. Total sales as they are made them 1.5 times too fast. The MMM's trend smooths away the wiggles altogether, so sales looked twenty times too slow. In 12 of 20 data sets it also made sales look like the leader, and it put the long run 17% high.

What to do: use total sales minus the MMM's short-term contributions, unsmoothed. Never use the MMM's trend. Trust the long-run return more than any speed or half-life.

## 10. Performance spend that chases demand

Performance budgets follow demand. When more people are in the market, search volume rises and search spend with it. An MMM then sees search spend move with sales the brand produced, and can credit search with some of them.

So we made search spend chase demand, at three strengths. At the strongest, its correlation with demand was about 0.85. Its true effect on sales stayed the same. We fitted an MMM to total sales, took base sales as total minus its contributions, and fitted the VECM, ten times at each strength.

[[chart fig_perf_feedback | MMM credit to paid search, the long-run lift from brand spend and the 4-week lift, as search spend chases demand more strongly | Ten simulated brands at each strength. Left: the median MMM contribution of paid search, against the truth. Middle and right: the median across data sets of each fit's median and 90% interval, divided by the truth.]]

**The MMM gets search wrong.** With no feedback it put search's contribution about 13% low. Once search chased demand it put it 30 to 40% too high.

**The brand's long run doesn't notice.** The long-run lift from brand spend came within 3% of the truth at every strength, and its intervals always held it. The demand handed to search was weeks of extra growth here and there, not a change in the level base sales settle at. The short run slipped a little: four weeks after spend, the lift went from about 13% low to about 25% low.

**A check needs the tracker.** Ask whether past demand predicts search spend. Asked with past sales, the test flagged the feedback in only one or two data sets in ten. Asked with past consideration, it flagged it every time.

## 11. Other things that can go wrong

| Problem, and what we built in | What happened | What to do |
|---|---|---|
| **A channel left out.** PR and sponsorship dropped from the model, on monthly data | Brand spend took their effect and came out about 40% too high | Include every brand activity, even ones without a budget line |
| **Carryover or diminishing returns left out** | Without carryover the model missed about half of brand spend's effect. A straight line put the effect of doubling spend at more than twice the truth | Include both. Trust the curve most near the spend levels the data has seen |
| **The wrong instrument.** Brand spend also reached consideration directly | The same-week link came out too large, and the test for this rarely noticed | Choose instruments on business grounds and say why |

## 12. Letting lifts fade

In the VECM so far nothing pulls a lift back. Once a campaign moves awareness, the ties carry the whole lift down to sales and keep it there. That is why the 5-year return in section 7 is so large.

It is a strong assumption. It says a campaign that ran five years ago lifts this week's sales as much as it ever did. People forget brands, and rivals answer campaigns with their own.

So I added a term to the model that lets a lift fade, which I call the drain. Each week a little of the lift that past spend created drains away. Two numbers set it: the share of each lift that eventually drains, and the half-life of the drain. With a share of zero nothing drains and the model is the VECM from Part 1. The priors are loose: the share is flat between zero and one, and the half-life is centred on a year.

This differs from a long adstock. A long adstock is fitted against the trend in sales, so it soaks up drift (section 2). Here the ties handle the trends first, and the drain only asks how much of a driver's lift is later given back.

We built two more versions of the brand, where half or all of each lift drains with a half-life of a year, and fitted all three with the drain.

[[chart fig08_response_curves | Response curves for a 13-week campaign within one and five years, for a brand whose lifts last, half fade or all fade | Extra sales against no campaign, indexed so the true first-year return of a campaign at current spend, if lifts last, is 1. Each brand fitted with the drain; medians with 89% intervals; dashed lines show the truth.]]

Over the first year the three brands look much alike, and the intervals hold the truth for all three. Over five years the true returns are 6.2, 4.3 and 2.1. The model gives 4.0, 3.7 and 3.2: it pulls all three towards the middle. On the lasting brand that costs something, 4.0 against 5.2 without the drain.

Leaving the drain out is worse. On brands whose lifts fade with a half-life of six months to two years, the VECM without it overstated the 5-year return by 2 to 6 times, and its intervals missed the truth every time. The drain model ran 1.3 to 3 times high, but its intervals mostly held the truth.

### What settles it: going dark

A dark spell is the clearest evidence there is. If lifts fade, sales sag once spend stops; if they last, sales hold. Switching brand spend off for a year halved the uncertainty in how much fades, and brought the 5-year return from 2 times the truth to 1.3. Six months helped less. Running brand in two 13-week pulses a year barely helped: thirteen-week gaps are short next to a year's half-life.

Keep any dark weeks in the data. A planned dark spell in one region, longer than you expect a lift to last, is worth more than any modelling choice.

## 13. Brands without a tracker: share of search

Everything so far assumes a brand tracker with years of weekly data. Many brands have nothing like it. Share of search is a cheap stand-in: a brand's searches as a share of all searches for brands in its category. Les Binet [introduced it at the IPA in 2020](https://ipa.co.uk/news/binet-presents-fast-cheap-predictive-share-of-search-metric/). In cars, energy and mobile phones it tracked share of market and moved ahead of it, by up to a year for cars.

It suits this method:

- **It has the history.** Google Trends gives weekly data for up to five years and monthly data back to 2004.
- **Google's scaling cancels out.** Trends rescales every request to run from 0 to 100. A share divides that out, as long as every brand comes from the same request.
- **It trends**, as awareness does, so it can be tied to base sales.

<figure class="fig diagram"><div class="fig-h">A brand system without a tracker</div><div class="system"><div class="sys-col"><div class="node drv">Brand spend</div><div class="node drv">PR reach</div><div class="node drv">Sponsorship reach</div></div><div class="arrow">→</div><div class="sys-col"><div class="node aw">Share of search<small>leads, if it works</small></div></div><div class="arrow">↔</div><div class="sys-col macro"><div class="node drv">Category searches</div><div class="varrow">↓</div><div class="node sa">Base sales<small>follows share of search</small></div></div></div><figcaption class="fig-cap">A sketch, not a fitted model. Two brand series with at most one tie. Category searches stand in for how many people are in the market.</figcaption></figure>

Two questions decide whether it works:

- **Does it lead?** Some branded searches come from existing customers looking for the website. That part follows sales. The VECM can test this: if share of search closes part of its gap to sales each week, it is a follower.
- **Brand or activation?** Binet found advertising moves share of search twice: a large short-term effect that fades fast, and a smaller one that builds up. The short-term part is activation inside a series meant to measure the brand, the base sales problem of section 9 in a new place.

## 14. The first year is firm. The fifth is a range.

A marketing mix model sees the short-term lift that follows a campaign and misses the slow climb in base sales. This article set out to measure that climb.

**The VECM** found the ties, picked out awareness as the leader and recovered the response curve: 0.8 within a year and 5.2 within five, against a true 1.0 and 6.2, with both intervals holding the truth.

**On realistic data** the long-run return held up. Less data and smaller effects widened its interval, and fewer years hid the ties. A rough base sales series, even with search spend chasing demand, left the long run within a few per cent of the truth. What it damaged was the short run.

**The drain** lets a lift fade. On brands whose lifts last, half fade or all fade, the true 5-year returns are 6.2, 4.3 and 2.1, and the drain model gives 4.0, 3.7 and 3.2. The 1-year return is firm whichever way the brand works. The 5-year return is a range, and a long dark spell is the best evidence for narrowing it. For brands without a tracker, share of search could stand in, if it leads.

### Try it yourself

Everything in this article was fitted with [bayesian_vecm](https://github.com/raz1470/bayesian_vecm), an open-source Python package. To run it on your own brand, start with [the notebook built for your own data](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/08_response_curves_on_your_data.ipynb): edit one cell and it runs every check in order and draws the response curves.

### What comes next

- **Pooling across brands and markets.** One brand rarely has enough history to pin down the drain. A hierarchical model across regions or sister brands could share it.
- **One model instead of two.** Fitting the MMM and the VECM together would carry the uncertainty in base sales through, rather than losing it at the handover.
- **Testing share of search** on simulated and then real brands.

The method is Peter Cain's, and the package is an open Bayesian version of it. Read [*Modelling short- and long-term marketing effects in the consumer purchase journey*](https://doi.org/10.1016/j.ijresmar.2021.06.006) for the method on real brand data.

### References

- Binet, L. and Field, P. (2013). *The Long and the Short of It: Balancing Short and Long-Term Marketing Strategies.* IPA.
- Binet, L. (2020). Share of Search as a predictive measure. IPA EffWorks Global 2020.
- Cain, P. M. (2022). Modelling short- and long-term marketing effects in the consumer purchase journey. *International Journal of Research in Marketing*, 39(1), 96–116.
- Cain, P. M. (2025). Long-term advertising effects: The Adstock illusion. *Applied Marketing Analytics*, 11(1), 23–42.
- Engle, R. F. and Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251–276.
- Granger, C. W. J. and Newbold, P. (1974). Spurious regressions in econometrics. *Journal of Econometrics*, 2(2), 111–120.
- Johansen, S. (1991). Estimation and hypothesis testing of cointegration vectors in Gaussian vector autoregressive models. *Econometrica*, 59(6), 1551–1580.
- Murray, M. P. (1994). A drunk and her dog: An illustration of cointegration and error correction. *The American Statistician*, 48(1), 37–39.
- Pesaran, H. H. and Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. *Economics Letters*, 58(1), 17–29.
