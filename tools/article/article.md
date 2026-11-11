<!-- Source text for docs/_extra/walkthrough.html. Build with tools/article/build.py. -->
<!-- Charts: a line [[chart NAME | ALT | CAPTION]] places the chart in tools/article/data/NAME.json. -->

# How Long Does Brand Last?

*Why a marketing mix model undercounts brand, and how cointegration measures it*

## 1. Introduction

Brand teams and marketing mix models disagree about time. Brand teams say a campaign builds the brand for years. A marketing mix model (MMM) usually says its effect is gone within a couple of months.

Both can be right. An MMM measures the short-term lift in sales that follows a burst of spend. What it struggles to see is the slow climb: more people who know the brand, more who would consider it, and a higher level of sales that stays.

Les Binet and Peter Field made this case with the IPA databank in [*The Long and the Short of It*](https://ipa.co.uk/knowledge/effectiveness-research-analysis/les-binet-peter-field) (2013). Measuring the effect for one brand from its own data is harder. Peter Cain has shown how. In [*Modelling short- and long-term marketing effects in the consumer purchase journey*](https://doi.org/10.1016/j.ijresmar.2021.06.006) (2022) he ties brand metrics to base sales with cointegration, so a lasting lift in awareness carries through to sales.

The chart shows three ways a campaign can play out:

- **Adstock.** The lift fades back to zero within weeks. This is what an MMM assumes.
- **A lasting lift.** Awareness moves to a new level and sales follow it up for good. This is what the method in Part 1 assumes.
- **A fading lift.** Awareness rises and then drains slowly back towards where it was heading. For most brands this is probably the more realistic case. The method in Part 1 doesn't allow it, so this article adds it.

[[chart fig01_three_ways | Three sketches of sales after a 13-week burst of spend: adstock fading to zero, a lasting lift, a fading lift | A sketch, not an estimate. Grey bars: 13 weeks of brand spend.]]

The gap between these is the whole question. Over five years a lasting lift earns many times what an MMM credits it with. A fading lift earns less, but still more than adstock suggests. Telling the two apart from the data is hard.

This article has three aims:

1. **Explain the method**, a vector error correction model (VECM) in which every lift lasts (Part 1).
2. **Ask whether the data a real brand has can recover it** (Part 2). Most brands have five years of data or less, base sales handed over from an MMM, and performance budgets that chase demand.
3. **Improve it** (Part 3). The method assumes every lift lasts, so I add a term that lets a lift fade. I call it the drain, and it is mine, not Cain's. For brands without a tracker I sketch the same system built on share of search.

The short answer: the 1-year return is firm. It stays close to the truth with less data, with a rough base sales series and with performance spend that chases demand. Those problems cost precision, and they distort how fast sales respond, but not how far. The 5-year return isn't firm, because the data say little about how much of a lift fades. A long dark spell is the best evidence there is, and even a year dark doesn't settle it alone.

We illustrate each step on a brand we simulated, so we always know the right answer.

## 2. The problem with adstock

An MMM explains weekly sales with spend, price, seasonality and a baseline. Spend enters through adstock: each week keeps a share of the previous week's effect, so a burst of TV fades out over a few weeks. That suits the question an MMM is built for: how did this quarter's spend move this quarter's sales?

It doesn't suit brand. The usual fix is a second, much longer adstock for brand spend: a short decay for activation and a long one for brand building. Peter Cain calls this [the adstock illusion](https://ideas.repec.org/a/aza/ama000/y2025v11i1p23-42.html).

The problem is the trend. Sales drift over years. A slowly decaying spend series drifts too. Put one against the other and they line up whether or not brand drives sales. The long adstock ends up fitting the drift in the baseline, and it tells us nothing about how brand building works.

The chart shows it. The baseline is a slow trend we built with no link to spend at all. The long adstock of brand spend climbs alongside it, and a regression happily credits brand with the climb.

[[chart fig02_adstock_trend | Five years of flighted brand spend, a baseline trend built with no link to spend, and the long adstock of spend fitted to it | Brand spend grows slowly over the five years, as budgets often do. Its adstock (decay 0.97) climbs with it. The baseline is a random walk with drift that never sees spend, yet the regression slope has a t-statistic of 34.]]

Cain's alternative, a VECM, separates the two. Short-term activation is the week-to-week movement around base sales. Long-term brand building is the base itself: the slow level of sales, which moves with awareness and consideration.

That needs three things an MMM leaves out:

- **The brand metrics**, alongside sales.
- **The trends treated as information**, not something to remove.
- **A way for a change in one series to carry through to the others.**

## 3. The brand we built

A brand works through a funnel. Marketing makes people aware of the brand. Some of those people come to consider it, and some of those buy. Each step feeds the next, so a lasting rise at the top should show up lower down.

Brand trackers measure the middle of the funnel:

- **Awareness** is the share of people who know the brand.
- **Consideration** is the share who would think about buying it.

Together with sales these form the system. Sales here means base sales: sales without the short-term lift from performance media and promotions. Sections 9 and 10 cover how to get that series right.

Around the system sit the drivers. Brand spend, PR and sponsorship push on awareness. Interest rates and consumer confidence push on consideration from outside. The drivers move the brand, but the brand does not move them.

<figure class="fig diagram"><div class="fig-h">Marketing moves awareness. Consideration and sales follow.</div><div class="system"><div class="sys-col"><div class="node drv">Brand spend</div><div class="node drv">PR reach</div><div class="node drv">Sponsorship reach</div></div><div class="arrow">→</div><div class="sys-col"><div class="node aw">Awareness<small>leads the system</small></div></div><div class="arrow">→</div><div class="sys-col macro"><div class="node drv">Interest rate</div><div class="varrow">↓</div><div class="node co">Consideration<small>follows awareness</small></div><div class="varrow">↑</div><div class="node drv">Consumer confidence</div></div><div class="arrow">→</div><div class="sys-col"><div class="node sa">Base sales<small>follows consideration</small></div></div></div><figcaption class="fig-cap">The brand we built. Drivers act from outside. The three brand series trend and are tied together.</figcaption></figure>

To show each step of the method we simulated five years of weekly data for a brand like this. Because we built it, we know the right answer at every step. This is what we built in:

| What we built in | Setting |
|---|---|
| Funnel order | Drivers → awareness → consideration → sales |
| How far a lift carries | A lasting 1% rise in awareness ends up as about 0.9% in consideration and about 1.1% in sales |
| Who catches up | Sales close about a quarter of any gap to consideration each week. Consideration closes about a fifth of its gap to awareness. Awareness never catches up to anything: it leads |
| Weekly surprises | News and competitors hit all three series in the same week |
| Does a lift last? | Yes, for good. Part 3 adds two versions where lifts fade |

The rest of the article shows how the method recovers each of these from the data alone.

## 4. Why trending data fools regression

Most analysts would start with a regression. Regression assumes each series is stationary: it moves around a stable average, and after a shock it comes back.

Brand series don't behave like that. Each week starts from where the last one ended, and nothing pulls the series back to a fixed level. Awareness that rises after a strong year stays risen until something else moves it. A series like this is non-stationary. Economists say it has a unit root.

Regress one trending series on another and the two will often look related when they aren't. This is a spurious regression, first shown by [Granger and Newbold (1974)](https://doi.org/10.1016/0304-4076(74)90034-7).

The chart shows two random walks that we built completely independently. Regress one on the other and the slope comes out highly significant. The leftover gap between them wanders as much as the series do, which gives the game away.

[[chart fig03_spurious | Two unrelated random walks, and what is left over after regressing one on the other | Two random walks built independently. The slope has a t-statistic of 15.9 and an R² of 0.50. The leftover never settles back to zero.]]

This isn't bad luck. Across 1,000 pairs of unrelated random walks of five years each, 84% gave a significant slope.

The usual fix is to difference the data: model the weekly change instead of the level. Weekly changes are stationary, so the regression is safe. But differencing throws away the levels, and the levels hold the long-run story. A campaign that lifts awareness for good shows up only in the level.

This gives a simple way to label series. A series that is stationary as it stands is I(0), "integrated of order zero". A series that becomes stationary once differenced is I(1). Trending brand series are usually I(1). The method in the rest of this article is built for I(1) series.

To sort the series we use the Augmented Dickey-Fuller (ADF) test. It starts from the assumption that a series has a unit root and asks whether the data are strong enough to reject it. A small p-value means the series is stationary, so I(0). A large one means we can't rule out a unit root. We then test the weekly changes: if those are stationary, the series is I(1).

| Series | ADF p-value (level) | Verdict |
|---|---|---|
| Base sales | 0.80 | Trending, I(1) |
| Awareness | 0.74 | Trending, I(1) |
| Consideration | 0.73 | Trending, I(1) |
| Interest rate | 0.58 | Trending, I(1) |
| Brand spend | <0.001 | Stable, I(0) |
| PR reach | <0.001 | Stable, I(0) |
| Sponsorship reach | <0.001 | Stable, I(0) |
| Consumer confidence | <0.001 | Stable, I(0) |

On the brand we built, the three brand series all fail to reject a unit root, and their weekly changes are clearly stationary, so all three are I(1). Brand spend, PR, sponsorship and consumer confidence move around a stable level, so they're I(0). The interest rate trends like the brand series, so it's I(1).

So we need a method that keeps the levels without being fooled by them. That method rests on cointegration.

## 5. Cointegration: the drunk and her dog

Some trending series are tied together. Each one wanders, but a weighted gap between them does not: it keeps coming back to a stable level. Series like these are cointegrated. The gap is the cointegrating relationship, and its stable level is the long-run equilibrium. The idea comes from [Engle and Granger (1987)](https://doi.org/10.2307/1913236).

Michael Murray gave it its best-known picture in "A Drunk and Her Dog" (1994). A drunk leaves the pub and wanders home. Her path is a random walk. So is her dog's. Watch either alone and you can't say where it will be in ten minutes. But when they drift too far apart, she calls and the dog comes back towards her. Both paths wander, but the distance between them does not.

In our brand one such tie runs between consideration and sales. Each series trends, but the gap between log sales and about 1.25 times log consideration stays close to its average for all five years.

[[chart fig04_cointegration | Base sales and 1.27 times consideration over five years, and the gap between them | Left: both series as % from their average, in logs. The data put the multiple at 1.27; we built in 1.25. Right: the gap stays within 8% of zero.]]

We can test for this with the ADF test from section 4: take the gap and ask whether it is stationary. For the two unrelated random walks in section 4 it is not. For sales and consideration it clearly is. This two-step check is the Engle-Granger test.

A second tie runs between awareness and consideration. Two ties among three series leave one shared trend. In our brand that trend belongs to awareness, and it is why a lasting lift in awareness carries through to sales. Section 6 shows why.

The number of ties is the cointegration rank. With more than two series, Johansen's trace test finds it by asking in turn: none, at most one, at most two? It stops at the first answer it can't reject. On our brand it stops at two, the number we built in.

The test is a guide rather than a ruling. On short histories it often finds too few ties (section 8). A business view of the funnel is a useful check: three brand series sharing one trend should have two ties.

## 6. The model in one equation

A vector error correction model (VECM) describes the weekly change in each brand series and splits it into parts. Each part has a plain reading for the brand.

<figure class="fig eqfig"><div class="fig-h">The VECM, term by term</div><div class="eqn"><span class="term"><span class="sym">Δ<b>y</b><sub>t</sub></span><span class="lab">Weekly change</span></span><span class="op">=</span><span class="term"><span class="sym"><b>αβ</b>′<b>y</b><sub>t−1</sub></span><span class="lab">Error correction</span></span><span class="op">+</span><span class="term"><span class="sym"><b>Γ</b>Δ<b>y</b><sub>t−1</sub></span><span class="lab">Momentum</span></span><span class="op">+</span><span class="term"><span class="sym"><b>Bx</b><sub>t</sub></span><span class="lab">Drivers</span></span><span class="op">+</span><span class="term"><span class="sym"><b>c</b></span><span class="lab">Growth</span></span><span class="op">+</span><span class="term"><span class="sym"><b>ε</b><sub>t</sub></span><span class="lab">Shock</span></span></div><figcaption class="fig-cap">y<sub>t</sub> holds the logs of base sales, awareness and consideration. x<sub>t</sub> holds the drivers.</figcaption></figure>

The left side is this week's change in log sales, log awareness and log consideration. Changes in logs read as percentage changes. Bold letters stand for all three series at once, so this is really three equations fitted together.

The parts, in order:

- **Error correction (αβ′y).** β′y<sub>t−1</sub> measures how far each tie from section 5 was stretched last week, for example how far sales sat from 1.25 times consideration. α turns that stretch into this week's move. This is what keeps the series together.
- **Momentum (ΓΔy).** Last week's changes carry into this week. If awareness jumped last week, some of that jump shows up in consideration now. These effects fade within weeks.
- **Drivers (Bx).** This week's brand spend, PR and other drivers push on each series. Drivers that move around a stable level enter as they are. A trending driver, like the interest rate, enters as its weekly change: a trend can't explain movement in a gap that keeps coming back.
- **Growth (c).** A constant that lets each series grow at a steady rate.
- **Shock (ε).** The part of this week that nothing in the model predicts. The three shocks are correlated, because news and competitors hit the series in the same week.

**Who closes the gap.** Go back to the drunk and her dog. When they drift apart, someone has to move, and α says who. A series with a large α is a follower: when a tie is stretched, it moves to close the gap. A series with α near zero on every tie ignores the gaps. It leads.

In the brand we built, sales close about a quarter of any gap to consideration each week. Consideration closes about a fifth of its gap to awareness. Awareness closes neither, and the fitted model finds the same: its α is close to zero on both ties.

This is why the question "does brand last?" runs through awareness. A lift that lands on sales alone stretches a tie, and sales are pulled back within weeks. A lift that lands on awareness is never pulled back. Consideration and then sales climb to meet it and stay there. So a campaign that moves awareness keeps paying, and section 7 adds up how much.

Two choices remain: how many weeks of momentum to include and how many ties. Information criteria such as Hannan-Quinn choose the momentum lags, and the rank test from section 5 chooses the ties. For our brand that gives one week of momentum and two ties.

We fit the model with Bayesian sampling, so every estimate comes with an interval.

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

The chart shows the response curve for the brand we built, fitted with the VECM. Every lift lasts, so every week after the campaign adds the same extra sales, and the 5-year curve sits far above the 1-year curve.

[[chart fig08a_lasting_curve | Response curves for a 13-week campaign within one and five years, for a brand whose lifts last, fitted with the VECM | Extra sales against no campaign, indexed so the true first-year return of a campaign at current spend (about £950 a week) is 1. Fitted without a drain; medians with 89% intervals; dashed lines show the truth.]]

This is the payoff of Part 1. Within a year a campaign at current spend earns 0.8 on the chart's scale, against a true 1.0. Within five years it earns 5.2, against a true 6.2. Both intervals hold the truth. Five years earn about six times the first year, because the lift never goes away.

That is the method on clean data: five years of weekly figures, a true base sales series and drivers set from outside the brand. Real brands rarely have all three. Part 2 asks what happens when they don't. Part 3 then relaxes the method's strongest assumption, that every lift lasts.

## 8. How much data is enough?

Part 1 ran on five years of weekly data. Most brands have less. A tracker may have started three years ago, or it may report monthly. So we cut the brand we built four ways:

- **Weekly, five years**: 261 weeks, as in Part 1.
- **Weekly, three years**: the first 157 weeks.
- **Monthly, five years**: the same weeks summed into 60 months.
- **Monthly, three years**: 36 months.

We did this for 20 versions of the brand, each with its own random shocks. On each we ran the rank test, then fitted the VECM with the two ties we built in and measured the 5-year return on brand spend.

[[chart fig_data_size | The 5-year return as a share of the truth, and how often the rank test found two ties, for weekly and monthly data over three and five years | Left: the median across 20 data sets of each fit's median and 89% interval, divided by the true return. Right: the share of data sets where the Johansen trace test found the two ties we built in.]]

Two things happen, and they depend on different parts of the data.

**Years find the ties.** The rank test looks for gaps that keep coming back, and a gap needs time to wander off and return. With five years it found the two ties in about nine data sets in ten on weekly data, and almost as often on monthly. With three years it found them in only two in five, weekly or monthly. Most misses found too few ties.

**Data points set the precision.** The return stayed close to the truth in every version, and its intervals held the truth nine times in ten or more. What changed was their width. Five years of weekly data put the 5-year return within about a quarter of the truth either way. Three years of weekly data widened that to about a third. Five years of monthly data, only 60 points, did worse than three years of weekly: about half the truth either way. Three years of monthly data gave an interval running from almost nothing to nearly twice the truth, and its middle drifted about a sixth low.

That last case is where the priors do most of the work. With 36 months and a model of three equations, there is little data per number being estimated. A Bayesian fit admits this with a wide interval. A classical fit doesn't: on 60 periods its intervals for the ties held the truth only about 60% of the time when they should have held it 90%. The Bayesian intervals stayed close to 90%. That is the main practical reason to fit the model by sampling.

What to do:

- **Use weekly data if it exists**, even if the tracker reports monthly figures for the board.
- **Four years or more** if the rank test is to be trusted. With less, set the rank from the funnel: three brand series sharing one trend have two ties.
- **With monthly data, expect wide intervals** and report them as they are.

## 9. Base sales from an MMM

The VECM models base sales: sales without the short-term lift from performance media and promotions. Nobody observes that series. In practice it comes from an MMM, and there is more than one way to take it.

We built a brand whose total sales add a short-term layer to the base: brand spend's same-week hit, paid search and paid social with carryover and diminishing returns, promotions, seasonality and noise. Then we gave the VECM four versions of base sales, on 20 data sets each:

- **The true base**, which only a simulation has.
- **Total sales minus the MMM's short-term contributions**: channels, promotions and seasonality taken off, week by week.
- **Total sales as they are**, with the short-term drivers added to the VECM's sales equation instead.
- **The MMM's trend**: the smooth, slowly changing baseline many MMMs report.

[[chart fig_base_sales | The long-run lift from brand spend and the speed at which sales catch up, for four versions of base sales | Twenty simulated brands. The median across data sets of each fit's median and 90% interval, divided by the truth. "Minus MMM" is total sales minus the MMM's short-term contributions.]]

**The long run survives a rough base.** The first three versions put the long-run lift from brand spend within 2% of the truth, with intervals that held it nine times in ten or more. The long run is read from the level base sales settle at, and short-term noise around that level averages out.

**The speed doesn't.** How fast sales close their gap to consideration depends on every wiggle in the series:

- **Minus the MMM's contributions** was close: about 13% fast. The noise the MMM can't take out makes sales look a little quicker than they are.
- **Total sales as they are** made sales about 1.5 times too fast, and its intervals held the truth only three times in ten.
- **The MMM's trend** was the worst. It smooths away the week-to-week movement the VECM learns speed from, so sales looked twenty times too slow. In 12 of 20 data sets sales looked like the leader of the brand, a link that doesn't exist. It also put the long-run lift about 17% high.

What to do:

- **Use total sales minus the MMM's short-term contributions**, and don't smooth it.
- **Never use the MMM's trend** as base sales.
- **Trust the long-run return more than any speed or half-life.**

## 10. Performance spend that chases demand

Performance budgets follow demand. When more people are in the market, more of them search for the brand, and search spend rises with the searches. An MMM fitted to total sales then sees search spend move with sales the brand produced. It can credit search with some of that demand, and take it out of base sales.

So we made search spend chase demand. Each week its spend rose and fell with how fast consideration had grown over the past month, which base sales follow. We tried three strengths: none, moderate and strong. At moderate the correlation between search spend and demand was about 0.65, and at strong about 0.85. The true effect of search on sales stayed the same throughout. We then fitted an MMM to total sales, built base sales by taking off its short-term contributions, and fitted the VECM to that, ten times at each strength.

[[chart fig_perf_feedback | MMM credit to paid search, the long-run lift from brand spend and the 4-week lift, as search spend chases demand more strongly | Ten simulated brands at each strength. Left: the median MMM contribution of paid search, against the truth. Middle and right: the median across data sets of each fit's median and 90% interval, divided by the truth.]]

**The MMM gets search wrong.** With no feedback it put search's contribution about 13% low. Once search chased demand it put it 30 to 40% too high. That is a problem for anyone reading the MMM's return on search.

**The brand's long-run return doesn't notice.** The long-run lift from brand spend came within 3% of the truth at every strength, and its intervals held the truth every time. The demand the MMM gave to search was weeks of extra growth here and there. It did not move the level that base sales settle at, and the long run is read from that level.

**The short run slips a little.** Four weeks after spend, the lift came out about 13% low with no feedback and about 25% low with strong feedback.

**A check needs the tracker.** Before fitting, ask whether past demand helps predict search spend. Asked with past sales, the test flagged the feedback in only one or two data sets in ten, about as often as it did when there was none. Asked with past consideration, it flagged it every time, with two false alarms in ten when there was no feedback.

What to do:

- **Run the check with the brand metrics, not sales.** If past consideration predicts search spend, search is chasing demand.
- **Read the MMM's search return with care** when it does. The VECM's long-run brand return is safe.

## 11. Other things that can go wrong

Sections 8 to 10 took the three problems most brands will hit. The rest are quicker to state. For each we built the problem into the simulation, refitted, and looked at what broke.

| Problem, and what we built in | What happened | What to do |
|---|---|---|
| **Awareness looks like a follower.** Nothing changed: this happens by chance | In about one data set in ten the model gave awareness a small but clear α, which the truth doesn't have | If awareness's α is small next to the followers', treat awareness as the leader |
| **A channel left out.** PR and sponsorship dropped from the model, on monthly data | Brand spend took their effect and came out about 40% too high, against about 12% too low with every channel in | Include every brand activity, even ones without a budget line |
| **Carryover or diminishing returns left out.** Fitted without carryover, or with a straight line instead of a saturation curve | Without carryover the model missed about half of brand spend's effect. The straight line put the effect of doubling spend at more than twice the truth. Near current spend the two barely differed | Include both. Trust the curve most near the spend levels the data has seen |
| **A same-week link with the wrong instrument.** Brand spend also reached consideration directly, which the instrument assumes it doesn't | The same-week link came out too large, and the test for this rarely noticed | Choose instruments on business grounds and say why. For a campaign's return alone, the link can be left out |
| **Seasonality.** A seasonal pattern added to sales | With the pattern left in, the rank test found the right number of ties one time in five. On a deseasonalised copy it was four in five | Give the sales equation seasonal terms. Run the rank test on a deseasonalised copy |
| **Brand budgets that follow the brand.** Not built in: our drivers are set outside the brand | Not tested here. If brand spend rises after good weeks, its effect is overstated | Run the same check as section 10 on brand spend: do past brand metrics predict it? |

## 12. Letting lifts fade

In the VECM so far nothing pulls a lift back. Once a campaign moves awareness, the ties carry the whole lift down to sales and keep it there. That is why the 5-year return in section 7 is so large.

It is a strong assumption. It says a campaign that ran five years ago lifts this week's sales as much as it ever did. People forget brands, and rivals answer campaigns with their own. The fading lift in section 1 is the alternative: awareness rises, then drains slowly back.

So I added a term that lets a lift fade. I call it the drain. This extension is mine, not Cain's. With the drain switched off, the model is the VECM from Part 1.

<figure class="fig eqfig"><div class="fig-h">The VECM with a drain</div><div class="eqn"><span class="term"><span class="sym">Δ<b>y</b><sub>t</sub></span><span class="lab">Weekly change</span></span><span class="op">=</span><span class="term"><span class="sym"><b>αβ</b>′<b>y</b><sub>t−1</sub></span><span class="lab">Error correction</span></span><span class="op">+</span><span class="term"><span class="sym"><b>Γ</b>Δ<b>y</b><sub>t−1</sub></span><span class="lab">Momentum</span></span><span class="op">+</span><span class="term"><span class="sym"><b>Bx</b><sub>t</sub></span><span class="lab">Drivers</span></span><span class="op">−</span><span class="term new"><span class="sym"><b>drain</b><sub>t</sub></span><span class="lab">Drain</span></span><span class="op">+</span><span class="term"><span class="sym"><b>c</b></span><span class="lab">Growth</span></span><span class="op">+</span><span class="term"><span class="sym"><b>ε</b><sub>t</sub></span><span class="lab">Shock</span></span></div><figcaption class="fig-cap">The drain is the only new term. With a drain share of zero it vanishes and the model is the one in section 6.</figcaption></figure>

Each week a little of the lift that past spend created drains away. Two numbers set it: the share of each lift that eventually drains, and the half-life of the drain. With a share of zero nothing drains and every lift lasts, as in Part 1. With a share of one the whole lift drains away over time.

This is where a fading lift differs from a long adstock. A long adstock in an MMM is fitted against the trend in sales, so it soaks up drift (section 2). Here the trends are handled by the ties first. The drain only asks how much of a driver's lift is later given back.

The priors are deliberately loose. The drain share is flat between zero and one. The half-life is centred on a year but allows anything from a few months to several years.

To test it we built two more versions of the brand, the same as before apart from the fade. In one, half of each lift drains away. In the other, all of it does. The draining share has a half-life of a year. We fitted all three versions with the drain, including the brand whose lifts last:

- **If lifts last**, the 5-year curve still sits far above the 1-year curve, as in section 7.
- **If half of each lift fades**, the 1-year curve barely moves and the 5-year curve comes down.
- **If all of each lift fades**, the 1-year return is about a fifth lower than for a lasting brand. The 5-year return is about a third of the lasting brand's, because most of the lift drains away within a couple of years.

[[chart fig08_response_curves | Response curves for a 13-week campaign within one and five years, for a brand whose lifts last, half fade or all fade | Extra sales against no campaign, indexed so the true first-year return of a campaign at current spend, if lifts last, is 1. Each brand fitted with the drain; medians with 89% intervals; dashed lines show the truth.]]

Over the first year the three brands look much alike, and the model's intervals hold the truth for all three. Over five years they differ by three times. At current spend the true 5-year returns are 6.2, 4.3 and 2.1 on the chart's scale. The model's are 4.0, 3.7 and 3.2. It pulls all three towards the middle. On the lasting brand that costs something: the VECM without a drain gave 5.2 in section 7, and the drain model gives 4.0.

The alternative is worse. On brands whose lifts fade with a half-life of six months, a year or two, the VECM without a drain overstated the 5-year return by 2 to 6 times, and its intervals missed the truth every time. The drain model ran 1.3 to 3 times high, but its intervals held the truth in most data sets. Five years of data say little about how much fades, and the drain at least says so.

### What settles it: going dark

A dark spell is the clearest evidence there is. If lifts fade, sales sag once spend stops. If they last, sales hold. We switched brand spend off mid-sample on the brand whose lifts fade:

- **Six months off** narrowed the interval on how much fades by about a fifth, and brought the 5-year return from 2 times the truth to 1.5.
- **A year off** halved that interval and brought the 5-year return to 1.3 times the truth. Even a year didn't settle it alone.
- **Running brand in pulses**, two 13-week bursts a year with the same budget, barely narrowed it. Thirteen-week gaps are short next to a year's half-life. Pulses did pin down the 1-year return, with intervals about half as wide.

Keep any dark weeks in the data. If the business can test, a planned dark spell in one region, longer than you expect a lift to last, is worth more than any modelling choice.

## 13. Brands without a tracker: share of search

Everything so far assumes a brand tracker: weekly awareness and consideration going back years. Many brands, especially smaller ones, have nothing like it. Others have it monthly, or only since the last change of agency, which section 8 showed is not enough.

Share of search is a cheap stand-in. It is a brand's searches as a share of all searches for brands in its category. Les Binet [introduced it at the IPA's EffWorks Global in 2020](https://ipa.co.uk/news/binet-presents-fast-cheap-predictive-share-of-search-metric/). In cars, energy and mobile phones it tracked share of market and moved ahead of it, by up to a year for cars.

It suits this method for three reasons:

- **It has the history.** Google Trends gives weekly data for windows of up to five years and monthly data back to 2004. That is the length section 8 asked for.
- **Google's scaling cancels out.** Trends rescales every request to run from 0 to 100, so its numbers aren't search counts. A share divides one brand by its category, so the scale drops out, as long as every brand comes from the same request.
- **It trends.** A brand's share of search drifts with the brand, as awareness does. So it can be tied to base sales with cointegration.

The sketch below is the system we would fit. It hasn't been tested yet.

<figure class="fig diagram"><div class="fig-h">A brand system without a tracker</div><div class="system"><div class="sys-col"><div class="node drv">Brand spend</div><div class="node drv">PR reach</div><div class="node drv">Sponsorship reach</div></div><div class="arrow">→</div><div class="sys-col"><div class="node aw">Share of search<small>leads, if it works</small></div></div><div class="arrow">↔</div><div class="sys-col macro"><div class="node drv">Category searches</div><div class="varrow">↓</div><div class="node sa">Base sales<small>follows share of search</small></div></div></div><figcaption class="fig-cap">A sketch, not a fitted model. Two brand series with at most one tie. Category searches stand in for how many people are in the market.</figcaption></figure>

- **Two series, one tie.** Share of search takes the place of awareness and consideration. With two series there is at most one tie, so the model is smaller and needs less data.
- **The same drivers.** Brand spend, PR and sponsorship push on share of search.
- **Category searches from outside.** Total searches for the category measure how many people are in the market. They push on base sales without being part of the brand.
- **Performance spend stays in the MMM**, for the reasons in section 10.

Four questions decide whether it works:

- **Does share of search lead?** Some branded searches come from people who already buy the brand and search its name to reach the website. That part follows sales rather than leading them. The VECM can check this. If share of search closes part of its gap to sales each week, it is a follower, and a lift in it won't carry through.
- **Brand or activation?** Binet found advertising moves share of search twice: a large short-term effect that fades fast, and a smaller one that builds up. The short-term part is activation sitting inside the series meant to measure the brand. That is the base sales problem of section 9 in a new place.
- **Noisy data.** Trends is built from a sample of searches, so the same request can give slightly different numbers on different days. Small brands are rounded to zero in quiet weeks. Averaging several pulls helps.
- **Which brands make the category?** The share depends on the list, and the list is a judgement call. Fix it before fitting and keep it fixed.

The next step is to simulate a brand like this, run the same checks as Part 2, and then fit it to real share of search.

## 14. The first year is firm. The fifth is a range.

A marketing mix model sees the short-term lift that follows a campaign. It misses the slow climb in base sales that comes when a brand grows. This article set out to measure that climb, with three aims.

**Explain the VECM.** Brand metrics and sales trend, and cointegration ties them together so a lasting lift in awareness carries down to sales. On the brand we built, the method found the two ties, picked out awareness as the leader and recovered the response curve. At current spend a campaign earned 0.8 within a year and 5.2 within five on the model, against a true 1.0 and 6.2. Both intervals held the truth.

**Can a real brand's data recover it?** Mostly, yes, as long as every lift lasts. Less data widened the intervals without moving the answer much: from about a quarter either side of the truth on five years of weekly data to most of the way to zero on three years of monthly data. Fewer years also hid the ties: with three years the rank test found them only two times in five. Base sales taken from an MMM's short-term contributions, even with search spend chasing demand, left the long-run return within a few per cent of the truth. What a rough base damaged was the short run: how fast sales respond, and, with the MMM's smooth trend, whether sales look like the leader. So trust the long run more than any speed or half-life.

**Improve it.** The drain, my extension, lets part of each lift drain away. On three versions of the brand, where lifts last, half fade or all fade, the true 5-year returns are 6.2, 4.3 and 2.1. The drain model gives 4.0, 3.7 and 3.2, and its 1-year intervals hold the truth for all three. The data say little about how much fades. A year dark halved that uncertainty, and even that didn't settle it. For brands without a tracker, share of search could take the place of awareness and consideration. Whether it leads sales is the first thing to test.

### Try it yourself

Everything in this article was fitted with [bayesian_vecm](https://github.com/raz1470/bayesian_vecm), an open-source Python package. To run it on your own brand, start with [the notebook built for your own data](https://github.com/raz1470/bayesian_vecm/blob/main/notebooks/guides/08_response_curves_on_your_data.ipynb): edit one cell and it runs every check in order and draws the response curves.

### What comes next

The method answers a lot, but the hardest question is still the one in the title. A few directions would help:

- **Telling lasting from fading.** The data alone struggle to say how much of a lift drains away. Planned tests would add the missing evidence: going dark in some regions and not others for longer than the lift is expected to last. So would priors on fade rates drawn from many brands, such as the IPA databank.
- **Pooling across brands and markets.** One brand rarely has enough history to pin down the drain. A hierarchical model fitted across regions or sister brands could share that information.
- **One model instead of two.** Base sales are handed over from an MMM, and sections 9 and 10 showed that is where the short run goes wrong. Fitting the short-term and long-term parts together would carry the uncertainty through rather than losing it at the handover.
- **Ties that change.** A pandemic, a change to the tracker or a repositioning can move the long-run links. Allowing the ties to shift at known dates, or slowly over time, would make the model safer on long histories.
- **Competitors and the category.** Our brand moves on its own. Real brands share a market, and part of a lasting lift may come at a rival's expense or from growth in the category.

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
