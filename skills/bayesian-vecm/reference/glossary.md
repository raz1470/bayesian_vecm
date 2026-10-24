# Glossary

One plain sentence, one brand example and why it matters. Use the plain words with the user and introduce the term once.

| Term | Plain English | Brand example | Why you care |
|---|---|---|---|
| Trending series (I(1), unit root) | A series with no fixed level that it returns to. Its week-to-week changes are stable. | Sales grow for years and never settle back to an old level. | Ordinary regression on trending series finds ties that are not there. |
| Stationary series (I(0)) | A series that keeps returning to the same average. | Weekly brand spend moves around a budget level. | These go in as drivers, not as trending brand metrics. |
| Cointegration (long-run tie) | Two or more trending series that never drift far apart. | Sales and consideration both grow, but sales stay about 1.25 times consideration in logs. | It is what lets a lift in a brand metric become a lasting lift in sales. |
| Cointegrating relation (β) | The equation for one long-run tie. | sales - 1.25 x consideration stays roughly constant. | Tells you which metrics are tied and by how much. |
| Rank | How many long-run ties there are. | Three series sharing one underlying trend have two ties. | Wrong rank changes every long-run answer. |
| Adjustment speed (α) | How fast a series closes a gap when the tie is stretched. | Sales close 24% of their gap to consideration each week. | Who leads and who follows. |
| Weakly exogenous | A series that does not adjust to the ties: it leads. | Awareness moves and the others follow it. | A weakly exogenous metric is where lasting change enters the system. |
| Short-run dynamics (Γ) | How last week's changes move this week's. | A jump in awareness last week lifts consideration this week. | Shapes the path in the first weeks. |
| Drivers (exog, B) | Series set from outside the system. | Spend, PR reach, interest rates. | Their effects are what you value. |
| Error correction | The pull that closes a gap in a long-run tie. | After awareness rises, consideration and then sales catch up. | Why brand effects last instead of fading. |
| Adstock | Spend that keeps working for weeks after it runs. | A TV burst lifts awareness this week and less each week after. | Ignoring it misses about half of a channel's effect. |
| Saturation | Each extra £ buys less as spend grows. | Doubling brand spend does not double its effect. | Needed for any big budget move. |
| Same-week link | One brand metric moving another within the same week. | A lift in awareness lifts consideration in the same week. | Decides how fast a campaign reaches the bottom of the funnel. |
| Instrument, exclusion | A driver that moves one metric but reaches another only through it. | Brand TV moves awareness and reaches consideration only via awareness. | The only way to measure a same-week link. It is an assumption. |
| Impulse response (IRF, GIRF) | The path of every series after a one-off shock. | Sales over the year after awareness jumps 1%. | The main picture of how long brand lasts. |
| Return per £ (`exog_irf`) | Extra sales from one more unit of a driver, summed over a window. | £2.20 in year one, £12.50 over five years, per £1 of brand spend. | The number a budget decision needs. |
| Granger causality | Whether one series helps predict another. | Past awareness helps predict consideration. | A prediction test, not proof of cause. |
| Evidence score | How clearly a posterior rules out zero, from 0 to 1. | 0.99 for awareness -> consideration. | Says whether there is an effect, not how big it is. |
| FEVD | How much of the uncertainty in a forecast comes from each series. | A year ahead, half of sales uncertainty traces back to awareness. | How much each metric matters. |
| Posterior, interval | What the data and the prior say together, as a range. | £1,600 to £2,800, 90% interval. | Every number comes with a range. |
| Divergences, r-hat, ESS | Checks that the sampler explored the posterior properly. | 0 divergences, r-hat 1.01 or below, ESS in the hundreds. | Results are unreliable until these pass. |
| Base sales | Sales without the short-term effect of marketing. | Sales minus the MMM's weekly media contributions. | The usual sales input to a long-term model. |
| UCM | Unobserved components model: a series split into level, seasonality, drivers and noise. | `statsmodels` `UnobservedComponents`. | One clean way to build base sales. |
| Extrapolation | An answer about conditions the data never saw. | Switching brand spend off when it never fell below 40% of average. | The range is wide and the shape is assumed. |
