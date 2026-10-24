# <Brand>: how long does brand marketing last?

<!-- Fill every section. Lead with the answer. Every number has its 90% range. Plain English first, terms second. -->

## The answer

<Two or three sentences answering the question from step 1 of the workflow. Example: "One more £1 of brand TV returns about £2.20 in the first year and £12.50 over five years (very likely between £9.30 and £16.00). Less than a fifth arrives in year one, which is why the MMM's figure of £1.80 undervalues it.">

## What we did

- **Data:** <weekly / monthly>, <dates>, <series>. Sales are <total / base sales built by ...>.
- **Model:** a Bayesian vector error correction model (`bayesian_vecm` <version>). It learns how sales and brand metrics move together over years and how spend moves them.
- **Options:** <adstock / saturation / same-week links / none>, because <reason>.

## What we found

### The long-run ties
<Which metrics are tied, in what proportion, and who leads. One sentence each.>

### How long effects last
<The impulse response in words, with a chart.>

### What each channel is worth
| Channel | Unit | 1 year | 5 years | Share in year 1 |
|---|---|---|---|---|
| | | £ (range) | £ (range) | |

### <Response curve / same-week links, if used>
<Sentences from reference/interpretation.md.>

## How sure we are

- **Sampler checks:** <divergences, r-hat, ESS>.
- **Residual checks:** <result>.
- **Hold-out year:** <forecast against random walk>.
- **Sensitivity:** <return per £ under the neighbouring rank, one more lag, with and without each option>.

## Decisions and assumptions

| Decision | Choice | Why | Who decided |
|---|---|---|---|
| Variables | | | |
| Rank | | | |
| Lags | | | |
| Options | | | |
| Exclusions (if links) | | | |

Assumptions the results rest on:
- Spend is set ahead of time and does not respond to the same week's sales or brand health.
- <Each exclusion, in business words.>
- <Anything else from the decision log.>

## Limits

<Not optional. Take each relevant row of reference/pitfalls.md and say how it applies. At minimum cover four things: prediction is not proof of cause; any extrapolated figure; any channel missing from the data; the data length.>

## Next steps

<One to three, concrete.>
