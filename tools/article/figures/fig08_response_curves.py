"""Section 8 chart: what a 13-week brand campaign earns, if lifts last, half fade or all fade.

Three versions of the simulated brand, the same apart from the fade: every lift
lasts; half of each lift fades; all of it fades (half-life 52 weeks for the
fading share). Brand spend has carryover (0.5) and saturation (5). Each is
fitted with the drain on the three channels, as section 9 recommends. Curves are
extra sales against no campaign, summed over 1 and 5 years, indexed so the true
1-year return of a campaign at current spend, if lifts last, is 1.
"""

import warnings

import numpy as np
from _spec import r as r_
from _spec import save

from bayesian_vecm import BayesianVECM, simulate_brand_data
from bayesian_vecm._saturation import logistic_saturation

warnings.filterwarnings("ignore")

CH = ["brand_spend", "pr_reach", "sponsorship_reach"]
SEED, WEEKS, HORIZON = 1, 13, 260
MULT = np.linspace(0, 1.5, 7)
VERSIONS = {
    "If lifts last": {},
    "If half of each lift fades": {"exog_fade": dict.fromkeys(CH, 52.0), "exog_fade_share": 0.5},
    "If all of each lift fades": {"exog_fade": dict.fromkeys(CH, 52.0)},
}


def true_sales(sim, plan, mean):
    """Log sales gap when brand spend follows ``plan`` instead of its average."""
    t, j = sim.truth, 0
    w, lam, scale = t.adstock[j], t.saturation[j], t.spend_scale[j]
    stock = sim.exog["brand_spend"].iloc[0]
    for x in sim.exog["brand_spend"].iloc[1:]:
        stock = (1 - w) * x + w * stock

    def growth(path):
        a, out = stock, []
        for x in path:
            a = (1 - w) * x + w * a
            out.append(t.b_saturation[:, j] * logistic_saturation(a / scale, lam))
        return np.array(out)

    shift = growth(plan) - growth(np.full(len(plan), mean))
    delta = t.fade_decay[j]
    share = t.fade_share if np.isfinite(t.fade_half_life[j]) else 0.0
    pi = t.alpha @ t.beta.T
    y, dy, fade, out = np.zeros(3), np.zeros(3), 0.0, []
    for k in range(len(plan)):
        if k > 0:
            fade = delta * fade + shift[k - 1]
        dy = pi @ y + t.gamma @ dy + shift[k] - share * (1 - delta) * fade
        y = y + dy
        out.append(y[0])
    return np.array(out)


def curves(name, kw):
    sim = simulate_brand_data(
        seed=SEED,
        exog_adstock={"brand_spend": 0.5},
        exog_saturation={"brand_spend": 5.0},
        **kw,
    )
    endog, exog = sim.endog, sim.exog.copy()
    other = ["interest_rate", "consumer_confidence"]
    exog[other] = exog[other] - exog[other].mean()
    model = BayesianVECM(
        k_ar_diff=1,
        coint_rank=2,
        deterministic="co",
        adstock=["brand_spend"],
        saturation=["brand_spend"],
        fading=CH,
    )
    model.fit(
        endog,
        exog=exog,
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.95,
        random_seed=SEED,
        progressbar=False,
    )
    names = list(exog.columns)
    j = names.index("brand_spend")
    mean = float(exog["brand_spend"].mean())
    base_row = exog.mean().to_numpy().copy()
    base_row[[names.index(c) for c in ["pr_reach", "sponsorship_reach", *other]]] = 0.0
    fit, true = [], []
    for m in MULT:
        base = np.tile(base_row, (HORIZON, 1))
        plan = base.copy()
        plan[:WEEKS, j] = m * mean
        paths = []
        for x in (base, plan):
            fc = model.sample_posterior_predictive(HORIZON, exog_future=x, random_seed=0)
            y = fc.posterior_predictive["y"].sel(variable="organic_sales").values
            paths.append(y.reshape(-1, HORIZON))
        fit.append(np.expm1(paths[1] - paths[0]))  # weeks of sales, per draw
        tplan = np.full(HORIZON, mean)
        tplan[:WEEKS] = m * mean
        true.append(np.expm1(true_sales(sim, tplan, mean)))
    fit, true = np.array(fit), np.array(true)
    out = {}
    for label, w in (("1 year", 52), ("5 years", HORIZON)):
        f = fit[..., :w].sum(axis=-1)
        t = true[:, :w].sum(axis=-1)
        out[label] = (f - f[0], t - t[0])
    share = model.idata_.posterior["fade_share"].values[..., 0].ravel()
    print(name, "fade share", np.round(np.percentile(share, [5, 50, 95]), 2))
    return out


res = {name: curves(name, kw) for name, kw in VERSIONS.items()}
mean = float(
    simulate_brand_data(
        seed=SEED, exog_adstock={"brand_spend": 0.5}, exog_saturation={"brand_spend": 5.0}
    )
    .exog["brand_spend"]
    .mean()
)
i1 = int(np.argmin(np.abs(MULT - 1)))
norm = res["If lifts last"]["1 year"][1][i1]
for name, r in res.items():
    for label, (f, t) in r.items():
        med = np.median(f, axis=1)
        print(name, label, "model", np.round(med[i1] / norm, 2), "truth", round(t[i1] / norm, 2))

X = MULT * mean * 1000.0  # £ a week; brand spend is in £1,000s
CURRENT = mean * 1000.0


def curve_layers(r):
    layers = [
        {"type": "vline", "x": CURRENT, "label": "current"},
    ]
    for label, colour in (("5 years", "blue"), ("1 year", "orange")):
        f, t = r[label]
        lo, med, hi = np.percentile(f / norm, [5.5, 50, 94.5], axis=1)
        layers += [
            {"type": "band", "x": r_(X), "lo": r_(lo), "hi": r_(hi), "color": colour},
            {
                "type": "line",
                "x": r_(X),
                "y": r_(t / norm),
                "color": "ink",
                "dash": True,
                "width": 1.2,
                "name": f"Truth, {label}",
            },
            {
                "type": "line",
                "x": r_(X),
                "y": r_(med),
                "color": colour,
                "width": 2.4,
                "name": f"Within {label}",
                "band": [r_(lo), r_(hi)],
            },
        ]
    return layers


XAXIS = {
    "min": 0,
    "max": float(X[-1]),
    "ticks": [[v, f"£{v:,.0f}"] for v in (0, 500, 1000)],
    "label": "Brand spend a week during the campaign",
    "hover": "gbp_week",
}
LEGEND = [
    {"label": "Earned within 5 years", "color": "blue", "kind": "line"},
    {"label": "Earned within 1 year", "color": "orange", "kind": "line"},
    {"label": "Truth", "color": "ink", "kind": "dash"},
    {"label": "89% interval", "color": "muted", "kind": "band"},
]

save(
    "fig08_response_curves",
    {
        "title": "Over five years the drain model pulls all three brands towards the middle",
        "legend": LEGEND,
        "height": 260,
        "shareY": True,
        "minPanel": 200,
        "panels": [
            {
                "title": name,
                "x": {**XAXIS, "label": "Brand spend a week"},
                "y": {
                    "min": 0,
                    "label": "Extra sales (index)",
                },
                "layers": curve_layers(r),
            }
            for name, r in res.items()
        ],
    },
)
