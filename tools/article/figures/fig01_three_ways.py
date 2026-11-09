"""Section 1 chart: three ways a campaign can play out. A sketch, not an estimate."""

import numpy as np
from _spec import r, save

W, START, LEN = 156, 8, 13
t = np.arange(W)
spend = np.where((t >= START) & (t < START + LEN), 1.0, 0.0)

# Adstock: geometric carryover of the burst, back to zero within weeks.
adstock = np.zeros(W)
for k in range(1, W):
    adstock[k] = 0.7 * adstock[k - 1] + spend[k]
# Lasting lift: each week of spend adds a step that stays; the level eases in.
lasting = np.zeros(W)
target = np.cumsum(spend)
for k in range(1, W):
    lasting[k] = lasting[k - 1] + 0.25 * (target[k] - lasting[k - 1])
# Fading lift: the same steps, drained back towards trend with a half-life of a year.
delta = 0.5 ** (1 / 52)
fading = np.zeros(W)
stock = 0.0
for k in range(1, W):
    stock = delta * stock + spend[k]
    fading[k] = fading[k - 1] + 0.25 * (stock - fading[k - 1])

scale = lasting.max()
panels = [
    ("Adstock", "fades to zero within weeks", adstock / adstock.max() * scale, "orange"),
    ("A lasting lift", "steps up and stays", lasting, "blue"),
    ("A fading lift", "rises, then drains back", fading, "aqua"),
]
save(
    "fig01_three_ways",
    {
        "title": "The same campaign can fade in weeks, last for good, or drain away over years",
        "height": 200,
        "shareY": True,
        "minPanel": 180,
        "panels": [
            {
                "title": title,
                "subtitle": sub,
                "titleColor": colour,
                "x": {
                    "min": 0,
                    "max": W,
                    "ticks": [[52 * k, str(k)] for k in range(4)],
                    "label": "Years",
                },
                "y": {
                    "min": 0,
                    "max": float(scale) * 1.08,
                    "ticks": [],
                    "label": "Extra sales" if k == 0 else "",
                },
                "layers": [
                    {"type": "bars", "x": r(t), "y": r(spend * 0.25 * scale), "color": "grey"},
                    {"type": "line", "x": r(t), "y": r(y), "color": colour, "width": 2.6},
                ]
                + (
                    [
                        {
                            "type": "text",
                            "x": START + LEN + 6,
                            "y": 0.18 * float(scale),
                            "text": "13 weeks of spend",
                            "color": "muted",
                        }
                    ]
                    if k == 0
                    else []
                ),
            }
            for k, (title, sub, y, colour) in enumerate(panels)
        ],
    },
)
