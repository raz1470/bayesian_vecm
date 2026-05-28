"""Package-wide constants shared across modules.

Keeping shared literals here prevents the duplication that arises when two
modules both need the same sentinel values (e.g. valid deterministic-term
codes) and avoids any risk of the two copies drifting apart.
"""

from __future__ import annotations

#: Deterministic-term codes accepted in v0.
#:
#: * ``"n"``  — no deterministic terms (default).
#: * ``"co"`` — constant *outside* the cointegration relation; appended as a
#:   column to ``delta_x``.
#: * ``"ci"`` — constant *inside* the cointegration relation; appended as a
#:   column to ``y_lag1``.
#: * ``"lo"`` — linear trend *outside* the cointegration relation; appended
#:   as a column to ``delta_x``.
#: * ``"li"`` — linear trend *inside* the cointegration relation; appended as
#:   a column to ``y_lag1``.
#:
#: Compound Johansen codes (cases 4 and 5, e.g. ``"colo"``, ``"cili"``) are a
#: planned follow-up and are rejected in v0.
VALID_DETERMINISTIC: frozenset[str] = frozenset({"n", "co", "ci", "lo", "li"})
