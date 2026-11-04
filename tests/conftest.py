"""Shared pytest configuration.

Tests that sample a PyMC model take seconds each; the rest take milliseconds.
Any test whose body, or any fixture it uses, calls ``.fit(`` or ``pm.sample(``
is marked ``slow`` automatically, so ``pytest -m "not slow"`` gives a quick run
and CI can schedule the sampling tests separately.
"""

from __future__ import annotations

import inspect

import pytest

_SAMPLING_CALLS = (".fit(", "pm.sample(")


def _samples(obj: object) -> bool:
    try:
        source = inspect.getsource(obj)
    except (OSError, TypeError):
        return False
    return any(call in source for call in _SAMPLING_CALLS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    cache: dict[object, bool] = {}
    for item in items:
        funcs = [getattr(item, "function", None)]
        info = getattr(item, "_fixtureinfo", None)
        if info is not None:
            for defs in info.name2fixturedefs.values():
                funcs.extend(d.func for d in defs)
        for func in funcs:
            if func is None:
                continue
            if func not in cache:
                cache[func] = _samples(func)
            if cache[func]:
                item.add_marker(pytest.mark.slow)
                break
