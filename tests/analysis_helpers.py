"""Helpers the `anb.analysis` tests share: the warnings a call issued, and LaTeX compared without whitespace."""

import re
import warnings


def expected_warnings(func, *args, **kwargs):
    """`func(...)` and the categories it warned with (the stray-file warning of the index left out)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = func(*args, **kwargs)
    return result, {w.category for w in caught if "legacy naming scheme" not in str(w.message)}


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)
