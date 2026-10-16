import pytest

from legacy_oracle import load_legacy


@pytest.fixture
def legacy():
    """A fresh legacy `ANBmodel` in ensemble mode (`detail=False`)."""
    return load_legacy()
