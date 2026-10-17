"""Adaptive Networks of Beliefs testbed (see docu/spec-anb-package.md).

`ANBmodel.py` stays the frozen reference model; this package is its refactor.
"""

import logging

# Silent unless the caller configures logging, e.g. `logging.basicConfig(level=logging.INFO)`.
logging.getLogger(__name__).addHandler(logging.NullHandler())
