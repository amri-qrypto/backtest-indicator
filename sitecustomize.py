"""Project-wide interpreter tweaks for compatibility."""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

# Ensure the ``src`` directory is importable regardless of the working
# directory used to launch Python or Jupyter.  This keeps imports such as
# ``from backtest import run_backtest`` functional from notebooks and scripts
# without requiring manual ``PYTHONPATH`` management.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    src_str = str(SRC_PATH)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

# The "matplotlib_inline" backend bundled with some notebook environments
# expects ``matplotlib.rcParams`` to expose a private ``_get`` method.  Older
# Matplotlib releases don't provide it which results in an ``AttributeError``
# the moment ``matplotlib.pyplot`` is imported.  The notebook execution in this
# project therefore fails before any plots can be created.
#
# To keep the notebooks environment agnostic we patch the missing attribute at
# interpreter start-up (Python automatically imports ``sitecustomize`` if it is
# available on the ``sys.path``).  When running against a newer Matplotlib
# nothing happens as the attribute already exists.
with contextlib.suppress(Exception):
    import matplotlib

    rc_params = getattr(matplotlib, "rcParams", None)
    if rc_params is not None and not hasattr(rc_params, "_get"):
        # Matplotlib exposes ``dict.get`` so we simply mirror the behaviour to
        # satisfy the inline backend without modifying downstream code.
        rc_params._get = rc_params.get  # type: ignore[attr-defined]
