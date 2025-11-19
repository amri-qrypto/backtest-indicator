"""Project-wide interpreter tweaks for compatibility."""
from __future__ import annotations

import contextlib
import sys
import warnings
from pathlib import Path

# Ensure the ``src`` directory is importable regardless of the working
# directory used to launch Python or Jupyter.  This keeps imports such as
# ``from backtest import run_backtest`` functional from notebooks and scripts
# without requiring manual ``PYTHONPATH`` management.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"

# ``import src`` expects the *parent* directory (the project root) to be on
# ``sys.path``.  Previously we were inserting ``PROJECT_ROOT / "src"`` which
# makes Python look for ``src/src`` and therefore breaks every notebook / CLI
# execution unless the working directory already happened to be the root.  By
# inserting the project root itself (and keeping the ``src`` sub-folder as a
# fallback) we guarantee deterministic imports regardless of where Python is
# launched from.
for candidate in (PROJECT_ROOT, SRC_PATH):
    if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

# ``qf_lib`` 1.x still instantiates ``pd.Series`` with the ``fastpath`` keyword,
# which pandas has deprecated.  The warning is harmless but extremely noisy
# because every import of ``qf_lib.containers.series.qf_series`` triggers it.
# Filtering it here (before any project modules import qf_lib) keeps both the
# CLI and ``pytest`` output clean without having to touch third-party code.
warnings.filterwarnings(
    "ignore",
    message="The 'fastpath' keyword in pd.Series is deprecated",
    category=DeprecationWarning,
    module=r"qf_lib\.containers\.series\.qf_series",
)

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
