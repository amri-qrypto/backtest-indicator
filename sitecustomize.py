"""Project-wide interpreter tweaks for compatibility."""
from __future__ import annotations

import contextlib

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
