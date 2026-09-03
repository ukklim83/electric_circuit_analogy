"""Compatibility facade for the unified circuit solver and optimizers.

The implementation lives in ``codes.electric_analogy``. Keeping this facade at
the repository root preserves ``import electric_analogy`` without duplicating
the legacy solver.
"""

from codes.electric_analogy import *
