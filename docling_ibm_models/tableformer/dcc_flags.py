"""Switches for the TableFormer speed-ups of the DCC docling-serve image.

``DCC_TABLES_<NAME>=0`` turns one off (the upstream code path), ``=1`` turns one on.
All give the same tables; RESIZE is off by default because it gained nothing measurable.
"""

import os

_DEFAULTS = {"MATCH": "1", "DECODER": "1", "PREP": "1", "RESIZE": "0"}


def on(name: str) -> bool:
    return os.environ.get(f"DCC_TABLES_{name}", _DEFAULTS.get(name, "0")) == "1"
