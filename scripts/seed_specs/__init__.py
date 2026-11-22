"""Seed instance specs, one module per batch (batch_00.py, batch_01.py, ...).

Each module defines `SPECS: list[dict]`. load_specs() discovers all batches in name order.
"""
from __future__ import annotations

import importlib
import pkgutil


def batch_names() -> list[str]:
    return sorted(m.name for m in pkgutil.iter_modules(__path__) if m.name.startswith("batch_"))


def load_specs(batches: list[str] | None = None) -> list[dict]:
    specs = []
    for name in batches or batch_names():
        specs.extend(importlib.import_module(f"{__name__}.{name}").SPECS)
    return specs
