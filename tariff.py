# Copyright (c) 2026 Anurak
"""Slab-based electricity bill calculation."""

# PLACEHOLDER slabs: (units in slab, rate per unit). None = "all remaining units".
# Replace with your own utility's tariff.
DEFAULT_SLABS = [(100.0, 0.0), (100.0, 4.0), (200.0, 6.0), (None, 8.0)]


def bill(units, slabs=DEFAULT_SLABS, fixed=0.0):
    """Total charge for `units` kWh: each slab is billed at its own rate."""
    remaining, total = max(float(units), 0.0), float(fixed)
    for size, rate in slabs:
        if remaining <= 0:
            break
        used = remaining if size is None else min(remaining, size)
        total += used * rate
        remaining -= used
    return round(total, 2)