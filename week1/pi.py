"""Estimate pi by throwing random darts at the unit square."""

import random


def estimate_pi(n, seed):
    """Return 4 * (darts within distance 1 of the origin) / n."""
    rng = random.Random(seed)
    inside = 0
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1:
            inside += 1
    return 4 * inside / n
