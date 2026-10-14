"""Trajectory recording for the separate 8-sample, 1000-step DDPM run.

Frames are written at t = 1000, 960, ..., 0. Each frame holds the current
noisy image x and the predicted clean image x0 as flat sample-major arrays of
n*size*size values, row-major within each image, rounded to 3 decimals. At
t = 0 both arrays contain the final image.
"""

from __future__ import annotations

import numpy as np


def trajectory_levels(t_steps: int = 1000) -> list[int]:
    return list(range(t_steps, 0, -40))


def _flatten(x) -> list[float]:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr.reshape(arr.shape[0], -1).reshape(-1)
    return np.round(arr, 3).tolist()


class TrajectoryRecorder:
    def __init__(self, n: int, size: int, levels):
        self.n = n
        self.size = size
        self.levels = [int(t) for t in levels]
        self._frames: list[dict] = []
        self._seen: set[int] = set()

    def observe(self, t, x, x0) -> None:
        t = int(t)
        if t in self.levels and t not in self._seen:
            self._seen.add(t)
            self._frames.append({"t": t, "x": _flatten(x), "x0": _flatten(x0)})

    def final(self, x) -> None:
        self._frames.append({"t": 0, "x": _flatten(x), "x0": _flatten(x)})

    def frames(self) -> list[dict]:
        return list(self._frames)