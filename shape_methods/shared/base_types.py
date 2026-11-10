import numpy as np
from dataclasses import dataclass

@dataclass
class PathResult:
    exited: bool
    x_final: np.ndarray
    exit_time: float | None = None
    x_boundary: np.ndarray | None = None
    num_steps: int = 0

@dataclass
class Trajectory:
    T: np.ndarray  # length N + 1
    X: np.ndarray  # shape of (N + 1, d)
    alpha: np.ndarray  # length N
    R: np.ndarray  # length N