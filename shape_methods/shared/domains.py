import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# Defining the Domain as a generic class that requires 3 functions to operate
class Domain:
    def distance_to_boundary(self, x: np.ndarray) -> float:
        raise NotImplementedError
    def project_to_boundary(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    def is_inside(self, x: np.ndarray) -> bool:
        raise NotImplementedError
    
# Defining the Domain as a Circle in dimension d
@dataclass
class BallDomain(Domain):
    R: float = 1.0

    def distance_to_boundary(self, x: np.ndarray) -> float:
        return self.R - np.linalg.norm(x)

    def project_to_boundary(self, x: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(x)
        if r == 0.0:
            out = np.zeros_like(x)
            out[0] = self.R
            return out
        return (self.R / r) * x

    def is_inside(self, x: np.ndarray) -> bool:
        return np.linalg.norm(x) < self.R

# Defining the Domain as a Rectangle/Hyperrectangle in dimension d
@dataclass
class BoxDomain(Domain):
    a: np.ndarray
    b: np.ndarray

    def distance_to_boundary(self, x: np.ndarray) -> float:
        return float(np.min(np.minimum(x - self.a, self.b - x)))

    def project_to_boundary(self, x: np.ndarray) -> np.ndarray:
        lower = x - self.a
        upper = self.b - x
        nearest = np.minimum(lower, upper)
        idx = np.argmin(nearest)

        x_proj = x.copy()
        x_proj[idx] = self.a[idx] if lower[idx] <= upper[idx] else self.b[idx]

        return x_proj

    def is_inside(self, x: np.ndarray) -> bool:
        return bool(np.all(self.a <= x) and np.all(x <= self.b))

# Defining the Domain as an L-shape in Dimension d
@dataclass
class LShapeDomain(Domain):
    a: np.ndarray
    b: np.ndarray
    cut: np.ndarray  # corner of the removed region, e.g. np.array([0.0, 0.0])

    def is_inside(self, x: np.ndarray) -> bool:
        in_rect = bool(np.all(self.a <= x) and np.all(x <= self.b))
        in_cut  = bool(x[0] > self.cut[0] and x[1] > self.cut[1])
        return bool(in_rect and not in_cut)

    def distance_to_boundary(self, x: np.ndarray) -> float:
        if not self.is_inside(x):
            return 0.0

        candidates = list(np.minimum(x - self.a, self.b - x))

        # distance to vertical inner edge: x[0] = cut[0], y >= cut[1]
        if x[1] >= self.cut[1]:
            candidates.append(abs(x[0] - self.cut[0]))

        # distance to horizontal inner edge: y[1] = cut[1], x >= cut[0]
        if x[0] >= self.cut[0]:
            candidates.append(abs(x[1] - self.cut[1]))

        # distance to re-entrant corner
        if x[0] <= self.cut[0] and x[1] <= self.cut[1]:
            candidates.append(np.linalg.norm(x - self.cut))
        return float(np.min(candidates))

    def project_to_boundary(self, x: np.ndarray) -> np.ndarray:
        if not self.is_inside(x):
            return x.copy()

        candidates = []

        # outer rectangle boundaries
        for i in range(len(x)):
            p = x.copy()
            p[i] = self.a[i]
            candidates.append((abs(x[i] - self.a[i]), p))

            p = x.copy()
            p[i] = self.b[i]
            candidates.append((abs(self.b[i] - x[i]), p))

        # inner vertical edge: x = cut[0], y >= cut[1]
        if x[1] >= self.cut[1]:
            p = x.copy()
            p[0] = self.cut[0]
            candidates.append((abs(x[0] - self.cut[0]), p))

        # inner horizontal edge: y = cut[1], x >= cut[0]
        if x[0] >= self.cut[0]:
            p = x.copy()
            p[1] = self.cut[1]
            candidates.append((abs(x[1] - self.cut[1]), p))

        # re-entrant corner
        if x[0] <= self.cut[0] and x[1] <= self.cut[1]:
            candidates.append((np.linalg.norm(x - self.cut), self.cut.copy()))

        _, x_proj = min(candidates, key=lambda t: t[0])
        return x_proj
    
@dataclass
class HalfSphereDomain(Domain):
    R: float = 1.0

    def is_inside(self, x: np.ndarray) -> bool:
        return bool(np.linalg.norm(x) < self.R and x[0] > 0.0)
    
    def distance_to_boundary(self, x: np.ndarray) -> float:
        if not self.is_inside(x):
            return 0.0

        dist_to_sphere = self.R - np.linalg.norm(x)
        dist_to_plane = x[0]

        return float(min(dist_to_sphere, dist_to_plane))
    
    def project_to_boundary(self, x: np.ndarray) -> np.ndarray:
        if not self.is_inside(x):
            return x.copy()

        dist_to_sphere = self.R - np.linalg.norm(x)
        dist_to_plane = x[0]

        if dist_to_sphere <= dist_to_plane:
            r = np.linalg.norm(x)

            if r == 0.0:
                out = np.zeros_like(x)
                out[0] = self.R
                return out
            
            return (self.R / r) * x
        
        else:
            out = x.copy()
            out[0] = 0.0
            return out

# domain = LShapeDomain(a=np.array([-1.0, -1.0]), b=np.array([1.0, 1.0]), cut=np.array([0.0, 0.0]))
#
# x1 = np.array([-0.5, 0.49])
# x2 = np.array([0.49, 0.51])
# x3 = np.array([0.51, 0.49])
# x4 = np.array([0.75, 0.49])
# x5 = np.array([0.25, 0.35])
#
# for x in [x1, x2, x3, x4, x5]:
#     print(f"x={x}, is_inside={domain.is_inside(x)}, distance_to_boundary={domain.distance_to_boundary(x)}, projected={domain.project_to_boundary(x)}")
#
# domain = LShapeDomain(
#     a=np.array([-1.0, -1.0, -1.0]),
#     b=np.array([ 1.0,  1.0,  1.0]),
#     cut=np.array([0.0, 0.0, 0.0]),
# )
#
# for _ in range(10_000):
#     x = np.random.uniform(domain.a, domain.b)
#
#     if domain.is_inside(x):
#         dist = domain.distance_to_boundary(x)
#         proj = domain.project_to_boundary(x)
#
#         assert dist >= -1e-12
#         assert np.isclose(dist, np.linalg.norm(x - proj), atol=1e-10)
#
#         # projected point should lie on boundary: inside or exactly on cut/box boundary
#         assert domain.is_inside(proj)
#
# d = 5
# domain = LShapeDomain(
#     a=-np.ones(d),
#     b=np.ones(d),
#     cut=np.zeros(d),
# )
#
# test_points = [
#     -0.5 * np.ones(d),              # far from cut
#     np.array([-0.01] * d),          # near re-entrant corner
#     np.array([0.2, -0.01, 0.2, 0.2, 0.2]),  # near one cut face
# ]
#
# for x in test_points:
#     dist = domain.distance_to_boundary(x)
#     proj = domain.project_to_boundary(x)
#
#     print(x)
#     print("inside:", domain.is_inside(x))
#     print("dist:", dist)
#     print("proj:", proj)
#
#     assert domain.is_inside(proj)
#     assert dist >= -1e-12
#     assert np.isclose(dist, np.linalg.norm(x - proj), atol=1e-10)
