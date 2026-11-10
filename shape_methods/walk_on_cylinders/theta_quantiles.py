import warnings
import numpy as np
from scipy.stats import norm
from scipy.special import jv, loggamma
from scipy.optimize import brentq

from .bessel_zeros_refactored import get_bessel_zeros


def H_theta(dim: int, theta: float, zeros: np.ndarray) -> float:
    """H_theta = P(H_1 <= theta) from the Bessel-Fourier series (2.3) of the paper."""
    nu = dim / 2.0 - 1.0
    z = np.asarray(zeros, dtype=float)
    # 1 - sum_n z^(nu-1) / (2^(nu-1) Gamma(nu+1) J_{nu+1}(z)) * exp(-z^2 theta / 2), computed in logs
    J = jv(nu + 1.0, z)
    log_mag = (nu - 1.0) * np.log(z) - (nu - 1.0) * np.log(2.0) - loggamma(nu + 1.0) - np.log(np.abs(J)) - 0.5 * z**2 * theta
    return float(1.0 - np.sum(np.sign(J) * np.exp(log_mag)))


# The alternating Bessel series loses accuracy in double precision for large d (checked up to d = 200). Above this
# dimension the normal approximation theta_p(d) ~ (1/d) (1 + z_p sqrt(2/(d+2))) of the paper (Sec. 6.2) is used instead.
SERIES_MAX_DIM = 200


def theta_quantile_normal(dim: int, p: float) -> float:
    """Normal approximation of theta_p(d); accurate for large d only."""
    return float((1.0 + norm.ppf(p) * np.sqrt(2.0 / (dim + 2.0))) / dim)


def theta_quantile(dim: int, p: float, n_zeros: int = 60) -> float:
    """theta_p(d): the root of H_theta = p (normal approximation above SERIES_MAX_DIM)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0, 1)")
    if dim > SERIES_MAX_DIM:
        warnings.warn(f"theta_quantile: dim={dim} > {SERIES_MAX_DIM}, using the normal approximation instead of the Bessel series.")
        return theta_quantile_normal(dim, p)
    # H_theta is increasing in theta; for p in [0.1, 0.9] the root sits between about 0.3/d and 2/d.
    # Scan down from the top of that range and bracket the root as soon as H_theta drops below p. This stays
    # clear of tiny theta, where the alternating series suffers cancellation in double precision (large d).
    hi = 3.0 / dim
    lo_floor = 0.05 / dim
    n = n_zeros
    while True:
        zeros = np.asarray(get_bessel_zeros(dim, n), dtype=float)
        if 0.5 * zeros[-1] ** 2 * lo_floor >= 40.0 or n >= 4000:
            break
        n *= 2
    f = lambda th: H_theta(dim, th, zeros) - p
    b = hi
    if f(b) < 0:
        raise ValueError(f"theta_quantile: H_theta(3/d) < p for dim={dim}, p={p}")
    a = b
    while f(a) > 0:
        b = a
        a *= 0.95
        if a < lo_floor:
            raise ValueError(f"theta_quantile: could not bracket the root for dim={dim}, p={p}")
    return float(brentq(f, a, b, xtol=1e-12))


def get_theta_quantiles(dim: int, ps=(0.1, 0.5, 0.9)) -> np.ndarray:
    return np.array([theta_quantile(dim, p) for p in ps])
