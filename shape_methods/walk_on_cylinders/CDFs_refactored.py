import numpy as np
from scipy.special import jv, jvp, gamma, loggamma

def build_cdfs(dim, theta, zeros, INV_R=2000, INV_T=2000):
    """
    Same API as your original:
      returns r_star, cdf_r, p_surv0, t_star, raw_t
    with numerically safer internals for large dimensions.
    """
    # ----- setup -----
    nu  = dim/2.0 - 1.0
    x   = np.asarray(zeros, dtype=np.float64)                 # z_{ν,n}
    
    lam = (x*x) / 2.0    # eigenvalues of (1/2)Δ on unit ball = z_{ν,n}^2 / 2

    # grids: keep uniform to preserve your API/behavior
    r_star = np.linspace(0.0, 1.0, int(INV_R), dtype=np.float64)
    t_star = np.linspace(0.0, 1.0, int(INV_T), dtype=np.float64)

    # ----- stable J_{ν+1}(z_{ν,n}) at roots via derivative identity -----
    # At zeros of J_ν, we have J_{ν+1}(z) = - J'_ν(z) [Watson treatise page 45 (4)]
    Jnu1_roots = -jvp(nu, x, 1)  # derivative w.r.t. argument

    # ----- weights (temporal & spatial) -----
    # coef = 1 / (2^{ν-1} Γ(ν+1))
    coef = 1.0 / (2.0**(nu - 1.0) * gamma(nu + 1.0))

    # A_den = coef * x^{ν-1} / J_{ν+1}(x)
    # Use logs for A_den downstream; still keep a non-log version for shapes/checks if needed
    # log A_den = (nu-1)*log x - (nu-1)*log 2 - log Γ(nu+1) - log J_{ν+1}(x)
    eps = np.finfo(np.float64).tiny
    logA_den = (nu - 1.0) * np.log(np.maximum(x, eps)) \
               - (nu - 1.0) * np.log(2.0) \
               - loggamma(nu + 1.0) \
               - np.log(np.maximum(np.abs(Jnu1_roots), eps))
    # Keep the correct sign for A_den (J_{ν+1} may change sign):
    sign_A_den = np.sign(Jnu1_roots)
    # log with sign: A_den = (coef / J_{ν+1}) * x^{ν-1}; 'coef' part goes into logA_den above implicitly.
    # Since coef > 0, sign is just sign of 1/J_{ν+1} = sign(J_{ν+1})
    # We'll incorporate sign when summing (below we use pure log-sum-exp on magnitudes, so A_den must be ≥0).
    # For time/survival series the standard closed form yields positive terms, so force nonnegative by magnitude:
    # (If your zeros/ordering create alternating signs numerically, taking abs here is the safe practical choice.)
    logA_den_mag = logA_den  # magnitude in log-domain (we treat terms as positive in the survival/time sums)


    # ----- exponentials for full nondimensional step theta -----
    exp_theta = np.exp(-lam * theta)  # e^{- (z^2/2) theta }

    # ----- survival probability p_surv0 with log-sum-exp -----
    # p_surv0 = Σ_n A_den * e^{-lam*theta}
    # => log terms = logA_den_mag - lam*theta
    log_terms_surv = logA_den_mag - lam * theta
    m = np.max(log_terms_surv)
    p_surv0 = float(np.exp(m) * np.sum(sign_A_den*np.exp(log_terms_surv - m))) #Here is where sign_A_den is multiplied back in.
    # Guard against complete underflow
    if not np.isfinite(p_surv0) or p_surv0 <= 0.0:
        p_surv0 = 0.0

    # ----- spatial numerator (Case 2) with columnwise scaling (handles sign + magnitude) -----
    # A_num = 2 / ( x [J_{ν+1}(x)]^2 )
    # Use derivative-based J_{ν+1} at roots for stability:
    A_num = np.maximum(x, eps)**(nu-1) / np.maximum(Jnu1_roots*Jnu1_roots, eps)
    # F_unnorm(ρ) = Σ_n [ A_num[n] · (r*)^{ν+1} · J_{ν+1}(x_n r*) · e^{- (z_n^2/2) theta} ]
    # Compute J_{ν+1}(x r*) on outer grid:
    Jp1 = jv(nu + 1.0, np.outer(x, r_star))  # shape (N_roots, INV_R)

    # radial weight
    rwt = r_star**(nu + 1.0)
    # terms matrix, shape (N_roots, INV_R)
    terms = (A_num[:, None] * exp_theta[:, None]) * Jp1 
    # Columnwise scaled summation to avoid overflow/underflow while keeping signs:
    scale = np.max(np.abs(terms), axis=0)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    sum_scaled = np.sum(terms / safe_scale, axis=0, dtype=np.float64)
    cdf_num = coef*sum_scaled * safe_scale* rwt
    

    # ----- conditional spatial CDF -----
    if p_surv0 > 0.0:
        cdf_r = cdf_num / p_surv0
    else:
        # if survival ~ 0 (extreme theta or dimension), the conditional is undefined; return zeros
        cdf_r = np.zeros_like(r_star)

    # clip + enforce monotonicity (robust against tiny oscillations)
    cdf_r = np.clip(cdf_r, 0.0, 1.0)
    cdf_r = np.maximum.accumulate(cdf_r)

    # ----- raw exit-time CDF on t* grid with log-sum-exp -----
    # raw_t(t*) = 1 - Σ_n A_den * e^{-lam*theta*t}
    # Vectorized log-sum-exp across n for each t*
    log_terms_time = logA_den_mag[:, None] - (lam[:, None] * (theta * t_star[None, :]))
    m_t = np.max(log_terms_time, axis=0)
    sum_time = np.exp(m_t) * np.sum(sign_A_den[:,None]*np.exp(log_terms_time - m_t[None, :]), axis=0)
    raw_t_num = 1.0 - sum_time
    raw_t_num = np.minimum.accumulate(raw_t_num[::-1])[::-1]
    raw_t_num = np.clip(raw_t_num, 0.0, 1-p_surv0)
    if p_surv0 >= 1.0:
        raw_t = np.zeros_like(raw_t_num)
    else:
        raw_t = raw_t_num/(1-p_surv0)

    return r_star, cdf_r, p_surv0, t_star, raw_t
