from ..shared.domains import *
from ..shared.base_types import PathResult
from .bessel_zeros_refactored import get_bessel_zeros
from .CDFs_refactored import build_cdfs as _build_cdfs


class WalkOnCylinders:
    def __init__(self, 
                 dim: int, 
                 domain: Domain, 
                 theta: float, 
                 N_ZEROS: int, 
                 INV_R: int, 
                 INV_T: int, 
                 _eps: float,  
                 max_steps: int = 1_000_000,
                 N_cutoff: int = 100,
                 seed: Optional[int] = None, 
                 use_spatial_boundary: bool = True
        ):
        if dim < 1:
            raise ValueError(f"Dimension must be at least 1, currently Dimension = {dim}")
        if domain is None:
            raise ValueError("Domain cannot be None")
        
        # Make parameters accessible as instance variables
        self.dim = int(dim)
        self.domain = domain
        self.theta = theta
        self.N_ZEROS = N_ZEROS
        self.INV_R = INV_R
        self.INV_T = INV_T
        self._eps = _eps
        self.max_steps = max_steps
        self.use_spatial_boundary = use_spatial_boundary
        self.rng = np.random.default_rng(seed)

        self.N_cutoff = int(N_cutoff)
        if self.N_cutoff <= 0:
            raise ValueError(f"N_cutoff must be a positive integer, currently N_cutoff = {self.N_cutoff}")
        

        # Precompute the Bessel zeros, CDFs, and their inverses
        self.n_bessel_zeros()
        self.build_validated_cdfs()
        self.precompute_inverse_cdfs()

        if isinstance(self.domain, BallDomain):
            if self.domain.R <= 0:
                raise ValueError(f"Radius of the ball must be greater than 0, currently R = {self.domain.R}")
            
        if isinstance(self.domain, BoxDomain):
            if np.any(self.domain.b <= self.domain.a):
                raise ValueError(f"Upper bounds must be greater than lower bounds for BoxDomain, currently a = {self.domain.a}, b = {self.domain.b}")
            if self.domain.a.shape != (self.dim,) or self.domain.b.shape != (self.dim,):
                raise ValueError(
                    f'For dimension {self.dim}, BoxDomain bounds must have shape ({self.dim},), currently a.shape = {self.domain.a.shape}, b.shape = {self.domain.b.shape}'
                )


    def build_cdfs_checked(self, theta, zeros, INV_R, INV_T):
        out = _build_cdfs(self.dim, theta, zeros, INV_R, INV_T)
        if out is None or not isinstance(out, tuple) or len(out) != 5:
            raise RuntimeError("CDFs3_working.build_cdfs must return 5 outputs")
        
        # Build the CDFs
        r_star, cdf_r, p_surv0, t_star, raw_t = out

        # quick shape/type sanity
        for name, arr in [("r_star", r_star), ("cdf_r", cdf_r), ("t_star", t_star), ("raw_t", raw_t)]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be numpy array, got {type(arr)}")
        if not np.isfinite(p_surv0):
            raise ValueError(f"p_surv0 is not finite: {p_surv0}")
        

        return r_star, cdf_r, p_surv0, t_star, raw_t

    # Get the first N_ZEROS zeros of the Bessel function of order ν = dim/2 - 1
    def n_bessel_zeros(self) -> np.ndarray:
        self.zeros = get_bessel_zeros(self.dim, self.N_ZEROS)
        return self.zeros
    
    def build_validated_cdfs(self) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float]:
        # Build the CDFs with validation checks
        r_star, cdf_r, p_surv0, t_star, raw_t = self.build_cdfs_checked(self.theta, self.zeros, self.INV_R, self.INV_T)

        self.v_grid = r_star
        self.cdf_V = cdf_r

        self.w_grid = t_star
        self.cdf_W = raw_t

        self.p_terminal = float(np.clip(p_surv0, 0.0, 1.0)) # P(tau_R > T)
        self.H_s = 1.0 - self.p_terminal                    # P(tau_R <= T)
        self.p_lateral = self.H_s

        return self.v_grid, self.cdf_V, self.p_lateral, self.w_grid, self.cdf_W, self.p_terminal

    def precompute_inverse_cdfs(self) -> Tuple[np.ndarray, np.ndarray]:

        u_v = np.linspace(0.0, 1.0, self.INV_R)
        self.cdf_V = np.maximum.accumulate(np.clip(self.cdf_V, 0.0, 1.0))
        self.V_inv = np.interp(u_v, self.cdf_V, self.v_grid)

        u_w = np.linspace(0.0, 1.0, self.INV_T)
        if self.p_lateral > 0.0:
            self.cdf_W = np.maximum.accumulate(np.clip(self.cdf_W, 0.0, 1.0))
            self.W_inv = np.interp(u_w, self.cdf_W, self.w_grid)
        else:
            self.W_inv = np.zeros_like(u_w)

        return self.V_inv, self.W_inv

    def simulate_path(self, T_rem: float, x_n: np.ndarray, tol_space: float, tol_time: float) -> PathResult:

        center = x_n.copy()

        num_steps = 0

        #H_s_threshold = 1 - delta
        #in_saturation = (self.H_s >= self.H_s_threshold) and (self.d2H_s_dtheta2 < 0.0)

        # Theorem 2 in the paper:
        # E[N] = 1 / (1 - H_s) = 1 / p_terminal
        # Activate the terminal epsilon when E[N] >= N_cutoff, i.e., when p_terminal <= 1/N_cutoff
        #p_cutoff = 1.0 / self.N_cutoff
        #use_terminal_eps = self.p_terminal <= p_cutoff
        #effective_tol_time = tol_time if use_terminal_eps else 0.0
        effective_tol_time = tol_time

        #effective_tol_time = tol_time if self.H_s >= self.H_s_threshold else 0.0
        
        self.effective_tol_time = effective_tol_time
        #self.use_terminal_eps = use_terminal_eps
        self.expected_steps = 1.0 / max(self.p_terminal, 1e-15)  # Avoid division by zero

        for _ in range(self.max_steps):
        #while True:
            # --------------------------
            # Time Stopping Criterion
            # --------------------------
            #if T_rem <= tol_time:
            if T_rem <= effective_tol_time:
                return PathResult(exited=False, x_final=center, num_steps=num_steps)


            # R so that t* = T_rem/R^2 = theta
            # Radius of the cylinder for the current remaining time
            #R = np.sqrt(max(T_rem, 0.0) / self.theta)
            # --------------------------
            # Cylinder Radius and Time Height
            # --------------------------
            R_time = np.sqrt(max(T_rem, 0.0) / self.theta)

            if self.use_spatial_boundary:
                if not self.domain.is_inside(center):
                    raise ValueError("Current position is outside the domain, cannot continue walk.")
                
                R_space = self.domain.distance_to_boundary(center)

                if R_space <= tol_space:
                    x_bdry = self.domain.project_to_boundary(center)

                    return PathResult(exited=True, x_final=center, exit_time=T_rem, x_boundary=x_bdry, num_steps=num_steps)
                
                R = min(R_time, R_space)
            else:
                R = R_time

            # local cylinder time height
            dt_cyl = self.theta * R ** 2



            # --------------------------
            # Random Direction
            # --------------------------
            # Random direction on unit sphere
            direction = self.rng.normal(size=self.dim)

            # Normalize to get a random point on the unit sphere (handle degenerate case)
            nrm = np.linalg.norm(direction)
            if nrm <= self._eps:
                direction = np.zeros(self.dim)
                direction[0] = 1.0
            else:
                direction /= nrm

            # --------------------------
            # Degenerate Case: Always Lateral Cylinder Exit
            # --------------------------

            if self.p_terminal <= 1e-15:
                # Sample the random uniform point for time exit
                u_e = self.rng.random()

                # Inverse CDF lookup the index for the corresponding t* (clamp to max index)
                idx = min(int(u_e * (self.INV_T - 1)), self.INV_T - 1)

                # Get the corresponding t*
                W_s = self.W_inv[idx]

                # Compute the fraction of remaining time that gets spent before next time-step update
                tau = W_s * dt_cyl

                # Update position to the exit point on the time boundary
                center += R * direction

                # Update remaining time
                T_rem  -= tau
                T_rem = max(T_rem, 0.0)

                num_steps += 1
                continue

            # --------------------------
            # Degenerate Case: Always Terminal Face Exit
            # --------------------------

            # Always exit through the time boundary of the cylinder
            if self.p_terminal >= 1.0 - 1e-15:
                # Sample the random uniform point for spatial exit
                u_s = self.rng.random()

                # Inverse CDF lookup the index for the corresponding r* (clamp to max index)
                idx = min(int(u_s * (self.INV_R - 1)), self.INV_R - 1)

                # Get the corresponding r*
                V_s = self.V_inv[idx]

                # Update position to the exit point on the spatial boundary
                center += (V_s * R) * direction
                T_rem -= dt_cyl
                T_rem = max(T_rem, 0.0)

                num_steps += 1

                if self.use_spatial_boundary:
                    continue
                else:
                    return PathResult(exited=False, x_final=center, num_steps=num_steps)
                


            """ General case with non-degenerate probabilities """

            # --------------------------
            # Exit Type Decision
            # --------------------------

            # Generate a uniform random variable to decide exit type
            u_exit_type = self.rng.random()


            if u_exit_type <= self.H_s:
                # Lateral side event: tau_R <= T
                u_w = u_exit_type / self.H_s

                idx = min(int(u_w * (self.INV_T - 1)), self.INV_T - 1)
                W_s = self.W_inv[idx]

                tau = W_s * dt_cyl

                center += R * direction
                T_rem -= tau

                num_steps += 1
                continue

            else:
                u_v = (u_exit_type - self.H_s) / self.p_terminal

                idx = min(int(u_v * (self.INV_R - 1)), self.INV_R - 1)
                V_s = self.V_inv[idx]

                center += (V_s * R) * direction
                T_rem -= dt_cyl

                num_steps += 1

                if self.use_spatial_boundary:
                    continue
                else:
                    return PathResult(exited=False, x_final=center, num_steps=num_steps)
            
            
            
            



        raise RuntimeError(f'Walk on Cylinders path simulation failed to exit after {self.max_steps} steps, last position: {center}, remaining time: {T_rem}')
    

