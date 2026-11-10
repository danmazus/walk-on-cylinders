from ..shared.domains import *
from ..shared.base_types import Trajectory


class WalkOnHeatBalls():
    def __init__(self, dim: int, domain: Domain, max_steps: int=1_000_000, seed: Optional[int]=None, use_spatial_boundary: bool=True):
        if dim < 1:
            raise ValueError(f"Dimension must be greater than 0, currently Dimension = {dim}")
        if domain is None:
            raise ValueError("Domain must be specified.")

        self.d = int(dim)
        self.domain = domain
        self.max_steps = int(max_steps)
        self.rng = np.random.default_rng(seed)
        self.use_spatial_boundary = use_spatial_boundary

        if self.use_spatial_boundary and domain is None:
            raise ValueError("Spatial boundary handling is enabled, but no domain was provided.")
        
        if isinstance(self.domain, BallDomain):
            if self.domain.R <= 0:
                raise ValueError(f"Ball radius must be > 0, currently R = {self.domain.R}")

        if isinstance(self.domain, BoxDomain):
            if self.domain.a.shape != (self.d,) or self.domain.b.shape != (self.d,):
                raise ValueError(
                    f"For dim={self.d}, BoxDomain bounds must have shape ({self.d},), "
                    f"got a.shape={self.domain.a.shape}, b.shape={self.domain.b.shape}"
                )
            if not np.all(self.domain.a < self.domain.b):
                raise ValueError("BoxDomain requires strict bounds a < b componentwise.")

    def alpha_func(self, u: float, v: np.ndarray) -> float:
        if not self.use_spatial_boundary:
            return float(u)
        
        delta = self.domain.distance_to_boundary(v)
        return min(u, (np.e / (self.d)) * delta ** 2)

    def sample_Un(self) -> np.ndarray:
        k = self.d // 2 + 1
        return self.rng.random(k)

    @staticmethod
    def PiU(U: np.ndarray) -> float:
        return float(np.prod(U))

    def sample_Gn(self) -> float:
        return float(self.rng.standard_normal())

    def sample_Vn(self) -> np.ndarray:
        Z = self.rng.standard_normal(self.d)
        Norm_z = np.linalg.norm(Z)
        if Norm_z == 0.0:
            Z[0] = 1.0
            Norm_z = 1.0
        return Z / Norm_z

    def R_n_plus1(self) -> float:
        U = self.sample_Un()
        G = self.sample_Gn()

        Pi = self.PiU(U)

        return float((Pi ** (2/self.d)) * np.exp(-(1 - ((2/self.d) * (self.d // 2))) * (G ** 2)))

    def psi_d(self, t: float) -> float:
        return float(np.sqrt(t * np.log(t ** (-self.d / 2))))

    def one_sample(self, t0: float,
                   x0: Union[np.ndarray, list, tuple],
                   epsi: float = 1e-6,
                   store_trajectory: bool=False,
                   verbose: bool = False
                   ) -> Union[Tuple[float, np.ndarray, int], Tuple[float, np.ndarray, int, Trajectory]]:
        
        if epsi <= 0:
            raise ValueError(f"Epsilon must be greater than 0, currently Epsilon = {epsi}")

        T = float(t0)
        X = np.array(x0, dtype=float).reshape(self.d)

        if T < 0:
            raise ValueError(f"Initial time t0 must be >= 0, currently t0 = {t0}")
        
        if X.shape != (self.d,):
            raise ValueError(f"Initial position x0 must have shape ({self.d},), currently shape = {X.shape}")
        if self.use_spatial_boundary:
            if not self.domain.is_inside(X):
                raise ValueError("Initial position x0 must lie inside the domain.")

        n = 0

        if store_trajectory:
            traj_T = [T]
            traj_X = [X.copy()]
            traj_alpha = []
            traj_R = []
        else:
            traj_T = traj_X = traj_alpha = traj_R = None

        while True:
            alpha = self.alpha_func(T, X)

            if verbose:
                dist = self.domain.distance_to_boundary(X)
                print(f'Step {n}: T={T:.6f}, ||X||={np.linalg.norm(X):.6f}, delta = {dist:.6f}, alpha = {alpha:.6f}')

            if alpha <= epsi or n >= self.max_steps:
                if verbose:
                    print(f'Stopping: alpha = {alpha:.6f}, epsilon = {epsi:.6f}')
                break

            R = self.R_n_plus1()
            V = self.sample_Vn()

            if store_trajectory:
                traj_alpha.append(alpha)
                traj_R.append(R)

            T = T - alpha * R
            X = X + np.sqrt(2 * alpha) * self.psi_d(R) * V
            n += 1

            if store_trajectory:
                traj_T.append(T)
                traj_X.append(X.copy())

        if self.use_spatial_boundary:
            delta = max(self.domain.distance_to_boundary(X), 0.0)

            if delta ** 2 <= (epsi * self.d) / np.e:
                # stopped near spatial boundary, project to boundary
                X_eps = self.domain.project_to_boundary(X)
                T_eps = T
            
            else:
                # reached terminal time, use current position
                X_eps = X
                T_eps = 0.0
        else:
            X_eps = X
            T_eps = 0.0

        if not store_trajectory:
            return T_eps, X_eps, n

        traj = Trajectory(
            T = np.array(traj_T, dtype=float), 
            X = np.array(traj_X, dtype=float),
            alpha = np.array(traj_alpha, dtype=float), 
            R = np.array(traj_R, dtype=float)
        )

        return T_eps, X_eps, n, traj



# Fix certain heatball size