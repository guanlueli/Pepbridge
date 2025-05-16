import torch
import os
from data import utils as data_utils
import logging


def igso3_expansion_batch(omega: torch.Tensor, eps: torch.Tensor, L: int = 1000) -> torch.Tensor:
    """Truncated sum of IGSO(3) distribution for batch processing.

    Args:
        omega: Rotation angles of shape [batch_size, num_points] or [batch_size]
        eps: Scale parameter of shape [batch_size, num_points] or [batch_size]
        L: Truncation level
    Returns:
        Power series sum of shape [batch_size, num_points] or [batch_size]
    """
    ls = torch.arange(L, device=omega.device)

    if len(omega.shape) == 4:  # [batch_size, num_points, 3, 1]
        ls = ls[None, None, None, None]  # [1, 1, 1, 1, L]
        omega = omega.unsqueeze(-1)  # [batch_size, num_points, 3, 1, 1]
        eps = eps.view(-1, 1, 1, 1, 1)  # [batch_size, 1, 1, 1, 1]
    elif len(omega.shape) == 1:  # [batch_size]
        ls = ls[None]  # [1, L]
        omega = omega[..., None]  # [batch_size, 1]
        eps = eps[..., None]  # [batch_size, 1]
    else:
        raise ValueError(f"Unexpected omega shape: {omega.shape}")

    p = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * eps**2 / 2) * torch.sin(omega * (ls + 1/2)) / torch.sin(omega / 2)
    return p.sum(dim=-1)


def density_batch(expansion: torch.Tensor, omega: torch.Tensor, marginal: bool = True) -> torch.Tensor:
    """Batch-oriented IGSO(3) density.

    Args:
        expansion: Power series sum of shape [batch_size, num_points] or [batch_size]
        omega: Rotation angles of same shape as expansion
        marginal: If True, compute marginal density over rotation angle
    Returns:
        Density values of same shape as inputs
    """
    if marginal:
        return expansion * (1 - torch.cos(omega)) / torch.pi
    else:
        return expansion / 8 / (torch.pi ** 2)


def score_batch(exp: torch.Tensor, omega: torch.Tensor, eps: torch.Tensor, L: int = 1000) -> torch.Tensor:
    """Batch-oriented score computation.

    Args:
        exp: Power series sum [batch_size, num_points, 3, 1]
        omega: Rotation angles [batch_size, num_points, 3]
        eps: Scale parameter [batch_size, 1]
        L: Truncation level
    """
    ls = torch.arange(L, device=omega.device)
    ls = ls[None, None, None, :]  # [1, 1, 1, L]

    # Add dimension for L
    omega = omega.unsqueeze(-1)  # [batch_size, num_points, 3, 1]
    eps = eps.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]

    hi = torch.sin(omega * (ls + 1 / 2))  # [batch_size, num_points, 3, L]
    dhi = (ls + 1 / 2) * torch.cos(omega * (ls + 1 / 2))
    lo = torch.sin(omega / 2)  # [batch_size, num_points, 3, 1]
    dlo = 1 / 2 * torch.cos(omega / 2)

    # Calculate dSigma using broadcasting
    dSigma = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * eps ** 2 / 2) * \
             (lo * dhi - hi * dlo) / (lo ** 2 + 1e-10)

    dSigma = dSigma.sum(dim=-1)  # Sum over L dimension
    return dSigma / (exp.squeeze(-1) + 1e-4)

class SO3Diffuser:
    def __init__(self, so3_conf, device=None):

        self.device = device
        print('self.device',self.device)

        self.schedule = so3_conf.schedule
        self.min_sigma = so3_conf.min_sigma
        self.max_sigma = so3_conf.max_sigma
        self.num_sigma = so3_conf.num_sigma
        self.use_cached_score = so3_conf.use_cached_score
        self._log = logging.getLogger(__name__)

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.discrete_omega = torch.linspace(0, torch.pi, so3_conf.num_omega + 1)[1:]

        # Precompute IGSO3 values.
        replace_period = lambda x: str(x).replace('.', '_')
        cache_dir = os.path.join(
            so3_conf.cache_dir,
            f'eps_{so3_conf.num_sigma}_omega_{so3_conf.num_omega}_min_sigma_{replace_period(so3_conf.min_sigma)}_max_sigma_{replace_period(so3_conf.max_sigma)}_schedule_{so3_conf.schedule}'
        )

        # If cache directory doesn't exist, create it
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        pdf_cache = os.path.join(cache_dir, 'pdf_vals.pt')
        cdf_cache = os.path.join(cache_dir, 'cdf_vals.pt')
        score_norms_cache = os.path.join(cache_dir, 'score_norms.pt')

        if all(os.path.exists(f) for f in [pdf_cache, cdf_cache, score_norms_cache]):
            self._log.info(f'Using cached IGSO3 in {cache_dir}')
            self._pdf = torch.load(pdf_cache).to(self.device)
            self._cdf = torch.load(cdf_cache).to(self.device)
            self._score_norms = torch.load(score_norms_cache).to(self.device)
        else:
            self._log.info(f'Computing IGSO3. Saving in {cache_dir}')
            # Compute the expansion of the power series in batched form
            exp_vals = igso3_expansion_batch(
                omega=self.discrete_omega.unsqueeze(0).expand(self.num_sigma, -1),  # Shape: [num_sigma, num_omega]
                eps=self.discrete_sigma.unsqueeze(1),  # Shape: [num_sigma, 1]
            )

            # Compute the pdf and cdf values in batched form
            self._pdf = density_batch(exp_vals, self.discrete_omega, marginal=True)
            self._cdf = torch.cumsum(self._pdf, dim=1) / so3_conf.num_omega * torch.pi
            self._pdf =  self._pdf.to(self.device)
            self._cdf =  self._cdf.to(self.device)
            # Compute score norms in batched form
            self._score_norms = score_batch(
                exp_vals,
                self.discrete_omega.unsqueeze(0).expand(self.num_sigma, -1),
                self.discrete_sigma.unsqueeze(1)
            )

            # Cache the precomputed values
            torch.save(self._pdf, pdf_cache)
            torch.save(self._cdf, cdf_cache)
            torch.save(self._score_norms, score_norms_cache)

        self._score_scaling = torch.sqrt(torch.abs(
            torch.sum(self._score_norms ** 2 * self._pdf, dim=-1) /
            torch.sum(self._pdf, dim=-1)
        )) / torch.sqrt(torch.tensor(3.0))

    @property
    def discrete_sigma(self):
        return self.sigma(torch.linspace(0.0, 1.0, self.num_sigma).to(self.device))

    def sigma_idx(self, sigma: torch.Tensor):
        """Calculates the index for discretized sigma during IGSO(3) initialization.
        Supports batched input of shape [..., N]."""
        return torch.bucketize(sigma.flatten(), self.discrete_sigma) - 1

    def sigma(self, t: torch.Tensor):
        """Extract \sigma(t) corresponding to chosen sigma schedule.
        Supports batched input of shape [..., N]."""
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'logarithmic':
            return torch.log(
                t * torch.exp(torch.tensor(self.max_sigma)) +
                (1 - t) * torch.exp(torch.tensor(self.min_sigma))
            ).to(self.device)
        else:
            raise ValueError(f'Unrecognized schedule {self.schedule}')

    def diffusion_coef(self, t: torch.Tensor):
        """Compute diffusion coefficient (g_t).
        Supports batched input of shape [..., N]."""
        if self.schedule == 'logarithmic':
            sigma_t = self.sigma(t)
            g_t = torch.sqrt(
                2 * (torch.exp(torch.tensor(self.max_sigma)) -
                     torch.exp(torch.tensor(self.min_sigma))) *
                sigma_t / torch.exp(sigma_t)
            )
        else:
            raise ValueError(f'Unrecognized schedule {self.schedule}')
        return g_t

    def t_to_idx(self, t: torch.Tensor):
        """Helper function to go from time t to corresponding sigma_idx.
        Supports batched input of shape [..., N]."""
        return self.sigma_idx(self.sigma(t))

    def sample_igso3(self, t: torch.Tensor, batch_shape: torch.Size):
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t: tensor of shape [...] or scalar
            batch_shape: desired shape of the batch [..., N]

        Returns:
            Tensor of rotation angles with shape batch_shape
        """
        t = torch.as_tensor(t)

        # Generate random samples
        x = torch.rand(batch_shape, device=t.device)  # Shape: [16, 152, 3]

        # Get CDF for time t
        cdf_t = self._cdf[self.t_to_idx(t)]  # Shape: [15, 1000]

        # Use first row of cdf_t (or whichever row is appropriate for your case)
        cdf_values = cdf_t[0]  # Shape: [1000]

        # Flatten x
        x_flat = x.reshape(-1)  # Shape: [16*152*3]

        # Find indices where x_flat would be inserted into cdf_values to maintain order
        indices = torch.searchsorted(cdf_values, x_flat)
        indices = torch.clamp(indices, 1, len(self.discrete_omega) - 1)

        # Get surrounding values for interpolation
        cdf_low = cdf_values[indices - 1]
        cdf_high = cdf_values[indices]
        omega_low = self.discrete_omega[indices - 1]
        omega_high = self.discrete_omega[indices]

        # Compute interpolation weights
        alpha = (x_flat - cdf_low) / (cdf_high - cdf_low)

        # Interpolate
        result_flat = torch.lerp(omega_low, omega_high, alpha)

        # Reshape back to original batch shape
        result = result_flat.reshape(batch_shape)

        return result

    def sample(self, t: torch.Tensor, batch_shape: torch.Size):
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: tensor of shape [...] or scalar
            batch_shape: desired shape of the batch [..., N]

        Returns:
            Tensor of rotation vectors with shape batch_shape + [3]
        """
        # Generate random directions for the entire batch
        x = torch.randn(*batch_shape, 3, device=t.device)
        x = x / torch.norm(x, dim=-1, keepdim=True)

        # Sample angles and broadcast
        angles = self.sample_igso3(t, batch_shape)
        return x * angles.unsqueeze(-1)

    def sample_ref(self, batch_shape: torch.Size):
        """Generate reference samples with shape batch_shape + [3]."""
        return self.sample(torch.tensor(1.0), batch_shape)

    def score(self, vec: torch.Tensor, t: torch.Tensor, eps: float = 1e-6):
        """Computes the score of IGSO(3) density as a rotation vector.
        Supports batched input vec of shape [..., 3]."""
        # if t.ndim > 0:
        #     raise ValueError(f't must be a scalar, got shape {t.shape}')
        return self.torch_score(vec, t.unsqueeze(0), eps)

    def torch_score(self, vec: torch.Tensor, t: torch.Tensor, eps: float = 1e-6):
        """Computes the score of IGSO(3) density as a rotation vector.
        Supports batched input vec of shape [..., 3] and t of shape [...]."""
        # Compute norms for the entire batch
        omega = torch.norm(vec, dim=-1) + eps
        batch_shape = omega.shape

        if self.use_cached_score:
            score_norms_t = self._score_norms[self.t_to_idx(t)]
            omega_idx = torch.bucketize(omega.flatten(), self.discrete_omega[:-1])
            omega_scores_t = torch.gather(score_norms_t, -1, omega_idx).reshape(batch_shape)
        else:
            sigma = self.discrete_sigma[self.t_to_idx(t)]
            omega_vals = igso3_expansion_batch(omega.unsqueeze(-1), sigma.unsqueeze(-1))
            omega_scores_t = score_batch(omega_vals, omega, sigma.unsqueeze(-1))

        return omega_scores_t.unsqueeze(-1) * vec / (omega.unsqueeze(-1) + eps)

    def score_scaling(self, t: torch.Tensor):
        """Calculates scaling used for scores during training.
        Supports batched input t of shape [...]."""
        return self._score_scaling[self.t_to_idx(t)]

    def forward_marginal(self, rot_0: torch.Tensor, t: torch.Tensor):
        """Samples from the forward diffusion process at time index t.
        Supports batched input rot_0 of shape [..., 3]."""
        batch_shape = rot_0.shape[:-1]

        # Sample rotations for the entire batch
        sampled_rots = self.sample(t, batch_shape)
        rot_score = self.score(sampled_rots, t)

        # Right multiply for the entire batch
        rot_t = data_utils.compose_rotvec(rot_0, sampled_rots)
        return rot_t, rot_score

    def reverse(self, rot_t: torch.Tensor, score_t: torch.Tensor,
                t: torch.Tensor, dt: float, mask: torch.Tensor = None,
                noise_scale: float = 1.0):
        """Simulates the reverse SDE for 1 step using the Geodesic random walk.
        Supports batched inputs of shape [..., 3]."""
        if t.ndim > 0:
            raise ValueError(f't must be a scalar, got shape {t.shape}')

        g_t = self.diffusion_coef(t)
        z = noise_scale * torch.randn_like(score_t)
        perturb = (g_t ** 2) * score_t * dt + g_t * torch.sqrt(torch.tensor(dt)) * z

        if mask is not None:
            perturb = perturb * mask.unsqueeze(-1)

        # Right multiply for the entire batch
        rot_t_1 = data_utils.compose_rotvec(rot_t, perturb)
        return rot_t_1