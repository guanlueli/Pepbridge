import numpy as np
import torch
import logging
import os


class AngleDiffuser:
    def __init__(self, conf):
        """Initialize AngleDiffuser for diffusion on periodic angular space.

        Args:
            conf: Configuration object with:
                - min_sigma: Minimum noise level
                - max_sigma: Maximum noise level
                - num_sigma: Number of sigma discretization steps
                - schedule: Noise schedule type ('logarithmic')
        """
        self.schedule = conf.schedule
        self.min_sigma = conf.min_sigma
        self.max_sigma = conf.max_sigma
        self.num_sigma = conf.num_sigma
        self._log = logging.getLogger(__name__)

    def sigma(self, t: torch.Tensor):
        """Compute sigma(t) based on chosen noise schedule.

        Args:
            t: Time values in [0,1]
        Returns:
            Corresponding sigma values
        """
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')

        if self.schedule == 'logarithmic':
            return torch.log(t * torch.exp(torch.tensor(self.max_sigma)) +
                           (1 - t) * torch.exp(torch.tensor(self.min_sigma)))
        else:
            raise ValueError(f'Unrecognized schedule {self.schedule}')

    def diffusion_coef(self, t):
        """Compute diffusion coefficient g(t).

        Args:
            t: Time values in [0,1]
        Returns:
            Diffusion coefficients
        """
        if self.schedule == 'logarithmic':
            g_t = torch.sqrt(2 * (torch.exp(torch.tensor(self.max_sigma)) -
                                 torch.exp(torch.tensor(self.min_sigma))) *
                            self.sigma(t) / torch.exp(self.sigma(t)))
        else:
            raise ValueError(f'Unrecognized schedule {self.schedule}')
        return g_t

    def forward_marginal(self, angles_0: torch.Tensor, t: float):
        """Sample from forward diffusion at time t.

        Args:
            angles_0: Initial angles in radians [..., n_angles]
            t: Time in [0,1]
        Returns:
            angles_t: Diffused angles
            angle_score: Score for diffused angles
        """
        sigma_t = self.sigma(torch.tensor(t))
        noise = torch.randn_like(angles_0) * torch.exp(sigma_t)
        angles_t = angles_0 + noise

        # Compute score (gradient of log density)
        angle_score = -noise / (torch.exp(2 * sigma_t))

        # Wrap to [-pi, pi]
        angles_t = torch.remainder(angles_t + torch.pi, 2 * torch.pi) - torch.pi

        return angles_t, angle_score

    def reverse(self, angles_t: torch.Tensor, score_t: torch.Tensor,
                t: float, dt: float, mask: torch.Tensor = None,
                noise_scale: float = 1.0):
        """Simulate reverse diffusion for one step.

        Args:
            angles_t: Current angles at time t
            score_t: Score function at time t
            t: Current time in [0,1]
            dt: Time step
            mask: Boolean mask for which angles to diffuse
            noise_scale: Scale factor for noise
        Returns:
            angles_t_1: Updated angles
        """
        if not isinstance(t, (int, float)):
            raise ValueError(f't must be scalar, got {t}')

        # Compute diffusion updates
        g_t = self.diffusion_coef(torch.tensor(t))
        z = noise_scale * torch.randn_like(score_t)
        perturb = (g_t ** 2) * score_t * dt + g_t * torch.sqrt(torch.tensor(dt)) * z

        if mask is not None:
            perturb *= mask

        # Update and wrap angles
        angles_t_1 = angles_t + perturb
        angles_t_1 = torch.remainder(angles_t_1 + torch.pi, 2 * torch.pi) - torch.pi

        return angles_t_1

    def sample(self, t: float, n_samples: int = 1):
        """Sample random angles from the distribution at time t.

        Args:
            t: Time in [0,1]
            n_samples: Number of samples
        Returns:
            Random angles sampled from wrapped normal distribution
        """
        sigma_t = self.sigma(torch.tensor(t))
        angles = torch.randn(n_samples) * torch.exp(sigma_t)
        angles = torch.remainder(angles + torch.pi, 2 * torch.pi) - torch.pi
        return angles

    def torch_score(self, angles: torch.Tensor, t: torch.Tensor):
        """PyTorch version of score computation.

        Args:
            angles: Angles in radians
            t: Time values
        Returns:
            Score values
        """
        sigma_t = self.sigma(t)

        # Compute score for wrapped normal
        score = -angles / (torch.exp(2 * sigma_t)[:, None])
        return score

    def score_scaling(self, t: torch.Tensor):
        """Compute score scaling factor at time t."""
        sigma_t = self.sigma(t)
        return 1.0 / torch.exp(sigma_t)