import torch
import torch.nn as nn


class R3Diffuser:
    def __init__(self, r3_conf):
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b

    def _scale(self, x):
        # [B, N, 3] -> [B, N, 3]
        return x * self._r3_conf.coordinate_scaling

    def _unscale(self, x):
        # [B, N, 3] -> [B, N, 3]
        return x / self._r3_conf.coordinate_scaling

    def b_t(self, t):
        # t: [B]
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return self.min_b + t * (self.max_b - self.min_b)

    def diffusion_coef(self, t):
        # t: [B]
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x, t):
        # x: [B, N, 3], t: [B]
        return -0.5 * self.b_t(t).unsqueeze(-1).unsqueeze(-1) * x

    def sample_ref(self, batch_size: int, n_points: int):
        # Returns: [B, N, 3]
        return torch.randn(batch_size, n_points, 3)

    def marginal_b_t(self, t):
        # t: [B]
        return t * self.min_b + (0.5) * (t ** 2) * (self.max_b - self.min_b)

    def calc_trans_0(self, score_t, x_t, t):
        # score_t: [B, N, 3], x_t: [B, N, 3], t: [B]
        beta_t = self.marginal_b_t(t)  # [B]
        beta_t = beta_t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        cond_var = 1 - torch.exp(-beta_t)
        return (score_t * cond_var + x_t) / torch.exp(-0.5 * beta_t)

    def forward(self, x_t_1: torch.Tensor, t: torch.Tensor, num_t: int):
        # x_t_1: [B, N, 3], t: [B]
        x_t_1 = self._scale(x_t_1)
        b_t = (self.marginal_b_t(t) / num_t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        z_t_1 = torch.randn_like(x_t_1)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        # x_t: [B, N, 3], score_t: [B, N, 3], t: [B], mask: [B, N]
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        f_t = self.drift_coef(x_t, t)
        std = g_t * torch.sqrt(torch.tensor(dt, device=x_t.device))
        mu = x_t - (f_t - g_t ** 2 * score_t) * dt
        if mask is not None:
            mu = mu * mask.unsqueeze(-1)
        return mu, std

    def forward_marginal(self, x_0: torch.Tensor, t: torch.Tensor):
        # x_0: [B, N, 3], t: [B]
        x_0 = self._scale(x_0)
        marg_b_t = self.marginal_b_t(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        loc = torch.exp(-0.5 * marg_b_t) * x_0
        scale = torch.sqrt(1 - torch.exp(-marg_b_t))
        x_t = loc + scale * torch.randn_like(x_0)
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: torch.Tensor):
        # t: [B]
        return 1 / torch.sqrt(self.conditional_var(t))

    def reverse(
            self,
            x_t: torch.Tensor,
            score_t: torch.Tensor,
            t: torch.Tensor,
            dt: float,
            mask: torch.Tensor = None,
            center: bool = True,
            noise_scale: float = 1.0,
    ):
        # x_t: [B, N, 3], score_t: [B, N, 3], t: [B], mask: [B, N]
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * torch.randn_like(score_t)
        perturb = (f_t - g_t ** 2 * score_t) * dt + g_t * torch.sqrt(torch.tensor(dt, device=x_t.device)) * z

        if mask is not None:
            perturb = perturb * mask.unsqueeze(-1)
        else:
            mask = torch.ones_like(x_t[..., 0])
        x_t_1 = x_t - perturb

        if center:
            # Compute center of mass for each batch
            mask_sum = torch.sum(mask, dim=-1, keepdim=True)  # [B, 1]
            com = torch.sum(x_t_1 * mask.unsqueeze(-1), dim=1) / mask_sum.unsqueeze(-1)  # [B, 3]
            x_t_1 = x_t_1 - com.unsqueeze(1)

        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t):
        # t: [B]
        marg_b_t = self.marginal_b_t(t)  # [B, 1]
        return 1 - torch.exp(-marg_b_t)

    def score(self, x_t, x_0, t, scale=False):
        # x_t: [B, N, 3], x_0: [B, N, 3], t: [B]
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)

        marg_b_t = self.marginal_b_t(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        cond_var = self.conditional_var(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        return -(x_t - torch.exp(-0.5 * marg_b_t) * x_0) / cond_var