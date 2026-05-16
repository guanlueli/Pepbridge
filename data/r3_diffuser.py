import torch
import torch.nn as nn
import math

class R3Diffuser:
    def __init__(self, r3_conf, num_t):
        self._r3_conf = r3_conf
        self.min_b = r3_conf.min_b
        self.max_b = r3_conf.max_b
        self.num_t = num_t
        self.coordinate_scaling = r3_conf.coordinate_scaling
        self.schedule = getattr(r3_conf, 'schedule', 'cosine')
        self.use_heun = r3_conf.use_heun
        if self.schedule == 'cosine':
            self.s = getattr(r3_conf, 'cosine_s', 0.008)
            # precompute normalization
            self.f0 = math.cos((self.s / (1 + self.s)) * math.pi / 2) ** 2
        self.dt = 1.0 / self.num_t

    def _scale(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.coordinate_scaling

    def _unscale(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.coordinate_scaling

    def _broadcast_time(self, t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """Helper to broadcast time tensor to match target tensor dimensions"""
        # Reshape t to [batch_size, 1, 1] for broadcasting with [batch_size, n_points, 3]
        return t.view(-1, *([1] * (len(target_shape) - 1)))

    def b_t(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise rate β(t)"""
        if torch.any((t < 0) | (t > 1)):
            raise ValueError(f't must be in [0,1], got {t}')
        if self.schedule == 'linear':
            return self.min_b + t * (self.max_b - self.min_b)
        f_t = torch.cos(((t + self.s) / (1 + self.s)) * math.pi / 2) ** 2
        alpha_bar = f_t / self.f0
        t_prev = torch.clamp(t - self.dt, min=0.0)
        f_prev = torch.cos(((t_prev + self.s) / (1 + self.s)) * math.pi / 2) ** 2
        alpha_bar_prev = f_prev / self.f0
        beta = (alpha_bar_prev - alpha_bar) / alpha_bar_prev
        return torch.clamp(beta, min=1e-6)

    def marginal_b_t(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative variance β̄(t) = -log ᾱ(t)"""
        if self.schedule == 'linear':
            return t * self.min_b + 0.5 * t.pow(2) * (self.max_b - self.min_b)
        f_t = torch.cos(((t + self.s) / (1 + self.s)) * math.pi / 2) ** 2
        alpha_bar = f_t / self.f0
        return -torch.log(alpha_bar)

    def diffusion_coef(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.b_t(t))

    def drift_coef(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return -0.5 * self._broadcast_time(self.b_t(t), x.shape) * x

    def conditional_var(self, t: torch.Tensor) -> torch.Tensor:
        return 1 - torch.exp(-self.marginal_b_t(t))

    def score_scaling(self, t: torch.Tensor) -> torch.Tensor:
        """Scaling term for transition loss: 1 / sqrt(Var[x_t | x0])"""
        var = self.conditional_var(t)
        return 1.0 / torch.sqrt(var)

    def sample_ref(self, batch_size: int, n_points: int) -> torch.Tensor:
        return torch.randn(batch_size, n_points, 3, device=self._r3_conf.device)

    def forward(self, x_prev: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_prev = self._scale(x_prev)
        b = self._broadcast_time(self.b_t(t) * self.dt, x_prev.shape)
        z = torch.randn_like(x_prev)
        return torch.sqrt(1 - b) * x_prev + torch.sqrt(b) * z

    def heun_step(self, x_prev: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        drift1 = self.drift_coef(x_prev, t) - self._broadcast_time(self.diffusion_coef(t)**2, x_prev.shape) * self.score(x_prev, t)
        x_pred = x_prev + drift1 * self.dt
        drift2 = self.drift_coef(x_pred, t - self.dt) - self._broadcast_time(self.diffusion_coef(t - self.dt)**2, x_pred.shape) * self.score(x_pred, t - self.dt)
        return x_prev + 0.5 * (drift1 + drift2) * self.dt

    def calc_trans_0(self, score_t: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        beta = self._broadcast_time(self.marginal_b_t(t), x_t.shape)
        cond_var = 1 - torch.exp(-beta)
        return (score_t * cond_var + x_t) * torch.exp(0.5 * beta)

    def forward_marginal(self, x0: torch.Tensor, t: torch.Tensor) -> tuple:
        
        x0_s = self._scale(x0)
        beta_bar = self._broadcast_time(self.marginal_b_t(t), x0_s.shape)
        mean = torch.exp(-0.5 * beta_bar) * x0_s
        std = torch.sqrt(1 - torch.exp(-beta_bar))
        xt = mean + std * torch.randn_like(x0_s)
        score_t = self.score(xt, x0_s, t, scale=False)
        
        return self._unscale(xt), score_t

    def distribution(self, x_t: torch.Tensor, score_t: torch.Tensor, t: torch.Tensor, mask: torch.Tensor, dt: float) -> tuple:
        x_t_s = self._scale(x_t)
        g = self._broadcast_time(self.diffusion_coef(t), x_t_s.shape)
        f = self.drift_coef(x_t_s, t)
        std = g * math.sqrt(dt)
        mu = x_t_s - (f - g**2 * score_t) * dt
        if mask is not None:
            mu = mu * mask.unsqueeze(-1)
        return mu, std

    def reverse(self, x_t: torch.Tensor, score_t: torch.Tensor, t: torch.Tensor, dt: float = None, pred_x_0: torch.Tensor=None, mask: torch.Tensor=None, center: bool=False, noise_scale: float=1.0, use_heun: bool=False) -> torch.Tensor:
        
        x_s = self._scale(x_t)
        if use_heun:
            return self._unscale(self.heun_step(x_s, t))
        g = self._broadcast_time(self.diffusion_coef(t), x_s.shape)
        f = self.drift_coef(x_s, t)
        z = noise_scale * torch.randn_like(x_s)
        dt_step = dt
        sde_perturb = (f - g**2 * score_t) * dt_step + g * torch.sqrt(dt_step) * z
        sde_step = x_s - sde_perturb
        
        if pred_x_0 is not None:
            pred_x_s = self._scale(pred_x_0)
            ode_perturb = -(pred_x_s - x_s) * dt_step  # Negative because we're moving towards pred_x_s
            ode_step = x_s - ode_perturb
            alpha = torch.clamp(1.0 - t, 0.1, 0.9).view(-1, 1, 1)
            x_prev = alpha * ode_step + (1 - alpha) * sde_step
        else:
            x_prev = sde_step

        if mask is not None:
            delta = x_prev - x_s
            masked_delta = delta * mask.unsqueeze(-1)
            x_prev = x_s + masked_delta
        
        # if center:
        #     mask_u = mask.unsqueeze(-1) if mask is not None else torch.ones_like(x_prev)
        #     com = (x_prev * mask_u).sum(dim=1, keepdim=True) / mask_u.sum(dim=1, keepdim=True)
        #     x_prev = x_prev - com
        
        return self._unscale(x_prev)

    # def score(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    #     beta_bar = self.marginal_b_t(t).view(-1,1,1)
    #     mean_coef = torch.exp(-0.5 * beta_bar)
    #     std2 = 1 - torch.exp(-beta_bar)
    #     # analytic score for Gaussian perturbation
    #     return -(x_t - mean_coef * x_t) / std2
    
    def score(self, x_t, x_0, t, scale=False):
        """Clean score computation with proper broadcasting"""
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)
        
        # Get time-dependent values and broadcast them properly
        beta_bar = self._broadcast_time(self.marginal_b_t(t), x_t.shape)
        cond_var = self._broadcast_time(self.conditional_var(t), x_t.shape)
        
        # Compute score with proper broadcasting
        exp_term = torch.exp(-0.5 * beta_bar)
        return -(x_t - exp_term * x_0) / cond_var

    def calc_trans_score(self, trans_t, trans_0, t, scale=True):
        return self.score(trans_t, trans_0, t, scale=scale)
