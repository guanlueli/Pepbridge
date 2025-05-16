import torch
import torch.nn as nn

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


class BridgeDiffuser:
    def __init__(self, bridge_conf):
        self._bridge_conf = bridge_conf
        self.beta_d = bridge_conf.beta_d
        self.beta_min = bridge_conf.beta_min
        self.sigma_max = bridge_conf.sigma_max
        self.sigma_min = bridge_conf.sigma_min
        self.pred_mode = bridge_conf.pred_mode
        self.sigma_data = bridge_conf.sigma_data
        self.sigma_data_end = bridge_conf.sigma_data
        self.cov_xy = bridge_conf.cov_xy
        self.c = 1
        self.weight_schedule = bridge_conf.weight_schedule

    def _scale(self, x):
        # [B, N, 3] -> [B, N, 3]
        return x * self._bridge_conf.coordinate_scaling

    def _unscale(self, x):
        # [B, N, 3] -> [B, N, 3]
        return x / self._bridge_conf.coordinate_scaling

    def forward(self, x_t_1: torch.Tensor, t: torch.Tensor, num_t: int):
        # x_t_1: [B, N, 3], t: [B]
        x_t_1 = self._scale(x_t_1)
        b_t = (self.marginal_b_t(t) / num_t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        z_t_1 = torch.randn_like(x_t_1)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def get_snr(self, sigmas):
        if self.pred_mode.startswith('vp'):
            return self.vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
        else:
            return sigmas ** -2

    def vp_logsnr(self, t, beta_d, beta_min):  # logarithm of the signal-to-noise ratio (logSNR)
        t = torch.as_tensor(t)
        return - torch.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)

    def vp_logs(self, t, beta_d, beta_min):  # standard deviation (logs)
        t = torch.as_tensor(t)
        return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min

    def get_weightings(self, sigma):
        # sigma: [B]
        snrs = self.get_snr(sigma)  # [B]

        if self.weight_schedule == "snr":
            weightings = snrs
        elif self.weight_schedule == "snr+1":
            weightings = snrs + 1
        elif self.weight_schedule == "karras":
            weightings = snrs + 1.0 / self.sigma_data ** 2
        elif self.weight_schedule.startswith("bridge_karras"):
            if self.pred_mode == 've':
                A = sigma ** 4 / self.sigma_max ** 4 * self.sigma_data_end ** 2 + (
                        1 - sigma ** 2 / self.sigma_max ** 2) ** 2 * self.sigma_data ** 2 + 2 * sigma ** 2 / self.sigma_max ** 2 * (
                            1 - sigma ** 2 / self.sigma_max ** 2) * self.cov_xy + self.c ** 2 * sigma ** 2 * (
                            1 - sigma ** 2 / self.sigma_max ** 2)
                weightings = A / ((sigma / self.sigma_max) ** 4 * (
                        self.sigma_data_end ** 2 * self.sigma_data ** 2 - self.cov_xy ** 2) + self.sigma_data ** 2 * self.c ** 2 * sigma ** 2 * (
                                          1 - sigma ** 2 / self.sigma_max ** 2))

            elif self.pred_mode == 'vp':
                logsnr_t = self.vp_logsnr(sigma, self.beta_d, self.beta_min)
                logsnr_T = self.vp_logsnr(1, self.beta_d, self.beta_min)
                logs_t = self.vp_logs(sigma, self.beta_d, self.beta_min)
                logs_T = self.vp_logs(1, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
                b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()

                A = a_t ** 2 * self.sigma_data_end ** 2 + b_t ** 2 * self.sigma_data ** 2 + 2 * a_t * b_t * self.cov_xy + self.c ** 2 * c_t
                weightings = A / (a_t ** 2 * (
                        self.sigma_data_end ** 2 * self.sigma_data ** 2 - self.cov_xy ** 2) + self.sigma_data ** 2 * self.c ** 2 * c_t)

            elif self.pred_mode == 'vp_simple' or self.pred_mode == 've_simple':
                weightings = torch.ones_like(snrs)
        elif self.weight_schedule == "truncated-snr":
            weightings = torch.clamp(snrs, min=1.0)
        elif self.weight_schedule == "uniform":
            weightings = torch.ones_like(snrs)
        else:
            raise NotImplementedError()

        return weightings  # [B]

    def get_bridge_scalings(self, sigma):
        # sigma: [B]
        if self.pred_mode == 've':
            A = sigma ** 4 / self.sigma_max ** 4 * self.sigma_data_end ** 2 + (
                    1 - sigma ** 2 / self.sigma_max ** 2) ** 2 * self.sigma_data ** 2 + 2 * sigma ** 2 / self.sigma_max ** 2 * (
                        1 - sigma ** 2 / self.sigma_max ** 2) * self.cov_xy + self.c ** 2 * sigma ** 2 * (
                        1 - sigma ** 2 / self.sigma_max ** 2)
            c_in = 1 / (A) ** 0.5
            c_skip = ((
                              1 - sigma ** 2 / self.sigma_max ** 2) * self.sigma_data ** 2 + sigma ** 2 / self.sigma_max ** 2 * self.cov_xy) / A
            c_out = ((sigma / self.sigma_max) ** 4 * (
                    self.sigma_data_end ** 2 * self.sigma_data ** 2 - self.cov_xy ** 2) + self.sigma_data ** 2 * self.c ** 2 * sigma ** 2 * (
                             1 - sigma ** 2 / self.sigma_max ** 2)) ** 0.5 * c_in
            return c_skip, c_out, c_in  # [B], [B], [B]

        elif self.pred_mode == 'vp':
            logsnr_t = self.vp_logsnr(sigma, self.beta_d, self.beta_min)
            logsnr_T = self.vp_logsnr(1, self.beta_d, self.beta_min)
            logs_t = self.vp_logs(sigma, self.beta_d, self.beta_min)
            logs_T = self.vp_logs(1, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            c_t = -torch.expm1(logsnr_T - logsnr_t) * (2 * logs_t - logsnr_t).exp()

            A = a_t ** 2 * self.sigma_data_end ** 2 + b_t ** 2 * self.sigma_data ** 2 + 2 * a_t * b_t * self.cov_xy + self.c ** 2 * c_t

            c_in = 1 / (A) ** 0.5
            c_skip = (b_t * self.sigma_data ** 2 + a_t * self.cov_xy) / A
            c_out = (a_t ** 2 * (
                    self.sigma_data_end ** 2 * self.sigma_data ** 2 - self.cov_xy ** 2) + self.sigma_data ** 2 * self.c ** 2 * c_t) ** 0.5 * c_in
            return c_skip, c_out, c_in  # [B], [B], [B]

        elif self.pred_mode == 've_simple' or self.pred_mode == 'vp_simple':
            c_in = torch.ones_like(sigma)
            c_out = torch.ones_like(sigma)
            c_skip = torch.zeros_like(sigma)
            return c_skip, c_out, c_in  # [B], [B], [B]

    def forward_marginal(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor):
        # x_0: [B, N, 3], x_T: [B, N, 3], t: [B]
        noise = torch.randn_like(x_0)

        # Broadcast t to match batch dimension
        t = torch.minimum(t, torch.ones_like(t) * self.sigma_max)  # [B]

        if self.pred_mode.startswith('ve'):
            # Expand t for broadcasting
            t_expand = t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

            std_t = t_expand * torch.sqrt(1 - t_expand ** 2 / self.sigma_max ** 2)
            mu_t = t_expand ** 2 / self.sigma_max ** 2 * x_T + (1 - t_expand ** 2 / self.sigma_max ** 2) * x_0
            samples = (mu_t + std_t * noise)
        elif self.pred_mode.startswith('vp'):
            logsnr_t = self.vp_logsnr(t, self.beta_d, self.beta_min)
            logsnr_T = self.vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
            logs_t = self.vp_logs(t, self.beta_d, self.beta_min)
            logs_T = self.vp_logs(self.sigma_max, self.beta_d, self.beta_min)

            # Expand for broadcasting
            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            b_t = (-torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()
            std_t = std_t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

            samples = a_t * x_T + b_t * x_0 + std_t * noise

        x_t = samples

        c_skip, c_out, c_in = self.get_bridge_scalings(t)
        # Reshape for broadcasting
        c_in = c_in.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        x_t = c_in * x_t

        weights = self.get_weightings(t)
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

        return x_t, weights

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
        marg_b_t = self.marginal_b_t(t)  # [B]
        return 1 - torch.exp(-marg_b_t)

    def score(self, x_t, x_0, t, scale=False):
        # x_t: [B, N, 3], x_0: [B, N, 3], t: [B]
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)

        marg_b_t = self.marginal_b_t(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        cond_var = self.conditional_var(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        return -(x_t - torch.exp(-0.5 * marg_b_t) * x_0) / cond_var