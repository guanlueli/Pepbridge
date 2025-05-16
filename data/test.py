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
        # self.min_b = bridge_conf.min_b
        # self.max_b = bridge_conf.max_b
        self.beta_d = bridge_conf.beta_d
        self.beta_min = bridge_conf.beta_min
        self.sigma_max = bridge_conf.sigma_max
        self.sigma_min = bridge_conf.sigma_min
        self.pred_mode = bridge_conf.pred_mode

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
            return sigmas**-2

    def vp_logsnr(self, t, beta_d, beta_min):   # logarithm of the signal-to-noise ratio (logSNR)
        t = torch.as_tensor(t)
        return - torch.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)

    def vp_logs(self, t, beta_d, beta_min):  # standard deviation (logs)
        t = torch.as_tensor(t)
        return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min

    def get_weightings(self, sigma):
        snrs = self.get_snr(sigma)

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

        return weightings

    def get_bridge_scalings(self, sigma):
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
            return c_skip, c_out, c_in

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
            return c_skip, c_out, c_in


        elif self.pred_mode == 've_simple' or self.pred_mode == 'vp_simple':

            c_in = torch.ones_like(sigma)
            c_out = torch.ones_like(sigma)
            c_skip = torch.zeros_like(sigma)
            return c_skip, c_out, c_in

    def forward_marginal(self, x_0: torch.Tensor, x_T: torch.Tensor, t: torch.Tensor):

        # x_0: [B, N, 3], t: [B]
        # x_0 = self._scale(x_0)
        # marg_b_t = self.marginal_b_t(t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        # loc = torch.exp(-0.5 * marg_b_t) * x_0
        # scale = torch.sqrt(1 - torch.exp(-marg_b_t))
        # x_t = loc + scale * torch.randn_like(x_0)
        # score_t = self.score(x_t, x_0, t)
        # x_t = self._unscale(x_t)

        # t = append_dims(t, dims)
        # std_t = th.sqrt(t)* th.sqrt(1 - t / self.sigma_max)
        noise = torch.randn_like(x_0)

        t = torch.minimum(t, torch.ones_like(t)* self.sigma_max)

        if self.pred_mode.startswith('ve'):
            std_t = t * torch.sqrt(1 - t ** 2 / self.sigma_max ** 2)
            mu_t = t ** 2 / self.sigma_max ** 2 * x_T + (1 - t ** 2 / self.sigma_max ** 2) * x_0
            samples = (mu_t + std_t * noise)
        elif self.pred_mode.startswith('vp'):
            print('t', t)
            logsnr_t = self.vp_logsnr(t, self.beta_d, self.beta_min)
            logsnr_T = self.vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
            logs_t = self.vp_logs(t, self.beta_d, self.beta_min)
            logs_T = self.vp_logs(self.sigma_max, self.beta_d, self.beta_min)

            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t / 2).exp()

            a1 = a_t * x_T
            a2 = b_t * x_0
            a3 = std_t * noise
            samples = a_t * x_T + b_t * x_0 + std_t * noise

        x_t = samples

        c_skip, c_out, c_in = [append_dims(x, x_t.ndim) for x in self.get_bridge_scalings(t)]

        # rescaled_t = 1000 * 0.25 * torch.log(t + 1e-44)

        x_t = c_in * x_t

        weights = self.get_weightings(t)
        weights = append_dims((weights), x_0.ndim)

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