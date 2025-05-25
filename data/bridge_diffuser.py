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

def append_zero(x):
    """Appends a zero to the end of a tensor's first dimension."""
    return torch.cat([x, x.new_zeros([1] + list(x.shape[1:]))], dim=0)


def mean_flat(tensor):
    """Takes the mean over all dimensions except the first batch dimension."""
    return tensor.mean(dim=list(range(1, tensor.ndim)))

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

    # def forward(self, x_t_1: torch.Tensor, t: torch.Tensor, num_t: int):
        # x_t_1: [B, N, 3], t: [B]
        x_t_1 = self._scale(x_t_1)
        b_t = (self.marginal_b_t(t) / num_t).unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        z_t_1 = torch.randn_like(x_t_1)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def get_snr(self, sigmas):
        if self.pred_mode.startswith('vp'):
            a = self.vp_logsnr(sigmas, self.beta_d, self.beta_min)
            a1 = self.vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
            return self.vp_logsnr(sigmas, self.beta_d, self.beta_min).exp()
        else:
            return sigmas ** -2

    def vp_logsnr(self, t, beta_d, beta_min):  # logarithm of the signal-to-noise ratio (logSNR)
        t = torch.as_tensor(t)
        a = 0.5 * beta_d * (t ** 2) + beta_min * t
        a1 = (0.5 * beta_d * (t ** 2) + beta_min * t).exp()
        a2 = - torch.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)
        return - torch.log((0.5 * beta_d * (t ** 2) + beta_min * t).exp() - 1)

    def vp_logs(self, t, beta_d, beta_min):  # standard deviation (logs)
        t = torch.as_tensor(t)
        return -0.25 * t ** 2 * (beta_d) - 0.5 * t * beta_min

    def get_sigmas(self, n=None, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        n = self.num_timesteps if n is None else n
        ramp = torch.linspace(0, 1, n)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return append_zero(sigmas).to(device)
    
    def get_weightings(self, sigma):
        # sigma: [B]
        snrs = self.get_snr(sigma)  # [B]

        if self.weight_schedule == "snr":
            weightings = snrs
        elif self.weight_schedule == "snr+1":
            weightings = snrs + 1
        elif self.weight_schedule == "karras":
            log_snr = torch.log(snrs + 1)
            weightings = log_snr + 1.0 / self.sigma_data ** 2
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
        t = torch.minimum(t, torch.ones_like(t) * self.sigma_max)  # [B] # sigmas

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

        return x_t

    
    def get_loss(self, denoised, x_start, sigmas):
        weights = self.get_weightings(sigmas)
        weights = append_dims(weights, x_start.ndim)

        xs_mse = mean_flat((denoised - x_start) ** 2)
        mse = mean_flat(weights * (denoised - x_start) ** 2)
        
        if self.pred_mode.startswith('ve'):
            loss = xs_mse + mse
        else:
            loss = mse
            
        return loss


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
        # x_t = self._scale(x_t)
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

        # x_t_1 = self._unscale(x_t_1)
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
    
    def bridge_sample(self, x0, xT, t, noise=None):
        """Generate a noisy sample from the bridge distribution."""
        dims = x0.ndim
        if noise is None:
            noise = torch.randn_like(x0)
            
        t = append_dims(t, dims)
        
        # Sample from the bridge distribution based on prediction mode
        if self.pred_mode.startswith('ve'):
            std_t = t * torch.sqrt(1 - t**2 / self.sigma_max**2)
            mu_t = t**2 / self.sigma_max**2 * xT + (1 - t**2 / self.sigma_max**2) * x0
            samples = mu_t + std_t * noise
        elif self.pred_mode.startswith('vp'):
            logsnr_t = self.vp_logsnr(t)
            logsnr_T = self.vp_logsnr(self.sigma_max)
            logs_t = self.vp_logs(t)
            logs_T = self.vp_logs(self.sigma_max)

            a_t = (logsnr_T - logsnr_t + logs_t - logs_T).exp()
            b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
            std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
            
            samples = a_t * xT + b_t * x0 + std_t * noise
            
        return samples
    
    def sample_heun(self, denoiser, x, sigmas, progress=False, callback=None, 
                   churn_step_ratio=0., guidance=1):
        """Heun's method sampler following Karras et al. (2022)."""
        x_T = x
        path = [x]
        
        s_in = x.new_ones([x.shape[0]])
        indices = range(len(sigmas) - 1)
        
        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        nfe = 0
        assert churn_step_ratio < 1

        for j, i in enumerate(indices):
            # Optional stochastic sampling step (churn)
            if churn_step_ratio > 0:
                sigma_hat = (sigmas[i+1] - sigmas[i]) * churn_step_ratio + sigmas[i]
                
                denoised = denoiser(x, sigmas[i] * s_in, x_T)
                if self.pred_mode == 've':
                    d_1, gt2 = self.to_d_ve(x, sigmas[i], denoised, x_T, guidance=guidance, stochastic=True)
                elif self.pred_mode.startswith('vp'):
                    d_1, gt2 = self.to_d_vp(x, sigmas[i], denoised, x_T, guidance=guidance, stochastic=True)
                
                dt = (sigma_hat - sigmas[i])
                x = x + d_1 * dt + torch.randn_like(x) * ((dt).abs() ** 0.5) * gt2.sqrt()
                
                nfe += 1
                path.append(x.detach().cpu())
            else:
                sigma_hat = sigmas[i]
            
            # Regular Heun step
            denoised = denoiser(x, sigma_hat * s_in, x_T)
            if self.pred_mode == 've':
                d = self.to_d_ve(x, sigma_hat, denoised, x_T, guidance=guidance)
            elif self.pred_mode.startswith('vp'):
                d = self.to_d_vp(x, sigma_hat, denoised, x_T, guidance=guidance)
                
            nfe += 1
            
            if callback is not None:
                callback({
                    "x": x,
                    "i": i,
                    "sigma": sigmas[i],
                    "sigma_hat": sigma_hat,
                    "denoised": denoised,
                })
                
            dt = sigmas[i + 1] - sigma_hat
            
            if sigmas[i + 1] == 0:
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in, x_T)
                
                if self.pred_mode == 've':
                    d_2 = self.to_d_ve(x_2, sigmas[i + 1], denoised_2, x_T, guidance=guidance)
                elif self.pred_mode.startswith('vp'):
                    d_2 = self.to_d_vp(x_2, sigmas[i + 1], denoised_2, x_T, guidance=guidance)
                
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
                nfe += 1
                
            path.append(x.detach().cpu())
            
        return x, path, nfe
    
    def to_d_ve(self, x, sigma, denoised, x_T, guidance=1, stochastic=False):
        """Convert denoised output to ODE derivative for variance-exploding (VE) formulation."""
        sigma = append_dims(sigma, x.ndim)
        grad_pxtlx0 = (denoised - x) / (sigma**2)
        grad_pxTlxt = (x_T - x) / (torch.ones_like(sigma) * self.sigma_max**2 - sigma**2)
        gt2 = 2 * sigma
        d = -(0.5 if not stochastic else 1) * gt2 * (grad_pxtlx0 - guidance * grad_pxTlxt * (0 if stochastic else 1))
        
        if stochastic:
            return d, gt2
        else:
            return d

    def to_d_vp(self, x, sigma, denoised, x_T, guidance=1, stochastic=False):
        """Convert denoised output to ODE derivative for variance-preserving (VP) formulation."""
        # Get all required sigma-dependent values
        logsnr_t = self.vp_logsnr(sigma)
        logsnr_T = self.vp_logsnr(self.sigma_max)
        logs_t = self.vp_logs(sigma)
        logs_T = self.vp_logs(self.sigma_max)
        
        # Calculate VP SDE coefficients
        vp_snr_sqrt_reciprocal = lambda t: (torch.exp(0.5 * self.beta_d * (t ** 2) + self.beta_min * t) - 1) ** 0.5
        vp_snr_sqrt_reciprocal_deriv = lambda t: 0.5 * (self.beta_min + self.beta_d * t) * (vp_snr_sqrt_reciprocal(t) + 1 / vp_snr_sqrt_reciprocal(t))
        s = lambda t: (1 + vp_snr_sqrt_reciprocal(t) ** 2).rsqrt()
        s_deriv = lambda t: -vp_snr_sqrt_reciprocal(t) * vp_snr_sqrt_reciprocal_deriv(t) * (s(t) ** 3)
        std = lambda t: vp_snr_sqrt_reciprocal(t) * s(t)
        
        # Reshape sigma for proper broadcasting
        sigma_t = append_dims(sigma, x.ndim)
        
        # Calculate mixing coefficients for the VP model
        a_t = append_dims((logsnr_T - logsnr_t + logs_t - logs_T).exp(), x.ndim)
        b_t = append_dims(-torch.expm1(logsnr_T - logsnr_t) * logs_t.exp(), x.ndim)
        
        # Calculate mean for current timestep
        mu_t = a_t * x_T + b_t * denoised
        
        # Calculate std_t for current timestep
        std_t = append_dims(std(sigma), x.ndim)
        
        # Calculate gradient terms
        grad_logq = -(x - mu_t) / std_t**2 / (-torch.expm1(logsnr_T - logsnr_t))
        grad_logpxTlxt = -(x - torch.exp(logs_t-logs_T)*x_T) / std_t**2 / torch.expm1(logsnr_t - logsnr_T)
        
        # Calculate drift and diffusion coefficients
        f = append_dims(s_deriv(sigma) * (-logs_t).exp(), x.ndim) * x
        gt2 = 2 * append_dims((logs_t).exp()**2 * sigma * vp_snr_sqrt_reciprocal_deriv(sigma), x.ndim)
        
        # Calculate the derivative
        d = f - gt2 * ((0.5 if not stochastic else 1) * grad_logq - guidance * grad_logpxTlxt)
        
        if stochastic:
            return d, gt2
        else:
            return d