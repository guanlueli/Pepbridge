"""Diffusion model for clamped_one_hot residue sequences."""
import numpy as np
import torch
import torch.nn.functional as F
import logging
from typing import Dict, Optional, Union, Any, Tuple


class SeqDiffuser:

    def __init__(self, seq_conf):
       
        self._seq_conf = seq_conf
        self.k = seq_conf.k
        self.K = seq_conf.K
        self.min_t = getattr(seq_conf, 'min_t', 1e-5)
        self.max_t = getattr(seq_conf, 'max_t', 1.0)
        
        # Set up noise schedule
        self.schedule_type = getattr(seq_conf, 'schedule', 'cosine')
        schedule_kwargs = getattr(seq_conf, 'schedule_kwargs', {})
        self.noise_schedule = self._get_noise_schedule(
            self.schedule_type, **schedule_kwargs)
    
    def _get_noise_schedule(self, schedule_type, **kwargs):
        """
        Create noise variance schedule.
        
        Args:
            schedule_type: Type of schedule ('linear', 'cosine', etc.)
            **kwargs: Additional parameters for the schedule
            
        Returns:
            Function that takes time t and returns the noise level
        """
        if schedule_type == 'linear':
            beta_start = kwargs.get('beta_start', 1e-4)
            beta_end = kwargs.get('beta_end', 0.02)
            
            def linear_schedule(t):
                return beta_start + (beta_end - beta_start) * t
            
            return linear_schedule
        
        elif schedule_type == 'cosine':
            s = kwargs.get('s', 0.008)
            
            def cosine_schedule(t):
                return torch.cos((t + s) / (1 + s) * torch.pi / 2) ** 2
            return cosine_schedule
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def _get_alpha_sigma(self, t, device=None):
        """
        Get alpha and sigma values for time t.
        
        Args:
            t: Float or tensor of shape [...] with times in [0, 1]
            
        Returns:
            alpha: Forward process scaling factor
            sigma: Forward process noise level
        """
        if isinstance(t, torch.Tensor):
            t = t.reshape(-1, 1, 1)
            device = t.device
        else:
            # ensure it goes to the right device
            t = torch.tensor(t, device=device).reshape(1, 1, 1)
            
        # Get noise level from schedule
        noise_level = self.noise_schedule(t)
        # Calculate alpha (scaling) and sigma (noise) for diffusion
        alpha = torch.sqrt(1 - noise_level)
        sigma = torch.sqrt(noise_level)
        return alpha, sigma
    
    def clamped_to_probs(self, clamped_seqs: torch.Tensor) -> torch.Tensor:
        """
        Convert clamped one-hot representation to probability distribution.
        
        Args:
            clamped_seqs: [..., L, K] clamped one-hot encodings
            
        Returns:
            probs: [..., L, K] probability distributions over amino acids
        """
        # Rescale from [-k, k] to [0, 1]
        seqs_scaled = (clamped_seqs + self.k) / (2 * self.k)
        # Ensure values are properly clamped between 0 and 1
        seqs_scaled = torch.clamp(seqs_scaled, 0, 1)
        
        # Normalize to ensure each position sums to 1
        probs = seqs_scaled / seqs_scaled.sum(dim=-1, keepdim=True)
        return probs
    
    def probs_to_clamped(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Convert probability distribution to clamped one-hot representation.
        
        Args:
            probs: [..., L, K] probability distributions over amino acids
            
        Returns:
            clamped_seqs: [..., L, K] clamped one-hot encodings
        """
        # Scale from [0, 1] to [-k, k]
        clamped_seqs = probs * self.k * 2 - self.k
        return clamped_seqs
        
    def forward_marginal(
            self,
            seqs_0: torch.Tensor,
            t: Union[float, torch.Tensor],
            diffuse_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, Any]:

        alpha, sigma = self._get_alpha_sigma(t, device=seqs_0.device)
        
        probs_0 = self.clamped_to_probs(seqs_0)
        
        # Create noise

        noise = torch.randn_like(probs_0)
        
        # Forward process: interpolate between data and noise
        # q(x_t | x_0) = alpha_t * x_0 + sigma_t * epsilon
        probs_t = alpha * probs_0 + sigma * noise
        
        # Renormalize to ensure valid probability distributions
        probs_t = probs_t / probs_t.sum(dim=-1, keepdim=True)
        
        # Calculate score: -noise / sigma
        score = -noise / sigma
        
        # Convert back to clamped one-hot encoding
        seqs_t = self.probs_to_clamped(probs_t)
        
        # Apply mask if provided
        if diffuse_mask is not None:
            # Add channel dimension for broadcasting
            mask = diffuse_mask.unsqueeze(-1)
            seqs_t = mask * seqs_t + (1 - mask) * seqs_0
            score = mask * score + (1 - mask) * torch.zeros_like(score)
        
        # Score scaling factor for loss weighting
        score_scaling = 1.0 / sigma.reshape(-1)
        
        return seqs_t, score, score_scaling
    
    def score(
            self,
            seqs_0: torch.Tensor,
            seqs_t: torch.Tensor,
            t: Union[float, torch.Tensor],
        ) -> torch.Tensor:
        """
        Calculate score function for a noised sequence given the original.
        
        Args:
            seqs_0: [..., L, K] original sequence
            seqs_t: [..., L, K] noised sequence at time t
            t: continuous time in [0, 1]
            
        Returns:
            score: [..., L, K] score function ∇ log q(x_t | x_0)
        """
        alpha, sigma = self._get_alpha_sigma(t, device=seqs_0.device)
        
        # Convert to probability space
        probs_0 = self.clamped_to_probs(seqs_0)
        probs_t = self.clamped_to_probs(seqs_t)
        
        # Compute implied noise
        noise = (probs_t - alpha * probs_0) / sigma
        
        # Score is negative noise divided by sigma
        score = -noise / sigma
        
        return score
    
    def score_scaling(self, t: Union[float, torch.Tensor], device = None) -> torch.Tensor:
        """
        Get score scaling factor for time t.
        
        Args:
            t: continuous time in [0, 1]
            
        Returns:
            score_scaling: Scaling factor for score
        """
        _, sigma = self._get_alpha_sigma(t, device=device) 
        return 1.0 / sigma.reshape(-1)
    
    def reverse(
            self,
            seqs_t: torch.Tensor,
            score_t: torch.Tensor,
            t: float,
            dt: float,
            diffuse_mask: Optional[torch.Tensor] = None,
            noise_scale: float = 1.0,
        ) -> torch.Tensor:
        """
        Reverse sampling function from time t to time (t-dt).
        
        Args:
            seqs_t: [..., L, K] sequence at time t
            score_t: [..., L, K] score function at time t
            t: continuous time in [0, 1]
            dt: continuous time step size
            diffuse_mask: [..., L] which residues to update (1) vs keep fixed (0)
            noise_scale: scaling factor for the noise (1.0 = standard sampling, 0.0 = deterministic)
            
        Returns:
            seqs_t_dt: [..., L, K] sequence at time t-dt
        """
        device = seqs_t.device
        # Get alpha and sigma for current and next timestep
        alpha_t, sigma_t = self._get_alpha_sigma(t, device=device)
        alpha_t_dt, sigma_t_dt = self._get_alpha_sigma(max(t - dt, 0.0), device=device)
        
        # Convert to probability space
        probs_t = self.clamped_to_probs(seqs_t)
        
        # Generate random noise for stochastic sampling
        z = torch.randn_like(probs_t) * noise_scale
        
        # Reverse process step
        # Euler-Maruyama step for reverse diffusion
        score_coef = (sigma_t_dt**2 - sigma_t**2) / sigma_t

        # if torch.isnan(probs_t).any() or torch.isinf(probs_t).any():
        #         print('probs_t nan or inf')
        
        # if torch.isnan(score_coef).any() or torch.isinf(score_coef).any():
        #         print('score_coef nan or inf')

        # if torch.isnan(sigma_t_dt).any() or torch.isinf(sigma_t_dt).any():
        #         print('sigma_t_dt nan or inf')

        # if torch.isnan(sigma_t).any() or torch.isinf(sigma_t).any():
        #         print('sigma_t nan or inf')
        
        diff_sigma_sq = sigma_t_dt**2 - sigma_t**2
        diff_sigma_sq = torch.clamp(diff_sigma_sq, min=0.0)
        noise_term = torch.sqrt(diff_sigma_sq) * z
        probs_t_dt = probs_t + score_coef * score_t + noise_term

        # if torch.isnan(probs_t_dt).any() or torch.isinf(probs_t_dt).any():
                # print('probs_t_dt nan or inf')
        
        # Ensure valid probability distributions
        probs_t_dt = torch.clamp(probs_t_dt, 1e-6, 1.0)

        # if torch.isnan(probs_t_dt).any() or torch.isinf(probs_t_dt).any():
        #         print('probs_t_dt nan or inf')

        probs_t_dt = probs_t_dt / probs_t_dt.sum(dim=-1, keepdim=True)
        
        # Convert back to clamped one-hot representation
        # if torch.isnan(probs_t_dt).any() or torch.isinf(probs_t_dt).any():
        #         print('probs_t_dt nan or inf')
        
        seqs_t_dt = self.probs_to_clamped(probs_t_dt)

        # if torch.isnan(seqs_t_dt).any() or torch.isinf(seqs_t_dt).any():
        #         print('seqs_t_dt nan or inf')
        
        # Apply mask if provided
        if diffuse_mask is not None:
            mask = diffuse_mask.unsqueeze(-1)
            seqs_t_dt = mask * seqs_t_dt + (1 - mask) * seqs_t
            
        # if torch.isnan(seqs_t_dt).any() or torch.isinf(seqs_t_dt).any():
        #         print('seqs_t_dt nan or inf')
        
        return seqs_t_dt
    
    def sample_ref(
            self,
            shape: Tuple[int, ...],
            impute: Optional[torch.Tensor] = None,
            diffuse_mask: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
        """
        Sample from the reference (prior) distribution.
        
        Args:
            shape: Tuple (B, L, K) for batch size, sequence length, and alphabet size
            impute: Optional tensor to use for imputation values
            diffuse_mask: [..., L] which residues to sample (1) vs impute (0)
            
        Returns:
            Dict containing:
                seqs_t: [B, L, K] sampled sequences from reference distribution
        """
        # Sample from uniform distribution over amino acids
        probs_ref = torch.ones(shape) / shape[-1]
        
        # Convert to clamped one-hot encoding
        seqs_ref = self.probs_to_clamped(probs_ref)
        
        # Apply mask if provided (and imputation values)
        if diffuse_mask is not None and impute is not None:
            mask = diffuse_mask.unsqueeze(-1)
            seqs_ref = mask * seqs_ref + (1 - mask) * impute
            
        return {'seqs_t': seqs_ref}
    
    def get_logits_from_seqs(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Convert clamped one-hot encodings to logits for categorical sampling.
        
        Args:
            seqs: [..., L, K] sequence in clamped one-hot encoding
            
        Returns:
            logits: [..., L, K] unnormalized log probabilities
        """
        # Rescale from [-k, k] to probabilities
        probs = self.clamped_to_probs(seqs)
        
        # Convert to logits (add small epsilon to avoid log(0))
        logits = torch.log(probs + 1e-10)
        
        return logits
    
    def sample_categorical(self, seqs: torch.Tensor) -> torch.Tensor:
        """
        Sample categorical amino acids from clamped one-hot encodings.
        
        Args:
            seqs: [..., L, K] sequence in clamped one-hot encoding
            
        Returns:
            sequences: [..., L] sequence of amino acid indices
        """
        logits = self.get_logits_from_seqs(seqs)
        
        # Sample categorically
        sequences = torch.distributions.Categorical(logits=logits).sample()
        
        return sequences
    
    def categorical_to_clamped_one_hot(self, cat_seqs: torch.Tensor) -> torch.Tensor:
        """
        Convert categorical sequences to clamped one-hot encoding.
        
        Args:
            cat_seqs: [..., L] sequence of amino acid indices
            
        Returns:
            clamped_seqs: [..., L, K] sequence in clamped one-hot encoding
        """
        # Create one-hot encoding
        shape = cat_seqs.shape
        one_hot = F.one_hot(cat_seqs, num_classes=self.K).float()
        
        # Apply clamping transformation
        clamped_seqs = one_hot * self.k * 2 - self.k
        
        return clamped_seqs