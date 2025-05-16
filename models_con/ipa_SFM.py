import numpy as np
import torch
import math
from scipy.stats import truncnorm
import torch.nn as nn
from typing import Optional, Callable, List, Sequence
from openfold.utils.rigid_utils import Rigid
from data import all_atom

def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)

def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f

def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        if init_fn is not None:
            init_fn(self.weight, self.bias)
        else:
            if init == "default":
                lecun_normal_init_(self.weight)
            elif init == "relu":
                he_normal_init_(self.weight)
            elif init == "glorot":
                glorot_uniform_init_(self.weight)
            elif init == "gating":
                gating_init_(self.weight)
                if bias:
                    with torch.no_grad():
                        self.bias.fill_(1.0)
            elif init == "normal":
                normal_init_(self.weight)
            elif init == "final":
                final_init_(self.weight)
            else:
                raise ValueError("Invalid init string.")


class InvariantPointAttention_SURF_BB_SFM(nn.Module):
    """
    Implements interaction between protein backbone and surface point cloud.
    """

    def __init__(
            self,
            ipa_conf,
            inf: float = 1e5,
            eps: float = 1e-8,
    ):
        """
        Args:
            ipa_conf: Configuration object with the following attributes:
                c_s: Single representation channel dimension
                c_z: Pair representation channel dimension
                c_hidden: Hidden channel dimension
                no_heads: Number of attention heads
                no_qk_points: Number of query/key points to generate
                no_v_points: Number of value points to generate
                surface_channels: Number of channels for surface point cloud features
        """
        super(InvariantPointAttention_SURF_BB_SFM, self).__init__()
        self._ipa_conf = ipa_conf

        self.c_s = ipa_conf.c_s
        self.c_z = ipa_conf.c_z
        self.c_hidden = ipa_conf.c_hidden
        self.no_heads = ipa_conf.no_heads
        self.no_qk_points = ipa_conf.no_qk_points
        self.no_v_points = ipa_conf.no_v_points
        self.surface_channels = getattr(ipa_conf, 'surface_channels', 16)  # Default if not specified
        self.inf = inf
        self.eps = eps

        # Backbone representation processing
        hc = self.c_hidden * self.no_heads
        self.linear_q = Linear(self.c_s, hc)
        self.linear_kv = Linear(self.c_s, 2 * hc)

        hpq = self.no_heads * self.no_qk_points * 3
        self.linear_q_points = Linear(self.c_s, hpq)

        hpkv = self.no_heads * (self.no_qk_points + self.no_v_points) * 3
        self.linear_kv_points = Linear(self.c_s, hpkv)

        # Surface point cloud processing
        self.surface_encoder = Linear(self.surface_channels, self.c_s)

        # Cross-attention between backbone and surface
        self.linear_surface_q = Linear(self.c_s, hc)
        self.linear_surface_k = Linear(self.c_s, hc)
        self.linear_surface_v = Linear(self.c_s, hc)

        # Pair representation processing
        self.linear_b = Linear(self.c_z, self.no_heads)
        self.down_z = Linear(self.c_z, self.c_z // 4)

        # Attention weights
        self.head_weights = nn.Parameter(torch.zeros((ipa_conf.no_heads)))
        ipa_point_weights_init_(self.head_weights)

        # Surface-backbone interaction weights
        self.surface_backbone_weights = nn.Parameter(torch.zeros((1)))
        ipa_point_weights_init_(self.surface_backbone_weights)

        # Output projections
        concat_out_dim = (
                self.c_z // 4 + self.c_hidden + self.no_v_points * 4 + self.c_hidden  # Added surface features
        )
        self.linear_out = Linear(self.no_heads * concat_out_dim, self.c_s, init="final")

        # Surface output projection
        self.linear_surface_out = Linear(2 * self.c_s, self.surface_channels, init="final")

        self.softmax = nn.Softmax(dim=-1)
        self.softplus = nn.Softplus()

    def forward(
            self,
            s: torch.Tensor,
            z: Optional[torch.Tensor],
            r: Rigid,
            mask: torch.Tensor,
            surface_points: torch.Tensor,  # [*, N_surface, 3] coordinates
            surface_features: Optional[torch.Tensor] = None,  # [*, N_surface, C_surface]
            surface_mask: Optional[torch.Tensor] = None,  # [*, N_surface]
            _offload_inference: bool = False,
            _z_reference_list: Optional[Sequence[torch.Tensor]] = None,
    ):
        """
        Args:
            s: [*, N_res, C_s] single backbone representation
            z: [*, N_res, N_res, C_z] pair representation
            r: [*, N_res] transformation object for backbone
            mask: [*, N_res] backbone mask
            surface_points: [*, N_surface, 3] surface point cloud coordinates
            surface_features: [*, N_surface, C_surface] features for surface points
            surface_mask: [*, N_surface] mask for surface points
            _offload_inference: Whether to offload inference
            _z_reference_list: Reference list for z
        Returns:
            Tuple of:
                [*, N_res, C_s] updated backbone representation
                [*, N_surface, C_s] updated surface representation
        """
        if _offload_inference:
            z = _z_reference_list
        else:
            z = [z]

        # Process surface points
        batch_dims = surface_points.shape[:-2]
        n_surface = surface_points.shape[-2]
        n_res = s.shape[-2]

        if surface_features is None:
            # Default to a simple feature if none provided
            surface_features = torch.ones(*batch_dims, n_surface, 1,
                                          device=surface_points.device)

        if surface_mask is None:
            surface_mask = torch.ones(*batch_dims, n_surface,
                                      device=surface_points.device)

        # Convert to proper dimensions if needed
        if len(surface_features.shape) != len(s.shape):
            # Add batch dimensions if needed
            for _ in range(len(s.shape) - len(surface_features.shape)):
                surface_features = surface_features.unsqueeze(0)
                surface_mask = surface_mask.unsqueeze(0)

        # Process surface features if needed
        if surface_features.shape[-1] != self.surface_channels:
            # Create a temporary encoder
            tmp_encoder = nn.Linear(
                surface_features.shape[-1], self.surface_channels,
                device=surface_features.device
            )
            surface_features = tmp_encoder(surface_features)

        # Encode surface features
        surface_feat = self.surface_encoder(surface_features)  # [*, N_surface, C_s]

        # Calculate distances between backbone residues and surface points
        backbone_pos = r.get_trans()  # [*, N_res, 3]

        # Ensure proper broadcasting by adding explicit dimensions
        # [*, N_res, 1, 3] - [*, 1, N_surface, 3] = [*, N_res, N_surface, 3]
        backbone_surface_displacement = backbone_pos.unsqueeze(-2) - surface_points.unsqueeze(-3)

        # [*, N_res, N_surface]
        backbone_surface_dist_sq = torch.sum(backbone_surface_displacement ** 2, dim=-1)

        #######################################
        # Generate backbone scalar and point activations
        #######################################
        # [*, N_res, H * C_hidden]
        q = self.linear_q(s)
        kv = self.linear_kv(s)

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, 2 * C_hidden]
        kv = kv.view(kv.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, H, C_hidden]
        k, v = torch.split(kv, self.c_hidden, dim=-1)

        # [*, N_res, H * P_q * 3]
        q_pts = self.linear_q_points(s)

        # [*, N_res, H * P_q, 3]
        q_pts = torch.split(q_pts, q_pts.shape[-1] // 3, dim=-1)
        q_pts = torch.stack(q_pts, dim=-1)
        q_pts = r[..., None].apply(q_pts)

        # [*, N_res, H, P_q, 3]
        q_pts = q_pts.view(
            q_pts.shape[:-2] + (self.no_heads, self.no_qk_points, 3)
        )

        # [*, N_res, H * (P_q + P_v) * 3]
        kv_pts = self.linear_kv_points(s)

        # [*, N_res, H * (P_q + P_v), 3]
        kv_pts = torch.split(kv_pts, kv_pts.shape[-1] // 3, dim=-1)
        kv_pts = torch.stack(kv_pts, dim=-1)
        kv_pts = r[..., None].apply(kv_pts)

        # [*, N_res, H, (P_q + P_v), 3]
        kv_pts = kv_pts.view(kv_pts.shape[:-2] + (self.no_heads, -1, 3))

        # [*, N_res, H, P_q/P_v, 3]
        k_pts, v_pts = torch.split(
            kv_pts, [self.no_qk_points, self.no_v_points], dim=-2
        )

        ##########################
        # Compute attention scores for backbone-backbone interaction
        ##########################
        # [*, N_res, N_res, H]
        b = self.linear_b(z[0])

        if (_offload_inference):
            z[0] = z[0].cpu()

        # [*, H, N_res, N_res]
        a = torch.matmul(
            permute_final_dims(q, (1, 0, 2)),  # [*, H, N_res, C_hidden]
            permute_final_dims(k, (1, 2, 0)),  # [*, H, C_hidden, N_res]
        )
        a *= math.sqrt(1.0 / (3 * self.c_hidden))
        a += (math.sqrt(1.0 / 3) * permute_final_dims(b, (2, 0, 1)))

        # [*, N_res, N_res, H, P_q, 3]
        pt_displacement = q_pts.unsqueeze(-4) - k_pts.unsqueeze(-5)
        pt_att = pt_displacement ** 2

        # [*, N_res, N_res, H, P_q]
        pt_att = sum(torch.unbind(pt_att, dim=-1))
        head_weights = self.softplus(self.head_weights).view(
            *((1,) * len(pt_att.shape[:-2]) + (-1, 1))
        )
        head_weights = head_weights * math.sqrt(
            1.0 / (3 * (self.no_qk_points * 9.0 / 2))
        )
        pt_att = pt_att * head_weights

        # [*, N_res, N_res, H]
        pt_att = torch.sum(pt_att, dim=-1) * (-0.5)
        # [*, N_res, N_res]
        square_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        square_mask = self.inf * (square_mask - 1)

        # [*, H, N_res, N_res]
        pt_att = permute_final_dims(pt_att, (2, 0, 1))

        a = a + pt_att
        a = a + square_mask.unsqueeze(-3)
        a = self.softmax(a)

        ################
        # Compute outputs for backbone self-attention
        ################
        # [*, N_res, H, C_hidden]
        o_backbone_self = torch.matmul(
            a, v.transpose(-2, -3)
        ).transpose(-2, -3)

        # [*, N_res, H * C_hidden]
        o_backbone_self = flatten_final_dims(o_backbone_self, 2)

        # [*, H, 3, N_res, P_v]
        o_pt = torch.sum(
            (
                    a[..., None, :, :, None]
                    * permute_final_dims(v_pts, (1, 3, 0, 2))[..., None, :, :]
            ),
            dim=-2,
        )

        # [*, N_res, H, P_v, 3]
        o_pt = permute_final_dims(o_pt, (2, 0, 3, 1))
        o_pt = r[..., None, None].invert_apply(o_pt)

        # [*, N_res, H * P_v]
        o_pt_dists = torch.sqrt(torch.sum(o_pt ** 2, dim=-1) + self.eps)
        o_pt_norm_feats = flatten_final_dims(
            o_pt_dists, 2)

        # [*, N_res, H * P_v, 3]
        o_pt = o_pt.reshape(*o_pt.shape[:-3], -1, 3)

        if (_offload_inference):
            z[0] = z[0].to(o_pt.device)

        # [*, N_res, H, C_z // 4]
        pair_z = self.down_z(z[0])
        o_pair = torch.matmul(a.transpose(-2, -3), pair_z)

        # [*, N_res, H * C_z // 4]
        o_pair = flatten_final_dims(o_pair, 2)

        ##########################
        # Surface-backbone interaction (simpler implementation)
        ##########################
        # Process surface points for attention
        # [*, N_surface, H * C_hidden]
        surface_q = self.linear_surface_q(surface_feat)
        surface_k = self.linear_surface_k(surface_feat)
        surface_v = self.linear_surface_v(surface_feat)

        # [*, N_surface, H, C_hidden]
        surface_q = surface_q.view(surface_q.shape[:-1] + (self.no_heads, -1))
        surface_k = surface_k.view(surface_k.shape[:-1] + (self.no_heads, -1))
        surface_v = surface_v.view(surface_v.shape[:-1] + (self.no_heads, -1))

        # Compute distance-based weights for surface-backbone interaction
        # Scale distances by a learned weight
        dist_weight = self.softplus(self.surface_backbone_weights)
        weighted_dists = -dist_weight * backbone_surface_dist_sq

        # Apply backbone mask and surface mask
        backbone_mask_2d = mask.unsqueeze(-1)  # [*, N_res, 1]
        surface_mask_2d = surface_mask.unsqueeze(-2)  # [*, 1, N_surface]

        # Create combined mask for backbone-surface interaction
        bs_mask = backbone_mask_2d * surface_mask_2d  # [*, N_res, N_surface]
        bs_mask_penalty = self.inf * (bs_mask - 1)

        # Add mask penalty to distance weights
        masked_weighted_dists = weighted_dists + bs_mask_penalty

        # Backbone → Surface attention weights (distance-based)
        backbone_to_surface_weights = self.softmax(masked_weighted_dists)  # [*, N_res, N_surface]

        # Surface → Backbone attention weights (transposed)
        surface_to_backbone_weights = self.softmax(
            masked_weighted_dists.transpose(-1, -2)
        )  # [*, N_surface, N_res]

        # Backbone attending to surface (gathering surface features)
        # [*, N_res, N_surface, C_s] x [*, N_surface, C_s] → [*, N_res, C_s]
        backbone_surface_interaction = torch.matmul(
            backbone_to_surface_weights.unsqueeze(-2),  # [*, N_res, 1, N_surface]
            surface_feat.unsqueeze(-3)  # [*, 1, N_surface, C_s]
        ).squeeze(-2)  # [*, N_res, C_s]

        # Process gathered surface features through the same dimension as other backbone features
        o_backbone_surface = backbone_surface_interaction
        if o_backbone_surface.shape[-1] != o_backbone_self.shape[-1]:
            # Project to match dimensions if needed
            tmp_proj = nn.Linear(
                o_backbone_surface.shape[-1], o_backbone_self.shape[-1],
                device=o_backbone_self.device
            )
            o_backbone_surface = tmp_proj(o_backbone_surface)

        # Final backbone features
        o_backbone_feats = [
            o_backbone_self,
            *torch.unbind(o_pt, dim=-1),
            o_pt_norm_feats,
            o_pair,
            o_backbone_surface  # Add surface interaction features
        ]

        # Updated backbone representation
        s_update = self.linear_out(
            torch.cat(o_backbone_feats, dim=-1)
        )

        # Surface attending to backbone (gathering backbone features)
        # [*, N_surface, N_res, C_s] x [*, N_res, C_s] → [*, N_surface, C_s]
        surface_backbone_interaction = torch.matmul(
            surface_to_backbone_weights.unsqueeze(-2),  # [*, N_surface, 1, N_res]
            s.unsqueeze(-3)  # [*, 1, N_res, C_s]
        ).squeeze(-2)  # [*, N_surface, C_s]

        # Update surface features
        surface_update = self.linear_surface_out(
            torch.cat([surface_backbone_interaction, surface_feat], dim=-1)
        )

        # Apply surface mask
        surface_update = surface_update * surface_mask.unsqueeze(-1)

        return s_update, surface_update