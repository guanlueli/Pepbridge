from typing import List, Dict, Any
from openfold.utils import rigid_utils as ru
from data import residue_constants
import numpy as np
import collections
import string
import pickle
import os
import torch
# from torch_scatter import scatter_add, scatter
from Bio.PDB.Chain import Chain
from data import protein
import dataclasses
from Bio import PDB
import io
from scipy.spatial.transform import Rotation


"""From https://github.com/microsoft/protein-frame-flow"""

Rigid = ru.Rigid
Protein = protein.Protein

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + ' '
CHAIN_TO_INT = {
    chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)
}
INT_TO_CHAIN = {
    i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)
}

NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE

CHAIN_FEATS = [
    'atom_positions', 'aatype', 'atom_mask', 'residue_index', 'b_factors'
]

to_numpy = lambda x: x.detach().cpu().numpy()
aatype_to_seq = lambda aatype: ''.join([
        residue_constants.restypes_with_x[x] for x in aatype])

def create_rigid(rots, trans):
    rots = ru.Rotation(rot_mats=rots)
    return Rigid(rots=rots, trans=trans)


def batch_align_structures(pos_1, pos_2, mask=None):
    if pos_1.shape != pos_2.shape:
        raise ValueError('pos_1 and pos_2 must have the same shape.')
    if pos_1.ndim != 3:
        raise ValueError(f'Expected inputs to have shape [B, N, 3]')
    num_batch = pos_1.shape[0]
    device = pos_1.device
    batch_indices = (
        torch.ones(*pos_1.shape[:2], device=device, dtype=torch.int64) 
        * torch.arange(num_batch, device=device)[:, None]
    )
    flat_pos_1 = pos_1.reshape(-1, 3)
    flat_pos_2 = pos_2.reshape(-1, 3)
    flat_batch_indices = batch_indices.reshape(-1)
    if mask is None:
        aligned_pos_1, aligned_pos_2, align_rots = align_structures(
            flat_pos_1, flat_batch_indices, flat_pos_2)
        aligned_pos_1 = aligned_pos_1.reshape(num_batch, -1, 3)
        aligned_pos_2 = aligned_pos_2.reshape(num_batch, -1, 3)
        return aligned_pos_1, aligned_pos_2, align_rots

    flat_mask = mask.reshape(-1).bool()
    _, _, align_rots = align_structures(
        flat_pos_1[flat_mask],
        flat_batch_indices[flat_mask],
        flat_pos_2[flat_mask]
    )
    aligned_pos_1 = torch.bmm(
        pos_1,
        align_rots
    )
    return aligned_pos_1, pos_2, align_rots


def skew_symmetric(v):
    """Convert vector to skew symmetric matrix."""
    batch_dims = v.shape[:-1]
    zeros = torch.zeros(*batch_dims, 1, device=v.device, dtype=v.dtype)

    a1, a2, a3 = torch.unbind(v, dim=-1)

    return torch.stack([
        torch.cat([zeros, -a3.unsqueeze(-1), a2.unsqueeze(-1)], dim=-1),
        torch.cat([a3.unsqueeze(-1), zeros, -a1.unsqueeze(-1)], dim=-1),
        torch.cat([-a2.unsqueeze(-1), a1.unsqueeze(-1), zeros], dim=-1)
    ], dim=-2)


def axis_angle_to_matrix(angle_axis):
    """Convert rotation vector to rotation matrix using Rodrigues formula."""
    batch_dims = angle_axis.shape[:-1]
    theta = torch.norm(angle_axis, dim=-1, keepdim=True)
    theta = torch.clamp(theta, min=1e-6)

    k = angle_axis / theta
    k_skew = skew_symmetric(k)
    k_dot = torch.matmul(k.unsqueeze(-2), k.unsqueeze(-1))

    eye = torch.eye(3, device=angle_axis.device, dtype=angle_axis.dtype)
    eye = eye.view(*[1] * len(batch_dims), 3, 3).expand(*batch_dims, 3, 3)

    R = eye + torch.sin(theta).unsqueeze(-1) * k_skew + (1 - torch.cos(theta)).unsqueeze(-1) * (k_dot - eye)
    return R


def matrix_to_axis_angle(matrix):
    """Convert rotation matrix to rotation vector."""
    batch_dims = matrix.shape[:-2]

    # Handle singularity when rotation angle is 0
    eye = torch.eye(3, device=matrix.device, dtype=matrix.dtype)
    eye = eye.view(*[1] * len(batch_dims), 3, 3).expand(*batch_dims, 3, 3)

    cos_theta = (torch.diagonal(matrix, dim1=-2, dim2=-1).sum(-1) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1, 1)
    theta = torch.acos(cos_theta)

    # Handle singularity when rotation angle is π
    sin_theta = torch.sin(theta)

    # Define a small epsilon to handle numerical stability
    eps = 1e-6

    # When sin(theta) is close to 0
    mask_zero = torch.abs(sin_theta) < eps
    mask_pi = cos_theta < -0.99

    axis = torch.zeros(*batch_dims, 3, device=matrix.device, dtype=matrix.dtype)

    # Regular case
    where_regular = ~(mask_zero | mask_pi)
    if where_regular.any():
        axis_regular = torch.stack([
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1]
        ], dim=-1)
        axis[where_regular] = axis_regular[where_regular] / (2 * sin_theta[where_regular].unsqueeze(-1))

    # When angle is close to π
    if mask_pi.any():
        diag = torch.diagonal(matrix[mask_pi], dim1=-2, dim2=-1)
        axis_pi = torch.stack([
            torch.sqrt((diag[..., 0] + 1) / 2),
            torch.sqrt((diag[..., 1] + 1) / 2),
            torch.sqrt((diag[..., 2] + 1) / 2)
        ], dim=-1)
        axis[mask_pi] = axis_pi * theta[mask_pi].unsqueeze(-1)

    # When angle is close to 0
    if mask_zero.any():
        axis[mask_zero] = torch.zeros_like(axis[mask_zero])

    return axis * theta.unsqueeze(-1)


def compose_rotvec(r1: torch.Tensor, r2: torch.Tensor) -> torch.Tensor:
    """Compose batches of rotation vectors in axis-angle format.

    Args:
        r1: Batch of rotation vectors of shape (B, N, 3) or (B, 3)
        r2: Batch of rotation vectors of shape (B, N, 3) or (B, 3)

    Returns:
        Composed rotation vectors of shape (B, N, 3) or (B, 3)
    """
    # Add batch dimension if not present
    if r1.dim() == 2:
        r1 = r1.unsqueeze(0)
    if r2.dim() == 2:
        r2 = r2.unsqueeze(0)

    # Convert to rotation matrices
    R1 = axis_angle_to_matrix(r1)  # (B, N, 3, 3) or (B, 3, 3)
    R2 = axis_angle_to_matrix(r2)  # (B, N, 3, 3) or (B, 3, 3)

    # Batch matrix multiplication
    cR = torch.matmul(R1, R2)  # (B, N, 3, 3) or (B, 3, 3)

    # Convert back to axis-angle
    result = matrix_to_axis_angle(cR)  # (B, N, 3) or (B, 3)

    # Remove batch dimension if input was unbatched
    if r1.size(0) == 1 and r2.size(0) == 1:
        result = result.squeeze(0)

    return result

