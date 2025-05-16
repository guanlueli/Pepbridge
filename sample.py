import sys
sys.path.append('../')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import math
from tqdm.auto import tqdm
import functools
import os
import argparse
import pandas as pd
from copy import deepcopy
import open3d as o3d

from models_con.pep_dataloader import PepDataset

from pepbridge.utils.train import recursive_to

from pepbridge.modules.common.geometry import reconstruct_backbone, reconstruct_backbone_partially, align, batch_align
from pepbridge.modules.protein.writers import save_pdb, save_pdb_chains

from pepbridge.utils.data import PaddingCollate

from models_con.utils import process_dic

from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask
from data.utils_pymol import preprocess_surface_single, plot_surface, get_symmetric_interface_masks, order_point_clouds_globally
# from torch_scatter import scatter_sum, scatter_mean


collate_fn = PaddingCollate(eight=False)

import argparse


def item_to_batch(item, nums=32):
    data_list = [deepcopy(item) for i in range(nums)]
    return collate_fn(data_list)

def sample_for_data_bb(data, model, device, save_root, num_steps=200, sample_structure=True, sample_sequence=True, nums=8):
    if not os.path.exists(os.path.join(save_root,data["id"])):
        os.makedirs(os.path.join(save_root,data["id"]))
    batch = recursive_to(item_to_batch(data, nums=nums),device=device)
    traj = model.sample(batch, num_steps=num_steps, sample_structure=sample_structure, sample_sequence=sample_sequence)
    final = recursive_to(traj[-1], device=device)
    pos_bb = reconstruct_backbone(R=final['rotmats'],t=final['trans'],aa=final['seqs'],chain_nb=batch['chain_nb'],res_nb=batch['res_nb'],mask=batch['res_mask']) # (32,L,4,3)
    pos_ha = F.pad(pos_bb, pad=(0,0,0,15-4), value=0.) # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
    mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
    mask_bb_atoms[:,:,:4] = True
    mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])
    aa_new = final['seqs']

    chain_nb = torch.LongTensor([0 if gen_mask else 1 for gen_mask in data['generate_mask']])
    chain_id = ['A' if gen_mask else 'B' for gen_mask in data['generate_mask']]
    icode = [' ' for _ in range(len(data['icode']))]
    for i in range(nums):
        ref_bb_pos = data['pos_heavyatom'][i][:,:4].cpu()
        pred_bb_pos = pos_new[i][:,:4].cpu()
        data_saved = {
                      'chain_nb':data['chain_nb'],'chain_id':data['chain_id'],'resseq':data['resseq'],'icode':data['icode'],
                      'aa':aa_new[i].cpu(), 'mask_heavyatom':mask_new[i].cpu(), 'pos_heavyatom':pos_new[i].cpu(),
                    }

        save_pdb(data_saved,path=os.path.join(save_root,data["id"],f'{data["id"]}_{i}.pdb'))
    save_pdb(data,path=os.path.join(save_root,data["id"],f'{data["id"]}_gt.pdb'))

def save_samples_bb(samples,save_dir):
    # meta data
    batch = recursive_to(samples['batch'],'cpu')
    chain_id = [list(item) for item in zip(*batch['chain_id'])][0] # fix chain id in collate func
    icode = [' ' for _ in range(len(chain_id))] # batch icode have same problem
    nums = len(batch['id'])
    id = batch['id'][0]
    # batch convert
    # aa=batch['aa] if only bb level
    pos_bb = reconstruct_backbone(R=samples['rotmats'],t=samples['trans'],aa=samples['seqs'],chain_nb=batch['chain_nb'],res_nb=batch['res_nb'],mask=batch['res_mask']) # (32,L,4,3)
    pos_ha = F.pad(pos_bb, pad=(0,0,0,15-4), value=0.) # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
    mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
    mask_bb_atoms[:,:,:4] = True
    mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])
    aa_new = samples['seqs']
    for i in range(nums):
        data_saved = {
                      'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                      'aa':aa_new[i], 'mask_heavyatom':mask_new[i], 'pos_heavyatom':pos_new[i],
                    }
        save_pdb(data_saved,path=os.path.join(save_dir,f'sample_{i}.pdb'))
    data_saved = {
                    'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                    'aa':batch['aa'][0], 'mask_heavyatom':batch['mask_heavyatom'][0], 'pos_heavyatom':batch['pos_heavyatom'][0],
                }
    save_pdb(data_saved,path=os.path.join(save_dir,f'gt.pdb'))

def save_samples_sc(samples, save_dir):
    # meta data
    batch = recursive_to(samples['batch'],'cpu')
    chain_id = [list(item) for item in zip(*batch['chain_id'])][0] # fix chain id in collate func
    icode = [' ' for _ in range(len(chain_id))] # batch icode have same problem
    nums = len(batch['id'])
    id = batch['id'][0]
    # batch convert
    # aa=batch['aa] if only bb level
    pos_ha, _, _ = full_atom_reconstruction(R_bb=samples['rotmats'],t_bb=samples['trans'],angles=samples['angles'],aa=samples['seqs']) # (32,L,14,3), instead of 15, ignore OXT masked
    pos_ha = F.pad(pos_ha, pad=(0,0,0,15-14), value=0.) # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
    mask_new = get_heavyatom_mask(samples['seqs'])
    aa_new = samples['seqs']
    for i in range(nums):
        data_saved = {
                      'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                      'aa':aa_new[i], 'mask_heavyatom':mask_new[i], 'pos_heavyatom':pos_new[i],
                    }
        save_pdb(data_saved, path=os.path.join(save_dir,f'sample_{i}.pdb'))
    data_saved = {
                    'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                    'aa':batch['aa'][0], 'mask_heavyatom':batch['mask_heavyatom'][0], 'pos_heavyatom':batch['pos_heavyatom'][0],
                }
    save_pdb(data_saved,path=os.path.join(save_dir,f'gt.pdb'))


def save_samples_surf(samples, save_dir):
    # meta data
    batch = recursive_to(samples['batch'],'cpu')
    chain_id = [list(item) for item in zip(*batch['chain_id'])][0] # fix chain id in collate func
    icode = [' ' for _ in range(len(chain_id))] # batch icode have same problem
    nums = len(batch['id'])
    id = batch['id'][0]
    # batch convert
    # aa=batch['aa] if only bb level
    pos_ha, _, _ = full_atom_reconstruction(R_bb=samples['rotmats'],t_bb=samples['trans'],angles=samples['angles'],aa=samples['seqs']) # (32,L,14,3), instead of 15, ignore OXT masked
    pos_ha = F.pad(pos_ha, pad=(0,0,0,15-14), value=0.) # (32,L,A,3) pos14 A=14
    pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
    mask_new = get_heavyatom_mask(samples['seqs'])
    aa_new = samples['seqs']

    for i in range(nums):
        data_saved = {
                      'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                      'aa':aa_new[i], 'mask_heavyatom':mask_new[i], 'pos_heavyatom':pos_new[i],
                    }
        save_pdb(data_saved, path=os.path.join(save_dir,f'sample_{i}.pdb'))
    data_saved = {
                    'chain_nb':batch['chain_nb'][0],'chain_id':chain_id,'resseq':batch['resseq'][0],'icode':icode,
                    'aa':batch['aa'][0], 'mask_heavyatom':batch['mask_heavyatom'][0], 'pos_heavyatom':batch['pos_heavyatom'][0],
                }
    save_pdb(data_saved,path=os.path.join(save_dir,f'gt.pdb'))


def tensor_to_o3d_point_cloud(points):
    """
    Convert tensor or numpy array to Open3D point cloud

    Parameters:
    points: torch.Tensor or numpy.ndarray of shape (N, 3)

    Returns:
    o3d.geometry.PointCloud
    """
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if isinstance(points, np.ndarray):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        return pcd

    return points  # Return as-is if already Open3D point cloud

def o3d_point_cloud_to_tensor(pcd, device='cuda'):
    """
    Convert Open3D point cloud to tensor

    Parameters:
    pcd: o3d.geometry.PointCloud
    device: torch device ('cuda' or 'cpu')

    Returns:
    torch.Tensor of shape (N, 3)
    """
    points = np.asarray(pcd.points)
    return torch.from_numpy(points).float().to(device)

def align_point_clouds(source_points, target_points, threshold=0.02, trans_init=None, device='cuda'):
    """
    Align source point cloud to target point cloud using ICP

    Parameters:
    source_points: Source points (torch.Tensor, numpy.ndarray, or o3d.geometry.PointCloud)
    target_points: Target points (torch.Tensor, numpy.ndarray, or o3d.geometry.PointCloud)
    threshold: Distance threshold for ICP
    trans_init: Initial transformation matrix (4x4 numpy array)
    device: torch device for output tensor

    Returns:
    transformation: The transformation matrix (numpy array)
    source_transformed: Transformed source points (same type as input)
    """
    # Store input type to return the same type
    input_is_tensor = isinstance(source_points, torch.Tensor)

    # Convert inputs to Open3D point clouds
    source_cloud = tensor_to_o3d_point_cloud(source_points)
    target_cloud = tensor_to_o3d_point_cloud(target_points)

    # If no initial transformation is provided, use identity matrix
    if trans_init is None:
        trans_init = np.identity(4)

    # Estimate normals
    source_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Apply ICP
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_cloud, target_cloud,
        threshold,
        trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

    # Transform the source point cloud
    source_transformed = source_cloud.transform(reg_p2p.transformation)

    # Convert back to tensor if input was tensor
    if input_is_tensor:
        source_transformed = o3d_point_cloud_to_tensor(source_transformed, device)

    return reg_p2p.transformation, source_transformed

def apply_transformation(points, transformation, device='cuda'):
    """
    Apply transformation matrix to points

    Parameters:
    points: torch.Tensor of shape (N, 3)
    transformation: numpy array of shape (4, 4)
    device: torch device

    Returns:
    torch.Tensor of shape (N, 3)
    """
    # Convert transformation to tensor
    transform = torch.from_numpy(transformation).float().to(device)

    # Add homogeneous coordinate
    ones = torch.ones(points.shape[0], 1, device=device)
    points_homogeneous = torch.cat([points, ones], dim=1)

    # Apply transformation
    points_transformed = torch.matmul(points_homogeneous, transform.t())

    # Return to 3D coordinates
    return points_transformed[:, :3]

def rotate_pointcloud(points, angles):
    """
    Rotate point cloud by given angles (in radians)

    Args:
        points: numpy array of shape (N, 3) where N is number of points
        angles: list/array of 3 angles [rx, ry, rz] for rotation around x, y, z axes

    Returns:
        rotated_points: numpy array of same shape as input with rotation applied
    """
    # Rotation matrix around x-axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angles[0]), -np.sin(angles[0])],
        [0, np.sin(angles[0]), np.cos(angles[0])]
    ])

    # Rotation matrix around y-axis
    Ry = np.array([
        [np.cos(angles[1]), 0, np.sin(angles[1])],
        [0, 1, 0],
        [-np.sin(angles[1]), 0, np.cos(angles[1])]
    ])

    # Rotation matrix around z-axis
    Rz = np.array([
        [np.cos(angles[2]), -np.sin(angles[2]), 0],
        [np.sin(angles[2]), np.cos(angles[2]), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Apply rotation
    rotated_points = points @ R.T

    return rotated_points

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--SAMPLEDIR', type=str, default='/mnt/storage1/gli/1Data/data_surface/pepbridge')
    parser = args.parse_args()
    SAMPLE_DIR = parser.SAMPLEDIR
    output_numb = '02'
    save_dir_name = "outputs_2"
    names = [n.split('.')[0] for n in os.listdir(os.path.join(SAMPLE_DIR, save_dir_name))][16:]
    for name in tqdm(names):
        sample = torch.load(os.path.join(SAMPLE_DIR,save_dir_name,f'{name}.pt'))
        os.makedirs(os.path.join(SAMPLE_DIR,f'pdbs{output_numb}',name),exist_ok=True)
        save_samples_surf(sample,os.path.join(SAMPLE_DIR,f'pdbs{output_numb}',name))