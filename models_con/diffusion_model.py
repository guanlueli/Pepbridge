import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_con.edge import EdgeEmbedder
from models_con.node import NodeEmbedder
from pepbridge.modules.common.layers import sample_from, clampped_one_hot
from models_con.sfencoder import SFEncoder
from pepbridge.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
from pepbridge.modules.common.geometry import construct_3d_basis

from pepbridge.modules.so3.dist import centered_gaussian, uniform_so3
from pepbridge.modules.common.geometry import batch_align, align
from data import utils as data_utils
from tqdm import tqdm

from data import so3_utils
from data import all_atom
from data import so3_diffuser
from data import r3_diffuser
from data import bridge_diffuser
from data import torus_diffuser
from data import seq_diffuser

from models_con.torsion import get_torsion_angle, torsions_mask
import models_con.torus as torus

from copy import deepcopy
from pepbridge.utils.data import PaddingCollate
collate_fn = PaddingCollate(eight=False)
from pepbridge.utils.train import recursive_to

resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

class DiffusionModel(nn.Module):
    def __init__(self,cfg, device):
        super().__init__()
        self._conf = cfg
        self._model_cfg = cfg.encoder
        self._interpolant_cfg = cfg.interpolant

        self.node_embedder = NodeEmbedder(cfg.encoder.node_embed_size,max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.encoder.edge_embed_size,max_num_heavyatoms)
        self.ipa_sutf_bb_sfm = SFEncoder(cfg.encoder)

        self.surf_node_mlp = nn.Sequential(
            nn.Linear(cfg.encoder.surf.node_in_dim, cfg.encoder.surf.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.surf.node_hidden_dim, cfg.encoder.surf.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.surf.node_hidden_dim, cfg.encoder.surf.node_out_dim)
        )

        # Edge embedding network
        self.surf_edge_mlp = nn.Sequential(
            nn.Linear(cfg.encoder.surf.edge_in_dim, cfg.encoder.surf.edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.surf.edge_hidden_dim, cfg.encoder.surf.edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.surf.edge_hidden_dim, cfg.encoder.surf.edge_out_dim)
        )

        # Position embedder (projects coordinates to higher dimension)
        self.pos_embedder = nn.Sequential(
            nn.Linear(3, cfg.encoder.surf.pos_embed_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.surf.pos_embed_dim, cfg.encoder.surf.pos_embed_dim)
        )

        # Edge attention network (converts attention logits to edge features)
        self.edge_attn_network = nn.Sequential(
            nn.Linear(1, cfg.encoder.surf.edge_hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.encoder.surf.edge_hidden_dim, cfg.encoder.surf.edge_out_dim)
        )

        self.k_nearest = cfg.encoder.surf.k_nearest

        self.sample_structure = self._interpolant_cfg.sample_structure
        self.sample_sequence = self._interpolant_cfg.sample_sequence

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value

        self._diffuse_surface = cfg.diffuser.diffuse_surface
        self._diffuse_trans = cfg.diffuser.diffuse_trans
        self._diffuse_rot = cfg.diffuser.diffuse_rot
        self._diffuse_angle = cfg.diffuser.diffuse_angle

        self._diffuse_trans_cfg = cfg.diffuser.r3
        self._diffuse_rot_cfg = cfg.diffuser.so3
        
        self._so3_diffuser = so3_diffuser.SO3Diffuser(cfg.diffuser.so3, device)
        self._surf_diffuser = bridge_diffuser.BridgeDiffuser(cfg.diffuser.bridge)
        self._r3_diffuser = r3_diffuser.R3Diffuser(cfg.diffuser.r3, num_t=cfg.interpolant.sampling.num_timesteps)
        self._tour_diffuser = torus_diffuser.AngleDiffuser(cfg.diffuser.tour)
        self._seq_diffuser = seq_diffuser.SeqDiffuser(cfg.diffuser.seq)

    def to(self, device):
        super().to(device)
        self.device = device
        # Make sure SO3Diffuser's tensors are also moved to the new device
        self._so3_diffuser.device = device
        self._r3_diffuser.device = device
        self._tour_diffuser.device = device
        self._seq_diffuser.device = device
        self._so3_diffuser.discrete_omega = self._so3_diffuser.discrete_omega.to(device)
        return self

    def encode(self, batch):

        rotmats_1 =  construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N] )
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        seqs_1 = batch['aa']

        angles_1 = batch['torsion_angle']

        context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
        structure_mask = context_mask if self.sample_structure else None
        sequence_mask = context_mask if self.sample_sequence else None
        node_embed = self.node_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'],
                                        batch['mask_heavyatom'], structure_mask=structure_mask, 
                                        sequence_mask=sequence_mask)
        edge_embed = self.edge_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'],
                                        batch['mask_heavyatom'], structure_mask=structure_mask, 
                                        sequence_mask=sequence_mask)

        return rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed

    def build_adjacency_mask(self, pos):
        """
        Builds an adjacency mask based on k-nearest neighbors

        Args:
            pos: (N, L, 3) point cloud coordinates

        Returns:
            mask: (N, L, L) boolean mask where True indicates valid edges
        """
        N, L, _ = pos.shape

        # Calculate pairwise distances
        x_i = pos.unsqueeze(2)  # (N, L, 1, 3)
        x_j = pos.unsqueeze(1)  # (N, 1, L, 3)

        # Squared L2 distance
        dist = torch.sum((x_i - x_j) ** 2, dim=-1)  # (N, L, L)

        # Create k-nearest neighbor mask
        _, nn_idx = torch.topk(dist, k=min(self.k_nearest, L), dim=-1, largest=False)

        # Create adjacency mask
        adj_mask = torch.zeros(N, L, L, dtype=torch.bool, device=pos.device)

        batch_indices = torch.arange(N, device=pos.device).view(-1, 1, 1)
        row_indices = torch.arange(L, device=pos.device).view(1, -1, 1)
        adj_mask[batch_indices, row_indices, nn_idx] = True

        # Ensure symmetry (if i is connected to j, then j is connected to i)
        adj_mask = adj_mask | adj_mask.transpose(1, 2)

        return adj_mask

    def encoder_surf(self, batch, surf_t):

        h_pep = torch.cat([batch['surf_hbond'].unsqueeze(-1), batch['surf_hp'].unsqueeze(-1)], dim=-1)
        node_feat = torch.cat([surf_t, h_pep], dim=-1)
        node_embed = self.surf_node_mlp(node_feat)  # MLP: (N, L, H_node)

        # surf_node_embed = node_embed * batch['mask_surf'][..., None]
        surf_node_embed = node_embed

        surf_edge = False
        if surf_edge:
            pos_proj = self.pos_embedder(surf_t)  # (N, L, H)
            attn_logits = torch.einsum('nld,nmd->nlm', pos_proj, pos_proj)  # (N, L, L)

            # Apply adjacency mask
            adj_mask = self.build_adjacency_mask(surf_t)  # (N, L, L)
            attn_logits = attn_logits.masked_fill(~adj_mask, -1e9)

            # Convert attention logits to edge features
            edge_embed = self.edge_attn_network(attn_logits.unsqueeze(-1))  # (N, L, L, H_edge)
            surf_edge_embed = edge_embed
        else:
            surf_edge_embed = None

        return surf_node_embed, surf_edge_embed

    def zero_center_part(self,pos,gen_mask,res_mask):
        """
        move pos by center of gen_mask
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        center = torch.sum(pos * gen_mask[...,None], dim=1) / (torch.sum(gen_mask,dim=-1,keepdim=True) + 1e-8) # (B,N,3)*(B,N,1)->(B,3)/(B,1)->(B,3)
        center = center.unsqueeze(1) # (B,1,3)
        # center = 0. it seems not center didnt influence the result, but its good for training stabilty
        pos = pos - center
        pos = pos * res_mask[...,None]
        return pos,center
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)
    
    def forward(self, batch):
        
        num_batch, num_res = batch['aa'].shape
        gen_mask, res_mask, angle_mask = batch['generate_mask'].long(), batch['res_mask'].long(), batch['torsion_angle_mask'].long()

        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch) # no generate mask t=T

        # prepare for denoise
        trans_1_c, _ = self.zero_center_part(trans_1, gen_mask, res_mask) # t=T
        trans_1_c = trans_1 
        
        seqs_1_simplex = self.seq_to_simplex(seqs_1)  # t=T
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)  # t=T

        with torch.no_grad():
            t = torch.rand((num_batch, 1), device=batch['aa'].device)
            t = t * (1-2 * self._interpolant_cfg.min_t) + self._interpolant_cfg.min_t # avoid 0
            
            # Add noise to surface
            if self._diffuse_surface:
                surf_T = batch['pts_rec_mask_symmetric']
                surf_0 = batch['surf_pos']
                surf_t = self._surf_diffuser.forward_marginal(surf_0, surf_T, t.squeeze(-1))
                c_skip, c_out, c_in = self._surf_diffuser.get_bridge_scalings(t)
                # rescaled_t = 1000 * 0.25 * torch.log(t + 1e-44)
                c_in = c_in.unsqueeze(-1)  # [B, 1, 1]
                surf_t = c_in * surf_t
                weights = self._surf_diffuser.get_weightings(t)
                weights = weights.unsqueeze(-1)  # [B, 1, 1]
            else:
                surf_t = batch['surf_pos'].detach().clone()  # t=t

            # Add noise to positions
            if self._diffuse_trans:
                trans_t, gt_trans_score = self._r3_diffuser.forward_marginal(trans_1_c, t.squeeze(-1))
                trans_score_scaling = self._r3_diffuser.score_scaling(t.squeeze(-1))
                trans_t_c = torch.where(batch['generate_mask'][...,None], trans_t, trans_1_c)
            else:
                trans_t_c = trans_1_c.detach().clone()  # t=t

            # Add noise to rotations
            if self._diffuse_rot:
                rotvec_1 = data_utils.matrix_to_axis_angle(rotmats_1)
                rotvec_t, gt_rot_score = self._so3_diffuser.forward_marginal(rotvec_1, t)
                rot_score_scaling = self._so3_diffuser.score_scaling(t)
                rotvec_t = torch.where(batch['generate_mask'][..., None], rotvec_t, rotvec_1)
                rotmats_t = data_utils.axis_angle_to_matrix(rotvec_t)
            else:
                rot_score_scaling = self._so3_diffuser.score_scaling(t)
                rotmats_t = rotmats_1.detach().clone()   # t=t

            #  Add noise to angles
            if self._diffuse_angle:
                angles_t, angles_score = self._tour_diffuser.forward_marginal(angles_1, t)
                angles_score_scaling = self._tour_diffuser.score_scaling(t)
                angles_t = torch.where(batch['generate_mask'][..., None], angles_t, angles_1)
            else:
                angles_t = angles_1.detach().clone()   # t=t

            if self.sample_sequence:
                # Add noise to sequence components
                seqs_t_simplex, score, score_scaling = self._seq_diffuser.forward_marginal(seqs_1_simplex, t)
                seqs_t_simplex = torch.where(batch['generate_mask'][..., None], seqs_t_simplex, seqs_1_simplex)
                seqs_t_prob = F.softmax(seqs_t_simplex, dim=-1)
                seqs_t = sample_from(seqs_t_prob)
                seqs_t = torch.where(batch['generate_mask'], seqs_t, seqs_1)    # t=t
            else:
                seqs_t = seqs_1.detach().clone()
                seqs_t_simplex = seqs_1_simplex.detach().clone()
                seqs_t_prob = seqs_1_prob.detach().clone()

        surf_node_embed, surf_edge_embed = self.encoder_surf(batch, surf_t)  # no generate mask
     
        pred_trans_1, pred_rotmats_1, pred_angles_1, pred_seqs_1_prob, pred_surf, init_rigids, curr_rigids  = self.ipa_sutf_bb_sfm(
                                                                                         t,
                                                                                         rotmats_t,
                                                                                         trans_t_c,
                                                                                         angles_t,
                                                                                         seqs_t,
                                                                                         node_embed,
                                                                                         edge_embed,
                                                                                         gen_mask,
                                                                                         res_mask,
                                                                                         surf_node_embed,
                                                                                         surf_edge_embed,
                                                                                         surf_t)
        
        pred_rot_score = self._so3_diffuser.calc_rot_score(
                    init_rigids.get_rots(),
                    curr_rigids.get_rots(),
                    t,
                )

        curr_rigids = self.ipa_sutf_bb_sfm.unscale_rigids(curr_rigids)  
        
        pred_trans_score = self._r3_diffuser.calc_trans_score(
            init_rigids.get_trans(),
            curr_rigids.get_trans(),
            t[:, None, None],
        )
           
        if self._diffuse_surface:
            denoised_surf = c_out.unsqueeze(-1) * pred_surf + c_skip.unsqueeze(-1) * surf_t
        
        pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'], pred_seqs_1, torch.clamp(seqs_1,0,19))
        pred_trans_1_c, _ = self.zero_center_part(pred_trans_1, gen_mask, res_mask)
        pred_trans_1_c = pred_trans_1 

        norm_scale = 1 / (1 - torch.min(t[...,None], torch.tensor(self._interpolant_cfg.t_normalization_clip))) # yim etal.trick, 1/1-t

        # surface_loss
        if self._diffuse_surface:
           
            w_mse_2 = weights * (denoised_surf - surf_0) ** 2
           
            mse_loss =  w_mse_2.mean(dim=list(range(1, len(w_mse_2.shape)))) 
            surface_loss = mse_loss.mean(dim=-1) # (B,)

        else:
            surface_loss = torch.tensor(0.0, device=pred_surf.device)

        surface_loss *= self._conf.loss_weights.surface_loss_weight
        raw_surf_loss = surface_loss/self._conf.loss_weights.surface_loss_weight
        
        if self._diffuse_trans:

            trans_score_mse = (gt_trans_score - pred_trans_score) ** 2 * gen_mask[..., None]
            trans_score_loss = torch.sum(trans_score_mse / trans_score_scaling[:, None, None]**2, dim=(-1, -2)) / (
                        torch.sum(gen_mask, dim=-1) + 1e-8)  # (B,)

            # Translation x0 loss
            trans_x0_loss = torch.sum((pred_trans_1_c - trans_1_c)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)

            trans_loss = (
                trans_score_loss * (t > self._diffuse_trans_cfg.trans_x0_threshold)
                + trans_x0_loss * (t <= self._diffuse_trans_cfg.trans_x0_threshold) * self._diffuse_trans_cfg.trans_x0_weight
            )  # x0

            trans_loss = torch.mean(trans_loss)
            trans_loss *= self._conf.loss_weights.trans_loss_weight

            raw_trans_loss = trans_loss / self._conf.loss_weights.trans_loss_weight

        else:
            trans_loss = 0

        if self._diffuse_rot:
            # rots score loss
            if torch.any(torch.isnan(gt_rot_score)) or torch.any(torch.isinf(gt_rot_score)):  
                raise ValueError(f'Invalid gt_rot_score: {gt_rot_score}')
           
            rot_mse = (gt_rot_score - pred_rot_score)**2 * gen_mask[..., None]
            rot_score_loss = torch.sum(rot_mse / rot_score_scaling[:, None, None]**2,dim=(-1, -2)) / (torch.sum(gen_mask, dim=-1) + 1e-8)
            
            # rots loss
            gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
            pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
            rot_x0_loss = torch.sum(((gt_rot_vf - pred_rot_vf) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)

            # rot_loss = (
            #     rot_score_loss * (t > self._diffuse_rot_cfg.rot_x0_threshold)
            #     + rot_x0_loss * (t <= self._diffuse_rot_cfg.rot_x0_threshold) * self._diffuse_rot_cfg.rot_x0_weight
            # )

            rot_loss = rot_score_loss
            
            rot_loss = torch.mean(rot_loss)
            rot_loss *= self._conf.loss_weights.rot_loss_weight
            
            raw_rot_loss = rot_loss / self._conf.loss_weights.rot_loss_weight

        else:
            rot_loss = torch.tensor(0.0, device=pred_rotmats_1.device)

        # bb aux loss
        gt_bb_atoms = all_atom.to_atom37(trans_1_c, rotmats_1)[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1_c, pred_rotmats_1)[:, :, :3]
        bb_atom_loss_type = 1
        if bb_atom_loss_type == 0:
            # gt_bb_atoms = all_atom.to_bb_atoms(trans_1_c, rotmats_1, angles_1[:,:,0]) # N,CA,C,O,CB
            # pred_bb_atoms = all_atom.to_bb_atoms(pred_trans_1_c, pred_rotmats_1, pred_angles_1[:,:,0])
            # print(gt_bb_atoms.shape)
            bb_atom_loss = torch.sum(
                (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None],
                dim=(-1, -2, -3)
            ) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
            bb_atom_loss = torch.mean(bb_atom_loss)
        elif bb_atom_loss_type == 1:
            squared_diffs = (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None]
            mse_per_sample = torch.sum(squared_diffs, dim=(-1, -2, -3)) / (torch.sum(gen_mask, dim=-1) * 3 + 1e-8)
            bb_atom_loss = torch.sqrt(mse_per_sample).mean()  # RMSD in Angstroms
        bb_atom_loss *= self._conf.loss_weights.bb_atom_loss_weight

        raw_bb_atom_loss = bb_atom_loss / self._conf.loss_weights.bb_atom_loss_weight
        
        if self.sample_sequence:
            #  seqs vf loss
            seqs_loss = F.cross_entropy(pred_seqs_1_prob.view(-1,pred_seqs_1_prob.shape[-1]),torch.clamp(seqs_1,0,19).view(-1), reduction='none').view(pred_seqs_1_prob.shape[:-1]) # (N,L), not softmax
            seqs_loss = torch.sum(seqs_loss * gen_mask, dim=-1) / (torch.sum(gen_mask,dim=-1) + 1e-8)
            seqs_loss = torch.mean(seqs_loss)
        else:
            seqs_loss = torch.tensor(0.0, device=pred_seqs_1.device)
        
        seqs_loss *= self._conf.loss_weights.seqs_loss_weight
        
        raw_seqs_loss = seqs_loss / self._conf.loss_weights.seqs_loss_weight

        if self._diffuse_angle:
            # angle vf loss
            angle_mask_loss = torsions_mask.to(batch['aa'].device)
            angle_mask_loss = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            angle_mask_loss = torch.cat([angle_mask_loss,angle_mask_loss],dim=-1) # (B,L,10)
            angle_mask_loss = torch.logical_and(batch['generate_mask'][...,None].bool(),angle_mask_loss)
            gt_angle_vf = torus.tor_logmap(angles_t, angles_1)
            gt_angle_vf_vec = torch.cat([torch.sin(gt_angle_vf),torch.cos(gt_angle_vf)],dim=-1)
            pred_angle_vf = torus.tor_logmap(angles_t, pred_angles_1)
            pred_angle_vf_vec = torch.cat([torch.sin(pred_angle_vf),torch.cos(pred_angle_vf)],dim=-1)
            angle_loss = torch.sum(((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale)**2*angle_mask_loss,dim=(-1,-2)) / (torch.sum(angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
            angle_loss = torch.mean(angle_loss)
            # angle aux loss
            angles_1_vec = torch.cat([torch.sin(angles_1),torch.cos(angles_1)],dim=-1)
            pred_angles_1_vec = torch.cat([torch.sin(pred_angles_1),torch.cos(pred_angles_1)],dim=-1)
            torsion_loss = torch.sum((pred_angles_1_vec - angles_1_vec)**2*angle_mask_loss,dim=(-1,-2)) / (torch.sum(angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
            torsion_loss = torch.mean(torsion_loss)
        else:
            angle_loss = torch.tensor(0.0, device=pred_angles_1.device)
            torsion_loss = torch.tensor(0.0, device=pred_angles_1.device)

        angle_loss *= self._conf.loss_weights.angle_loss_weight
        torsion_loss *= self._conf.loss_weights.torsion_loss_weight

        raw_angle_loss = angle_loss / self._conf.loss_weights.angle_loss_weight
        raw_torsion_loss = torsion_loss / self._conf.loss_weights.torsion_loss_weight
        
        final_loss = trans_loss + rot_loss + bb_atom_loss + seqs_loss + angle_loss + torsion_loss + surface_loss
        
        return final_loss , {
            "weighted_trans_loss": trans_loss,
            "raw_trans_loss": raw_trans_loss,
            "trans_score_loss": torch.mean(trans_score_loss),
            "trans_x0_loss": torch.mean(trans_x0_loss),
            'weighted_rot_loss': rot_loss,
            'raw_rot_loss': raw_rot_loss,
            'rot_score_loss': torch.mean(rot_score_loss),
            'rot_loss_x0': torch.mean(rot_x0_loss),
            'bb_atom_loss': bb_atom_loss,
            'raw_bb_atom_loss': raw_bb_atom_loss,
            'seqs_loss': seqs_loss,
            'raw_seqs_loss': raw_seqs_loss,
            'angle_loss': angle_loss,
            'raw_angle_loss': raw_angle_loss,
            'torsion_loss': torsion_loss,
            'raw_torsion_loss': raw_torsion_loss,
            'surface_loss': surface_loss,
            'raw_surface_loss': raw_surf_loss,
        } 
    
    @torch.no_grad()
    def sample(self, batch, num_steps = 100, sample_surf=True, sample_bb=True, sample_ang=True, sample_seq=True):

        num_batch, num_res = batch['aa'].shape
        batch_shape = batch['aa'].shape
        gen_mask, res_mask = batch['generate_mask'], batch['res_mask']
        K = self._interpolant_cfg.seqs.num_classes
        k = self._interpolant_cfg.seqs.simplex_value
        angle_mask_loss = torsions_mask.to(batch['aa'].device)

        # encode
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch)
        # trans_1_c,center = self.zero_center_part(trans_1,gen_mask,res_mask)  # todo test
        trans_1_c = trans_1
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)
        surf_1 = batch['surf_pos']

        #initial noise
        if sample_bb:
            rot_sample_mode = 1
            if rot_sample_mode == 0:
                rotmats_0 = uniform_so3(num_batch, num_res, device=batch['aa'].device)
            else:
                rotvec_0 = self._so3_diffuser.sample_ref(batch_shape = batch_shape)
                rotmats_0 = data_utils.axis_angle_to_matrix(rotvec_0)
                
            rotmats_0 = torch.where(batch['generate_mask'][...,None,None], rotmats_0, rotmats_1)
                
            trans_0 = torch.randn((num_batch,num_res,3), device=batch['aa'].device) 
            trans_0_c, center = self.zero_center_part(trans_0, gen_mask,res_mask)
            trans_0_c = torch.where(batch['generate_mask'][...,None],trans_0_c,trans_1_c)
        else:
            rotmats_0 = rotmats_1.detach().clone()
            trans_0_c = trans_1_c.detach().clone()
        if sample_ang:
            # angle noise
            angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype) # (B,L,5)
            angles_0 = torch.where(batch['generate_mask'][...,None],angles_0,angles_1)
        else:
            angles_0 = angles_1.detach().clone()
        if sample_seq:
            seqs_0_simplex = k * torch.randn((num_batch,num_res,K), device=batch['aa'].device)
            seqs_0_prob = F.softmax(seqs_0_simplex,dim=-1)
            seqs_0 = sample_from(seqs_0_prob)
            seqs_0 = torch.where(batch['generate_mask'],seqs_0,seqs_1)
            seqs_0_simplex = torch.where(batch['generate_mask'][...,None],seqs_0_simplex,seqs_1_simplex)
        else:
            seqs_0 = seqs_1.detach().clone()
            seqs_0_prob = seqs_1_prob.detach().clone()
            seqs_0_simplex = seqs_1_simplex.detach().clone()
        if sample_surf:
            surf_0 = batch['pts_rec_mask_symmetric']
            surf_0 = surf_0 - center  
        else:
            surf_0 = surf_1

        # Set-up time
        ts = torch.linspace(1.e-2, 1.0, num_steps, device=batch['aa'].device)
        # t_1 = ts[0]
        # ts = torch.linspace(1.0, 1e-2, num_steps, device=batch['aa'].device)  # reverse
        dt = ts[1] - ts[0]  # step size

        # prot_traj = [{'rotmats':rotmats_0,'trans':trans_0_c,'seqs':seqs_0,'seqs_simplex':seqs_0_simplex,'rotmats_1':rotmats_1,'trans_1':trans_1-center,'seqs_1':seqs_1}]
        clean_traj = []
        rotmats_t, trans_t_c, angles_t, seqs_t, seqs_t_simplex, surf_t = rotmats_0, trans_0_c, angles_0, seqs_0, seqs_0_simplex, surf_0

        # denoise loop
        # for t_2 in tqdm(ts[1:]):
        for t_1 in tqdm(ts):

            # t_batch = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
            t_batch = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
            surf_node_embed, surf_edge_embed = self.encoder_surf(batch, surf_t)  # no generate mask
            
            pred_trans_1, pred_rotmats_1, pred_angles_1, pred_seqs_1_prob, pred_surf_1, init_rigids, curr_rigids = self.ipa_sutf_bb_sfm(
                                                                                    t_batch,
                                                                                    rotmats_t,
                                                                                    trans_t_c,
                                                                                    angles_t,
                                                                                    seqs_t,
                                                                                    node_embed,
                                                                                    edge_embed,
                                                                                    batch['generate_mask'].long(),
                                                                                    batch['res_mask'].long(),
                                                                                    surf_node_embed,
                                                                                    surf_edge_embed,
                                                                                    surf_t,
                                                                                )            
            
            
            pred_rot_score = self._so3_diffuser.calc_rot_score(
                    init_rigids.get_rots(),
                    curr_rigids.get_rots(),
                    t_batch,
                )
            # rot_score = rot_score * node_mask[..., None]

            curr_rigids = self.ipa_sutf_bb_sfm.unscale_rigids(curr_rigids)  # what this for?
            
            pred_trans_score = self._r3_diffuser.calc_trans_score(
                init_rigids.get_trans(),
                curr_rigids.get_trans(),
                t_batch[:, None, None],
            )

            # todo
            # if self._diffuse_surface:
            #     denoised_surf = c_out.unsqueeze(-1) * pred_surf + c_skip.unsqueeze(-1) * surf_t
            
            pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None], pred_rotmats_1, rotmats_1)
            # trans, move center
            # pred_trans_1_c,center = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
            pred_trans_1_c = torch.where(batch['generate_mask'][...,None], pred_trans_1, trans_1_c) # move receptor also
            # angles
            pred_angles_1 = torch.where(batch['generate_mask'][...,None], pred_angles_1, angles_1)
            # seqs
            pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob, dim=-1))
            pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1, seqs_1)
            pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
            # seq-angle
            torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
            
            if not sample_bb:
                pred_trans_1_c = trans_1_c.detach().clone()
                # _,center = self.zero_center_part(trans_1,gen_mask,res_mask)
                pred_rotmats_1 = rotmats_1.detach().clone()
            if not sample_ang:
                pred_angles_1 = angles_1.detach().clone()
            if not sample_seq:
                pred_seqs_1 = seqs_1.detach().clone()
                pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
            if not sample_surf:
                pred_surf_1 = surf_1

            clean_traj.append({'rotmats':pred_rotmats_1.cpu(),'trans':pred_trans_1_c.cpu(),'angles':pred_angles_1.cpu(),'seqs':pred_seqs_1.cpu(),'seqs_simplex':pred_seqs_1_simplex.cpu(),
                               'surfs': pred_surf_1.cpu(), 'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'angles_1':angles_1.cpu(),'seqs_1':seqs_1.cpu(), 'surf_1':
                                surf_1.cpu()})
            
            trans_t_c = self._r3_diffuser.reverse(trans_t_c, pred_trans_score, t_batch, dt, trans_0_c)
            
            rotvec_t = data_utils.matrix_to_axis_angle(rotmats_t)
            rotvec_t = self._so3_diffuser.reverse(rot_t=rotvec_t,
                                                    score_t=pred_rot_score,
                                                    t=t_1,
                                                    dt=dt,
                                                    noise_scale = 0.1,
                                                    )
            
            trans_t_c = torch.where(batch['generate_mask'][..., None], trans_t_c, trans_1_c)
            rotmats_t = data_utils.axis_angle_to_matrix(rotvec_t)
            rotmats_t = torch.where(batch['generate_mask'][..., None, None], rotmats_t, rotmats_1)
            
            angles_t = self._tour_diffuser.reverse(angles_t, pred_angles_1, t_1, dt)

            angles_t = torch.where(batch['generate_mask'][..., None], angles_t, angles_1)
            # Apply sequence-dependent angle masking
            torsion_mask = angle_mask_loss[seqs_t.reshape(-1)].reshape(num_batch, num_res, -1)
            angles_t = torch.where(torsion_mask.bool(), angles_t, torch.zeros_like(angles_t))

            if torch.isnan(angles_t).any() or torch.isinf(angles_t).any():
                print('angles_t nan or inf')
          
            pred_seqs_simplex = self.seq_to_simplex(sample_from(F.softmax(pred_seqs_1_prob, dim=-1)))
            
            if torch.isnan(pred_seqs_simplex).any() or torch.isinf(pred_seqs_simplex).any():
                print('pred_seqs_simplex nan or inf')

            seqs_t_simplex = self._seq_diffuser.reverse(seqs_t_simplex, pred_seqs_simplex, t_1, dt)
            
            if torch.isnan(seqs_t_simplex).any() or torch.isinf(seqs_t_simplex).any():
                print('seqs_t_simplex nan or inf')
            
            seqs_t = sample_from(F.softmax(seqs_t_simplex, dim=-1))
            seqs_t = torch.where(batch['generate_mask'],seqs_t,seqs_1)
            # seqs_t_simplex = seqs_t_simplex + (pred_seqs_1_simplex - seqs_0_simplex) * d_t[...,None]
            # seqs_t = sample_from(F.softmax(seqs_t_simplex,dim=-1))
            # seqs_t = torch.where(batch['generate_mask'],seqs_t,seqs_1)
            
            # surf_t = self._surf_diffuser.reverse(surf_t, pred_surf_1, t, dt)
            # rotmats_t_1, trans_t_1_c, angles_t_1, seqs_t_1, seqs_t_1_simplex, surf_t_1 = rotmats_t_2, trans_t_2_c, angles_t_2, seqs_t_2, seqs_t_2_simplex, surf_t_2
            # t_1 = t_2

        # final step
        t = ts[-1]  # Final time step
        t_batch = torch.ones((num_batch, 1), device=batch['aa'].device) * t
       
        surf_node_embed, surf_edge_embed = self.encoder_surf(batch, surf_t)  # no generate mask
        pred_trans_1, pred_rotmats_1, pred_angles_1, pred_seqs_1_prob, pred_surf_1, init_rigids, curr_rigids = self.ipa_sutf_bb_sfm(
                                                                                            t_batch,
                                                                                            rotmats_t,
                                                                                            trans_t_c,
                                                                                            angles_t,
                                                                                            seqs_t,
                                                                                            node_embed,
                                                                                            edge_embed,
                                                                                            batch['generate_mask'].long(),
                                                                                            batch['res_mask'].long(),
                                                                                            surf_node_embed,
                                                                                            surf_edge_embed,
                                                                                            surf_t
                                                                                        )
        pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
        # pred_trans_1_c,center = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
        pred_trans_1_c = torch.where(batch['generate_mask'][...,None],pred_trans_1, trans_1_c) # move receptor also
        # angles
        pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
        # seqs
        pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob,dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
        pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
        # seq-angle
        torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
        if not sample_bb:
            pred_trans_1_c = trans_1_c.detach().clone()
            # _,center = self.zero_center_part(trans_1,gen_mask,res_mask)
            pred_rotmats_1 = rotmats_1.detach().clone()
        if not sample_ang:
            pred_angles_1 = angles_1.detach().clone()
        if not sample_seq:
            pred_seqs_1 = seqs_1.detach().clone()
            pred_seqs_1_simplex = seqs_1_simplex.detach().clone()
        if not sample_surf:
            pred_surf_1 = surf_1.detach().clone()
        clean_traj.append({'rotmats':pred_rotmats_1.cpu(),'trans':pred_trans_1_c.cpu(),'angles':pred_angles_1.cpu(),'seqs':pred_seqs_1.cpu(),'seqs_simplex':pred_seqs_1_simplex.cpu(), 'surf': pred_surf_1.cpu(),
                                'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'angles_1':angles_1.cpu(),'seqs_1':seqs_1.cpu(), 'surf_1': surf_1.cpu()
                           })
        
        return clean_traj
