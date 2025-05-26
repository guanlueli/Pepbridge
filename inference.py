import sys
sys.path.append('../')
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import functools
import os
import argparse
import pandas as pd
from copy import deepcopy

from Pepbridge.data.pep_dataloader import PepDataset
from pepbridge.utils.misc import load_config
from pepbridge.utils.train import recursive_to
from pepbridge.modules.common.geometry import reconstruct_backbone, reconstruct_backbone_partially, align, batch_align
from pepbridge.modules.protein.writers import save_pdb
from pepbridge.utils.data import PaddingCollate
from models_con.utils import process_dic
from pepbridge.utils.misc import seed_all
from models_con.torsion import full_atom_reconstruction, get_heavyatom_mask
from models_con.diffusion_model import DiffusionModel
collate_fn = PaddingCollate(eight=False)
import argparse
from math import ceil


BASE_DIR = '/home/gli/project/DVAE/000_PepBridge'
DATA_DIR = '/mnt/storage1/gli/1Data/data_surface/PepBridge'


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str, default=f'{BASE_DIR}/configs/learn_surf_angle.yaml')
    args.add_argument('--device', type=str, default='cuda:0')
    args.add_argument('--ckpt', type=str, default=f'{DATA_DIR}/logs/learn_surf_angle[1]_2025_05_19__23_43_14_n1_time1000_batch6_block6_withsurfnew03loss03_newr3diffscale0loss1_newseqdiff_newrotloss/checkpoints/1100000.pt')
    args.add_argument('--tag', type=str, default='t_0_11_withode_nocenter')
    args.add_argument('--output', type=str, default=f'{DATA_DIR}/learn_surf_angle')
    args.add_argument('--num_steps', type=int, default=1000)
    args.add_argument('--num_samples', type=int, default=40)
    args.add_argument('--mini_batch_size', type=int, default=10)
    args.add_argument('--sample_surf', type=bool, default=False)
    args.add_argument('--sample_bb', type=bool, default=True)
    args.add_argument('--sample_ang', type=bool, default=True)
    args.add_argument('--sample_seq', type=bool, default=True)
    parser = args.parse_args()

    config, cfg_name = load_config(parser.config)
    device = parser.device
    dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=4, pin_memory=True)
    ckpt = torch.load(parser.ckpt, map_location=device)

    seed_all(123123)
    model = DiffusionModel(config.model, device).to(device)
    model.load_state_dict(process_dic(ckpt['model']))
    model.eval()

    save_name = os.path.dirname(parser.ckpt).split('/')[-2]
    save_name = f'{save_name}_{parser.tag}'
    save_dir = f'{parser.output}/{save_name}'
    os.makedirs(save_dir, exist_ok=True)
    print(f'Output directory: {parser.ckpt}')
    print(f'Steps: {parser.num_steps}')

    dic = {'id':[],'len':[],'tran':[],'aar':[],'rot':[],'trans_loss':[],'rot_loss':[]}

    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        num_samples = parser.num_samples
        data_list = [deepcopy(item) for _ in range(num_samples)]
        num_batches = ceil(num_samples / parser.mini_batch_size)
        
        total_ca_dist, total_rot_dist, total_aar = 0, 0, 0
        total_trans_loss, total_rot_loss = 0, 0

        total_len = 0  
        sample_count = 0
        
        results_by_id = {}
        for j in range(num_batches):
            start = j * parser.mini_batch_size
            end = min((j + 1) * parser.mini_batch_size, num_samples)
            mini_batch_data = data_list[start:end]
        
            batch = recursive_to(collate_fn(mini_batch_data), device)
        
            with torch.no_grad():
                loss_dic = model(batch)
                traj_1 = model.sample(batch,
                                    num_steps=parser.num_steps,
                                    sample_surf=parser.sample_surf,
                                    sample_bb=parser.sample_bb,
                                    sample_ang=parser.sample_ang,
                                    sample_seq=parser.sample_seq)

                mask = batch['generate_mask']
                mask_sum = mask.sum().cpu().item()
                total_len += mask_sum
                sample_count += 1

                ca_dist = torch.sqrt(torch.sum((traj_1[-1]['trans'] - traj_1[-1]['trans_1'])**2 * mask[..., None].cpu().long()) / (mask_sum + 1e-8))
                rot_dist = torch.sqrt(torch.sum((traj_1[-1]['rotmats'] - traj_1[-1]['rotmats_1'])**2 * mask[..., None, None].cpu().long()) / (mask_sum + 1e-8))
                aar = torch.sum((traj_1[-1]['seqs'] == traj_1[-1]['seqs_1']) * mask.long().cpu()) / (mask_sum + 1e-8)

                total_ca_dist += ca_dist.item()
                total_rot_dist += rot_dist.item()
                total_aar += aar.item()
                total_trans_loss += loss_dic['trans_loss'].item()
                total_rot_loss += loss_dic['rot_loss'].item()

                traj_1[-1]['batch'] = batch
        
                item_id = item['id']
                
                if item_id not in results_by_id:
                    results_by_id[item_id] = []
                results_by_id[item_id].append(traj_1[-1])

        item_id = item['id'] 
        torch.save(results_by_id[item_id], f'{save_dir}/{item_id}.pt')
        
        avg_ca_dist = total_ca_dist / num_batches
        avg_rot_dist = total_rot_dist / num_batches
        avg_aar = total_aar / num_batches
        avg_trans_loss = total_trans_loss / num_batches
        avg_rot_loss = total_rot_loss / num_batches

        print(f'[Summary] tran: {avg_ca_dist:.4f}, rot: {avg_rot_dist:.4f}, aar: {avg_aar:.4f}, len: {total_len}')

        dic['tran'].append(avg_ca_dist)
        dic['rot'].append(avg_rot_dist)
        dic['aar'].append(avg_aar)
        dic['trans_loss'].append(avg_trans_loss)
        dic['rot_loss'].append(avg_rot_loss)
        dic['id'].append(item['id'])  
        dic['len'].append(total_len)

        del traj_1, batch, loss_dic
        # torch.cuda.empty_cache()
        # gc.collect()

    # Save results
    dic = pd.DataFrame(dic)
    dic.to_csv(f'{parser.output}/{save_name}.csv', index=None)