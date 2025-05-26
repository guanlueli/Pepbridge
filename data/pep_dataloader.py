"""pep-rec dataset"""
import os
import logging
import joblib
import pickle
import lmdb
from Bio import PDB
from Bio.PDB import PDBExceptions
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from pepbridge.modules.protein.parsers import parse_pdb
from pepbridge.modules.common.geometry import *
from pepbridge.modules.protein.constants import *
from pepbridge.utils.data import mask_select_data, find_longest_true_segment, PaddingCollate
from data.utils_pymol import preprocess_surface_single, get_symmetric_interface_masks, find_prominent_points, order_point_clouds_globally
from torch.utils.data import DataLoader

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler, dist

from pepbridge.utils.misc import load_config
from pepbridge.utils.train import recursive_to
from models_con.torsion import get_torsion_angle
import torch
from pepbridge.modules.protein.writers import save_pdb

DATA_DIR = './'

# testset
names = []
with open(f'{DATA_DIR}/pepbridge/names.txt','r') as f:
    for line in f:
        names.append(line.strip())
    
def preprocess_structure(task):

    try:
        # if task['id'] in names:
        #     raise ValueError(f'{task["id"]} not in names')
        pdb_path = task['pdb_path']
        # pep
        # process peptide and find center of mass
        pep = parse_pdb(os.path.join(pdb_path,'peptide.pdb'))[0]
        center = torch.sum(pep['pos_heavyatom'][pep['mask_heavyatom'][:, BBHeavyAtom.CA], BBHeavyAtom.CA], dim=0) / (torch.sum(pep['mask_heavyatom'][:, BBHeavyAtom.CA]) + 1e-8)
        pep['pos_heavyatom'] = pep['pos_heavyatom'] - center[None, None, :]
        pep['torsion_angle'],pep['torsion_angle_mask'] = get_torsion_angle(pep['pos_heavyatom'],pep['aa']) # calc angles after translation
        if len(pep['aa'])<3 or len(pep['aa'])>25:
            raise ValueError('peptide length not in [3,25]')
        # rec
        rec = parse_pdb(os.path.join(pdb_path,'pocket.pdb'))[0]
        rec['pos_heavyatom'] = rec['pos_heavyatom'] - center[None, None, :]
        rec['torsion_angle'],rec['torsion_angle_mask'] = get_torsion_angle(rec['pos_heavyatom'],rec['aa']) 
        rec['chain_nb'] += 1

        # process surface
        pts_pep, norms_pep, hp_pep, hbond_pep = preprocess_surface_single(structure_dir=pdb_path, pdb_fname=task['id'], type='peptide.pdb')
        pts_rec, norms_rec, hp_rec, hbond_rec = preprocess_surface_single(structure_dir=pdb_path, pdb_fname=task['id'], type='pocket.pdb')
        surface_intf_cutoff = 4
        distances = torch.cdist(pts_pep, pts_rec)
        intf_pep_mask = (distances < surface_intf_cutoff).sum(dim=1) > 0
        intf_rec_mask = (distances < surface_intf_cutoff).sum(dim=0) > 0

        pts_pep_sub = pts_pep[intf_pep_mask]

        intf_pep_mask_symmetric, intf_rec_mask_symmetric = get_symmetric_interface_masks(pts_pep, pts_rec, surface_intf_cutoff)
        # pts_pep_mask_symmetric = pts_pep[intf_pep_mask_symmetric]
        pts_rec_mask_symmetric = pts_rec[intf_rec_mask_symmetric]
        hp_rec_sub_symmetric = hp_rec[intf_rec_mask_symmetric]
        hbond_rec_sub_symmetric = hbond_rec[intf_rec_mask_symmetric]

        reordered_pts_rec_mask_symmetric_globally, col_ind = order_point_clouds_globally(fixed_cloud=pts_pep_sub,
                                                                                moving_cloud=pts_rec_mask_symmetric)

        reordered_hp_rec = hp_rec_sub_symmetric[col_ind]
        reordered_hbond_rec = hbond_rec_sub_symmetric[col_ind]

        # meta data
        data = {}
        data['id'] = task['id']
        data['generate_mask'] = torch.cat([torch.zeros_like(rec['aa']), torch.ones_like(pep['aa'])], dim=0).bool()
        # data['surf_mask'] = torch.cat([torch.zeros_like(rec['aa']), torch.ones_like(pep['aa'])], dim=0).bool()
        for k in rec.keys():
            if isinstance(rec[k], torch.Tensor):
                data[k] = torch.cat([rec[k], pep[k]], dim=0)
            elif isinstance(rec[k], list):
                data[k] = rec[k] + pep[k]
            else:
                raise ValueError(f'Unknown type of {rec[k]}')

        data['surf_pos'] = pts_pep_sub.squeeze(0)
        data['surf_hp'] = reordered_hp_rec
        data['surf_hbond'] = reordered_hbond_rec
        data['pts_rec_mask_symmetric'] = reordered_pts_rec_mask_symmetric_globally

        data['surf_pos'] = data['surf_pos'] - center[None, :]
        data['pts_rec_mask_symmetric'] = data['pts_rec_mask_symmetric'] - center[None, :]

        return data
        
    except (
        PDBExceptions.PDBConstructionException, 
        KeyError,
        ValueError,
        TypeError
    ) as e:
        logging.warning('[{}] {}: {}'.format(
            task['id'], 
            e.__class__.__name__, 
            str(e)
        ))
        return None


class PepDataset(Dataset):

    MAP_SIZE = 32*(1024*1024*1024)  # 32GB

    def __init__(self, structure_dir = "./Data/PepMerge_new/", dataset_dir = "./Data/",
                                            name = 'pep', transform=None, reset=False):

        super().__init__()
        self.structure_dir = structure_dir
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.name = name

        self.db_conn = None
        self.db_ids = None
        self._load_structures(reset)

    @property
    def _cache_db_path(self):
        return os.path.join(self.dataset_dir, f'{self.name}_structure_cache.lmdb')

    def _connect_db(self):
        self._close_db()
        self.db_conn = lmdb.open(
            self._cache_db_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db_conn.begin() as txn:
            keys = [k.decode() for k in txn.cursor().iternext(values=False)]
            self.db_ids = keys

    def _close_db(self):
        if self.db_conn is not None:
            self.db_conn.close()
        self.db_conn = None
        self.db_ids = None

    def _load_structures(self, reset):
        all_pdbs = os.listdir(self.structure_dir)

        if reset:
            if os.path.exists(self._cache_db_path):
                os.remove(self._cache_db_path)
                lock_file = self._cache_db_path + "-lock"
                if os.path.exists(lock_file):
                    os.remove(lock_file)
            self._close_db()
            todo_pdbs = all_pdbs
        else:
            if not os.path.exists(self._cache_db_path):
                todo_pdbs = all_pdbs
            else:
                todo_pdbs = []
                # self._connect_db()
                # processed_pdbs = self.db_ids
                # self._close_db()
                # todo_pdbs = list(set(all_pdbs) - set(processed_pdbs))

        if len(todo_pdbs) > 0:
            self._preprocess_structures(todo_pdbs)
    
    def _preprocess_structures(self, pdb_list):
        tasks = []
        for pdb_fname in pdb_list:
            pdb_path = os.path.join(self.structure_dir, pdb_fname)
            tasks.append({
                'id': pdb_fname,
                'pdb_path': pdb_path,
            })

        data = preprocess_structure(tasks[0])

        data_list = joblib.Parallel(
            n_jobs = max(joblib.cpu_count() // 2, 1),
        )(
            joblib.delayed(preprocess_structure)(task)
            for task in tqdm(tasks, dynamic_ncols=True, desc='Preprocess')
        )

        db_conn = lmdb.open(
            self._cache_db_path,
            map_size = self.MAP_SIZE,
            create=True,
            subdir=False,
            readonly=False,
        )
        ids = []
        with db_conn.begin(write=True, buffers=True) as txn:
            for data in tqdm(data_list, dynamic_ncols=True, desc='Write to LMDB'):
                if data is None:
                    continue
                ids.append(data['id'])
                txn.put(data['id'].encode('utf-8'), pickle.dumps(data))

    def __len__(self):
        self._connect_db() # make sure db_ids is not None
        return len(self.db_ids)

    def __getitem__(self, index):
        self._connect_db()
        id = self.db_ids[index]
        with self.db_conn.begin() as txn:
            data = pickle.loads(txn.get(id.encode()))
        if self.transform is not None:
            data = self.transform(data)
        return data


if __name__ == '__main__':
    device = 'cuda:1'
    config,cfg_name = load_config("./configs/data_process.yaml")
    dataset = PepDataset(structure_dir = "./Data/PepMerge_new/", dataset_dir = "./Data/Fixed_Data",
                                            name = 'pep_pocket_test', transform=None, reset=True)
    print(len(dataset))
    print(dataset[0])

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=PaddingCollate(eight=False))

    batch = next(iter(dataloader))
    print(batch['torsion_angle'].shape)
    print(batch['torsion_angle_mask'].shape)

