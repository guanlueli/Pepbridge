import pyrosetta
from pyrosetta import init, pose_from_pdb, get_fa_scorefxn
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import RestrictToRepacking
from pyrosetta.rosetta.protocols.minimization_packing import PackRotamersMover

import os
import pandas as pd
import subprocess
import numpy as np
import shutil
from tqdm import tqdm
import pickle

from joblib import delayed, Parallel
from utils import *

import glob

from Bio.PDB import PDBParser, Superimposer, is_aa, Select, NeighborSearch
import tmtools
import os
import numpy as np
import mdtraj as md
from Bio.SeqUtils import seq1

import warnings
from Bio import BiopythonWarning, SeqIO

import difflib
import torch

# ignore PDBConstructionWarning
warnings.filterwarnings('ignore', category=BiopythonWarning)

# energy

def get_chain_dic(input_pdb):
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb)
    chain_dic = {}
    for model in structure:
        for chain in model:
            chain_dic[chain.id] = len([res for res in chain if is_aa(res) and res.has_id('CA')])

    return chain_dic

def get_rosetta_score_base(pdb_path,chain_id='A'):
    try:
        init()
        pose = pyrosetta.pose_from_pdb(pdb_path)
        chains = list(get_chain_dic(pdb_path).keys())
        chains.remove(chain_id)
        interface = f'{chain_id}_{"".join(chains)}'
        fast_relax = FastRelax() # cant be pickled
        scorefxn = get_fa_scorefxn()
        fast_relax.set_scorefxn(scorefxn)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        stabs,binds = [],[]
        for i in range(5):
            fast_relax.apply(pose)
            stab = scorefxn(pose)
            mover.apply(pose)
            bind = pose.scores['dG_separated']
            stabs.append(stab)
            binds.append(bind)
        return {'name':pdb_path,'stab':np.array(stabs).mean(),'bind':np.array(binds).mean()}
    except:
        return {'name':pdb_path,'stab':999.0,'bind':999.0}


def get_rosetta_score(pdb_path,chain='A'):
    try:
        init()
        pose = pyrosetta.pose_from_pdb(pdb_path)
        # chains = list(get_chain_dic(os.path.join(input_dir,name,'pocket_merge_renum.pdb')).keys())
        # chains.remove(chain)
        # interface = f'{chain}_{"".join(chains)}'
        interface='A_B'
        fast_relax = FastRelax() # cant be pickled
        scorefxn = get_fa_scorefxn()
        fast_relax.set_scorefxn(scorefxn)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        fast_relax.apply(pose)
        energy = scorefxn(pose)
        mover.apply(pose)
        dg = pose.scores['dG_separated']
        return [pdb_path,energy,dg]
    except:
        return [pdb_path,999.0,999.0]

def pack_sc(name='1a1m_C',num_samples=10):
    try:
        if os.path.exists(os.path.join(output_dir,name,'rosetta')):
            shutil.rmtree(os.path.join(output_dir,name,'rosetta'))
        os.makedirs(os.path.join(output_dir,name,'rosetta'),exist_ok=True)
        init()
        tf = TaskFactory()
        tf.push_back(RestrictToRepacking())  # Only repack, don't change amino acid types
        packer = PackRotamersMover()
        packer.task_factory(tf)
        for i in range(num_samples):
            pose = pose_from_pdb(os.path.join(input_dir,name,f'pocket_merge_renum_bb.pdb'))
            packer.apply(pose)
            pose.dump_pdb(os.path.join(output_dir,name,'rosetta',f'packed_{i}.pdb'))
    except:
        return None

# geometry

def get_chain_from_pdb(pdb_path, chain_id='A'):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb_path)[0]
    for chain in structure:
        if chain.id == chain_id:
            # print(len(chain))
            return chain
    return None

def diff_ratio(str1, str2):
    # Create a SequenceMatcher object
    seq_matcher = difflib.SequenceMatcher(None, str1, str2)

    # Calculate the difference ratio
    return seq_matcher.ratio()

#RMSD

def align_chains(chain1, chain2):
    reslist1 = []
    reslist2 = []
    for residue1,residue2 in zip(chain1.get_residues(),chain2.get_residues()):
        if is_aa(residue1) and residue1.has_id('CA'): # at least have CA
            reslist1.append(residue1)
            reslist2.append(residue2)
    return reslist1,reslist2

def get_rmsd(chain1, chain2):
    # chain1 = get_chain_from_pdb(pdb1, chain_id1)
    # chain2 = get_chain_from_pdb(pdb2, chain_id2)
    if chain1 is None or chain2 is None:
        return None
    super_imposer = Superimposer()
    pos1 = np.array([atom.get_coord() for atom in chain1.get_atoms() if atom.name == 'CA'])
    pos2 = np.array([atom.get_coord() for atom in chain2.get_atoms() if atom.name == 'CA'])
    rmsd1 = np.sqrt(np.sum((pos1 - pos2)**2) / len(pos1))
    super_imposer.set_atoms([atom for atom in chain1.get_atoms() if atom.name == 'CA'],
                            [atom for atom in chain2.get_atoms() if atom.name == 'CA'])
    rmsd2 = super_imposer.rms
    return rmsd1,rmsd2

def get_tm(chain1,chain2):
    # chain1 = get_chain_from_pdb(pdb1, chain_id1)
    # chain2 = get_chain_from_pdb(pdb2, chain_id2)
    pos1 = np.array([atom.get_coord() for atom in chain1.get_atoms() if atom.name == 'CA'])
    pos2 = np.array([atom.get_coord() for atom in chain2.get_atoms() if atom.name == 'CA'])
    tm_results = tmtools.tm_align(pos1, pos2, 'A'*len(pos1), 'A'*len(pos2))
    # print(dir(tm_results))
    return tm_results.tm_norm_chain2

def get_traj_chain(pdb, chain):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    chain2id = {chain.id:i for i,chain in enumerate(structure)}
    traj = md.load(pdb)
    chain_indices = traj.topology.select(f"chainid {chain2id[chain]}")
    traj = traj.atom_slice(chain_indices)
    return traj

def get_second_stru(pdb,chain):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    chain2id = {chain.id:i for i,chain in enumerate(structure)}
    traj = md.load(pdb)
    chain_indices = traj.topology.select(f"chainid {chain2id[chain]}")
    traj = traj.atom_slice(chain_indices)
    return md.compute_dssp(traj,simplified=True)

def get_ss(traj1,traj2):
    # traj1,traj2 = get_traj_chain(pdb1,chain_id1),get_traj_chain(pdb2,chain_id2)
    ss1,ss2 = md.compute_dssp(traj1,simplified=True),md.compute_dssp(traj2,simplified=True)
    return (ss1==ss2).mean()

def get_bind_site(pdb,chain_id):
    parser = PDBParser()
    structure = parser.get_structure('X', pdb)[0]
    peps = [atom for res in structure[chain_id] for atom in res if atom.get_name() == 'CA']
    recs = [atom for chain in structure if chain.get_id()!=chain_id for res in chain for atom in res if atom.get_name() == 'CA']
    # print(recs)
    search = NeighborSearch(recs)
    near_res = []
    for atom in peps:
        near_res += search.search(atom.get_coord(), 10.0, level='R')
    near_res = set([res.get_id()[1] for res in near_res])
    return near_res

def get_bind_ratio(pdb1, pdb2, chain_id1, chain_id2):
    near_res1,near_res2 = get_bind_site(pdb1,chain_id1),get_bind_site(pdb2,chain_id2)
    # print(near_res1)
    # print(near_res2)
    return len(near_res1.intersection(near_res2))/(len(near_res2)+1e-10) # last one is gt

def get_dihedral(pdb,chain):
    traj = get_traj_chain(pdb,chain)
    #TODO: dihedral

def get_seq(pdb,chain_id):
    parser = PDBParser()
    chain = parser.get_structure('X', pdb)[0][chain_id]
    return seq1("".join([residue.get_resname() for residue in chain])) # ignore is_aa,used for extract seq from genrated pdb

def get_mpnn_seqs(path):
    fastas = []
    for record in SeqIO.parse(path, "fasta"):
        tmp = [c for c in str(record.seq)]
        fastas.append(tmp)
    return fastas


def analyze_protein_structure(pdb_file, reference_pdb=None, chain_id='A', ref_chain_id='A'):

    results = {
        'pdb_file': pdb_file,
        'chain_id': chain_id
    }

    # Get sequence
    try:
        sequence = get_seq(pdb_file, chain_id)
        results['sequence'] = sequence
        results['sequence_length'] = len(sequence)
    except Exception as e:
        print(f"Error getting sequence from {pdb_file}: {e}")
        results['sequence'] = None

    # Get chain
    try:
        chain = get_chain_from_pdb(pdb_file, chain_id)
        if chain is None:
            print(f"Chain {chain_id} not found in {pdb_file}")
            return results
    except Exception as e:
        print(f"Error getting chain from {pdb_file}: {e}")
        return results

    # Calculate comparative metrics if reference is provided
    if reference_pdb and os.path.isfile(reference_pdb):
        try:
            ref_chain = get_chain_from_pdb(reference_pdb, ref_chain_id)
            if ref_chain is not None:
                # Get RMSD
                rmsd1, rmsd2 = get_rmsd(chain, ref_chain)
                results['rmsd_before_superposition'] = rmsd1
                results['rmsd_after_superposition'] = rmsd2

                # Get TM-score
                tm_score = get_tm(chain, ref_chain)
                results['tm_score'] = tm_score

                # Get sequence similarity
                ref_sequence = get_seq(reference_pdb, ref_chain_id)
                results['reference_sequence'] = ref_sequence
                results['sequence_similarity'] = diff_ratio(sequence, ref_sequence)

                # Get secondary structure similarity
                try:
                    traj = get_traj_chain(pdb_file, chain_id)
                    ref_traj = get_traj_chain(reference_pdb, ref_chain_id)
                    ss_similarity = get_ss(traj, ref_traj)
                    results['ss_similarity'] = ss_similarity
                except Exception as e:
                    print(f"Error calculating secondary structure: {e}")

                # Get binding site overlap
                binding_ratio = get_bind_ratio(pdb_file, reference_pdb, chain_id, ref_chain_id)
                results['binding_site_overlap'] = binding_ratio
        except Exception as e:
            print(f"Error comparing with reference: {e}")

    return results

if __name__ == '__main__':

    input_dir = '/Results/pdb/'
    output_dir = '/Analysis/'

    pdb_files = glob.glob(os.path.join(input_dir, "*.pdb"))

    test_energy = False
    test_geometry = True
    if test_energy:
        # Process each file
        results_energy = []
        for pdb_file in pdb_files:
            result = get_rosetta_score(pdb_file)
            results_energy.append(result)
            print(f"Processed {pdb_file}: Energy={result[1]}, Binding={result[2]}")

        # Sort results by binding energy
        sorted_results = sorted(results_energy, key=lambda x: x[2])
        print("\nTop 5 structures by binding energy:")
        for i, result in enumerate(sorted_results[:5]):
            print(f"{i + 1}. {os.path.basename(result[0])}: Energy={result[1]}, Binding={result[2]}")
    if test_geometry:
        results_geometry = []
        for pdb_file in pdb_files:
            pdb_name = pdb_file[:4]
            reference_pdb = f'{data_base}/{pdb_name}'
            result = analyze_protein_structure(pdb_file, reference_pdb)
            results_geometry.append(result)

