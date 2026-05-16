import torch
from data.constants import AA

# Kyte Doolittle scale
kd_scale = {"ILE": 4.5,
            "VAL": 4.2,
            "LEU": 3.8,
            "PHE": 2.8,
            "CYS": 2.5,
            "MET": 1.9,
            "ALA": 1.8,
            "GLY": -0.4,
            "THR": -0.7,
            "SER": -0.8,
            "TRP": -0.9,
            "TYR": -1.3,
            "PRO": -1.6,
            "HIS": -3.2,
            "GLU": -3.5,
            "GLN": -3.5,
            "ASP": -3.5,
            "ASN": -3.5,
            "LYS": -3.9,
            "ARG": -4.5,
            "UNK": 0.0}   # assign 0 to unknown residues


kd_scale_NAME_NUMBER = {"ILE": 0,
                        "VAL": 1,
                        "LEU": 2,
                        "PHE": 3,
                        "CYS": 4,
                        "MET": 5,
                        "ALA": 6,
                        "GLY": 7,
                        "THR": 8,
                        "SER": 9,
                        "TRP": 10,
                        "TYR": 11,
                        "PRO": 12,
                        "HIS": 13,
                        "GLU": 14,
                        "GLN": 15,
                        "ASP": 16,
                        "ASN": 17,
                        "LYS": 18,
                        "ARG": 19,
                        "UNK": 20}   # assign 0 to unknown residues

# For each vertex in names, compute
def computeHydrophobicity(res_types, feature = 'number'):
    if feature == 'number':
        hp = torch.zeros(len(res_types), dtype=torch.long)
    elif feature == 'scale':
        hp = torch.zeros(len(res_types))

    for ix, res_int in enumerate(res_types):
        aa = AA(res_int).name
        if feature == 'number':
            hp[ix] = kd_scale_NAME_NUMBER[aa]
        elif feature == 'scale':
            hp[ix] = kd_scale[aa]
    return hp
