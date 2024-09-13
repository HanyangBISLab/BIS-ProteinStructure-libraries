from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO

import gzip
import os
import shutil
import zipfile
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import random

import concurrent.futures
import subprocess

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser

from scipy.spatial.distance import pdist, squareform

import residue_constants as residue_constants


res_types = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', \
           'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', 'UNK' : '-'}


def readMMCIF_label(mmcif_path):
    parser = MMCIFParser(auth_residues=False, QUIET=True)
    structure = parser.get_structure("structure", mmcif_path)
    residue_dict = {}    
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residue_dict[chain_id] = {}
            for residue in chain:
                res_name = residue.resname
                res_num = residue.id[1]
                if res_num in residue_dict[chain_id].keys(): continue
                if (res_name in res_types) and (residue.id[0]==' '): residue_dict[chain_id][res_num] = res_name
        break
    
    pdb_dict = MMCIF2Dict(mmcif_path)
    ATOM_CHECK = pdb_dict.get('_atom_site.group_PDB')
    auth_CHAIN_ID = pdb_dict.get('_atom_site.auth_asym_id')
    SEQUENCE_ID = pdb_dict.get('_atom_site.label_seq_id')
    
    chains = list(residue_dict.keys())
    
    residue_idx_edit = {}
    for c_id in chains:
        c = c_id
        residue_idx_edit[c] = {}
        
        if len(residue_dict[c].keys()) == 0 : continue
            
        idx_info_dict = {}
        for atom_check, chain_check, label_idx in zip(ATOM_CHECK, auth_CHAIN_ID, SEQUENCE_ID):
            if (atom_check == 'ATOM') and (chain_check == c):
                if label_idx not in idx_info_dict.keys():
                    idx_info_dict[int(label_idx)] = int(label_idx) - 1
        
        for r in sorted(list(residue_dict[c].keys())):
            idx = int(idx_info_dict[r])
            r_name = residue_dict[c][r]
            residue_idx_edit[c][r] = idx
    
    return model, chains, residue_dict, residue_idx_edit



def get_SS_ver5(chain_id, chain, res_length, residue_idx_edit, criteria):
    residue_length = int(res_length)
    
    calpha_coords = np.zeros([residue_length,3])
    
    sgamma_coords = np.zeros([residue_length,3])
    
    Cys_list = list()
    
    for residue in chain:
        if residue.id[0]!=' ': continue

        res_num_b = residue.id[1]
        
        res_name = residue.resname
        if res_name not in res_types: continue
            
        res_num = residue_idx_edit[chain_id][res_num_b]
        
        # For circulation
        if res_name in res_types.keys():
#             print(res_name)
            # for distogram
            if "CA" in residue:
                calpha_coords[res_num] = residue["CA"].get_coord()
            # for cysteine index
            if res_name == 'CYS':
                Cys_list.append(res_num)
            # for condition of S
            if criteria == 'SG':
                if "SG" in residue:
                    sgamma_coords[res_num] = residue["SG"].get_coord()
        
    real_distogram = squareform(pdist(calpha_coords, 'euclidean'))
    
    # for disulfide feature
    if criteria == 'CA':
        min_dist, max_dist = 3.0, 7.5
        distogram = real_distogram
    elif criteria == 'SG':
        min_dist, max_dist = 2.0, 3.0
        distogram = squareform(pdist(sgamma_coords, 'euclidean'))
        
    Cys_mask1 = np.zeros((distogram.shape))
    Cys_mask2 = np.zeros((distogram.shape))
    if len(Cys_list) != 0:
        Cys_mask1[np.array(Cys_list),:] = 1
        Cys_mask2[:,np.array(Cys_list)] = 1
    Cys_mask = Cys_mask1 * Cys_mask2
    
    filter_disto = distogram * Cys_mask
    filter_disto = np.where(filter_disto < min_dist, 0, filter_disto)
    filter_disto = np.where(filter_disto > max_dist, 0, filter_disto)
    
    # Reshape the matrix
    filter_disto = filter_disto[:,:,None]
    
    # for cysteine pair 
    pair_list = list()
    for index_i, index_j in zip(np.where(filter_disto != 0)[0], np.where(filter_disto != 0)[1]):
        pair = [int(index_i), int(index_j)]
    
        if pair not in pair_list and list(reversed(pair)) not in pair_list:
            pair_list.append(pair)
            
    pair_list = np.array(pair_list)
        
    
    return filter_disto, pair_list
