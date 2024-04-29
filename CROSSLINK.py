from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBIO

import gzip
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

import random

from DSSPparser import parseDSSP

import concurrent.futures
import subprocess

import os
import shutil
import zipfile
import pickle


from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
from argparse import ArgumentParser

from scipy.spatial.distance import pdist, squareform

res_types = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', \
           'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', 'UNK' : '-'}


SA_maximum_map = {'PHE' : 210, 'ILE' : 175, 'LEU' : 170, 'VAL' : 155, 'PRO' : 145, 'ALA' : 115, \
                  'GLY' : 75, 'MET' : 185, 'CYS' : 135, 'TRP' : 255, 'TYR' : 230, 'THR' : 140, \
                  'SER' : 115, 'GLN' : 180, 'ASN' : 160, 'GLU' : 190, 'ASP' : 150, 'HIS' : 195, \
                  'LYS' : 200, 'ARG' : 225}

def readPDB(pdb_dir):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('pdb', pdb_dir)
    residue_dict = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residue_dict[chain_id] = {}
            for residue in chain:
                res_name = residue.resname
                res_num = residue.id[1]
                if res_name in res_types : residue_dict[chain_id][res_num] = res_name
        break
    chains = list(residue_dict.keys())
    return model,chains, residue_dict

def readMMCIF(mmcif_path):
    parser = MMCIFParser()
    structure = parser.get_structure("structure", mmcif_path)
    residue_dict = {}
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residue_dict[chain_id] = {}
            for residue in chain:
                res_name = residue.resname
                res_num = residue.id[1]
                if res_name in res_types: residue_dict[chain_id][res_num] = res_name
        break
    chains = list(residue_dict.keys())
    return model, chains, residue_dict



def calculate_calpha_distogram(chain_id, chain, residue_dict):
    residue_length = np.array(list(residue_dict[chain_id].keys())).max()
    calpha_coords = np.zeros([residue_length,3])
    
    for residue in chain:
        res_num = residue.get_id()[1] - 1
        res_name = residue.resname
        
        if res_num >= residue_length: continue

        if res_name in res_types:
            if "CA" in residue:
                calpha_coords[res_num] = residue["CA"].get_coord()
            else : 
                calpha_coords[res_num] = np.array([np.nan,np.nan,np.nan])
        else : 
            calpha_coords[res_num] = np.array([np.nan,np.nan,np.nan])
    
    distogram = squareform(pdist(calpha_coords, 'euclidean'))
    return residue_length, distogram


def calculate_lys_leu_map(chain_id, chain, residue_dict):
    residue_length = np.array(list(residue_dict[chain_id].keys())).max()
    lys_leu_map = np.zeros(residue_length)
    
    for res_num in residue_dict[chain_id]:
        res_name = residue_dict[chain_id][res_num]
        res_num = res_num - 1
        if res_name in ['LYS', 'LEU']: lys_leu_map[res_num] = True
        else : lys_leu_map[res_num] = False
    
    lys_leu_map = np.logical_or(lys_leu_map[None],lys_leu_map[:,None])
    return residue_length, lys_leu_map


def load_list_from_file(file_path):
    with open(file_path, 'rb') as f:
        data_list = pickle.load(f)
    return data_list


def calculate_sa_map(chain_id, chain, residue_dict, solvent_raw_datas):
    residue_length = np.array(list(residue_dict[chain_id].keys())).max()
    sa_accs = np.zeros(residue_length)
    sa_accs[:] = np.nan
    sa_map = np.zeros([residue_length, residue_length])
    
    for res_num in residue_dict[chain_id]:
        res_type = residue_dict[chain_id][res_num]
        res_num = res_num - 1
        if len(solvent_raw_datas) > res_num : 
            sa_accs[res_num] = float(solvent_raw_datas[res_num][3])/SA_maximum_map[res_type]
    
    for i in range(residue_length):
        for j in range(residue_length):
            sa_map[i,j] = np.min([sa_accs[i],sa_accs[j]])
    return residue_length, sa_map


def calculate_tryptic_map(chain_id, chain, residue_dict):
    residue_length = np.array(list(residue_dict[chain_id].keys())).max()
    tryptic_num = np.zeros([residue_length])
    tryptic_map = np.zeros([residue_length, residue_length])
    
    for res_num in residue_dict[chain_id].keys():
        res_name = residue_dict[chain_id][res_num]
        res_num = res_num - 1
        if res_name in ['LYS', 'ARG']:
            tryptic_num[res_num+1:] += 1
    for i in range(residue_length):
        for j in range(residue_length):
            tryptic_map[i,j] = np.abs(tryptic_num[i] - tryptic_num[j]) > 1
    return residue_length, tryptic_map



def plot_cross_link(residue_length, distogram, lys_leu_map, sa_map, tryptic_map):
    plt.figure(figsize = (20,5), facecolor = 'white')

    plt.subplot(1,5,1)
    plt.title('Ca Distogrma, < 10amstrong')
    plt.imshow(distogram < 10)

    plt.subplot(1,5,2)
    plt.title('LYS_LEU_MAP')
    plt.imshow(lys_leu_map, cmap = 'binary')

    plt.subplot(1,5,3)
    plt.title('Solvent_Accs > 0.5')
    plt.imshow(sa_map > 0.5, cmap = 'binary')

    plt.subplot(1,5,4)
    plt.title('Tryptic map')
    plt.imshow(tryptic_map, cmap = 'binary')

    distogram = distogram < 10
    disto_lysleu = np.where(lys_leu_map, distogram, False)
    disto_lysleu_sa = np.where(sa_map > 0.5, disto_lysleu, False)
    disto_lysleu_sa_tryptic = np.where(tryptic_map, disto_lysleu_sa, False)
    
    plt.subplot(1,5,5)
    plt.title('Distogram * LYS_LEU * (SA > 0.5) * tryptic_map')
    plt.imshow(disto_lysleu_sa_tryptic[:50,:50])
    
    print(f'residue length : {residue_length}')
    print(f'total number of cross link :  {int(disto_lysleu_sa_tryptic.sum()/2)} ')
    
    plt.show()

def generate_random_numbers(n, subsample_rate):
    num_samples = int(n * subsample_rate)  # n의 10분의 1 개수만큼 뽑기
    random_numbers = random.sample(range(n), num_samples)
    return random_numbers
    
def get_crosslink(chain_id, chain, residue_dict, dist_thres = 10, falselink = True, fdr = 0.05, subsample_rate = 0.1, out = False):
    residue_length = np.array(list(residue_dict[chain_id].keys())).max()
    
    calpha_coords = np.zeros([residue_length,3])
    calpha_coords[:] = np.nan
    
    lys_leu_map = np.zeros(residue_length)
    tryptic_num = np.zeros(residue_length)
    
    tryptic_map = np.zeros([residue_length, residue_length])
    cross_link = []
    
    for residue in chain:
        res_num = residue.get_id()[1] - 1
        res_name = residue.resname
        
        if res_num >= residue_length: continue
        if res_num < 1: continue
      
        # FOR... Circulation....
        if res_name in res_types.keys():
        # for distogram
            if "CA" in residue : 
                calpha_coords[res_num] = residue["CA"].get_coord()
                
        # for lys-leu map.
            if res_name in ['LYS', 'LEU']: 
                lys_leu_map[res_num] = True
                
        # for tryptic-map....
            if res_name in ['LYS', 'ARG']: 
                tryptic_num[res_num+1:] += 1
        
    lys_leu_map = np.logical_or(lys_leu_map[None],lys_leu_map[:,None])
    distogram = squareform(pdist(calpha_coords, 'euclidean'))
    
    false_link_candidates = []
    for i in range(residue_length):
        for j in range(i+1,residue_length):
            if (lys_leu_map[i,j]) and (np.abs(tryptic_num[i] - tryptic_num[j]) > 1):
                if (distogram[i,j] < dist_thres):
                    cross_link.append([i+1, j+1, fdr])
                else : false_link_candidates.append([i+1,j+1,fdr])
                    
    subsampled_linker = generate_random_numbers(len(cross_link)-1,subsample_rate)
    cross_link = [cross_link[i] for i in subsampled_linker]
    
    if falselink    == True: 
        n_false_link = np.max([1, int(len(cross_link) * 0.05)])
        subsampled_false_linker = random.sample(range(len(false_link_candidates)-1), n_false_link)
        false_cross_link = [false_link_candidates[i] for i in subsampled_false_linker]
        
    elif falselink  == False : false_cross_link = []
    
    if out :  print(f''' N_True_link : {len(cross_link)}, N_False_link : {len(false_cross_link)}''')
    cross_link = cross_link + false_cross_link
                
    return cross_link
                
                
    
    
    
    
    
    
    
    
    
