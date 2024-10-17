import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
from scipy.spatial.distance import pdist, squareform
from multiprocessing import Pool
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

atom_types = {"N":0,"CA":1,"C":2,"CB":3,"O":4,"CG":5,"CG1":6,"CG2":7,"OG":8,"OG1":9,"SG":10,"CD":11,"CD1":12,"CD2":13,"ND1":14,"ND2":15,"OD1":16,"OD2":17,"SD":18, "CE":19,"CE1":20,"CE2":21,"CE3":22,"NE":23,"NE1":24,"NE2":25,"OE1":26,"OE2":27,"CH2":28,"NH1":29,"NH2":30,"OH":31,"CZ":32,"CZ2":33,"CZ3":34,"NZ":35,"OXT":36}

res_types = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', \
           'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', 'UNK' : '-'}

def restype_refer_atoms(restype):
    atoms = []
    if restype   == 'ALA' : atoms = [0,1,2,3,4]
    elif restype == 'ARG' : atoms = [0,1,2,3,4,5,11,23,29,30,32]
    elif restype == 'ASN' : atoms = [0,1,2,3,4,5,15,16]
    elif restype == 'ASP' : atoms = [0,1,2,3,4,5,16,17]
    elif restype == 'CYS' : atoms = [0,1,2,3,4,10]
    elif restype == 'GLN' : atoms = [0,1,2,3,4,5,11,25,26]
    elif restype == 'GLU' : atoms = [0,1,2,3,4,5,11,26,27]
    elif restype == 'GLY' : atoms = [0,1,2,3]
    elif restype == 'HIS' : atoms = [0,1,2,3,4,5,13,14,20,25]
    elif restype == 'ILE' : atoms = [0,1,2,3,4,6,7,12]
    elif restype == 'LEU' : atoms = [0,1,2,3,4,5,12,13]
    elif restype == 'LYS' : atoms = [0,1,2,3,4,5,11,19,35]
    elif restype == 'MET' : atoms = [0,1,2,3,4,5,18,19]
    elif restype == 'PHE' : atoms = [0,1,2,3,4,5,12,13,20,21,32]
    elif restype == 'PRO' : atoms = [0,1,2,3,4,5,11]
    elif restype == 'SER' : atoms = [0,1,2,3,4,8]
    elif restype == 'THR' : atoms = [0,1,2,3,4,7,9]
    elif restype == 'TRP' : atoms = [0,1,2,3,4,5,12,13,21,22,24,28,33,34]
    elif restype == 'TYR' : atoms = [0,1,2,3,4,5,12,13,20,21,31,32]
    elif restype == 'VAL' : atoms = [0,1,2,3,4,6,7]
    
    return atoms


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

def convert_string(s):
    # 알파벳 매핑: A는 0, B는 1, ..., Z는 23로 매핑
    mapping = {chr(i + 65): i for i in range(24)}
    
    # 입력 문자열을 매핑된 숫자로 변환
    result = ''.join(str(mapping[char]) for char in s)
    
    return result
    

def get_contact(pdb_path, residue_dict = None,coord_masks = None, interface = 'all', show = True):
    
    def get_coordinates(residue_length, model, residue_dict, residue_intervals):
        coord = np.zeros([residue_length,37, 3])
        coord_mask = np.zeros([residue_length, 37, 1])
    
        def get_atom_coords(atom):
            vec = atom.get_vector()
            return np.array([vec[0], vec[1], vec[2]])
    
        residue_interval = 0
        for i,chain in enumerate(model):
            if i >= len(residue_intervals) : break
            for residue in chain:
                res_num = residue.get_id()[1]-1
                if res_num < 0 : continue
                res_num += residue_interval
                if res_num > residue_interval + residue_intervals[i] : continue
                res_name = residue.resname
                for atom in residue:
                    atom_id = atom.get_id()
                    refer_atoms = restype_refer_atoms(res_name)
                    if res_name == 'MET' and atom_id == 'SE': atom_id = 'SD'
                    if atom_id in atom_types.keys():
                        coord[res_num,atom_types[atom_id]] = get_atom_coords(atom)
                        coord_mask[res_num,atom_types[atom_id]] = True
            residue_interval += residue_intervals[i]

        return np.array(coord),np.array(coord_mask)
    
    model,chains, temp_residue_dict = readPDB(pdb_path)
    
    if residue_dict == None: residue_dict = temp_residue_dict
    
    residue_length = 0
    residue_intervals = []    

    for chain in residue_dict.keys():
        residue_intervals.append(np.array(list(residue_dict[chain].keys())).max())
        residue_length += np.array(list(residue_dict[chain].keys())).max()
        
    contact_map = np.zeros([residue_length,residue_length])
    start_point = 0
    start_points = [0]
    
    for residue_interval in residue_intervals:
        contact_map[start_point:start_point + residue_interval,start_point:start_point + residue_interval] = np.nan
        start_point += residue_interval
        start_points.append(start_point)
    
    if interface != 'all':
        contact_map[:] = np.nan
        i_chains, j_chains = interface.split(':')
        i_chains, j_chains = i_chains.upper(), j_chains.upper()
        i_chain_nums, j_chain_nums = convert_string(i_chains), convert_string(j_chains)

        for i in range(len(i_chain_nums)):
            i_chain_num = int(i_chain_nums[i])
            j_chain_num = int(j_chain_nums[i])
            contact_map[start_points[i_chain_num]:start_points[i_chain_num] + residue_intervals[i_chain_num],start_points[j_chain_num]:start_points[j_chain_num] + residue_intervals[j_chain_num]] = 0
            contact_map[start_points[j_chain_num]:start_points[j_chain_num] + residue_intervals[j_chain_num],start_points[i_chain_num]:start_points[i_chain_num] + residue_intervals[i_chain_num]] = 0


    coords, temp_coord_masks = get_coordinates(residue_length, model, residue_dict, residue_intervals)
    if coord_masks is None: coord_masks = temp_coord_masks
    
    coords = np.where(coord_masks,coords,np.nan)
    distogram = squareform(pdist(coords[:,1,:], 'euclidean'))
    
    ca_mask = coord_masks[:,1,0][None,...] * coord_masks[:,1,0][...,None]
    ca_contact = np.where(ca_mask, distogram < 15, np.nan)
    
    contact_map += ca_contact

    residue_indices_1, residue_indices_2 = np.where(contact_map == 1)
    for residue_indice in tqdm(range(len(residue_indices_1))):
        residue_i, residue_j = residue_indices_1[residue_indice], residue_indices_2[residue_indice]
        
        atom_indices_1 = list(np.where(coord_masks[residue_i,:,0] == 1)[0])
        atom_indices_2 = list(np.where(coord_masks[residue_j,:,0] == 1)[0])
        
        for atom_indice_1 in atom_indices_1:
            distances = np.linalg.norm(coords[residue_i,atom_indice_1][None] - coords[residue_j,atom_indices_2], axis = -1)        
            atom_contact = np.sum(distances < 5)
            if atom_contact:  
                contact_map[residue_i,residue_j] += 1
                break
    
    if show == True:
        plt.figure(figsize=  (15,15))
        plt.title('Contacted residue pairs from another chains')
        plt.imshow(contact_map, cmap = 'viridis_r')
        plt.show()

        print(f'''candidate_residue_pairs : {(contact_map > 0).sum()}''')
        print(f'''contacted_residue_pairs : {(contact_map > 1).sum()}''')

    return contact_map,residue_dict, coord_masks

def get_ICS(native_contact_map, pred_contact_map):
    
    TP = ((pred_contact_map == 2) & (native_contact_map == 2)).sum()
    FP = ((pred_contact_map == 2) & (native_contact_map == 1)).sum()
    TN = ((pred_contact_map == 1) & (native_contact_map == 1)).sum()
    FN = ((pred_contact_map == 1) & (native_contact_map == 2)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1_score = 2 * (precision * recall) / (precision + recall)
    interface_similarity_score = np.max([f1_score,0])
    
    return interface_similarity_score

## for new patch setting..
def get_IPS(native_contact_map, pred_contact_map):    
    native_patches = (native_contact_map == 2).sum(axis=-1) > 0
    pred_patches    = (pred_contact_map   == 2).sum(axis=-1) > 0

    intersection = np.logical_and(native_patches, pred_patches)
    union = np.logical_or(native_patches, pred_patches)

    intersection_count = np.sum(intersection)
    union_count = np.sum(union)
    
    interface_patch_similarity = intersection_count/union_count

    return interface_patch_similarity


def eval_interface(native_pdb_path, pred_pdb_path,show = False, interface = 'all', print = False):
    
    native_contact_map, residue_dict, coord_masks = get_contact(native_pdb_path, residue_dict = None, coord_masks = None, interface = interface, show = show)
    pred_contact_map, _, _ = get_contact(pred_pdb_path, residue_dict = residue_dict, coord_masks = coord_masks, interface = interface, show = show)
    
    #print(np.isnan(native_contact_map).sum(), np.isnan(pred_contact_map).sum())
    ICS = get_ICS(native_contact_map, pred_contact_map)
    IPS = get_IPS(native_contact_map, pred_contact_map) 
    
    if np.isnan(ICS) : ICS = 0
    if np.isnan(IPS) : IPS = 0

    if print == True:
        print(F'Interface Similarity Score : {ICS:.5f}, Interface Patch Score : {IPS:.5f}')
    return ICS, IPS
