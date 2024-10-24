import os
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np
import torch
import pickle
import sys
from math import pi
from scipy.spatial.distance import pdist, squareform

sys.path.append('/gpfs/deepfold/users/casp/alphafold/')
import alphafold

plot_types = {'PHI_PSI' : [1,2], 'PHI_CHI1' : [1,3], 'PSI_CHI1' : [2,3], 'CHI1_CHI2' : [3, 4]}
    
  
def new_dihedral(p):
    """Praxeolitic formula
    1 sqrt, 1 cross product"""
    b0 = -1.0*(p[1] - p[0])
    b1 = p[2] - p[1]
    b2 = p[3] - p[2]

    # normalize b1 so that it does not influence m    agnitude of vector
    # rejections that come next
    b1 /= np.linalg.norm(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    angle = np.degrees(np.arctan2(y, x))
    
    return angle
  
def get_bondangle(p):
    axis_1 = (p[0] - p[1]) / np.linalg.norm(p[0] - p[1])
    axis_2 = (p[2] - p[1]) / np.linalg.norm(p[2] - p[1])
    bondangle = np.arccos(np.dot(axis_1 , axis_2)) * 180 / np.pi
    return bondangle

def angles_to_sincos(tor_angles):
    length = tor_angles.shape[1]
    tor_sincos = torch.zeros(len(tor_angles),length,2)
    tor_sincos[:,:,0] = torch.sin(torch.deg2rad(tor_angles))
    tor_sincos[:,:,1] = torch.cos(torch.deg2rad(tor_angles))
    return tor_sincos

def get_refer_atoms(restype,angletype):
    candidates = []
    if angletype == 0: candidates = [1,2,0,1]
    if angletype == 1: candidates = [2,0,1,2]
    if angletype == 2: candidates = [0,1,2,4]
    
    if angletype == 3:
        if   restype == 'ARG': candidates = [0,1,3,5]
        elif restype == 'ASN': candidates = [0,1,3,5]
        elif restype == 'ASP': candidates = [0,1,3,5]
        elif restype == 'CYS': candidates = [0,1,3,10]
        elif restype == 'GLN': candidates = [0,1,3,5]
        elif restype == 'GLU': candidates = [0,1,3,5]
        elif restype == 'HIS': candidates = [0,1,3,5]
        elif restype == 'ILE': candidates = [0,1,3,6]
        elif restype == 'LEU': candidates = [0,1,3,5]
        elif restype == 'LYS': candidates = [0,1,3,5]
        elif restype == 'MET': candidates = [0,1,3,5]
        elif restype == 'PHE': candidates = [0,1,3,5]
        elif restype == 'PRO': candidates = [0,1,3,5]
        elif restype == 'SER': candidates = [0,1,3,8]
        elif restype == 'THR': candidates = [0,1,3,9]
        elif restype == 'TRP': candidates = [0,1,3,5]
        elif restype == 'TYR': candidates = [0,1,3,5]
        elif restype == 'VAL': candidates = [0,1,3,6]
    
    elif angletype == 4:
        if   restype == 'ARG': candidates = [1,3,5,11]
        elif restype == 'ASN': candidates = [1,3,5,16,15]
        elif restype == 'ASP': candidates = [1,3,5,16,17]
        elif restype == 'GLN': candidates = [1,3,5,11]
        elif restype == 'GLU': candidates = [1,3,5,11]
        elif restype == 'HIS': candidates = [1,3,5,14,13]
        elif restype == 'ILE': candidates = [1,3,6,12]
        elif restype == 'LEU': candidates = [1,3,5,12]
        elif restype == 'LYS': candidates = [1,3,5,11]
        elif restype == 'MET': candidates = [1,3,5,18]
        elif restype == 'PHE': candidates = [1,3,5,12,13]
        elif restype == 'PRO': candidates = [1,3,5,11]
        elif restype == 'TRP': candidates = [1,3,5,12,13]
        elif restype == 'TYR': candidates = [1,3,5,12,13]
        
    elif angletype == 5:
        if   restype == 'ARG': candidates = [3,5,11,23]
        elif restype == 'GLN': candidates = [3,5,11,26,25]
        elif restype == 'GLU': candidates = [3,5,11,26,27]
        elif restype == 'LYS': candidates = [3,5,11,19]
        elif restype == 'MET': candidates = [3,5,10,19]
        
    if angletype == 6:
        if restype == 'ARG': candidates = [5,11,23,32]
        elif restype == 'LYS': candidates = [5,11,19,35]
    return candidates


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


atom_types = {"N":0,"CA":1,"C":2,"CB":3,"O":4,"CG":5,"CG1":6,"CG2":7,"OG":8,"OG1":9,"SG":10,"CD":11,"CD1":12,"CD2":13,"ND1":14,"ND2":15,"OD1":16,"OD2":17,"SD":18,\
            "CE":19,"CE1":20,"CE2":21,"CE3":22,"NE":23,"NE1":24,"NE2":25,"OE1":26,"OE2":27,"CH2":28,"NH1":29,"NH2":30,"OH":31,"CZ":32,"CZ2":33,"CZ3":34,"NZ":35,"OXT":36}

tor_types = {'OMEGA' : 0, 'PHI' : 1, 'PSI' : 2, 'CHI1' : 3, 'CHI2' : 4 , 'CHI3' : 5, 'CHI4' : 6}

res_map = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', \
           'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', 'UNK' : '-'}

def readPDB(pdb_path):
    parser = PDBParser(PERMISSIVE=1)
    structure = parser.get_structure('pdb', pdb_path)
    residues = {}
    for model_id in structure:
        for chain_id in model_id:
            chain = model_id[chain_id.id]
            for residue in chain_id:
                res_name = residue.resname.strip()
                res_id = residue.id[1]
                residues[res_id] = res_name
    return residues, chain  

def get_coordinates(final_residue, residues,chain):
    first_residue = list(residues.keys())[0]
    coord = np.zeros([final_residue,37, 3])
    coord_mask = np.zeros([final_residue, 37, 1])
    unexpected_atoms = {}
    
    for i in range(1, first_residue):
        unexpected_atoms[i] = {}
    
    def get_coordinates(atom):
        vec = atom.get_vector()
        return np.array([vec[0], vec[1], vec[2]])
    
    for residue in chain:
        res_num = residue.get_id()[1]-1
        if (res_num + 1 ) > final_residue: continue
        elif res_num < 0 : continue
        res_name = residue.resname
        unexpected_atoms[res_num+1] = {}
        for atom in residue:
            atom_id = atom.get_id()
            refer_atoms = restype_refer_atoms(res_name)
            if res_name == 'MET' and atom_id == 'SE': atom_id = 'SD'
            if atom_id not in atom_types.keys()         : unexpected_atoms[res_num+1][atom_id] = {}
            elif atom_types[atom_id] not in refer_atoms : unexpected_atoms[res_num+1][atom_id] = {}
            else :
                coord[res_num,atom_types[atom_id]] = get_coordinates(atom)
                coord_mask[res_num,atom_types[atom_id]] = True
    return np.array(coord),np.array(coord_mask),unexpected_atoms

  
def getDistogram(residues, atom_pos, atom_mask):
    final_residue = list(residues.keys())[-1]
    pairwise_dist  = np.zeros([len(atom_mask), len(atom_mask)])
    origin_coords  = np.zeros([len(atom_mask), 3])
    pairwise_dist[:] = np.nan
    origin_coords[:] = np.nan
    
    for res_num in residues.keys():
        if res_num > final_residue: continue
        elif res_num < 1 : continue
        res_type = residues[res_num]        
        if (res_type == 'GLY') : origin_atom = 'CA'
        else                   : origin_atom = 'CB'
        
        if atom_mask[res_num-1][atom_types[origin_atom]] : origin_coords[res_num-1] = np.array(atom_pos[res_num-1][atom_types[origin_atom]])
        else :                                             origin_coords[res_num-1] = np.array([np.nan, np.nan, np.nan])
    
    pairwise_dist = squareform(pdist(origin_coords, 'euclidean'))
    return pairwise_dist
  
def getInteracted(final_residue, residues, atom_pos, atom_mask):
    #final_residue = list(residues.keys())[-1]
    origin_coords  = np.zeros([len(atom_mask), 3])
    interacted = {}
    
    origin_coords[:] = np.nan
    
    for res_num in residues.keys():
        if res_num > final_residue: continue
        elif res_num < 1 : continue
        res_type = residues[res_num]        
        if (res_type == 'GLY') : origin_atom = 'CA'
        else                   : origin_atom = 'CB'
        
        if atom_mask[res_num-1][atom_types[origin_atom]] : origin_coords[res_num-1] = np.array(atom_pos[res_num-1][atom_types[origin_atom]])
        else :                                             origin_coords[res_num-1] = np.array([np.nan, np.nan, np.nan])
    
    pairwise_dist = squareform(pdist(origin_coords, 'euclidean'))
    
    for res_num in residues.keys():
        if res_num > final_residue: continue
        elif res_num < 1 : continue
        checker = pairwise_dist[res_num-1,:]
        checker[res_num-1] = 100
        interacted[res_num] = np.nanargmin(checker) + 1
    return interacted
  
  

def get_interacted_torsion(atom_mask, atom_pos, residues, as_tensor = False, angle = 'all'):
    n_angle = 3
    
    if angle == 'omega' or angle == 'phi' or angle == 'psi': n_angle = 1
    
    tor_masks = np.zeros([len(atom_mask),len(atom_mask),n_angle], dtype = np.bool_)
    tor_angles = np.zeros([len(atom_mask),len(atom_mask),n_angle])
    
    start_num = list(residues.keys())[0]
    last_num  = len(atom_mask)
    
    for origin_num in residues.keys():
        if origin_num > last_num: continue
        elif origin_num < 1 : continue
        for target_num in residues.keys():
            if target_num > last_num: continue
            elif target_num < 1 : continue
            
            origin_i = origin_num -1
            target_i = target_num -1
            origin_mask = atom_mask[origin_i]
            target_mask = atom_mask[target_i]
            if origin_i == target_i : continue
            
            
            if angle == 'omega' or angle == 'all':
              angle_index = 0
              if angle != 'all' : angle_index = 0
              if (origin_mask[1] and origin_mask[3] and target_mask[1] and target_mask[3]) : 
                  tor_masks[origin_i,target_i,angle_index] = True
                  tor_angles[origin_i,target_i,angle_index] = new_dihedral([atom_pos[origin_i,1], atom_pos[origin_i,3], atom_pos[target_i,3], atom_pos[target_i,1]])
                  
            if angle == 'phi' or angle == 'all':
              angle_index = 1
              if angle != 'all' : angle_index = 0
              if (origin_mask[0] and origin_mask[1] and origin_mask[3] and target_mask[3]) : 
                  tor_masks[origin_i,target_i,angle_index] = True
                  tor_angles[origin_i,target_i,angle_index] = new_dihedral([atom_pos[origin_i,0], atom_pos[origin_i,1], atom_pos[origin_i,3], atom_pos[target_i,3]])
              
            if angle == 'psi' or angle == 'all':
              angle_index = 2
              if angle != 'all' : angle_index = 0
              if (origin_mask[1] and origin_mask[3] and target_mask[3]) : 
                  tor_masks[origin_i,target_i,angle_index] = True
                  tor_angles[origin_i,target_i,angle_index] = get_bondangle([atom_pos[origin_i,1], atom_pos[origin_i,3], atom_pos[target_i,3]])
    
    if as_tensor == True:
        tor_angles = torch.tensor(tor_angles)
        tor_masks = torch.tensor(tor_masks)
    return tor_masks, tor_angles
  
def get_torsion(atom_mask, atom_pos, residues, as_tensor = False):
    tor_masks = np.zeros([len(atom_mask),7], dtype = np.bool_)
    tor_angles = np.zeros([len(atom_mask),7])
    
    start_num = list(residues.keys())[0]
    last_num  = len(atom_mask)
    
    for res_num in residues.keys():
        if (res_num ) > last_num: continue
        elif res_num < 1 : continue
        
        i = res_num -1
        if i!= start_num-1: prev_mask  = atom_mask[i-1]
        curr_mask                      = atom_mask[i]
        if i!= last_num-1 : next_mask  = atom_mask[i+1]
        
        if res_num == start_num : 
            if (curr_mask[0] and curr_mask[1] and curr_mask[2] and next_mask[0]) : 
                tor_masks[i,2] = True
                tor_angles[i,2] = new_dihedral([atom_pos[i,0], atom_pos[i,1], atom_pos[i,2], atom_pos[i+1,0]])
                
        elif res_num == last_num :
            if (prev_mask[1] and prev_mask[2] and curr_mask[0] and curr_mask[1]): 
                tor_masks[i,0] =  True
                tor_angles[i,0] = new_dihedral([atom_pos[i-1,1],atom_pos[i-1,2], atom_pos[i,0], atom_pos[i,1]])    
            if (prev_mask[2] and curr_mask[0] and curr_mask[1] and curr_mask[2]): 
                tor_masks[i,1] =  True
                tor_angles[i,1] = new_dihedral([atom_pos[i-1,2], atom_pos[i,0]  , atom_pos[i,1], atom_pos[i,2]])
        else : 
            if (prev_mask[1] and prev_mask[2] and curr_mask[0] and curr_mask[1]): 
                tor_masks[i,0] =  True
                tor_angles[i,0] = new_dihedral([atom_pos[i-1,1],atom_pos[i-1,2], atom_pos[i,0], atom_pos[i,1]])    
            if (prev_mask[2] and curr_mask[0] and curr_mask[1] and curr_mask[2]): 
                tor_masks[i,1] =  True
                tor_angles[i,1] = new_dihedral([atom_pos[i-1,2], atom_pos[i,0]  , atom_pos[i,1], atom_pos[i,2]])
            if (curr_mask[0] and curr_mask[1] and curr_mask[2] and next_mask[0]): 
                tor_masks[i,2] =  True
                tor_angles[i,2] = new_dihedral([atom_pos[i,0]  , atom_pos[i,1]  , atom_pos[i,2], atom_pos[i+1,0]])
                
        for side_angle in range(3,7):
            refer_atoms = get_refer_atoms(residues[res_num],side_angle)
            if refer_atoms != [] and (curr_mask[refer_atoms[0]] and curr_mask[refer_atoms[1]] and curr_mask[refer_atoms[2]] and curr_mask[refer_atoms[3]]):
                tor_masks[i,side_angle] = True
                tor_angles[i,side_angle] = new_dihedral([atom_pos[i,refer_atoms[0]], atom_pos[i,refer_atoms[1]], atom_pos[i,refer_atoms[2]], atom_pos[i,refer_atoms[3]]])
    
    if as_tensor == True:
        tor_angles = torch.tensor(tor_angles)
        tor_masks = torch.tensor(tor_masks)
    return tor_masks, tor_angles

  
def get_torsion_with_interacted(atom_mask, atom_pos, residues, interacted, as_tensor = False):
    tor_masks = np.zeros([len(atom_mask),7], dtype = np.bool_)
    tor_angles = np.zeros([len(atom_mask),7])
    interacted_masks = np.zeros([len(atom_mask), 3], dtype = np.bool_)
    interacted_angles = np.zeros([len(atom_mask), 3])
    
    start_num = list(residues.keys())[0]
    last_num  = len(atom_mask)
    
    for res_num in residues.keys():
        if (res_num ) > last_num: continue
        elif res_num < 1 : continue
        
        i = res_num -1
        if i!= start_num-1: prev_mask  = atom_mask[i-1]
        curr_mask                      = atom_mask[i]
        if i!= last_num-1 : next_mask  = atom_mask[i+1]
        
        if res_num == start_num : 
            if (curr_mask[0] and curr_mask[1] and curr_mask[2] and next_mask[0]) : 
                tor_masks[i,2] = True
                tor_angles[i,2] = new_dihedral([atom_pos[i,0], atom_pos[i,1], atom_pos[i,2], atom_pos[i+1,0]])
                
        elif res_num == last_num :
            if (prev_mask[1] and prev_mask[2] and curr_mask[0] and curr_mask[1]): 
                tor_masks[i,0] =  True
                tor_angles[i,0] = new_dihedral([atom_pos[i-1,1],atom_pos[i-1,2], atom_pos[i,0], atom_pos[i,1]])    
            if (prev_mask[2] and curr_mask[0] and curr_mask[1] and curr_mask[2]): 
                tor_masks[i,1] =  True
                tor_angles[i,1] = new_dihedral([atom_pos[i-1,2], atom_pos[i,0]  , atom_pos[i,1], atom_pos[i,2]])
        else : 
            if (prev_mask[1] and prev_mask[2] and curr_mask[0] and curr_mask[1]): 
                tor_masks[i,0] =  True
                tor_angles[i,0] = new_dihedral([atom_pos[i-1,1],atom_pos[i-1,2], atom_pos[i,0], atom_pos[i,1]])    
            if (prev_mask[2] and curr_mask[0] and curr_mask[1] and curr_mask[2]): 
                tor_masks[i,1] =  True
                tor_angles[i,1] = new_dihedral([atom_pos[i-1,2], atom_pos[i,0]  , atom_pos[i,1], atom_pos[i,2]])
            if (curr_mask[0] and curr_mask[1] and curr_mask[2] and next_mask[0]): 
                tor_masks[i,2] =  True
                tor_angles[i,2] = new_dihedral([atom_pos[i,0]  , atom_pos[i,1]  , atom_pos[i,2], atom_pos[i+1,0]])
        
        for side_angle in range(3,7):
            refer_atoms = get_refer_atoms(residues[res_num],side_angle)
            if refer_atoms != [] and (curr_mask[refer_atoms[0]] and curr_mask[refer_atoms[1]] and curr_mask[refer_atoms[2]] and curr_mask[refer_atoms[3]]):
                tor_masks[i,side_angle] = True
                tor_angles[i,side_angle] = new_dihedral([atom_pos[i,refer_atoms[0]], atom_pos[i,refer_atoms[1]], atom_pos[i,refer_atoms[2]], atom_pos[i,refer_atoms[3]]])
                
                
        if res_num not in interacted.keys(): continue
        
        
        if residues[interacted[res_num]] == 'GLY'  or residues[res_num] == 'GLY':
            interacted_masks[i][0]  = False
            interacted_masks[i][1]  = False
            interacted_masks[i][2]  = False
        
        elif residues[interacted[res_num]] != 'GLY' and residues[res_num] != 'GLY' :
            if curr_mask[0]  and curr_mask[1] and curr_mask[3] and interacted[res_num] >= 0 : 
                if atom_mask[interacted[res_num]-1][3]   :  
                    interacted_masks[i][0] = True
                    interacted_angles[i][0] = new_dihedral([atom_pos[i,0],atom_pos[i,1],atom_pos[i,3],atom_pos[interacted[res_num]-1,3]])
                  
                if atom_mask[interacted[res_num]-1][1] and  atom_mask[interacted[res_num]-1][3] :  
                    interacted_masks[i][1] = True
                    interacted_angles[i][1] = new_dihedral([atom_pos[i,1],atom_pos[i,3],atom_pos[interacted[res_num]-1,3],atom_pos[interacted[res_num]-1,1]])
                    
                if atom_mask[interacted[res_num]-1][3] :  
                    interacted_masks[i][2] = True
                    interacted_angles[i][2] = get_bondangle([atom_pos[i,1], atom_pos[i,3], atom_pos[interacted[res_num]-1,3]]) * 2 - 180
                
    if as_tensor == True:
        tor_angles = torch.tensor(tor_angles)
        tor_masks = torch.tensor(tor_masks)
        interacted_angles = torch.tensor(interacted_angles)
        interacted_masks = torch.tensor(interacted_masks)
    return tor_masks, tor_angles, interacted_masks, interacted_angles
  
  
def sidechain_sym_angle(target_residues,tor_masks, native_angles, target_angles, target_alter_angles):
    diff = np.array([0,0])
    tor_len = len(tor_masks)
    for res_num in target_residues.keys():
        i = res_num - 1
        if res_num > tor_len : continue
        elif res_num < 1 : continue
        
        for side_angle in range(3,7):
            residue = target_residues[res_num]
            if tor_masks[i,side_angle] and ((side_angle == 4 and residue in ['ASN','ASP','HIS','PHE','TRP','TYR']) or (side_angle == 5 and residue in ['GLN','GLU'])) :
                native_diff = np.abs(native_angles[i,side_angle] - target_angles[i,side_angle])
                alter_diff = np.abs(native_angles[i,side_angle] - target_alter_angles[i,side_angle])
                diff[0] = np.min([native_diff, 360-native_diff])
                diff[1] = np.min([alter_diff, 360-alter_diff])
                if diff[1] < diff[0] : target_angles[i,side_angle] = target_alter_angles[i,side_angle]
                    
                
    return target_angles  


def getTorsion_acc(target_residues, tor_masks, native_angles, target_angles, target_alter_angles, thres = 10, chi_dependent = True):
    
    final_residue = len(tor_masks)
    torsion_acc = {}
    bb_angles = ['OMEGA','PHI','PSI']
    side_angles = ['CHI1','CHI2','CHI3','CHI4']
    angles = bb_angles + side_angles
    for angle in angles:
        torsion_acc[angle] = {}
        for label in ['total', 'correct','accuracy']:
            torsion_acc[angle][label] = 0
    
    diff = np.array([0,0]) # can have 0 - 180

    for res_num in target_residues.keys():
        if res_num > final_residue: continue
        elif res_num < 1 : continue
        
        chi_check = True
        i = res_num - 1
        for j, angle in enumerate(angles):    
            if tor_masks[i,j]:
                diff = np.abs(native_angles[i,j] - target_angles[i,j])
                diff = np.min([diff, 360 - diff])
                
                torsion_acc[angle]['total'] += 1
                if (j < 3):
                    if diff < thres: torsion_acc[angle]['correct'] += 1
                else :
                    if chi_dependent == True:
                        if diff < thres and chi_check: torsion_acc[angle]['correct'] += 1
                        else : chi_check = False
                    elif chi_dependent == False:
                        if diff < thres : torsion_acc[angle]['correct'] += 1
    for angle in angles:
        if torsion_acc[angle]['total'] != 0: 
            torsion_acc[angle]['accuracy'] = torsion_acc[angle]['correct']/torsion_acc[angle]['total']
    
    return torsion_acc

def torsion_angle_loss(
    a,  # [*, N, 7, 2]
    a_gt,  # [*, N, 7, 2]
    tor_masks):
    
    
    bb_masks = torch.zeros(len(a),7)
    bb_masks[:,:3] = True
    bb_masks = torch.logical_and(tor_masks,bb_masks)
    
    side_masks = torch.zeros(len(a),7)
    side_masks[:,3:] = True
    side_masks = torch.logical_and(tor_masks,side_masks)
    
    norm = torch.norm(a, dim=-1)

    a = a / norm.unsqueeze(-1)

    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    
    min_diff = diff_norm_gt**2
    
    min_diff_total = torch.sum(min_diff * tor_masks)  / torch.sum(tor_masks)
    min_diff_bb    = torch.sum(min_diff * bb_masks)   / torch.sum(bb_masks)
    min_diff_side  = torch.sum(min_diff * side_masks) / torch.sum(side_masks)
    
    angle_norm = torch.abs(norm - 1)
     
    angle_norm_total = 0.02 * torch.sum(angle_norm * tor_masks)  / torch.sum(tor_masks)
    angle_norm_bb    = 0.02 * torch.sum(angle_norm * bb_masks)   / torch.sum(bb_masks)
    angle_norm_side  = 0.02 * torch.sum(angle_norm * side_masks) / torch.sum(side_masks)
    
    total_loss = (min_diff_total  + angle_norm_total).numpy()
    bb_loss    = (min_diff_bb     + angle_norm_bb).numpy()
    side_loss  = (min_diff_side   + angle_norm_side).numpy()
    
    loss = {}
    loss['total'] = total_loss
    loss['backbone']    = bb_loss
    loss['sidechain']  = side_loss
    
    return loss
