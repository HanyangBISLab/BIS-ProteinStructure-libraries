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
    """Calculate the dihedral angle between four points.

    Parameters
    ----------
    p : np.ndarray
        An array of shape (4, 3) representing the coordinates of the four points.

    Returns
    -------
    float
        The dihedral angle in degrees between the planes formed by the points.
    """
    b0 = -1.0*(p[1] - p[0])
    b1 = p[2] - p[1]
    b2 = p[3] - p[2]

    # normalize b1 so that it does not influence magnitude of vector rejections
    b1 /= np.linalg.norm(b1)

    # vector rejections
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    # angle between v and w in a plane is the torsion angle
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    angle = np.degrees(np.arctan2(y, x))
    
    return angle
  
def get_bondangle(p):
    """Calculate the bond angle formed by three points.

    Parameters
    ----------
    p : np.ndarray
        An array of shape (3, 3) representing the coordinates of the three points.

    Returns
    -------
    float
        The bond angle in degrees formed by the three points.
    """
    axis_1 = (p[0] - p[1]) / np.linalg.norm(p[0] - p[1])
    axis_2 = (p[2] - p[1]) / np.linalg.norm(p[2] - p[1])
    bondangle = np.arccos(np.dot(axis_1, axis_2)) * 180 / np.pi
    return bondangle

def angles_to_sincos(tor_angles):
    """Convert angles to sine and cosine values.

    Parameters
    ----------
    tor_angles : torch.Tensor
        A tensor of shape (n, m) containing angles in degrees.

    Returns
    -------
    torch.Tensor
        A tensor of shape (n, m, 2) where the last dimension contains the sine and cosine of the angles.
    """
    length = tor_angles.shape[1]
    tor_sincos = torch.zeros(len(tor_angles), length, 2)
    tor_sincos[:, :, 0] = torch.sin(torch.deg2rad(tor_angles))
    tor_sincos[:, :, 1] = torch.cos(torch.deg2rad(tor_angles))
    return tor_sincos

def get_refer_atoms(restype, angletype):
    """Get reference atom indices based on residue type and angle type.

    Parameters
    ----------
    restype : str
        The residue type (e.g., 'ARG', 'GLY').
    angletype : int
        The angle type index (0-6).

    Returns
    -------
    list
        A list of indices representing reference atoms for the given residue and angle types.
    """
    candidates = []
    if angletype == 0: candidates = [1, 2, 0, 1]
    if angletype == 1: candidates = [2, 0, 1, 2]
    if angletype == 2: candidates = [0, 1, 2, 4]

    if angletype == 3:
        if restype == 'ARG': candidates = [0, 1, 3, 5]
        elif restype == 'ASN': candidates = [0, 1, 3, 5]
        elif restype == 'ASP': candidates = [0, 1, 3, 5]
        elif restype == 'CYS': candidates = [0, 1, 3, 10]
        elif restype == 'GLN': candidates = [0, 1, 3, 5]
        elif restype == 'GLU': candidates = [0, 1, 3, 5]
        elif restype == 'HIS': candidates = [0, 1, 3, 5]
        elif restype == 'ILE': candidates = [0, 1, 3, 6]
        elif restype == 'LEU': candidates = [0, 1, 3, 5]
        elif restype == 'LYS': candidates = [0, 1, 3, 5]
        elif restype == 'MET': candidates = [0, 1, 3, 5]
        elif restype == 'PHE': candidates = [0, 1, 3, 5]
        elif restype == 'PRO': candidates = [0, 1, 3, 5]
        elif restype == 'SER': candidates = [0, 1, 3, 8]
        elif restype == 'THR': candidates = [0, 1, 3, 9]
        elif restype == 'TRP': candidates = [0, 1, 3, 5]
        elif restype == 'TYR': candidates = [0, 1, 3, 5]
        elif restype == 'VAL': candidates = [0, 1, 3, 6]

    elif angletype == 4:
        if restype == 'ARG': candidates = [1, 3, 5, 11]
        elif restype == 'ASN': candidates = [1, 3, 5, 16, 15]
        elif restype == 'ASP': candidates = [1, 3, 5, 16, 17]
        elif restype == 'GLN': candidates = [1, 3, 5, 11]
        elif restype == 'GLU': candidates = [1, 3, 5, 11]
        elif restype == 'HIS': candidates = [1, 3, 5, 14, 13]
        elif restype == 'ILE': candidates = [1, 3, 6, 12]
        elif restype == 'LEU': candidates = [1, 3, 5, 12]
        elif restype == 'LYS': candidates = [1, 3, 5, 11]
        elif restype == 'MET': candidates = [1, 3, 5, 18]
        elif restype == 'PHE': candidates = [1, 3, 5, 12, 13]
        elif restype == 'PRO': candidates = [1, 3, 5, 11]
        elif restype == 'TRP': candidates = [1, 3, 5, 12, 13]
        elif restype == 'TYR': candidates = [1, 3, 5, 12, 13]
        
    elif angletype == 5:
        if restype == 'ARG': candidates = [3, 5, 11, 23]
        elif restype == 'GLN': candidates = [3, 5, 11, 26, 25]
        elif restype == 'GLU': candidates = [3, 5, 11, 26, 27]
        elif restype == 'LYS': candidates = [3, 5, 11, 19]
        elif restype == 'MET': candidates = [3, 5, 10, 19]
        
    if angletype == 6:
        if restype == 'ARG': candidates = [5, 11, 23, 32]
        elif restype == 'LYS': candidates = [5, 11, 19, 35]
    return candidates

def restype_refer_atoms(restype):
    """Get reference atom indices for a given residue type.

    Parameters
    ----------
    restype : str
        The residue type (e.g., 'ALA', 'ARG').

    Returns
    -------
    list
        A list of indices representing the atoms associated with the residue type.
    """
    atoms = []
    if restype == 'ALA': atoms = [0, 1, 2, 3, 4]
    elif restype == 'ARG': atoms = [0, 1, 2, 3, 4, 5, 11, 23, 29, 30, 32]
    elif restype == 'ASN': atoms = [0, 1, 2, 3, 4, 5, 15, 16]
    elif restype == 'ASP': atoms = [0, 1, 2, 3, 4, 5, 16, 17]
    elif restype == 'CYS': atoms = [0, 1, 2, 3, 4, 10]
    elif restype == 'GLN': atoms = [0, 1, 2, 3, 4, 5, 11, 25, 26]
    elif restype == 'GLU': atoms = [0, 1, 2, 3, 4, 5, 11, 26, 27]
    elif restype == 'HIS': atoms = [0, 1, 2, 3, 4, 5, 11, 13, 14]
    elif restype == 'ILE': atoms = [0, 1, 2, 3, 4, 5, 6, 12]
    elif restype == 'LEU': atoms = [0, 1, 2, 3, 4, 5, 11]
    elif restype == 'LYS': atoms = [0, 1, 2, 3, 4, 5, 11, 19]
    elif restype == 'MET': atoms = [0, 1, 2, 3, 4, 5, 10, 18]
    elif restype == 'PHE': atoms = [0, 1, 2, 3, 4, 5, 12, 13]
    elif restype == 'PRO': atoms = [0, 1, 2, 3, 4, 5]
    elif restype == 'SER': atoms = [0, 1, 2, 3, 4, 8]
    elif restype == 'THR': atoms = [0, 1, 2, 3, 4, 9]
    elif restype == 'TRP': atoms = [0, 1, 2, 3, 4, 5, 12, 13]
    elif restype == 'TYR': atoms = [0, 1, 2, 3, 4, 5, 12, 13]
    elif restype == 'VAL': atoms = [0, 1, 2, 3, 4, 5, 6]
    return atoms



atom_types = {"N":0,"CA":1,"C":2,"CB":3,"O":4,"CG":5,"CG1":6,"CG2":7,"OG":8,"OG1":9,"SG":10,"CD":11,"CD1":12,"CD2":13,"ND1":14,"ND2":15,"OD1":16,"OD2":17,"SD":18,\
            "CE":19,"CE1":20,"CE2":21,"CE3":22,"NE":23,"NE1":24,"NE2":25,"OE1":26,"OE2":27,"CH2":28,"NH1":29,"NH2":30,"OH":31,"CZ":32,"CZ2":33,"CZ3":34,"NZ":35,"OXT":36}

tor_types = {'OMEGA' : 0, 'PHI' : 1, 'PSI' : 2, 'CHI1' : 3, 'CHI2' : 4 , 'CHI3' : 5, 'CHI4' : 6}

res_map = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', \
           'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', 'UNK' : '-'}

def readPDB(pdb_path):
    """
    Reads a PDB file and extracts residue information.

    Parameters
    ----------
    pdb_path : str
        The file path to the PDB file to be parsed.

    Returns
    -------
    residues : dict
        A dictionary mapping residue IDs to residue names.
    chain : Chain
        The chain object from the parsed structure.
    """
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

def get_coordinates(final_residue, residues, chain):
    """
    Retrieves the coordinates of atoms from the residues in a chain.

    Parameters
    ----------
    final_residue : int
        The total number of residues to be considered.
    residues : dict
        A dictionary mapping residue IDs to residue names.
    chain : Chain
        The chain object containing the residues.

    Returns
    -------
    coord : np.ndarray
        An array of shape (final_residue, 37, 3) containing the coordinates of the atoms.
    coord_mask : np.ndarray
        An array of shape (final_residue, 37, 1) indicating which atoms have valid coordinates.
    unexpected_atoms : dict
        A dictionary mapping residue IDs to unexpected atom IDs found during processing.
    """
    first_residue = list(residues.keys())[0]
    coord = np.zeros([final_residue, 37, 3])
    coord_mask = np.zeros([final_residue, 37, 1])
    unexpected_atoms = {}
    
    for i in range(1, first_residue):
        unexpected_atoms[i] = {}
    
    def get_coordinates(atom):
        vec = atom.get_vector()
        return np.array([vec[0], vec[1], vec[2]])
    
    for residue in chain:
        res_num = residue.get_id()[1]-1
        if (res_num + 1) > final_residue: continue
        elif res_num < 0: continue
        res_name = residue.resname
        unexpected_atoms[res_num+1] = {}
        for atom in residue:
            atom_id = atom.get_id()
            refer_atoms = restype_refer_atoms(res_name)
            if res_name == 'MET' and atom_id == 'SE': atom_id = 'SD'
            if atom_id not in atom_types.keys(): unexpected_atoms[res_num+1][atom_id] = {}
            elif atom_types[atom_id] not in refer_atoms: unexpected_atoms[res_num+1][atom_id] = {}
            else:
                coord[res_num, atom_types[atom_id]] = get_coordinates(atom)
                coord_mask[res_num, atom_types[atom_id]] = True
    return np.array(coord), np.array(coord_mask), unexpected_atoms

def getDistogram(residues, atom_pos, atom_mask):
    """
    Calculates the pairwise distance matrix for atom positions.

    Parameters
    ----------
    residues : dict
        A dictionary mapping residue IDs to residue names.
    atom_pos : np.ndarray
        An array containing the coordinates of the atoms.
    atom_mask : np.ndarray
        An array indicating which atoms have valid coordinates.

    Returns
    -------
    pairwise_dist : np.ndarray
        A square matrix of shape (n_atoms, n_atoms) containing pairwise distances.
    """
    final_residue = list(residues.keys())[-1]
    pairwise_dist = np.zeros([len(atom_mask), len(atom_mask)])
    origin_coords = np.zeros([len(atom_mask), 3])
    pairwise_dist[:] = np.nan
    origin_coords[:] = np.nan
    
    for res_num in residues.keys():
        if res_num > final_residue: continue
        elif res_num < 1: continue
        res_type = residues[res_num]        
        if (res_type == 'GLY'): origin_atom = 'CA'
        else: origin_atom = 'CB'
        
        if atom_mask[res_num-1][atom_types[origin_atom]]:
            origin_coords[res_num-1] = np.array(atom_pos[res_num-1][atom_types[origin_atom]])
        else:
            origin_coords[res_num-1] = np.array([np.nan, np.nan, np.nan])
    
    pairwise_dist = squareform(pdist(origin_coords, 'euclidean'))
    return pairwise_dist

def get_torsion(atom_mask, atom_pos, residues, as_tensor=False):
    """
    Computes the torsion angles for a set of residues.

    Parameters
    ----------
    atom_mask : np.ndarray
        An array indicating which atoms have valid coordinates.
    atom_pos : np.ndarray
        An array containing the coordinates of the atoms.
    residues : dict
        A dictionary mapping residue IDs to residue names.
    as_tensor : bool, optional
        If True, returns the angles and masks as PyTorch tensors (default is False).

    Returns
    -------
    tor_masks : np.ndarray
        A boolean array of shape (n_residues, 7) indicating valid torsion angles.
    tor_angles : np.ndarray
        An array of shape (n_residues, 7) containing the torsion angle values.
    """
    tor_masks = np.zeros([len(atom_mask), 7], dtype=np.bool_)
    tor_angles = np.zeros([len(atom_mask), 7])
    
    start_num = list(residues.keys())[0]
    last_num = len(atom_mask)
    
    for res_num in residues.keys():
        if (res_num) > last_num: continue
        elif res_num < 1: continue
        
        i = res_num - 1
        if i != start_num - 1: prev_mask = atom_mask[i - 1]
        curr_mask = atom_mask[i]
        if i != last_num - 1: next_mask = atom_mask[i + 1]
        
        if res_num == start_num: 
            if (curr_mask[0] and curr_mask[1] and curr_mask[2] and next_mask[0]): 
                tor_masks[i, 2] = True
                tor_angles[i, 2] = new_dihedral([atom_pos[i, 0], atom_pos[i, 1], atom_pos[i, 2], atom_pos[i + 1, 0]])
                
        elif res_num == last_num:
            if (prev_mask[1] and prev_mask[2] and curr_mask[0] and curr_mask[1]): 
                tor_masks[i, 0] = True
                tor_angles[i, 0] = new_dihedral([atom_pos[i - 1, 1], atom_pos[i - 1, 2], atom_pos[i, 0], atom_pos[i, 1]])    
            if (prev_mask[2] and curr_mask[0] and curr_mask[1] and curr_mask[2]): 
                tor_masks[i, 1] = True
                tor_angles[i, 1] = new_dihedral([atom_pos[i - 1, 2], atom_pos[i, 0], atom_pos[i, 1], atom_pos[i, 2]])
        else: 
            if (prev_mask[1] and prev_mask[2] and curr_mask[0] and curr_mask[1]): 
                tor_masks[i, 0] = True
                tor_angles[i, 0] = new_dihedral([atom_pos[i - 1, 1], atom_pos[i - 1, 2], atom_pos[i, 0], atom_pos[i, 1]])    
            if (prev_mask[2] and curr_mask[0] and curr_mask[1] and curr_mask[2]): 
                tor_masks[i, 1] = True
                tor_angles[i, 1] = new_dihedral([atom_pos[i - 1, 2], atom_pos[i, 0], atom_pos[i, 1], atom_pos[i, 2]])
            if (curr_mask[0] and curr_mask[1] and curr_mask[2] and next_mask[0]): 
                tor_masks[i, 2] = True
                tor_angles[i, 2] = new_dihedral([atom_pos[i, 0], atom_pos[i, 1], atom_pos[i, 2], atom_pos[i + 1, 0]])
                
        for side_angle in range(3, 7):
            refer_atoms = get_refer_atoms(residues[res_num], side_angle)
            if refer_atoms != [] and (curr_mask[refer_atoms[0]] and curr_mask[refer_atoms[1]] and curr_mask[refer_atoms[2]] and curr_mask[refer_atoms[3]]):
                tor_masks[i, side_angle] = True
                tor_angles[i, side_angle] = new_dihedral([atom_pos[i, refer_atoms[0]], atom_pos[i, refer_atoms[1]], atom_pos[i, refer_atoms[2]], atom_pos[i, refer_atoms[3]]])
    
    if as_tensor:
        tor_angles = torch.tensor(tor_angles)
        tor_masks = torch.tensor(tor_masks)
    return tor_masks, tor_angles


  
  
def sidechain_sym_angle(target_residues, tor_masks, native_angles, target_angles, target_alter_angles):
    """
    Adjusts the target angles of sidechain torsions based on their differences with native angles and alternative angles.
    
    Parameters
    ----------
    target_residues : dict
        A dictionary containing the target residues indexed by their residue numbers.
    tor_masks : np.ndarray
        A boolean array indicating the availability of torsion angles for each residue.
    native_angles : np.ndarray
        An array of native torsion angles.
    target_angles : np.ndarray
        An array of target torsion angles to be modified.
    target_alter_angles : np.ndarray
        An array of alternative target torsion angles for comparison.
        
    Returns
    -------
    np.ndarray
        The modified target angles after adjusting based on comparisons with native and alternative angles.
    """
    diff = np.array([0, 0])
    tor_len = len(tor_masks)
    for res_num in target_residues.keys():
        i = res_num - 1
        if res_num > tor_len:
            continue
        elif res_num < 1:
            continue
        
        for side_angle in range(3, 7):
            residue = target_residues[res_num]
            if tor_masks[i, side_angle] and ((side_angle == 4 and residue in ['ASN', 'ASP', 'HIS', 'PHE', 'TRP', 'TYR']) or (side_angle == 5 and residue in ['GLN', 'GLU'])):
                native_diff = np.abs(native_angles[i, side_angle] - target_angles[i, side_angle])
                alter_diff = np.abs(native_angles[i, side_angle] - target_alter_angles[i, side_angle])
                diff[0] = np.min([native_diff, 360 - native_diff])
                diff[1] = np.min([alter_diff, 360 - alter_diff])
                if diff[1] < diff[0]:
                    target_angles[i, side_angle] = target_alter_angles[i, side_angle]
    
    return target_angles  


def getTorsion_acc(target_residues, tor_masks, native_angles, target_angles, target_alter_angles, thres=10, chi_dependent=True):
    """
    Calculates the accuracy of torsion angles by comparing native angles with target angles and optional alternative angles.
    
    Parameters
    ----------
    target_residues : dict
        A dictionary containing the target residues indexed by their residue numbers.
    tor_masks : np.ndarray
        A boolean array indicating the availability of torsion angles for each residue.
    native_angles : np.ndarray
        An array of native torsion angles.
    target_angles : np.ndarray
        An array of target torsion angles for comparison.
    target_alter_angles : np.ndarray
        An array of alternative target torsion angles for comparison.
    thres : int, optional
        The threshold for considering an angle correct. Default is 10 degrees.
    chi_dependent : bool, optional
        If True, ensures that only the first valid chi angle is counted as correct. Default is True.
        
    Returns
    -------
    dict
        A dictionary containing the total, correct counts, and accuracy for each angle type (backbone and sidechain).
    """
    final_residue = len(tor_masks)
    torsion_acc = {}
    bb_angles = ['OMEGA', 'PHI', 'PSI']
    side_angles = ['CHI1', 'CHI2', 'CHI3', 'CHI4']
    angles = bb_angles + side_angles
    
    for angle in angles:
        torsion_acc[angle] = {}
        for label in ['total', 'correct', 'accuracy']:
            torsion_acc[angle][label] = 0
    
    diff = np.array([0, 0])  # can have 0 - 180

    for res_num in target_residues.keys():
        if res_num > final_residue:
            continue
        elif res_num < 1:
            continue
        
        chi_check = True
        i = res_num - 1
        for j, angle in enumerate(angles):    
            if tor_masks[i, j]:
                diff = np.abs(native_angles[i, j] - target_angles[i, j])
                diff = np.min([diff, 360 - diff])
                
                torsion_acc[angle]['total'] += 1
                if j < 3:
                    if diff < thres:
                        torsion_acc[angle]['correct'] += 1
                else:
                    if chi_dependent:
                        if diff < thres and chi_check:
                            torsion_acc[angle]['correct'] += 1
                        else:
                            chi_check = False
                    else:
                        if diff < thres:
                            torsion_acc[angle]['correct'] += 1
    
    for angle in angles:
        if torsion_acc[angle]['total'] != 0:
            torsion_acc[angle]['accuracy'] = torsion_acc[angle]['correct'] / torsion_acc[angle]['total']
    
    return torsion_acc


def torsion_angle_loss(a, a_gt, tor_masks):
    """
    Computes the loss for torsion angles based on the difference between predicted angles and ground truth angles, 
    including penalties for angle normalization.
    
    Parameters
    ----------
    a : torch.Tensor
        The predicted torsion angles, shape [*, N, 7, 2].
    a_gt : torch.Tensor
        The ground truth torsion angles, shape [*, N, 7, 2].
    tor_masks : torch.Tensor
        A boolean tensor indicating the validity of torsion angles.
        
    Returns
    -------
    dict
        A dictionary containing the total loss, backbone loss, and sidechain loss.
    """
    bb_masks = torch.zeros(len(a), 7)
    bb_masks[:, :3] = True
    bb_masks = torch.logical_and(tor_masks, bb_masks)
    
    side_masks = torch.zeros(len(a), 7)
    side_masks[:, 3:] = True
    side_masks = torch.logical_and(tor_masks, side_masks)
    
    norm = torch.norm(a, dim=-1)

    a = a / norm.unsqueeze(-1)

    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    
    min_diff = diff_norm_gt**2
    
    min_diff_total = torch.sum(min_diff * tor_masks) / torch.sum(tor_masks)
    min_diff_bb    = torch.sum(min_diff * bb_masks)   / torch.sum(bb_masks)
    min_diff_side  = torch.sum(min_diff * side_masks) / torch.sum(side_masks)
    
    angle_norm = torch.abs(norm - 1)
     
    angle_norm_total = 0.02 * torch.sum(angle_norm * tor_masks) / torch.sum(tor_masks)
    angle_norm_bb    = 0.02 * torch.sum(angle_norm * bb_masks)   / torch.sum(bb_masks)
    angle_norm_side  = 0.02 * torch.sum(angle_norm * side_masks) / torch.sum(side_masks)
    
    total_loss = (min_diff_total + angle_norm_total).numpy()
    bb_loss    = (min_diff_bb + angle_norm_bb).numpy()
    side_loss  = (min_diff_side + angle_norm_side).numpy()
    
    loss = {}
    loss['total'] = total_loss
    loss['backbone'] = bb_loss
    loss['sidechain'] = side_loss
    
    return loss
