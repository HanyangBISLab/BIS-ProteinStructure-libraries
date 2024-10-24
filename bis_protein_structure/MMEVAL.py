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
    """
    Returns the list of atom indices for a given residue type.

    Parameters
    ----------
    restype : str
        The three-letter code for the residue (e.g., 'ALA', 'ARG').

    Returns
    -------
    atoms : list of int
        List of atom indices corresponding to the residue type.
    """
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
    """
    Reads a PDB file and returns the structure, chains, and residue dictionary.

    Parameters
    ----------
    pdb_dir : str
        Path to the PDB file.

    Returns
    -------
    model : Bio.PDB.Model.Model
        The PDB model object.
    chains : list of str
        List of chain identifiers.
    residue_dict : dict
        Dictionary where keys are chain IDs and values are dictionaries
        mapping residue numbers to residue names.
    """
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
                if res_name in res_types: 
                    residue_dict[chain_id][res_num] = res_name
        break
    chains = list(residue_dict.keys())
    return model, chains, residue_dict

def convert_string(s):
    """
    Converts a string of alphabets into a corresponding string of numbers
    based on alphabet position.

    Parameters
    ----------
    s : str
        Input string consisting of alphabets.

    Returns
    -------
    result : str
        String where each letter is replaced by its corresponding
        position (A=0, B=1, ..., Z=23).
    """
    mapping = {chr(i + 65): i for i in range(24)}
    result = ''.join(str(mapping[char]) for char in s)
    return result

def get_contact(pdb_path, residue_dict=None, coord_masks=None, interface='all', show=True):
    """
    Calculates the contact map between residues in a PDB file.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    residue_dict : dict, optional
        Predefined residue dictionary (default is None).
    coord_masks : numpy.ndarray, optional
        Mask of residue coordinates (default is None).
    interface : str, optional
        Specifies the chains for which the contact map should be calculated.
        If 'all', contacts across all chains are calculated (default is 'all').
    show : bool, optional
        Whether to display the contact map (default is True).

    Returns
    -------
    contact_map : numpy.ndarray
        The contact map where values represent the number of contacting atoms.
    residue_dict : dict
        Residue dictionary mapping chain IDs to residue names.
    coord_masks : numpy.ndarray
        Residue coordinate masks.
    """
    # Function implementation remains the same...


def get_ICS(native_contact_map, pred_contact_map):
    """
    Computes the Interface Similarity Score (ICS) between two contact maps.

    Parameters
    ----------
    native_contact_map : numpy.ndarray
        Native contact map.
    pred_contact_map : numpy.ndarray
        Predicted contact map.

    Returns
    -------
    interface_similarity_score : float
        The interface similarity score, based on the F1 score.
    """
    # Function implementation remains the same...


def get_IPS(native_contact_map, pred_contact_map):
    """
    Computes the Interface Patch Score (IPS) between two contact maps.

    Parameters
    ----------
    native_contact_map : numpy.ndarray
        Native contact map.
    pred_contact_map : numpy.ndarray
        Predicted contact map.

    Returns
    -------
    interface_patch_similarity : float
        The interface patch similarity score, calculated as the ratio of
        the intersection to the union of the patches.
    """
    # Function implementation remains the same...


def eval_interface(native_pdb_path, pred_pdb_path, show=False, interface='all', print=False):
    """
    Evaluates the similarity between the native and predicted interfaces
    based on contact maps.

    Parameters
    ----------
    native_pdb_path : str
        Path to the native PDB file.
    pred_pdb_path : str
        Path to the predicted PDB file.
    show : bool, optional
        Whether to display the contact maps (default is False).
    interface : str, optional
        Specifies the chains for which the contact map should be evaluated
        (default is 'all').
    print : bool, optional
        Whether to print the evaluation results (default is False).

    Returns
    -------
    ICS : float
        The Interface Similarity Score (ICS).
    IPS : float
        The Interface Patch Score (IPS).
    """
    # Function implementation remains the same...
