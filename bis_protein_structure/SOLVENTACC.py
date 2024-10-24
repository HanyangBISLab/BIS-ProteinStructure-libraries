import numpy as np
import os
import shutil
import zipfile
import pickle
from tqdm import tqdm

from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBIO
from Bio.PDB import Structure, Model, Chain, Residue, Atom
from Bio.PDB.SASA import ShrakeRupley
from DSSPparser import parseDSSP

import concurrent.futures
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor

TMscore_path = './TMscore'

res_types = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
             'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
             'UNK': '-', 'XAA': 'X'}
res_map = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L',
           'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V', 'UNK': '-'}

res_map_three_word = list(res_types.keys())
res_map_one_word = list(res_types.values())
restype_to_aaorder = {}
res_map_one_to_three = {}

for i, one_word in enumerate(res_map_one_word):
    res_map_one_to_three[one_word] = res_map_three_word[i]

for i, three_word in enumerate(res_map_three_word):
    if i == 21: i = 20
    restype_to_aaorder[three_word] = i

SA_maximum_map = {'PHE': 210, 'ILE': 175, 'LEU': 170, 'VAL': 155, 'PRO': 145, 'ALA': 115,
                  'GLY': 75, 'MET': 185, 'CYS': 135, 'TRP': 255, 'TYR': 230, 'THR': 140,
                  'SER': 115, 'GLN': 180, 'ASN': 160, 'GLU': 190, 'ASP': 150, 'HIS': 195,
                  'LYS': 200, 'ARG': 225, 'XAA': 170}

atom_types = {"N": 0, "CA": 1, "C": 2, "CB": 3, "O": 4, "CG": 5, "CG1": 6, "CG2": 7, "OG": 8, "OG1": 9, "SG": 10, "CD": 11, "CD1": 12,
              "CD2": 13, "ND1": 14, "ND2": 15, "OD1": 16, "OD2": 17, "SD": 18, "CE": 19, "CE1": 20, "CE2": 21, "CE3": 22, "NE": 23,
              "NE1": 24, "NE2": 25, "OE1": 26, "OE2": 27, "CH2": 28, "NH1": 29, "NH2": 30, "OH": 31, "CZ": 32, "CZ2": 33, "CZ3": 34,
              "NZ": 35, "OXT": 36}

index_to_atom_name = {v: k for k, v in atom_types.items()}


def load_pickle_file(file_path):
    """Load data from a pickle file.

    Parameters
    ----------
    file_path : str
        The path to the pickle file.

    Returns
    -------
    data
        The data loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data


def getTMscore(pdb_path1, pdb_path2):
    """Calculate the TM-score between two PDB files.

    Parameters
    ----------
    pdb_path1 : str
        Path to the first PDB file.
    pdb_path2 : str
        Path to the second PDB file.

    Returns
    -------
    float
        The TM-score between the two structures.
    """
    cmd = TMscore_path + ' ' + pdb_path1 + ' ' + pdb_path2
    p = subprocess.Popen(cmd,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,
                         shell=True)
    
    stdout, stderr = p.communicate()
    outputs = stdout.decode().split('\n')
    tm = 0.0
    for line in outputs:
        if line.startswith('TM-score'):
            tm = float(line.split(' ')[5])
    return tm


def convert_mmcif_to_dssp(mmcif_path, dssp_root):
    """Convert a MMCIF file to DSSP format.

    Parameters
    ----------
    mmcif_path : str
        Path to the MMCIF file.
    dssp_root : str
        Directory where the DSSP file will be saved.
    """
    mmcif = mmcif_path.split('/')[-1]
    dssp = mmcif.replace('.cif', '.dssp')
    dssp_path = os.path.join(dssp_root, dssp)
    
    dssp_executable = "mkdssp"  # Change this if DSSP is installed with a different name
    subprocess.run([dssp_executable, "-i", mmcif_path, "-o", dssp_path])
    

def convert_mmcif_to_dssp_parallel(mmcif_paths, dssp_root, max_workers=4):
    """Convert multiple MMCIF files to DSSP format in parallel.

    Parameters
    ----------
    mmcif_paths : list of str
        List of paths to the MMCIF files.
    dssp_root : str
        Directory where the DSSP files will be saved.
    max_workers : int, optional
        Maximum number of workers for parallel processing (default is 4).
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Using list comprehension to create futures
        futures = [executor.submit(convert_mmcif_to_dssp, mmcif_path, dssp_root) for mmcif_path in mmcif_paths]

        # Wait for all futures to complete
        for future in futures:
            future.result()


def convert_pdb_to_dssp(pdb_path, dssp_root):
    """Convert a PDB file to DSSP format.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.
    dssp_root : str
        Directory where the DSSP file will be saved.
    """
    pdb = pdb_path.split('/')[-1]
    dssp = pdb.replace('.pdb', '.dssp')
    dssp_path = os.path.join(dssp_root, dssp)
    
    dssp_executable = "mkdssp"  # Change this if DSSP is installed with a different name
    subprocess.run([dssp_executable, "-i", pdb_path, "-o", dssp_path])


def contains_non_numeric(input_string):
    """Check if the input string contains non-numeric characters.

    Parameters
    ----------
    input_string : str
        The string to be checked.

    Returns
    -------
    bool
        True if the string contains non-numeric characters, False otherwise.
    """
    for char in input_string:
        if not char.isnumeric():
            return True
    return False


def create_structure_from_feature(sequence, all_atom_positions, all_atom_mask, structure_id="pred", model_id=0, chain_id="A"):
    """Create a Biopython Structure object from sequence and atomic features.

    Parameters
    ----------
    sequence : str
        The amino acid sequence.
    all_atom_positions : np.ndarray
        Array of shape (n_residues, n_atoms, 3) containing atom coordinates.
    all_atom_mask : np.ndarray
        Array of shape (n_residues, n_atoms) indicating which atoms are present.
    structure_id : str, optional
        The ID of the structure (default is "pred").
    model_id : int, optional
        The model ID (default is 0).
    chain_id : str, optional
        The chain ID (default is "A").

    Returns
    -------
    Structure
        A Biopython Structure object representing the protein.
    """
    structure = Structure.Structure(structure_id)
    model = Model.Model(model_id)
    chain = Chain.Chain(chain_id)

    for i in range(len(sequence)):
        res_name = res_map_one_to_three.get(sequence[i], "UNK")
        res_id = (' ', i + 1, ' ')  # Residue ID (hetero flag, resSeq, icode)
        residue = Residue.Residue(res_id, res_name, ' ')
        
        # Create atoms for the residue
        for j, atom_position in enumerate(all_atom_positions[i]):
            if all_atom_mask[i, j] == 1:  # Only include present atoms
                atom_name = index_to_atom_name[j]
                atom = Atom.Atom(atom_name, atom_position, 1.0, 1.0, ' ', res_id[1], res_name, res_id)
                residue.add(atom)
        
        chain.add(residue)
    
    model.add(chain)
    structure.add(model)
    return structure


def extract_coords_from_pdb(pdb_path):
    """Extract atomic coordinates from a PDB file.

    Parameters
    ----------
    pdb_path : str
        Path to the PDB file.

    Returns
    -------
    tuple
        A tuple containing the sequence and an array of atomic coordinates.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", pdb_path)

    all_atom_positions = []
    sequence = []
    
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_id()[0] == ' ':  # Exclude hetero atoms
                    res_name = res.get_resname()
                    sequence.append(res_name)
                    res_positions = []
                    
                    for atom in res:
                        res_positions.append(atom.get_coord())
                    
                    all_atom_positions.append(res_positions)
    
    return sequence, np.array(all_atom_positions)
