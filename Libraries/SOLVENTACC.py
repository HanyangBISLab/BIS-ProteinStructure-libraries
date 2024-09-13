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

res_types = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', \
             'LEU' : 'L', 'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', \
             'UNK' : '-', 'XAA' : 'X'}
res_map = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', \
           'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', 'UNK' : '-'}

res_map_three_word = list(res_types.keys())
res_map_one_word   = list(res_types.values())
restype_to_aaorder = {}
res_map_one_to_three = {}

for i,one_word in enumerate(res_map_one_word):
    res_map_one_to_three[one_word] = res_map_three_word[i]

for i, three_word in enumerate(res_map_three_word):
    if i == 21 : i = 20
    restype_to_aaorder[three_word] = i

SA_maximum_map = {'PHE' : 210, 'ILE' : 175, 'LEU' : 170, 'VAL' : 155, 'PRO' : 145, 'ALA' : 115, \
                  'GLY' : 75, 'MET' : 185, 'CYS' : 135, 'TRP' : 255, 'TYR' : 230, 'THR' : 140, \
                  'SER' : 115, 'GLN' : 180, 'ASN' : 160, 'GLU' : 190, 'ASP' : 150, 'HIS' : 195, \
                  'LYS' : 200, 'ARG' : 225, 'XAA' : 170}


atom_types = {"N":0,"CA":1,"C":2,"CB":3,"O":4,"CG":5,"CG1":6,"CG2":7,"OG":8,"OG1":9,"SG":10,"CD":11,"CD1":12,"CD2":13,"ND1":14,"ND2":15,"OD1":16,"OD2":17,"SD":18,\
            "CE":19,"CE1":20,"CE2":21,"CE3":22,"NE":23,"NE1":24,"NE2":25,"OE1":26,"OE2":27,"CH2":28,"NH1":29,"NH2":30,"OH":31,"CZ":32,"CZ2":33,"CZ3":34,"NZ":35,"OXT":36}

index_to_atom_name = {v: k for k, v in atom_types.items()}


def load_pickle_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def getTMscore(pdb_path1, pdb_path2):

    cmd = TMscore_path +' '+ pdb_path1 +' ' + pdb_path2
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


def convert_mmcif_to_dssp(mmcif_path,dssp_root):
    # Parse MMCIF file to obtain structure
    mmcif = mmcif_path.split('/')[-1]
    dssp = mmcif.replace('.cif', '.dssp')
    dssp_path = os.path.join(dssp_root, dssp)
    
    dssp_executable = "mkdssp"  # Change this if DSSP is installed with a different name
    subprocess.run([dssp_executable, "-i", mmcif_path, "-o", dssp_path])
    
def convert_mmcif_to_dssp_parallel(mmcif_paths, dssp_root, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Using list comprehension to create futures
        futures = [executor.submit(convert_mmcif_to_dssp, mmcif_path, dssp_root) for mmcif_path in mmcif_paths]

        # Wait for all futures to complete
        for future in futures:
            future.result()
            
def convert_pdb_to_dssp(pdb_path,dssp_root):
    # Parse MMCIF file to obtain structure
    pdb = pdb_path.split('/')[-1]
    dssp = pdb.replace('.pdb', '.dssp')
    dssp_path = os.path.join(dssp_root, dssp)
    
    dssp_executable = "mkdssp"  # Change this if DSSP is installed with a different name
    subprocess.run([dssp_executable, "-i", pdb_path, "-o", dssp_path])

        
def contains_non_numeric(input_string):
    for char in input_string:
        if not char.isnumeric():
            return True
    return False


def create_structure_from_feature(sequence, all_atom_positions, all_atom_mask, structure_id="pred", model_id=0, chain_id="A"):
    structure = Structure.Structure(structure_id)
    model = Model.Model(model_id)
    chain = Chain.Chain(chain_id)

    for i in range(len(sequence)):
        #print(residue_index)
        residue_id = (' ', i + 1, ' ')
        residue = Residue.Residue(residue_id, sequence[i], '')

        for j in range(all_atom_positions.shape[1]):
            if all_atom_mask[i, j] == 1:  # Only consider atoms where mask is 1
                atom_name = index_to_atom_name.get(j, f"X{j + 1}")  # Default to "X{j + 1}" if not found
                atom_coords = all_atom_positions[i, j]
                atom = Atom.Atom(atom_name, atom_coords, 1.0, 1.0, '', atom_name, j + 1, 'C')
                residue.add(atom)

        chain.add(residue)
    model.add(chain)
    structure.add(model)

    # Compute SASA using ShrakeRupley
    sr = ShrakeRupley()
    sr.compute(structure, level="R")

    # Add SASA values to each residue's xtra attribute
    for res in structure.get_residues():
        if 'EXP_NACCESS' in res.xtra:
            res.sasa = res.xtra['EXP_NACCESS']
        else:
            res.sasa = None

    return structure


def get_binned_solvents_for_input_feature(SA_root, feature, protein_name, save = False):
    
    thres = np.linspace(0,1,11)[:-1]    
    
     #1. extract features
    sequence = feature['sequence'][0].decode('utf-8')
    residue_indices = feature['residue_index']
    all_atom_positions = feature['all_atom_positions']
    all_atom_mask      = feature['all_atom_mask']

    residue_length = len(residue_indices)

    #2. reconstruct structure object
    reconstructed_structure = create_structure_from_feature(sequence, all_atom_positions, all_atom_mask)
    sr = ShrakeRupley()
    sr.compute(reconstructed_structure, level="R")

    #print(sequence)
    #print()

    binned_solvents     = np.zeros([residue_length,10])
    real_solvents       = np.zeros([residue_length])
    solvents_mask       = np.zeros([residue_length])        

    #3. get SA from reconstructed structure..
    for model in reconstructed_structure:
        for chain in model:
            for residue in chain:
                if residue.sasa is not None:
                    #print(f"Residue {residue.get_id()[1]}, {residue.resname} SA: {residue.sasa}")
                    #print(residue.resname,end = '')

                    if residue.resname.upper() in res_map_one_to_three.keys(): restype = res_map_one_to_three[residue.resname.upper()]
                    else : restype = 'XAA'
                    resnum = int(residue.get_id()[1]) -1 
                    
                    real_solvents[resnum] = residue.sasa
                    RSA = np.min([1, (residue.sasa / SA_maximum_map[restype])])
                    binned_solvents[resnum, (RSA>=thres).sum()-1] = True
                    binned_solvents[resnum, 0] = False
                    solvents_mask[resnum] = True
                    
    solvents = {}
    solvents['binned'] = binned_solvents
    solvents['real']   = real_solvents
    solvents['mask']   = solvents_mask
    sa_path = os.path.join(SA_root, protein_name +'.pkl')
    
    if save == True:
        with open(sa_path, "wb") as file:
            pickle.dump(solvents, file)   
            
    return solvents


def get_binned_solvents_for_casp_target(SA_root, dssp_file_path, residue_length, sequence, save = False):
    parser = parseDSSP(dssp_file_path)
    parser.parse()
    pddict = parser.dictTodataframe()
    chains   = set(pddict['rcsb_given_chain'].values)
    
    dssp = dssp_file_path.split('/')[-1].split('.dssp')[0].split('_')[0]
    
    thres = np.linspace(0,1,11)[:-1]
 
    chain_specific_pddict = pddict

    inscodes = chain_specific_pddict['inscode'].values
    resnums = chain_specific_pddict['resnum'].values
    restypes = chain_specific_pddict['aa'].values
    solvents = chain_specific_pddict['acc'].values
    
    binned_solvents     = np.zeros([residue_length,10])
    real_solvents       = np.zeros([residue_length])
    solvents_mask       = np.zeros([residue_length])    
    
    k = int(inscodes[0])-1
    i = 0
    while True:
        if i >= len(restypes) or k >= len(sequence): 
            break
                
        if restypes[i] == '!':
            k += int(inscodes[i+1]) - int(inscodes[i-1])
            i+=1
            #print('!!!!!@@@@@!!!!!!', end=' ')
            continue    
        if restypes[i].upper() in res_map_one_to_three.keys(): restype = res_map_one_to_three[restypes[i].upper()]
        else : restype = 'XAA'

        RSA = np.min([1,int(solvents[i]) / SA_maximum_map[restype] ])
        binned_solvents[k, (RSA>=thres).sum()-1] = True
        binned_solvents[k, 0] = False
        real_solvents[k] = int(solvents[i])
        solvents_mask[k] = True

        i+=1
        k+=1

    solvents = {}
    solvents['binned'] = binned_solvents
    solvents['real']   = real_solvents
    solvents['mask']   = solvents_mask
    
    sa_path = os.path.join(SA_root, dssp.split('.dssp')[0].split('_')[0] + '.pkl')
        
    if save == True:
        with open(sa_path, "wb") as file:
            pickle.dump(solvents, file)   
    return solvents



def get_binned_solvents_for_netsurfp(SA_root,  netsurfp_SA_path, residue_length, sequence, save = False):
    thres = np.linspace(0,1,11)[:-1]
     
    netsurfp = netsurfp_SA_path.split('/')[-1]
    
    binned_solvents     = np.zeros([residue_length,10])
    real_solvents       = np.zeros([residue_length])
    solvents_mask       = np.zeros([residue_length])  
    
    with open(netsurfp_SA_path, 'rb') as file:  # 'rb'는 바이너리 읽기 모드
        netsurfp_SA = pickle.load(file)
    
    for resnum in range(residue_length):
        if resnum >= residue_length: continue
        RSA = netsurfp_SA[resnum].sum()
        binned_solvents[resnum, (RSA>=thres).sum()-1] = True
        binned_solvents[resnum, 0] = False
        
        real_solvents[resnum] = RSA
        solvents_mask[resnum] = True
        
    solvents = {}
    solvents['binned'] = binned_solvents
    solvents['real']   = real_solvents
    solvents['mask']   = solvents_mask
    
    sa_path = os.path.join(SA_root, netsurfp)
    
    if save == True:
        with open(sa_path, "wb") as file:
            pickle.dump(binned_solvents, file)   
    return solvents


def process_feature_adding_SA(candidate, former_feature_root, binned_SA_root, after_feature_root):
    try:
        print(candidate)
        feature_path = os.path.join(former_feature_root, candidate)
        code_name, chain = candidate.split('.')[0].split('_')
        protein_name = code_name + '_' + chain

        with open(feature_path, 'rb') as file:  # 'rb'는 바이너리 읽기 모드
            feature = pickle.load(file)

        solvents = get_binned_solvents_for_input_feature(binned_SA_root, feature, protein_name, save=True)

        feature_with_binned_SA_path = os.path.join(after_feature_root, candidate)
        feature_with_binned_SA = feature
        feature_with_binned_SA['solvent_acc_binned'] = solvents['binned']
        feature_with_binned_SA['solvent_acc_real'] = solvents['real']
        feature_with_binned_SA['solvent_acc_mask'] = solvents['mask']
    
        with open(feature_with_binned_SA_path, "wb") as file:
            pickle.dump(feature_with_binned_SA, file)

        return (candidate, None)  # 성공적으로 처리된 경우

    except Exception as e:
        return (candidate, e)  # 예외가 발생한 경우

def DSSP_to_SA_parerall(former_features, former_feature_root, binned_SA_root, after_feature_root):
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_feature_adding_SA, candidate, former_feature_root, binned_SA_root, after_feature_root)
            for candidate in former_features
        ]

        for future in tqdm(as_completed(futures), total=len(futures)):
            candidate, exception = future.result()
            if exception is not None:
                print(candidate, exception)
                
