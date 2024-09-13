from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from Bio.PDB import Dice
import os
import pickle
import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')
from tqdm import tqdm
import re
import numpy as np

res_types = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', \
           'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', 'UNK' : '-'}

def extract(structure, chain_id, start, end, filename):
    """Write out selected portion to filename."""
    sel = Dice.ChainSelector(chain_id, start, end)
    io = MMCIFIO()
    io.set_structure(structure)
    io.save(filename, sel)
    
    
def split_the_pdb(cif_path, pdb, chain):
    p = MMCIFParser(QUIET=True)
    s = p.get_structure(f'{pdb}_{chain}', pdb_path)
    for m in s:
        for cn, c in enumerate(m):
            if chain != c.id : continue
                
            pdb_seq =''
            tmp_res = list()

            for i, res in enumerate(c):
                if res.resname in res_types:
                    pdb_seq += res_types[res.resname]
                    tmp_res.append(res.get_id()[1])

            if len(pdb_seq) == 0:
                continue

            starting_index = int(tmp_res[0])
            real_len = len(pdb_seq)
            index_len = real_len - 1

            filename = f'{new_path}/{pdb}_{c.id}.cif'
            extract(structure=s, chain_id=c.id, start=starting_index, end=starting_index + index_len, filename=filename)
