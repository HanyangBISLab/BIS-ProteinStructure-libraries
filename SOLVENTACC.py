from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBIO


import concurrent.futures
import subprocess

import os
import shutil
import zipfile

from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed

TMscore_path = './TMscore'

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

res_types = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', \
           'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', 'UNK' : '-', 'XAA' : 'X'}
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

def contains_non_numeric(input_string):
    for char in input_string:
        if not char.isnumeric():
            return True
    return False

def get_normalized_solvents_for_input_feature(dssp_file_path, save = False):
    parser = parseDSSP(dssp_file_path)
    parser.parse()
    pddict = parser.dictTodataframe()
    chains   = set(pddict['rcsb_given_chain'].values)
    #resnums = pddict['inscode'].values
    #restypes = pddict['aa'].values
    #solvents = pddict['acc'].values
    
    for chain in chains:
        if len(chain) == 0 : continue
        if chain == '' : continue
        
        chain_specific_pddict = pddict[pddict['rcsb_given_chain'] == chain]
        chain_specific_pddict = pddict
        resnums = chain_specific_pddict['inscode'].values
        restypes = chain_specific_pddict['aa'].values
        solvents = chain_specific_pddict['acc'].values
    
        numres_exist = 0
        residue_length = 0
        for resnum in resnums:
            if contains_non_numeric(resnum) : continue
            if len(resnum) < 1 : continue
            if int(resnum)>= residue_length : residue_length = int(resnum)

        normalized_solvents = np.zeros([residue_length,21])

        for i in range(len(resnums)):
            if contains_non_numeric(resnums[i]) : continue
            if len(resnums[i]) < 1 : continue
            resnum = int(resnums[i])-1

            if restypes[i].upper() in res_map_one_to_three.keys():
                restype = res_map_one_to_three[restypes[i].upper()]
            else : 
                restype = 'XAA'
            aaorder = restype_to_aaorder[restype]
            normalized_solvents[resnum,aaorder] = np.min([1,int(solvents[i]) / SA_maximum_map[restype] ])
            numres_exist += 1
            
        #sa_path = os.path.join(SA_root, dssp.replace('.dssp','.pkl'))
        sa_path = os.path.join(SA_root, dssp.split('.dssp')[0]+f'_{chain.lower()}.pkl')
        #sa_path = os.path.join(SA_root, dssp.split('.dssp')[0]+f'.pkl')
        
        if save == True:
            with open(sa_path, "wb") as file:
                pickle.dump(normalized_solvents, file)   
            
    return normalized_solvents, numres_exist


def parse_mmcif(path, file_id, chain_id, alignment_dir):
    with open(path, 'r') as f:
        mmcif_string = f.read()

    mmcif_object = mmcif_parsing.parse(
        file_id=file_id, mmcif_string=mmcif_string)
    
    # Crash if an error is encountered. Any parsing errors should have
    # been dealt with at the alignment stage.
    if(mmcif_object.mmcif_object is None):
        raise list(mmcif_object.errors.values())[0]

    mmcif_object = mmcif_object.mmcif_object

    data = data_pipeline.process_mmcif(
        mmcif=mmcif_object,
        alignment_dir=alignment_dir,
        chain_id=chain_id)
    
    return data


def generate_feature_dict(
    tag,
    seq,
    fasta_path,
    alignment_dir):
    
    local_alignment_dir = os.path.join(alignment_dir, tag)
    feature_dict = data_processor.process_fasta(
        fasta_path=fasta_path, alignment_dir=local_alignment_dir)

    return feature_dict


def parse_fasta(data):
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]

    tags = [t.split()[0] for t in tags]

    return tags, seqs


def read_fasta(file_path):
    with open(file_path, 'r') as fasta_file:
        sequence_id = None
        sequence = ''
        for line in fasta_file:
            line = line.strip()
            if line.startswith('>'):
                if sequence_id is not None:
                    print(f'Sequence ID: {sequence_id}')
                    print(f'Sequence: {sequence}')
                    print()  # Add an empty line between sequences
                sequence_id = line[1:]
                sequence = ''
            else:
                sequence += line
        # Print the last sequence in the file
        if sequence_id is not None:
            print(f'>{sequence_id}')
            print(f'{sequence}')
            
            
def cluster_and_save(sequences, threshold=0.3, output_file="cluster_info.json"):
    n_sequences = len(sequences)

    # 클러스터링을 위한 유사도 리스트 생성
    similarity_list = []

    # 시퀀스 간의 유사도 계산
    for i in tqdm(range(n_sequences), desc="Calculating sequence similarity"):
        similarity_row = []
        for j in range(i + 1, n_sequences):
            similarity = calculate_identity(sequences[i], sequences[j])
            similarity_row.append(similarity)
        similarity_list.append(similarity_row)

    # 클러스터링
    clustering = AgglomerativeClustering(n_clusters=None, affinity="precomputed", linkage="average", distance_threshold=threshold)
    clusters = clustering.fit_predict(similarity_list)

    # 클러스터 번호와 해당 클러스터에 속하는 각 fasta 파일의 이름 저장
    cluster_info = {}
    for i, cluster_label in enumerate(clusters):
        sequence_name = sequences[i].id  # fasta 파일의 이름(필요에 따라 수정)
        if cluster_label not in cluster_info:
            cluster_info[cluster_label] = []
        cluster_info[cluster_label].append(sequence_name)

    # 결과 저장
    with open(output_file, "w") as f:
        json.dump(cluster_info, f, indent=4)
