import os
from glob import glob
import numpy as np
from termcolor import colored
from scipy.spatial.distance import pdist, squareform


def get_original_path(models):
  ret = []
  for pdb in models:
    if os.path.islink(pdb):
      ret.append(os.readlink(pdb))
    else:
      ret.append(pdb)
    
  return ret


def get_pdbs(target, submit=False):
  
  if not submit:
    return np.sort(glob(f'./unrelaxed_model_?.pdb'))
  
  pdbs = glob('./model?.pdb')
  pdbs.sort()
  
  return get_original_path(pdbs)
  
  
def get_pdbs_multi(path):
    afs = np.sort(glob(f'{path}/model*pdb'))
    return afs
  
  
def get_pkls(target, submit=False):

  if not submit:
    return np.sort(glob(f'./result_model_?.pkl'))
  
  outputs = []
  models = get_pdbs(target, submit)
  for model in models:
    splits = model.split('/')
    directory = '/'.join(splits[:-1])

    last = splits[-1]
    output = f'result{last[7:-4]}.pkl'     
    output = os.path.join(directory, output)
    
    if os.path.exists(output):
      outputs.append(output)
      
  return outputs
    

def get_pkls_multi(path):
    afs = np.sort(glob(f'{path}/model*pkl'))
    return afs
  
  
def get_Neff(msa_feat):
    theta = 0.38
    #msa_feat = make_msa(msas,naln,nres)
    W = 1 / (1+np.sum(squareform(pdist(msa_feat,'hamming') < theta), 0))
    Neff_1 = np.log(np.sum(W))
    
    theta = 0.38
    W = 1 / (1+np.sum(squareform(pdist(msa_feat,'hamming') < theta), 0))
    Neff_2 = np.log2(np.sum(W))# / np.sqrt(nres)
    
    return Neff_1, Neff_2

  
def get_fasta(target):
  fasta_path = f'/gpfs/deepfold/casp/casp15/fasta/{target}.fasta'
  with open(fasta_path) as f:
    return f.readlines()[1].strip()

  
def get_fasta_multi(fasta_path):
    seqs = {}
    with open(fasta_path) as f:
        for line in f.readlines():
            if line.startswith('>'):
                key = line.split()[0][1:]
            else:
                seqs[key] = line.strip()
    return seqs

  
def print_fasta(target):
  fasta = get_fasta(target)
  
  n = len(fasta)
  idx = ''
  k = 1
  for i in range(1, n+1):
    if i % 10 == 0:
      
      if k > 10: 
        idx = idx[:-1]

      idx += f'{k}'
      k += 1
    else:
      idx += ' '
  
  print(idx)
  print(fasta)
  
  
def print_secondary(secondary):
  new_s = ''
  
  color_dict = {
    'H' : 'red',
    'B' : 'blue',
    'E' : 'blue',
  }
  
  for s in secondary:
    if s in list(color_dict.keys()):
      new_s += colored(s, color_dict[s])
    else:
      new_s += colored(s, 'white')
  print(new_s)

  
def get_secondary(fasta, pdb):
  dssp_ret = os.popen(f'dssp3 {pdb} 2>/dev/null').readlines()

  start = False
  chain = ''
  secondary = ''
  for line in dssp_ret:
    if line.startswith('  #'):
        start = True
        continue
    
    if start:
        chain += line[13]
        secondary += line[16]
        
  assert len(chain) == len(secondary)
  assert fasta == chain
  
  return secondary
