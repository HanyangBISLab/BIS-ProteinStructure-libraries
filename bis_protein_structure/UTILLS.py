import os
import json
import random
import itertools
from itertools import product
from Bio import SeqIO
from Bio.PDB.PDBParser import PDBParser
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import tensorflow as tf
import pickle
import torch

import subprocess
import Modules.TORSION2 as TORSION2
import Modules.RAMACHANDRAN as RAMACHANDRAN

atom_types = {"N":0,"CA":1,"C":2,"CB":3,"O":4,"CG":5,"CG1":6,"CG2":7,"OG":8,"OG1":9,"SG":10,"CD":11,"CD1":12,"CD2":13,"ND1":14,"ND2":15,"OD1":16,"OD2":17,"SD":18,\
            "CE":19,"CE1":20,"CE2":21,"CE3":22,"NE":23,"NE1":24,"NE2":25,"OE1":26,"OE2":27,"CH2":28,"NH1":29,"NH2":30,"OH":31,"CZ":32,"CZ2":33,"CZ3":34,"NZ":35,"OXT":36}
angle_types = {0 : 'OMEGA', 1 : 'PHI' , 2 : 'PSI', 3 : 'CHI1', 4 : 'CHI2', 5 : 'CHI3', 6 : 'CHI4'}
tor_types = {'OMEGA' : 0, 'PHI' : 1, 'PSI' : 2, 'CHI1' : 3, 'CHI2' : 4 , 'CHI3' : 5, 'CHI4' : 6}
plot_types = {'PHI_PSI' : [1,2], 'PHI_CHI1' : [1,4], 'PSI_CHI1' : [2,4], 'CHI1_CHI2' : [4, 5]}


def getBestModel(sequenceID:str, 
                 root='/user/deepfold/casp/casp13/'):
    json_dir = os.path.join(root, f'{sequenceID}/ranking_debug.json')
    with open(json_dir, 'r') as f: 
        ranking_data = json.load(f)
    ranking = ranking_data['order']
    best = ranking[0][6]

    return best


class sequenceInfo(object):
    
    def __init__(self, 
                 sequenceID:str, 
                 root='/user/deepfold/casp/casp13/'):
        
        self.sequenceID = sequenceID
        self.root = root
        
    def getDirectory(self) :
        fasta_dir = os.path.join(self.root, f'{self.sequenceID}/{self.sequenceID}.fasta')
        native_pdb_dir = os.path.join(self.root, f'native/{self.sequenceID}.pdb')
        
        year = self.root.split('/')[4]
        
        af_pkl_dir, af_pdb_dir, subs_pdb_dir = {}, {}, {}
        for f in os.listdir(os.path.join(self.root, self.sequenceID)):
            if f[:6] == 'result':
                af_pkl_dir[f[13]] = os.path.join(self.root, self.sequenceID, f)

        for f in os.listdir(os.path.join(self.root, self.sequenceID)):
            if f[:9] == 'unrelaxed':
                af_pdb_dir[f[16]] = os.path.join(self.root, self.sequenceID, f)
        
        if self.sequenceID in os.listdir('/gpfs/deepfold/casp/'+year +'/TS_as_submitted/'):
          
            for f in os.listdir('/gpfs/deepfold/casp/'+year +'/TS_as_submitted/' + self.sequenceID +'/'):
                if len(self.sequenceID) == 5:
                    if f[5:10] not in subs_pdb_dir.keys():
                        subs_pdb_dir[f[5:10]] = {}
                        subs_pdb_dir[f[5:10]][f[11]] = '/gpfs/deepfold/casp/'+ year +'/TS_as_submitted/' + self.sequenceID + '/' + f
                elif len(self.sequenceID) == 7:
                    if f[7:12] not in subs_pdb_dir.keys():
                        subs_pdb_dir[f[7:12]] = {}
                        subs_pdb_dir[f[7:12]][f[13]] = '/gpfs/deepfold/casp/'+ year +'/TS_as_submitted/' + self.sequenceID + '/' + f

        return {'fasta': fasta_dir, 
                'native_pdb': native_pdb_dir, 
                'af_pkl': af_pkl_dir, 
                'af_pdb': af_pdb_dir, 
                'subs_pdb': subs_pdb_dir}
    
    
    def getBestModel(self):

        json_dir = os.path.join(self.root, f'{self.sequenceID}/ranking_debug.json')
        with open(json_dir, 'r') as f: 
            ranking_data = json.load(f)
        ranking = ranking_data['order']
        best = ranking[0][6]

        return best
    
    
class Calculators(object):
    
    def __init__(self, 
                 sequenceID:str,  
                 groupNum=False, 
                 root='/user/deepfold/casp/casp13/'):
        
        self.sequenceID = sequenceID
        self.root = root
        self.groupNum = groupNum
        self.sInfo = sequenceInfo(self.sequenceID, self.root)
        self.d_dir = self.sInfo.getDirectory()
        self.year = root.split('/')[4]
        
        self.sampleGroup = self.sampleGroup(group_num=self.groupNum)
        self.validID = self.getValidID()
        self.parser = PDBParser(PERMISSIVE=1)
        
        self.readPDB = self.readPDB
        
        self.residues, self.chain = self.readPDB(self.d_dir['native_pdb'])
        
        self.final_residue = list(self.residues)[-1]
        self.coords, self.coords_mask, _ = TORSION2.get_coordinates(self.final_residue ,self.residues, self.chain)
        self.interacted = TORSION2.getInteracted(self.final_residue, self.residues, self.coords, self.coords_mask)
        
        self.tor_masks, self.tor_angles = TORSION2.get_torsion(self.coords_mask, self.coords, self.residues, as_tensor = True)
        self.tor_alter_angles = torch.where(self.tor_masks,self.tor_angles+180, self.tor_angles)
        self.tor_sincos, self.tor_alter_sincos = TORSION2.angles_to_sincos(self.tor_angles), TORSION2.angles_to_sincos(self.tor_alter_angles)
        
        
    def sampleGroup(self, 
                    group_num:int):
        
        groups = [name for name in self.d_dir["subs_pdb"].keys()]
        if group_num > 0:
            random.shuffle(groups)
            groups = groups[:group_num]
  
        return groups


    def getValidID(self):

        seq = SeqIO.read(self.d_dir["fasta"], 'fasta')

        validID = []
        for res_id, res_name in enumerate(seq.seq):
            validID.append(res_id+1)
 
        return validID

    
    def readPDB(self, 
                pdb_dir:dir):

        structure = self.parser.get_structure(self.sequenceID, pdb_dir)

        residues = {}
        for model_id in structure:
            for chain_id in model_id:
                chain = model_id[chain_id.id]
                for residue in chain_id:
                    res_name = residue.resname.strip()
                    res_id = residue.id[1]
                    residues[res_id] = res_name

        return residues, chain
      
    def getPairwiseDistance(self, coords):

        pairwise_dist = squareform(pdist(coords, 'euclidean'))

        return pairwise_dist
      
      
    def show_distogram(self, coords, coords_mask, contact = True):
        plt.figure(figsize=(18,6))
        
        coords = np.where(coords_mask[:,3],coords[:,3],np.nan)
        
        logit = self.getPairwiseDistance(coords)
        
        if contact == True: logit = np.where(logit<8, 1, 0)
        
        plt.matshow(logit, cmap='viridis_r', fignum=False)
        plt.title('contact map')
    
        plt.show()
        
        return logit
      
        
    def getSingleTMscore(self, pdb_dir:dir):

        cmd = 'tmscore'+' '+ self.d_dir['native_pdb'] +' ' + pdb_dir
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
    
    def getTMscore(self):

        d_tm = {}
        groups = self.sampleGroup

        # native TMscore
        d_tm['native_pdb'] = self.getSingleTMscore(self.d_dir["native_pdb"])

        # model TMscore
        d_tm['af_pdb'] = {}
        for model, model_dir in self.d_dir["af_pdb"].items():
            d_tm['af_pdb'][model] = self.getSingleTMscore(model_dir)

        # submissions TMscore
        d_tm['subs_pdb'] = {}
        for group in groups:
            
            if group not in d_tm.keys():
                d_tm['subs_pdb'][group] = {}
            d_model = self.d_dir["subs_pdb"][group]
            
            for model, model_dir in d_model.items():
                d_tm['subs_pdb'][group][model] = self.getSingleTMscore(model_dir)

            
        return d_tm
        
    def getSingleTorsion(self, pdb_path, use_pkl = False, loss_type = 'square', chi_dependent = False): 
        
        if use_pkl == False:
            target_residues, target_chain = self.readPDB(pdb_path)
            final_residue = np.min([list(self.residues)[-1], list(target_residues)[-1]])
            target_coords, target_coords_mask, _ = TORSION2.get_coordinates(final_residue, target_residues, target_chain)
            target_tor_masks, target_tor_angles  = TORSION2.get_torsion(target_coords_mask, target_coords, target_residues, as_tensor = True)
        
            target_tor_masks = torch.logical_and(self.tor_masks[:final_residue], target_tor_masks)
            target_tor_alter_angles = torch.where(target_tor_masks,target_tor_angles+180,target_tor_angles)
            target_tor_alter_angles = torch.where(target_tor_alter_angles>180,target_tor_alter_angles-360 ,target_tor_alter_angles)
        
            target_tor_angles = TORSION2.sidechain_sym_angle(target_residues, target_tor_masks, self.tor_angles[:final_residue], target_tor_angles, target_tor_alter_angles)
            target_tor_sincos = TORSION2.angles_to_sincos(target_tor_angles)
            loss = TORSION2.torsion_angle_each_loss(target_tor_sincos, self.tor_sincos[:final_residue],target_tor_masks, loss_type, chi_dependent)
            
        if use_pkl == True:
            target_residues, target_chain = self.readPDB(pdb_path)
            final_residue = np.min([list(self.residues)[-1], list(target_residues)[-1]])
            
            target_coords, target_coords_mask, _ = TORSION2.get_coordinates(final_residue, target_residues, target_chain)
            target_tor_masks, target_tor_angles  = TORSION2.get_torsion(target_coords_mask, target_coords, target_residues, as_tensor = True)
        
            target_tor_masks = torch.logical_and(self.tor_masks[:final_residue], target_tor_masks)
            target_tor_alter_angles = torch.where(target_tor_masks,target_tor_angles+180,target_tor_angles)
            target_tor_alter_angles = torch.where(target_tor_alter_angles>180,target_tor_alter_angles-360 ,target_tor_alter_angles)
            
            af_tor_angles = np.load(self.d_dir['af_pkl']['1'], allow_pickle = True)['structure_module']['sidechains']['angles_sin_cos'][7]
            af_tor_angles = torch.tensor(np.arctan2(af_tor_angles[:,:,0], af_tor_angles[:,:,1]) * 180 / np.pi ) 
            af_tor_masks  = self.tor_masks
            af_tor_sincos = TORSION2.angles_to_sincos(af_tor_angles)
            
            
            target_tor_angles = TORSION2.sidechain_sym_angle(target_residues, target_tor_masks[:final_residue], af_tor_angles[:final_residue], \
                                                             target_tor_angles[:final_residue], target_tor_alter_angles[:final_residue])
            target_tor_sincos = TORSION2.angles_to_sincos(target_tor_angles)
            
            loss = TORSION2.torsion_angle_each_loss(target_tor_sincos, af_tor_sincos[:final_residue],target_tor_masks, loss_type, chi_dependent)
        
        return loss
    
    def getSingle_inter_torsion(self, pdb_path, loss_type = 'square', chi_dependent = True): 
      
        target_residues, target_chain = self.readPDB(pdb_path)
        final_residue = np.min([list(self.residues)[-1], list(target_residues)[-1]])
        target_coords, target_coords_mask, _ = TORSION2.get_coordinates(final_residue, target_residues, target_chain)
        target_interacted = TORSION2.getInteracted(final_residue,target_residues, target_coords,target_coords_mask)
        
        target_tor_masks, target_tor_angles, target_interacted_masks, target_interacted_angles = TORSION2.get_torsion_with_interacted(target_coords_mask, target_coords, target_residues, target_interacted, as_tensor =True)
        
        ### torsion angle original
        target_tor_masks = torch.logical_and(self.tor_masks[:final_residue], target_tor_masks)
        target_tor_alter_angles = torch.where(target_tor_masks,target_tor_angles+180,target_tor_angles)
        target_tor_alter_angles = torch.where(target_tor_alter_angles>180,target_tor_alter_angles-360 ,target_tor_alter_angles)
        
        target_tor_angles = TORSION2.sidechain_sym_angle(target_residues, target_tor_masks, self.tor_angles[:final_residue], target_tor_angles, target_tor_alter_angles)
        target_tor_sincos = TORSION2.angles_to_sincos(target_tor_angles)
        
        ### torsion angle with inter-residue
        target_interacted_sincos = TORSION2.angles_to_sincos(target_interacted_angles)
        
        original_loss = TORSION2.torsion_angle_each_loss(target_tor_sincos, self.tor_sincos[:final_residue],target_tor_masks, loss_type, chi_dependent)
        inter_loss    = TORSION2.inter_torsion_angle_loss(self.interacted_sincos[:final_residue], target_interacted_sincos, target_interacted_masks[:final_residue], loss_type)

        return {'original' : original_loss, 'inter' : inter_loss}
    
    
      
    def getSingle_inter_torsion_2(self, pdb_path, contact = False, contact_range = 8, out =True, angle = 'omega'): 
      
        target_residues, target_chain = TORSION2.readPDB(pdb_path)
        final_residue = list(target_residues)[-1]
        target_coords, target_coords_mask, _ = TORSION2.get_coordinates(final_residue, target_residues, target_chain)
        
        target_tor_masks, target_tor_angles  = TORSION2.get_torsion(target_coords_mask, target_coords, target_residues, as_tensor = False)
            
        
        target_inter_tor_masks, target_inter_tor_angles  = TORSION2.get_interacted_torsion(target_coords_mask, target_coords, target_residues, as_tensor = True, angle = angle)       
        
        coords = np.where(target_coords_mask[:,3],target_coords[:,3],np.nan)
        distogram = self.getPairwiseDistance(coords)
        
        if contact == True : 
          distogram = np.where(distogram<contact_range,1,0)
          final_masks = np.logical_and(target_inter_tor_masks, np.repeat(distogram[:,:,None],3,axis=-1))
          target_inter_tor_angles = np.where(final_masks, target_inter_tor_angles, np.nan)
          
        elif contact == False:
          target_inter_tor_angles = np.where(target_inter_tor_masks, target_inter_tor_angles, np.nan)  
        
        if out == True :
            plt.figure(figsize=(36,12))
            plt.suptitle(f'{self.sequenceID}', size = 20)

            if angle == 'all' or angle == 'omega':
              plt.subplot(1,3,1)
              plt.title(f'Inter-omega', size = 15)
              plt.matshow(target_inter_tor_angles[:,:,0], cmap='viridis_r', fignum=False)
              plt.colorbar(fraction=0.046, pad=0.04)
            
            if angle == 'all' or angle == 'phi':
              plt.subplot(1,3,2)
              plt.title('Inter-phi', size = 15)
              plt.matshow(target_inter_tor_angles[:,:,1], cmap='viridis_r', fignum=False)
              plt.colorbar(fraction=0.046, pad=0.04)

            if angle == 'all' or angle == 'psi':
              plt.subplot(1,3,3)
              plt.title('Inter-psi', size = 15)
              plt.matshow(target_inter_tor_angles[:,:,2], cmap='viridis_r', fignum=False)
              plt.colorbar(fraction=0.046, pad=0.04)

            plt.show()
        
        
        return distogram, target_tor_masks, target_tor_angles, target_inter_tor_masks, target_inter_tor_angles
          
    def getTorsion(self, use_pkl = False, loss_type = 'square', chi_dependent = False):
        
        d_Torsion = {}
        groups = self.sampleGroup
        
        # native Torsion
        d_Torsion['native_pdb'] = self.getSingleTorsion(self.d_dir["native_pdb"], use_pkl, loss_type, chi_dependent)
        
        # af_pdb Torsion
        d_Torsion['af_pdb'] = {}
        for model, model_dir in self.d_dir["af_pdb"].items():
            d_Torsion['af_pdb'][model] = self.getSingleTorsion(model_dir, use_pkl, loss_type, chi_dependent)
        
        # submissions Torsion
        d_Torsion['subs_pdb'] = {}
        for group in groups:
            if group not in d_Torsion.keys():
                d_Torsion['subs_pdb'][group] = {}
            d_model = self.d_dir["subs_pdb"][group]
            
            for model, model_dir in d_model.items():
                d_Torsion['subs_pdb'][group][model] = self.getSingleTorsion(model_dir, use_pkl, loss_type, chi_dependent)

        
        return d_Torsion
      
    def getTorsion_Informaion(self, s_id, pdb_path, binned_ramachan, out = True):
        s_id = self.sequenceID
        residues, chain = self.readPDB(pdb_path)
        final_residue = list(residues)[-1]
        coords, coords_mask, _ = TORSION2.get_coordinates(final_residue ,residues, chain)
        tor_masks, tor_angles = TORSION2.get_torsion(coords_mask, coords, residues, as_tensor = True)
        if out == True: RAMACHANDRAN.plot_ramans_scatter(s_id, binned_ramachan, residues, tor_angles, tor_masks)
        
        return tor_masks, tor_angles
        
    def getSingleTorsion_acc(self, pdb_path, use_pkl = False, thres= 10, diverse_thres = False, chi_dependent = True):
        
        if use_pkl == False:
            target_residues, target_chain = self.readPDB(pdb_path)
            final_residue = np.min([list(self.residues)[-1], list(target_residues.keys())[-1]])
        
            target_coords, target_coords_mask, _= TORSION2.get_coordinates(final_residue, target_residues, target_chain)
        
            target_tor_masks, target_tor_angles = TORSION2.get_torsion(target_coords_mask, target_coords, target_residues, as_tensor = True)
        
            target_tor_masks = torch.logical_and(self.tor_masks[:final_residue],target_tor_masks)
        
            target_tor_alter_angles = torch.where(target_tor_masks,target_tor_angles+180,target_tor_angles)
            target_tor_alter_angles = torch.where(target_tor_alter_angles>180,target_tor_alter_angles-360 ,target_tor_alter_angles)
        
            target_tor_angles = TORSION2.sidechain_sym_angle(target_residues, target_tor_masks, self.tor_angles[:final_residue], target_tor_angles, target_tor_alter_angles)
            if diverse_thres == False:
                return TORSION2.getTorsion_acc(target_residues, target_tor_masks, self.tor_angles[:final_residue], target_tor_angles, target_tor_alter_angles, thres, chi_dependent)    
          
            elif diverse_thres == True:
                accs = {}
                for thres in [5,10,15,20,25]:
                    accs[thres] = TORSION2.getTorsion_acc(target_residues, target_tor_masks, self.tor_angles[:final_residue], target_tor_angles, target_tor_alter_angles, thres, chi_dependent)
          
        if use_pkl == True:
            target_residues, target_chain = self.readPDB(pdb_path)
            final_residue = np.min([list(self.residues)[-1], list(target_residues)[-1]])
            
            target_coords, target_coords_mask, _ = TORSION2.get_coordinates(final_residue, target_residues, target_chain)
            target_tor_masks, target_tor_angles  = TORSION2.get_torsion(target_coords_mask, target_coords, target_residues, as_tensor = True)
        
            target_tor_masks = torch.logical_and(self.tor_masks[:final_residue], target_tor_masks)
            target_tor_alter_angles = torch.where(target_tor_masks,target_tor_angles+180,target_tor_angles)
            target_tor_alter_angles = torch.where(target_tor_alter_angles>180,target_tor_alter_angles-360 ,target_tor_alter_angles)
            
            af_tor_angles = np.load(self.d_dir['af_pkl']['1'], allow_pickle = True)['structure_module']['sidechains']['angles_sin_cos'][7]
            af_tor_angles = torch.tensor(np.arctan2(af_tor_angles[:,:,0], af_tor_angles[:,:,1]) * 180 / np.pi ) 
            af_tor_masks  = self.tor_masks
        
            if diverse_thres == False:
                return TORSION2.getTorsion_acc(target_residues, target_tor_masks, af_tor_angles[:final_residue], target_tor_angles, target_tor_alter_angles, thres, chi_dependent)    
          
            elif diverse_thres == True:
                accs = {}
                for thres in [5,10,15,20,25]:
                    accs[thres] = TORSION2.getTorsion_acc(target_residues, target_tor_masks, af_tor_angles[:final_residue], target_tor_angles, target_tor_alter_angles, thres, chi_dependent)    
                    return accs
 
    def getTorsion_acc(self, use_pkl = False, thres=10, diverse_thres = False, chi_dependent = True):
        d_Torsion_acc = {}
        groups = self.sampleGroup
        
        # native Torsion
        d_Torsion_acc['native_pdb'] = self.getSingleTorsion_acc(self.d_dir["native_pdb"], use_pkl, thres, diverse_thres, chi_dependent)
        
        # af_pdb Torsion
        d_Torsion_acc['af_pdb'] = {}
        for model, model_dir in self.d_dir["af_pdb"].items():
            d_Torsion_acc['af_pdb'][model] = self.getSingleTorsion_acc(model_dir,use_pkl, thres, diverse_thres, chi_dependent)
        
        # submissions Torsion
        d_Torsion_acc['subs_pdb'] = {}
        for group in groups:
            if group not in d_Torsion_acc.keys():
                d_Torsion_acc['subs_pdb'][group] = {}
            d_model = self.d_dir["subs_pdb"][group]
            
            for model, model_dir in d_model.items():
                d_Torsion_acc['subs_pdb'][group][model] = self.getSingleTorsion_acc(model_dir, use_pkl, thres, diverse_thres, chi_dependent)
        
        return d_Torsion_acc
      
    def plot_Torsion_acc_infor(self, year, s_id, acc,save_path):
      plot_acc = {}

      for thres in [5,10,15,20,25]:
          plot_acc[thres] = np.zeros(7)
          for i, angle in enumerate(angle_types.values()):
              plot_acc[thres][i] = acc[thres][angle]['accuracy']
            
      plt.figure(figsize=(20,10),facecolor='white')
      for i, thres in enumerate([5,10,15,20,25]):
          plt.subplot(1,5,i+1)
          plt.title(f'thres : {thres}')
          plt.ylim(0,1.0)
          plt.bar(angle_types.values(), plot_acc[thres], label = 'accuracy')
          plt.legend()
      plt.suptitle(f'{year}_{s_id}_thresholds_infor')
      plt.savefig(save_path)
      plt.show()
      
    def getSingleTorsion_diff(self, pdb_path, use_pkl = False):
        
        target_residues, target_chain = self.readPDB(pdb_path)
        final_residue = np.min([list(self.residues)[-1], list(target_residues)[-1]])
            
        target_coords, target_coords_mask, _ = TORSION2.get_coordinates(final_residue, target_residues, target_chain)
        target_tor_masks, target_tor_angles  = TORSION2.get_torsion(target_coords_mask, target_coords, target_residues, as_tensor = True)
        
        target_tor_masks = torch.logical_and(self.tor_masks[:final_residue], target_tor_masks)
        target_tor_alter_angles = torch.where(target_tor_masks,target_tor_angles+180,target_tor_angles)
        target_tor_alter_angles = torch.where(target_tor_alter_angles>180,target_tor_alter_angles-360 ,target_tor_alter_angles)
        
        if use_pkl == True:
            af_tor_angles = np.load(self.d_dir['af_pkl']['1'], allow_pickle = True)['structure_module']['sidechains']['angles_sin_cos'][7]
            af_tor_angles = torch.tensor(np.arctan2(af_tor_angles[:,:,0], af_tor_angles[:,:,1]) * 180 / np.pi ) 
            af_tor_masks  = self.tor_masks
            af_tor_sincos = TORSION2.angles_to_sincos(af_tor_angles)
                
            target_tor_angles = TORSION2.sidechain_sym_angle(target_residues, target_tor_masks[:final_residue], af_tor_angles[:final_residue], \
                                                         target_tor_angles[:final_residue], target_tor_alter_angles[:final_residue])
            target_tor_sincos = TORSION2.angles_to_sincos(target_tor_angles)
        
            return TORSION2.getTorsion_diff(target_residues, target_tor_masks, af_tor_angles[:final_residue], target_tor_angles, target_tor_alter_angles) 
        
        elif use_pkl == False:
            target_tor_angles = TORSION2.sidechain_sym_angle(target_residues, target_tor_masks[:final_residue], self.tor_angles[:final_residue], \
                                                         target_tor_angles[:final_residue], target_tor_alter_angles[:final_residue])
            target_tor_sincos = TORSION2.angles_to_sincos(target_tor_angles)
            
            return TORSION2.getTorsion_diff(target_residues, target_tor_masks, self.tor_angles[:final_residue], target_tor_angles, target_tor_alter_angles) 

    def getTorsion_diff(self):
        d_Torsion_diff = {}
        groups = self.sampleGroup
        
        # native Torsion
        d_Torsion_diff['native_pdb'] = self.getSingleTorsion_diff(self.d_dir["native_pdb"])
        
        # af_pdb Torsion
        d_Torsion_diff['af_pdb'] = {}
        for model, model_dir in self.d_dir["af_pdb"].items():
            d_Torsion_diff['af_pdb'][model] = self.getSingleTorsion_diff(model_dir)
        
        # submissions Torsion
        d_Torsion_diff['subs_pdb'] = {}
        for group in groups:
            if group not in d_Torsion_diff.keys():
                d_Torsion_diff['subs_pdb'][group] = {}
            d_model = self.d_dir["subs_pdb"][group]
            
            for model, model_dir in d_model.items():
                d_Torsion_diff['subs_pdb'][group][model] = self.getSingleTorsion_diff(model_dir)
        
        return d_Torsion_diff
    
    def get_total_diff(self, TMscores, diffs):
        total_diff = {}
        groups = self.sampleGroup
        

        for angle in angle_types.values():
            total_diff[angle] = []
        for angle in angle_types.values(): total_diff[angle] += diffs['native_pdb'][angle]
        
        # af_pdb Torsion
        for model, model_dir in self.d_dir["af_pdb"].items():
            if TMscores['af_pdb'][model] > 0.75 : 
                for angle in angle_types.values(): total_diff[angle] += diffs['af_pdb'][model][angle]
                
        
        # submissions Torsion
        for group in groups:
            for model, model_dir in self.d_dir['subs_pdb'][group].items():
                if TMscores['subs_pdb'][group][model] > 0.75 :
                    for angle in angle_types.values(): total_diff[angle] += diffs['subs_pdb'][group][model][angle]
        return total_diff
    
    def plot_angle_diffs(self):
        TMscores = self.getTMscore()
        diffs = self.getTorsion_diff()
        total_diff = self.get_total_diff(TMscores, diffs)
        bins = np.linspace(0,180,37)[:-1]
        
        plt.figure(figsize = [20,10], facecolor = 'white')
        plot_no = 0
        for angle in angle_types.values():
            plot_no += 1
            if plot_no == 4: plot_no = 5
            plt.subplot(2,4,plot_no)
            plt.title(f'{angle}')
            plt.xlabel('angle_difference')
            plt.ylabel('amount')
            plt.hist(total_diff[angle], bins = bins)
        plt.suptitle(f'{self.sequenceID}', fontsize = 20)
        plt.savefig(f'/gpfs/deepfold/users/jsg/angledelta/{self.year}_{self.sequenceID}')
        plt.show()
        
    def get_lit_violation(self,binned_ramachan, pdb_path, bins = 36, lit_to_target = True):  
        target_residues, target_chain = self.readPDB(pdb_path)
        final_residue = np.min([list(self.residues)[-1], list(target_residues.keys())[-1]])
        #if final_residue != self.final_residue : return {'PHI_PSI' : np.nan, 'PHI_CHI0' : np.nan, 'PSI_CHI0' : np.nan, 'CHI0_CHI1' : np.nan, 'CHI1_CHI2' : np.nan}
        target_coords, target_coords_mask, _= TORSION2.get_coordinates(final_residue, target_residues, target_chain)
        
        target_tor_masks, target_tor_angles = TORSION2.get_torsion(target_coords_mask, target_coords, target_residues, as_tensor = True)
        
        target_tor_masks = torch.logical_and(target_tor_masks, self.tor_masks[:final_residue])
        target_tor_angles = TORSION2.sidechain_sym_angle_under_0(target_residues,target_tor_masks, target_tor_angles)
        
        violation = RAMACHANDRAN.get_lit_violation(binned_ramachan, target_residues, target_tor_masks, target_tor_angles, bins = bins, lit_to_target = lit_to_target)
        return violation
    
    
    def get_all_lit_violation(self,binned_ramachan, bins = 36, lit_to_target = True):
        d_lit_violation = {}
        groups = self.sampleGroup
        
        # native Torsion
        d_lit_violation['native_pdb'] = self.get_lit_violation(binned_ramachan, self.d_dir["native_pdb"],bins, lit_to_target)
        
        # af_pdb Torsion
        d_lit_violation['af_pdb'] = {}
        for model, model_dir in self.d_dir["af_pdb"].items():
            d_lit_violation['af_pdb'][model] = self.get_lit_violation(binned_ramachan, model_dir, bins, lit_to_target)
        
        # submissions Torsion
        d_lit_violation['subs_pdb'] = {}
        for group in groups:
            if group not in d_lit_violation.keys():
                d_lit_violation['subs_pdb'][group] = {}
            d_model = self.d_dir["subs_pdb"][group]
            
            for model, model_dir in d_model.items():
                d_lit_violation['subs_pdb'][group][model] = self.get_lit_violation(binned_ramachan, model_dir, bins, lit_to_target)
                
        return d_lit_violation
    
    
def plotTorsion_with_energy(figsize:tuple, 
                   sequenceID:str,
                   xdata:dict, 
                   ydata:dict, 
                   groupNum=False,
                   xlabel ='', 
                   ylabel = '',
                   save_path='',
                   year = 'casp', TMscores = 0):
    
    assert groupNum <= len(ydata['subs_pdb'].keys()), 'groupNum is greater than the number of submitted models'
    
    groups = [name for name in xdata["subs_pdb"].keys()]
    if groupNum is not False:
        random.shuffle(groups)
        groups = groups[:groupNum]
    
    markers = ['o', 'v', '^', '<', '>', 's', 'p', 'h', '+', 'x', 'X', 'D', 'd']
    colors = dict(mcolors.BASE_COLORS, **mcolors.TABLEAU_COLORS) # CSS4_COLORS
    del colors['w']; del colors['r']; del colors['tab:olive']; del colors['c']
    colors = [color for color in colors.keys()]
    random.shuffle(markers); random.shuffle(colors)
    cm_list = [colors, markers]  #cm_list = [markers, colors]
    cm_set = list(product(*cm_list))
    random.shuffle(cm_set)
    
    plt.figure(figsize=figsize, facecolor = 'white')
    plot_no = 1
    
    for plot_type in plot_types.keys():
        xlabel = plot_type
        plt.subplot(2,2,plot_no)
        plot_no +=1
        xdatas = []
        ydatas = []

        angle_1, angle_2 = plot_type.split('_')
        
        for key in ydata.keys(): 
            if key == 'native_pdb' and TMscores['native_pdb'] > 0.00 and xdata[key][plot_type] > 0.2:
                x=xdata[key][plot_type]
                y=ydata[key][angle_1]['accuracy'] + ydata[key][angle_2]['accuracy']
                plt.scatter(x, y, c='red', marker='*', s=200, label=key)
                xdatas.append(x)
                ydatas.append(y)
            
            elif key == 'af_pdb':
                i=1
                for model in ydata[key].keys():
                    if TMscores['af_pdb'][model] >= 0.00 and xdata[key][model][plot_type] > 0.2: 
                        cm = cm_set[i%len(cm_set)]
                        x=xdata[key][model][plot_type]
                        y=ydata[key][model][angle_1]['accuracy'] + ydata[key][model][angle_2]['accuracy']
                        plt.scatter(x, y, c=cm[0], marker='*', s=200, label=key+model)
                        i+=1
                        xdatas.append(x)
                        ydatas.append(y)
            
            elif key == 'subs_pdb':
                i = 0
                for group in groups:
                    cm = cm_set[i%len(cm_set)]
                    best_num = list(ydata[key][group].keys())[0]
                    
                    for model in ydata[key][group]:
                        if TMscores['subs_pdb'][group][model] >= 0.00 and xdata[key][group][model][plot_type] > 0.2:
                            if ydata[key][group][model][angle_1]['accuracy'] >= ydata[key][group][best_num][angle_1]['accuracy'] :
                                best_num = model
                            x=xdata[key][group][best_num][plot_type]
                            y=ydata[key][group][best_num][angle_1]['accuracy'] + ydata[key][group][best_num][angle_2]['accuracy']
                            plt.scatter(x, y, c=cm[0], marker=cm[1], s=50)
                            i += 1
                            xdatas.append(x)
                            ydatas.append(y)
            
           
        
        xdatas = np.array(xdatas)
        ydatas = np.array(ydatas)
        
        corr = np.corrcoef(xdatas,ydatas)[0][1]
        plt.annotate(f'corr : {corr:.3f}', xy = (xdatas.max(),ydatas.max()), size = 15)
                    
        plt.title(f'{plot_type}', fontsize=15)
        plt.xlabel(f'{plot_type} Distance', fontsize=10)
        plt.xticks(fontsize=10)
        plt.ylabel(f'{angle_1}_{angle_2}_accuracy', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='upper left', ncol=2)
        plt.ylim(-0.4,2.4)

    plt.suptitle(f'{year}_ {sequenceID}', fontsize=25)
    plt.savefig(save_path)
    plt.show()
    
    
    
    
def plotloss_with_acc(figsize:tuple, 
                   sequenceID:str,
                   xdata:dict, 
                   ydata:dict, 
                   groupNum=False,
                   xlabel ='', 
                   ylabel = '',
                   save_path='',
                   year = 'casp', title = ''):
    
    assert groupNum <= len(ydata['subs_pdb'].keys()), 'groupNum is greater than the number of submitted models'
    
    groups = [name for name in xdata["subs_pdb"].keys()]
    if groupNum is not False:
        random.shuffle(groups)
        groups = groups[:groupNum]
    
    markers = ['o', 'v', '^', '<', '>', 's', 'p', 'h', '+', 'x', 'X', 'D', 'd']
    colors = dict(mcolors.BASE_COLORS, **mcolors.TABLEAU_COLORS) # CSS4_COLORS
    del colors['w']; del colors['r']; del colors['tab:olive']; del colors['c']
    colors = [color for color in colors.keys()]
    random.shuffle(markers); random.shuffle(colors)
    cm_list = [colors, markers]  #cm_list = [markers, colors]
    cm_set = list(product(*cm_list))
    random.shuffle(cm_set)
    
    
    
    plt.figure(figsize=figsize, facecolor = 'white')
    plot_no = 1
    
    for angle_type in angle_types.values():
        xdatas = []
        ydatas = []
        xlabel = angle_type
        plt.subplot(2,4,plot_no)
        plot_no +=1 
        if plot_no == 4: plot_nno = 5
        for key in ydata.keys(): 
            if key == 'native_pdb':
                x=xdata[key][angle_type]
                y=ydata[key][angle_type]['accuracy']
                #plt.scatter(x, y, c='red', marker='*', s=200, label=key)
            
            elif key == 'af_pdb':
                i=1
                for model in ydata[key].keys():
                    cm = cm_set[i%len(cm_set)]
                    x=xdata[key][model][angle_type]
                    y=ydata[key][model][angle_type]['accuracy']
                    plt.scatter(x, y, c=cm[0], marker='*', s=200, label=key+model)
                    i+=1
                    xdatas.append(x)
                    ydatas.append(y)
            else:
                i = 0
                for group in groups:
                    cm = cm_set[i%len(cm_set)]
                    best_num = list(ydata[key][group].keys())[0]
                    
                    for model in ydata[key][group]:
                        if ydata[key][group][model][angle_type]['accuracy'] >= ydata[key][group][best_num][angle_type]['accuracy']:
                            best_num = model
                    x=xdata[key][group][best_num][angle_type]
                    y=ydata[key][group][best_num][angle_type]['accuracy']
                    plt.scatter(x, y, c=cm[0], marker=cm[1], s=50)
                    i += 1
                    xdatas.append(x)
                    ydatas.append(y)
            
        
        xdatas = np.array(xdatas)
        ydatas = np.array(ydatas)
        
        corr = np.corrcoef(xdatas,ydatas)[0][1]
        plt.annotate(f'corr : {corr:.3f}', xy = (xdatas.min(),ydatas.max()), size = 15)
        plt.title(f'{angle_type}', fontsize=15)
        plt.xlabel(f'{angle_type}_loss', fontsize=10)
        plt.xticks(fontsize=10)
        plt.ylabel(f'accuracy', fontsize=10)
        plt.yticks(fontsize=10)
        plt.legend(loc='upper right', ncol=2)
        plt.ylim(-0.2, 1.2)

    plt.suptitle(f'{year}_ {sequenceID}_{title}', fontsize=25)
    plt.savefig(save_path)
    plt.show()
    

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)
        
        
        
        
def padd_plot_confusion_matrix(conf_matrix, x_labels, y_labels, save_path = ''):    
    pad_conf_matrix = conf_matrix
    
    plt.imshow(pad_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.xlabel('Predicted Values', fontsize = 15)
    plt.ylabel('Actual Values', fontsize = 15)
    plt.tight_layout()
    plt.colorbar(fraction=0.046, pad=0.04)
    
    x_posits = np.arange(len(x_labels))
    #y_posits = np.arange(len(y_labels)+2)
    y_posits = np.arange(len(y_labels))
    
    plt.xticks(x_posits, x_labels, fontsize = 10)
    
    plt.yticks(y_posits, y_labels, rotation = 90, fontsize = 10)
    
    thresh = conf_matrix.max()/2.

    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            plt.text(j,i, f'{pad_conf_matrix[i,j]} \n ({int(pad_conf_matrix[i,j]/pad_conf_matrix.sum()*100)}%)', fontsize=15, horizontalalignment="center", color = 'black')
    #plt.savefig(save_path)
    #plt.show()        
