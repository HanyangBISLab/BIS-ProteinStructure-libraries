import Modules.TORSION2 as TORSION2
from Bio.PDB.PDBParser import PDBParser
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pickle
import gzip
import warnings
warnings.filterwarnings(action='ignore')



res_map = {'ALA' : 'A', 'ARG' : 'R', 'ASN' : 'N', 'ASP' : 'D', 'CYS' : 'C', 'GLN' : 'Q', 'GLU' : 'E', 'GLY' : 'G', 'HIS' : 'H', 'ILE' : 'I', 'LEU' : 'L', \
               'LYS' : 'K', 'MET' : 'M', 'PHE' : 'F', 'PRO' : 'P', 'SER' : 'S', 'THR' : 'T', 'TRP' : 'W', 'TYR' : 'Y', 'VAL' : 'V', 'TOTAL' : 'T'}
angle_types = {0 : 'OMEGA', 1 : 'PHI' , 2 : 'PSI', 3 : 'CHI1', 4 : 'CHI2', 5 : 'CHI3', 6 : 'CHI4'}
tor_types = {'OMEGA' : 0, 'PHI' : 1, 'PSI' : 2, 'CHI1' : 3, 'CHI2' : 4 , 'CHI3' : 5, 'CHI4' : 6}
plot_types = {'PHI_PSI' : [1,2], 'PHI_CHI1' : [1,3], 'PSI_CHI1' : [2,3], 'CHI1_CHI2' : [3, 4]}
    
def get_ramans(tors):
    ramachan_plot = {}
    for plot_type in plot_types:
        ramachan_plot[plot_type] = {}
        type_1, type_2 = plot_type.split('_')
        angle_1, angle_2 = tor_types[type_1], tor_types[type_2]
    
    
        for restype in res_map.keys():
            ramachan_plot[plot_type][restype] = []
            ramachan_plot[plot_type][restype].append([np.nan,np.nan])
    
        for pdb_name in tqdm(tors.keys()):
            tor_masks  = tors[pdb_name]['tor_masks']
            tor_angles = tors[pdb_name]['tor_angles']
            residues   = tors[pdb_name]['residues']
            tor_angles = TORSION2.sidechain_sym_angle_under_0(residues,tor_masks, tor_angles)
            tor_len = len(tor_masks)
        
            for res_num in residues.keys():
                i = res_num - 1
                if i < 0 or i >= tor_len : continue
                restype = residues[res_num]
                if tor_masks[i][angle_1] and tor_masks[i][angle_2] :
                    ramachan_plot[plot_type][restype].append([tor_angles[i][angle_1],tor_angles[i][angle_2]])
                    ramachan_plot[plot_type]['TOTAL'].append([tor_angles[i][angle_1],tor_angles[i][angle_2]])
                
    with gzip.open(f'''/dev/shm/jsg/raman/raman.pkl''', 'wb') as f:
        pickle.dump(ramachan_plot, f)
    return ramachan_plot 

def get_inter_statistic(tors):
    temp = 0
    inter_statistic = {}
    for restype_1 in res_map.keys():
        inter_statistic[restype_1] = {}
        for restype_2 in res_map.keys():
            inter_statistic[restype_1][restype_2]  = []
        
    for pdb_name in tqdm(tors.keys()):
        #temp +=1
        #if temp == 5: break
        tor_masks        = tors[pdb_name]['tor_masks']
        tor_angles       = tors[pdb_name]['tor_angles']
        residues         = tors[pdb_name]['residues']
        inter_tor_masks  = np.array(tors[pdb_name]['inter_tor_masks'])
        inter_tor_angles = np.array(tors[pdb_name]['inter_tor_angles'])
        distogram        = tors[pdb_name]['distogram']
        tor_len = len(tor_masks)

        for i in range(tor_len):
            res_num_1 = i + 1
            if i < 0 or i >= tor_len or res_num_1 not in residues.keys(): continue
            restype_1 = residues[res_num_1]
            chi1 = tor_angles[i][3]
            
            for j in range(i,tor_len):
                res_num_2 = j + 1
                if j < 0 or j >= tor_len or res_num_2 not in residues.keys() or i == j: continue
                restype_2    = residues[res_num_2]
                if restype_2 not in res_map.keys(): continue
                inter_statistic[restype_1][restype_2].append(np.array([distogram[i,j], tor_angles[i][3], inter_tor_angles[i,j,0]],dtype = np.float16) )

    for restype_1 in res_map.keys():
        with gzip.open(f'''/dev/shm/jsg/inter/inter_statistic_{restype_1}.pkl''', 'wb') as f:
            pickle.dump(inter_statistic[restype_1], f)
    return inter_statistic
  
  
"""
# for temp...
temp_res_map = ['HIS']
res_map = RAMACHANDRAN.res_map

def get_inter_statistic(tors):
    temp = 0
    inter_statistic = {}
    for restype_1 in res_map.keys():
        inter_statistic[restype_1] = {}
        for restype_2 in res_map.keys():
            inter_statistic[restype_1][restype_2]  = []
        
    for pdb_name in tqdm(tors.keys()):
        #temp +=1
        #if temp == 5: break
        tor_masks        = tors[pdb_name]['tor_masks']
        tor_angles       = tors[pdb_name]['tor_angles']
        residues         = tors[pdb_name]['residues']
        inter_tor_masks  = np.array(tors[pdb_name]['inter_tor_masks'])
        inter_tor_angles = np.array(tors[pdb_name]['inter_tor_angles'])
        distogram        = tors[pdb_name]['distogram']
        tor_len = len(tor_masks)

        for i in range(tor_len):
            res_num_1 = i + 1
            if i < 0 or i >= tor_len or res_num_1 not in residues.keys(): continue
            restype_1 = residues[res_num_1]
            if restype_1 not in temp_res_map : continue
            chi1 = tor_angles[i][3]
            
            for j in range(i,tor_len):
                res_num_2 = j + 1
                if j < 0 or j >= tor_len or res_num_2 not in residues.keys()or i == j : continue
                restype_2    = residues[res_num_2]
                if restype_2 not in res_map.keys(): continue
                inter_statistic[restype_1][restype_2].append(np.array([distogram[i,j], tor_angles[i][3], inter_tor_angles[i,j,0]],dtype = np.float16) )

    for restype_1 in temp_res_map:
        with gzip.open(f'''/dev/shm/jsg/inter/inter_statistic_{restype_1}.pkl''', 'wb') as f:
            pickle.dump(inter_statistic[restype_1], f)
    return inter_statistic
inter_statistic = get_inter_statistic(tors)
"""


def get_binned_infor(ramachan_plot, bins = 36):
    axis_bin = np.linspace(-180,180,bins+1)
    axis_1 = np.array(ramachan_plot)[:,0]
    axis_2 = np.array(ramachan_plot)[:,1]
    
    zs = np.zeros([bins,bins])
    
    for i in range(bins):
        for j in range(bins):
            min_1, max_1 = axis_bin[i], axis_bin[i+1]
            min_2, max_2 = axis_bin[j], axis_bin[j+1]
            count = np.sum(np.logical_and(
                np.logical_and(min_1<=axis_1, axis_1 < max_1),
                np.logical_and(min_2<=axis_2, axis_2 < max_2)
                )
            )
            zs[i][j] = count
    
    zs/= zs.sum()
    return zs

def get_binned_ramachan(ramachan_plot, bins = 36):
  
    bin_infor = np.linspace(-180,180,bins+1)
    binned_ramachan = {}
    for plot_type in plot_types:
        binned_ramachan[plot_type] = {}
    
        for res_type in tqdm(res_map.keys()):
            #z = get_binned_infor(ramachan_plot[plot_type][res_type], bins = bins)
            z = np.histogram2d(np.array(ramachan_plot[plot_type][res_type])[:,0],np.array(ramachan_plot[plot_type][res_type])[:,1] , [bin_infor,bin_infor])[0]
            binned_ramachan[plot_type][res_type] = z
            
    with gzip.open(f'/dev/shm/jsg/raman/binned_ramachan_{bins}.pkl', 'wb') as f:
        pickle.dump(binned_ramachan, f)
    
    return binned_ramachan
                
def plot_ramans_surface(binned_ramachan, bins = 36):
    for plot_type in plot_types.keys():
        
        plot_infors = binned_ramachan[plot_type]
        fig = plt.figure(figsize= (20,30), facecolor = 'white')
        
        for plot_no, res_type in enumerate(plot_infors.keys()):
            ax = fig.add_subplot(6, 4, plot_no+1, projection='3d')
            
            
            plt.title(f'{res_type}', fontdict = {'fontsize' : 15})
            plt.xlabel(f'''{plot_type.split('_')[0]}''',size = 10)
            plt.ylabel(f'''{plot_type.split('_')[1]}''', size = 10)
            
            add = 180/bins
            xs = np.linspace(-180,180,bins+1)[:-1]  + add
            ys = np.linspace(-180,180,bins+1)[:-1]  + add
            
            ys, xs = np.meshgrid(xs,ys)
            zs = binned_ramachan[plot_type][res_type]
            
            surf = ax.plot_surface(xs, ys, zs, cmap=cm.Accent_r,linewidth=0, antialiased=False,alpha = 1.0)
            fig.colorbar(surf, shrink=0.5, aspect=5)
            
            ax.set_xlim(180,-180)
            ax.set_ylim(180,-180)
            ax.view_init(90,90)
            
        plt.suptitle(f'{plot_type}_{over}_{under}', size = 25)
        plt.savefig(f'/user/deepfold/users/jsg/Images/ramachandran/{plot_type}.png')
        plt.show()
                    
def plot_ramans_scatter(s_id, binned_ramachan, residues, tor_angles, tor_masks):
    plt.figure(figsize= (30,5), facecolor = 'white')
    
    for plot_no, plot_type in enumerate(plot_types.keys()):
        plt.subplot(1,5,plot_no + 1)
        angles = plot_types[plot_type]
        angle_1 = angles[0]
        angle_2 = angles[1]
    
        for i in range(len(tor_masks)):
            if tor_masks[i][angle_1] and tor_masks[i][angle_2]:
                plt.scatter(tor_angles[i][angle_1],tor_angles[i][angle_2], color = 'blue', alpha = 0.5)
            plt.title(f'{plot_type}', size= 12)
            plt.xlabel(f'''{plot_type.split('_')[0]}''', size =10)
            plt.ylabel(f'''{plot_type.split('_')[1]}''', size =10)
            plt.xlim(-180,180)
            plt.ylim(-180,180)
            
    plt.subplot(1,5,5)
    
    lit_violation = get_lit_violation(binned_ramachan, residues, tor_masks, tor_angles)
    bar = plt.bar(lit_violation.keys(), lit_violation.values(), color = 'blue')
    for rect in bar:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.1f' % height, ha='center', va='bottom', size = 12)
    
    plt.xlabel('pair_type', size = 10)
    plt.ylabel('KLdivergence',size = 10)
    plt.title('literature_loss', size = 12)
    plt.ylim(0,20)
    
    plt.suptitle(f'{s_id}, Torsion angle information', size = 15)
    plt.show()
  
def plot_ramans_total(tor_angles, tor_masks, plot_type = 'PHI_PSI'):
    plt.figure(figsize= (10,10), facecolor = 'white')
    
    angles = plot_types[plot_type]
    angle_1 = angles[0]
    angle_2 = angles[1]
    
    for i in range(len(tor_masks)):
        if tor_masks[i][angle_1] and tor_masks[i][angle_2]:
            plt.scatter(tor_angles[i][angle_1],tor_angles[i][angle_2])
        plt.xlim(-180,180)
        plt.ylim(-180,180)
    plt.show()
    
    
    
def get_lit_violation(binned_ramachan, target_residues, target_tor_masks, target_tor_angles, bins = 36, lit_to_target = True):
    bin_infor = np.linspace(-180,180,bins+1)
    target_ramachan_infor = {}
    target_binned_ramachan = {}
    lit_violation = {}
    
    final_residue = len(target_tor_masks)
    nans = np.zeros(target_tor_masks.shape[0])
    nans[:] = np.nan
    
    for plot_type in plot_types:
        angle_1, angle_2 = plot_types[plot_type]
        lit_violation[plot_type] = 0
        
        target_ramachan_infor[plot_type] = [[np.nan,np.nan]]
        
        for res_num in target_residues.keys():
            if res_num > final_residue or res_num < 1 : continue
            i = res_num - 1
            if target_tor_masks[i][angle_1] and target_tor_masks[i][angle_2] :
                target_ramachan_infor[plot_type].append([target_tor_angles[i][angle_1],target_tor_angles[i][angle_2]])
                
        if (target_ramachan_infor[plot_type]) == [[np.nan,np.nan]] :  continue
        
        angles_1  = np.where(np.logical_and(target_tor_masks[:,angle_1],target_tor_masks[:,angle_2]), target_tor_angles[:,angle_1], nans)
        angles_2  = np.where(np.logical_and(target_tor_masks[:,angle_1],target_tor_masks[:,angle_2]), target_tor_angles[:,angle_2], nans)
    
        target_binned_ramachan[plot_type] = np.histogram2d(angles_1, angles_2, [bin_infor,bin_infor])[0]
        target_binned_ramachan[plot_type] /= np.sum(target_binned_ramachan[plot_type])
             
        target = (target_binned_ramachan[plot_type])
        lit    = (binned_ramachan[plot_type]['TOTAL'])
         
        if   lit_to_target == True : lit_violation[plot_type] = -np.sum(lit    * np.log((target+1e-10)/(lit+1e-10)))
        elif lit_to_target == False: lit_violation[plot_type] = -np.sum(target * np.log((lit+1e-10)  /(target+1e-10)))
         
    return lit_violation


