def readPDB(pdb_dir):
    """
    Reads a PDB file and extracts chain and residue information.

    Parameters
    ----------
    pdb_dir : str
        Path to the PDB file.

    Returns
    -------
    model : Bio.PDB.Model
        The first model from the parsed PDB file.
    chains : list of str
        List of chain identifiers.
    residue_dict : dict
        Dictionary mapping chain IDs to residues and their positions.
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


def readMMCIF(mmcif_path):
    """
    Reads an MMCIF file and extracts chain and residue information.

    Parameters
    ----------
    mmcif_path : str
        Path to the MMCIF file.

    Returns
    -------
    model : Bio.PDB.Model
        The first model from the parsed MMCIF file.
    chains : list of str
        List of chain identifiers.
    residue_dict : dict
        Dictionary mapping chain IDs to residues and their positions.
    """
    parser = MMCIFParser()
    structure = parser.get_structure("structure", mmcif_path)
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


def calculate_calpha_distogram(chain_id, chain, residue_dict):
    """
    Calculates the C-alpha distogram for a protein chain.

    Parameters
    ----------
    chain_id : str
        Chain identifier.
    chain : Bio.PDB.Chain
        Chain object from the parsed structure.
    residue_dict : dict
        Dictionary mapping chain IDs to residues and their positions.

    Returns
    -------
    residue_length : int
        Number of residues in the chain.
    distogram : numpy.ndarray
        A 2D matrix representing the pairwise C-alpha distances.
    """
    residue_length = np.array(list(residue_dict[chain_id].keys())).max()
    calpha_coords = np.zeros([residue_length, 3])
    
    for residue in chain:
        res_num = residue.get_id()[1] - 1
        res_name = residue.resname
        
        if res_num >= residue_length:
            continue

        if res_name in res_types:
            if "CA" in residue:
                calpha_coords[res_num] = residue["CA"].get_coord()
            else:
                calpha_coords[res_num] = np.array([np.nan, np.nan, np.nan])
        else:
            calpha_coords[res_num] = np.array([np.nan, np.nan, np.nan])
    
    distogram = squareform(pdist(calpha_coords, 'euclidean'))
    return residue_length, distogram


def calculate_lys_leu_map(chain_id, chain, residue_dict):
    """
    Generates a binary map indicating the presence of Lysine (LYS) or Leucine (LEU) residues.

    Parameters
    ----------
    chain_id : str
        Chain identifier.
    chain : Bio.PDB.Chain
        Chain object from the parsed structure.
    residue_dict : dict
        Dictionary mapping chain IDs to residues and their positions.

    Returns
    -------
    residue_length : int
        Number of residues in the chain.
    lys_leu_map : numpy.ndarray
        A 2D boolean array where True indicates the presence of LYS or LEU at corresponding residue positions.
    """
    residue_length = np.array(list(residue_dict[chain_id].keys())).max()
    lys_leu_map = np.zeros(residue_length)
    
    for res_num in residue_dict[chain_id]:
        res_name = residue_dict[chain_id][res_num]
        res_num = res_num - 1
        if res_name in ['LYS', 'LEU']:
            lys_leu_map[res_num] = True
        else:
            lys_leu_map[res_num] = False
    
    lys_leu_map = np.logical_or(lys_leu_map[None], lys_leu_map[:, None])
    return residue_length, lys_leu_map


def load_list_from_file(file_path):
    """
    Loads a list from a pickle file.

    Parameters
    ----------
    file_path : str
        Path to the pickle file.

    Returns
    -------
    data_list : list
        List of data loaded from the pickle file.
    """
    with open(file_path, 'rb') as f:
        data_list = pickle.load(f)
    return data_list


def calculate_sa_map(chain_id, chain, residue_dict, solvent_raw_datas):
    """
    Calculates the solvent accessibility (SA) map for a protein chain.

    Parameters
    ----------
    chain_id : str
        Chain identifier.
    chain : Bio.PDB.Chain
        Chain object from the parsed structure.
    residue_dict : dict
        Dictionary mapping chain IDs to residues and their positions.
    solvent_raw_datas : list
        List containing raw solvent accessibility data for each residue.

    Returns
    -------
    residue_length : int
        Number of residues in the chain.
    sa_map : numpy.ndarray
        A 2D array representing the solvent accessibility map, with values between 0 and 1.
    """
    residue_length = np.array(list(residue_dict[chain_id].keys())).max()
    sa_accs = np.zeros(residue_length)
    sa_accs[:] = np.nan
    sa_map = np.zeros([residue_length, residue_length])
    
    for res_num in residue_dict[chain_id]:
        res_type = residue_dict[chain_id][res_num]
        res_num = res_num - 1
        if len(solvent_raw_datas) > res_num:
            sa_accs[res_num] = float(solvent_raw_datas[res_num][3]) / SA_maximum_map[res_type]
    
    for i in range(residue_length):
        for j in range(residue_length):
            sa_map[i, j] = np.min([sa_accs[i], sa_accs[j]])
    return residue_length, sa_map


def calculate_tryptic_map(chain_id, chain, residue_dict):
    """
    Generates a tryptic map based on the presence of Lysine (LYS) or Arginine (ARG).

    Parameters
    ----------
    chain_id : str
        Chain identifier.
    chain : Bio.PDB.Chain
        Chain object from the parsed structure.
    residue_dict : dict
        Dictionary mapping chain IDs to residues and their positions.

    Returns
    -------
    residue_length : int
        Number of residues in the chain.
    tryptic_map : numpy.ndarray
        A 2D boolean array where True indicates tryptic cleavage points between residues.
    """
    residue_length = np.array(list(residue_dict[chain_id].keys())).max()
    tryptic_num = np.zeros([residue_length])
    tryptic_map = np.zeros([residue_length, residue_length])
    
    for res_num in residue_dict[chain_id].keys():
        res_name = residue_dict[chain_id][res_num]
        res_num = res_num - 1
        if res_name in ['LYS', 'ARG']:
            tryptic_num[res_num+1:] += 1
    for i in range(residue_length):
        for j in range(residue_length):
            tryptic_map[i, j] = np.abs(tryptic_num[i] - tryptic_num[j]) > 1
    return residue_length, tryptic_map


def plot_cross_link(residue_length, distogram, lys_leu_map, sa_map, tryptic_map):
    """
    Plots cross-linking analysis based on distance, LYS_LEU map, solvent accessibility, and tryptic map.

    Parameters
    ----------
    residue_length : int
        Number of residues in the chain.
    distogram : numpy.ndarray
        Pairwise C-alpha distance matrix.
    lys_leu_map : numpy.ndarray
        LYS and LEU residue map.
    sa_map : numpy.ndarray
        Solvent accessibility map.
    tryptic_map : numpy.ndarray
        Tryptic cleavage map.
    """
    plt.figure(figsize=(20, 5), facecolor='white')

    plt.subplot(1, 5, 1)
    plt.title('Ca Distogram, < 10 Å')
    plt.imshow(distogram < 10)

    plt.subplot(1, 5, 2)
    plt.title('LYS_LEU_MAP')
    plt.imshow(lys_leu_map, cmap='binary')

    plt.subplot(1, 5, 3)
    plt.title('Solvent_Accs > 0.5')
    plt.imshow(sa_map > 0.5, cmap='binary')

    plt.subplot(1, 5, 4)
    plt.title('Tryptic map')
    plt.imshow(tryptic_map, cmap='binary')

    distogram = distogram * lys_leu_map * (sa_map > 0.5) * tryptic_map
    plt.subplot(1, 5, 5)
    plt.title('CrossLink Candidates')
    plt.imshow(np.logical_and(distogram > 5, distogram < 25), cmap='binary')

    plt.tight_layout()
    plt.show()
