from Bio import SeqIO
from Bio.PDB import MMCIFParser, MMCIFIO
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB import Dice
from Bio.PDB.DSSP import DSSP


import concurrent.futures
from multiprocessing import Pool

def parse_mmcif(path, file_id, chain_id, alignment_dir):
    """
    Parses an MMCIF file and processes it into structured data.

    Parameters
    ----------
    path : str
        Path to the MMCIF file.
    file_id : str
        Unique identifier for the MMCIF file.
    chain_id : str
        Chain ID to process from the MMCIF file.
    alignment_dir : str
        Directory where the alignments are stored.

    Returns
    -------
    data : dict
        Processed data from the MMCIF file.
    
    Raises
    ------
    Exception
        If an error occurs during parsing or if the MMCIF object is None.
    """
    with open(path, 'r') as f:
        mmcif_string = f.read()

    mmcif_object = mmcif_parsing.parse(
        file_id=file_id, mmcif_string=mmcif_string)
    
    if mmcif_object.mmcif_object is None:
        raise list(mmcif_object.errors.values())[0]

    mmcif_object = mmcif_object.mmcif_object

    data = data_pipeline.process_mmcif(
        mmcif=mmcif_object,
        alignment_dir=alignment_dir,
        chain_id=chain_id)
    
    return data


def generate_feature_dict(tag, seq, fasta_path, alignment_dir):
    """
    Generates a feature dictionary from the given sequence and alignment.

    Parameters
    ----------
    tag : str
        Sequence identifier tag.
    seq : str
        The amino acid sequence.
    fasta_path : str
        Path to the FASTA file.
    alignment_dir : str
        Directory where alignments are stored.

    Returns
    -------
    feature_dict : dict
        Dictionary of sequence features.
    """
    local_alignment_dir = os.path.join(alignment_dir, tag)
    feature_dict = data_processor.process_fasta(
        fasta_path=fasta_path, alignment_dir=local_alignment_dir)

    return feature_dict


def parse_fasta(data):
    """
    Parses the contents of a FASTA file and extracts sequence tags and sequences.

    Parameters
    ----------
    data : str
        String content of the FASTA file.

    Returns
    -------
    tags : list of str
        List of sequence tags.
    seqs : list of str
        List of sequences corresponding to the tags.
    """
    data = re.sub('>$', '', data, flags=re.M)
    lines = [
        l.replace('\n', '')
        for prot in data.split('>') for l in prot.strip().split('\n', 1)
    ][1:]
    tags, seqs = lines[::2], lines[1::2]
    tags = [t.split()[0] for t in tags]

    return tags, seqs


def read_fasta(file_path):
    """
    Reads a FASTA file and prints out each sequence ID and sequence.

    Parameters
    ----------
    file_path : str
        Path to the FASTA file.

    Returns
    -------
    None
    """
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
        if sequence_id is not None:
            print(f'>{sequence_id}')
            print(f'{sequence}')


class AllResiduesSelector(Select):
    """
    Selector class for PDBIO to select all residues in a specific chain.

    Parameters
    ----------
    target_chain_id : str
        The ID of the target chain to select residues from.

    Methods
    -------
    accept_residue(residue)
        Returns True if the residue belongs to the target chain.
    """
    def __init__(self, target_chain_id):
        self.target_chain_id = target_chain_id

    def accept_residue(self, residue):
        return residue.get_parent().id == self.target_chain_id


def mmcif_to_pdbs(input_mmcif_file, output_pdb_root):
    """
    Converts an MMCIF file to separate PDB files for each chain.

    Parameters
    ----------
    input_mmcif_file : str
        Path to the input MMCIF file.
    output_pdb_root : str
        Directory where the output PDB files will be stored.

    Returns
    -------
    None
    """
    cif_name = input_mmcif_file.split('/')[-1].split('.cif')[0]
    mmcif_parser = MMCIFParser(QUIET=True)
    structure = mmcif_parser.get_structure("structure", input_mmcif_file)
    pdb_io = PDBIO()

    for model in structure:
        for chain in model:
            output_pdb_file = os.path.join(output_pdb_root, f'{cif_name}_{chain.id}.pdb')
            pdb_io.set_structure(structure)
            pdb_io.save(output_pdb_file, AllResiduesSelector(chain.id))


def process_mmcif_to_pdbs(mmcif, mmcif_root, output_root):
    """
    Processes an MMCIF file and converts it to PDB format.

    Parameters
    ----------
    mmcif : str
        Name of the MMCIF file.
    mmcif_root : str
        Directory where the MMCIF files are stored.
    output_root : str
        Directory where the PDB files will be saved.

    Returns
    -------
    None
    """
    try:
        mmcif_path = os.path.join(mmcif_root, mmcif)
        mmcif_to_pdbs(mmcif_path, output_root)
    except Exception as e:
        print(f"Error processing {mmcif}: {e}")


def parallel_processing(mmcifs, mmcif_root, output_root):
    """
    Processes multiple MMCIF files in parallel and converts them to PDB files.

    Parameters
    ----------
    mmcifs : list of str
        List of MMCIF file names to process.
    mmcif_root : str
        Directory where the MMCIF files are stored.
    output_root : str
        Directory where the PDB files will be saved.

    Returns
    -------
    None
    """
    with Pool() as pool:
        pool.starmap(process_mmcif_to_pdbs, [(mmcif, mmcif_root, output_root) for mmcif in mmcifs])


def create_structure_from_feature(sequence, all_atom_positions, all_atom_mask, structure_id="pred", model_id=0, chain_id="A"):
    """
    Creates a structure from sequence and atomic position information.

    Parameters
    ----------
    sequence : str
        Amino acid sequence of the protein.
    all_atom_positions : numpy.ndarray
        Array of atomic positions for the protein.
    all_atom_mask : numpy.ndarray
        Mask indicating valid atoms in the structure.
    structure_id : str, optional
        Identifier for the structure (default is 'pred').
    model_id : int, optional
        Model ID for the structure (default is 0).
    chain_id : str, optional
        Chain ID for the structure (default is 'A').

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        Generated structure object containing atomic coordinates.
    """
    structure = Structure.Structure(structure_id)
    model = Model.Model(model_id)
    chain = Chain.Chain(chain_id)

    for i in range(len(sequence)):
        residue_id = (' ', i + 1, ' ')
        residue = Residue.Residue(residue_id, sequence[i], '')

        for j in range(all_atom_positions.shape[1]):
            if all_atom_mask[i, j] == 1:
                atom_name = index_to_atom_name.get(j, f"X{j + 1}")
                atom_coords = all_atom_positions[i, j]
                atom = Atom.Atom(atom_name, atom_coords, 1.0, 1.0, '', atom_name, j + 1, 'C')
                residue.add(atom)

        chain.add(residue)
    model.add(chain)
    structure.add(model)

    sr = ShrakeRupley()
    sr.compute(structure, level="R")

    for res in structure.get_residues():
        if 'EXP_NACCESS' in res.xtra:
            res.sasa = res.xtra['EXP_NACCESS']
        else:
            res.sasa = None

    return structure
