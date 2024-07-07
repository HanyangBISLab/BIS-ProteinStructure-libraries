from Bio import SeqIO
from Bio.PDB import MMCIFParser, MMCIFIO
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB import Dice
from Bio.PDB.DSSP import DSSP


import concurrent.futures
from multiprocessing import Pool

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

            
            
class AllResiduesSelector(Select):
    def __init__(self, target_chain_id):
        self.target_chain_id = target_chain_id

    def accept_residue(self, residue):
        return residue.get_parent().id == self.target_chain_id

def mmcif_to_pdbs(input_mmcif_file, output_pdb_root):
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
    try:
        mmcif_path = os.path.join(mmcif_root, mmcif)
        mmcif_to_pdbs(mmcif_path, output_root)
    except Exception as e:
        print(f"Error processing {mmcif}: {e}")

def parallel_processing(mmcifs, mmcif_root, output_root):
    # CPU 코어 수만큼 프로세스 풀을 생성합니다.
    with Pool() as pool:
        # 각 MMCIF 파일을 병렬로 처리합니다.
        pool.starmap(process_mmcif_to_pdbs, [(mmcif, mmcif_root, output_root) for mmcif in mmcifs])
        

        
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

        
