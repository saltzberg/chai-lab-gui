import numpy as np
from Bio.PDB import *
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def align_and_get_rmsd(ref_structure, alt_structure):
    """
    Align two structures using CA atoms and return the aligned coordinates.
    
    Args:
        ref_structure: Reference structure
        alt_structure: Structure to align to reference
    
    Returns:
        tuple: (ref_coords, aligned_coords)
    """
    # Get CA atoms from both structures
    ref_atoms = []
    alt_atoms = []
    
    # Extract CA atoms from both structures
    for ref_chain, alt_chain in zip(ref_structure[0], alt_structure[0]):
        for ref_res, alt_res in zip(ref_chain, alt_chain):
            if 'CA' in ref_res and 'CA' in alt_res:
                ref_atoms.append(ref_res['CA'])
                alt_atoms.append(alt_res['CA'])
    
    # Assuming ref_atoms and alt_atoms are lists of Atom objects with identical order
    super_imposer = Superimposer()
    super_imposer.set_atoms(ref_atoms, alt_atoms)

    # Get the RMSD after alignment (optional for verification)
    rmsd = super_imposer.rms

    return rmsd

def compute_rmsd(structure_paths):
    """
    Compute pairwise RMSD between structures after alignment.
    
    Args:
        structure_paths: List of paths to structure files
    
    Returns:
        tuple: (average_rmsd, rmsd_matrix)
    """
    n_structures = len(structure_paths)
    rmsd_matrix = np.zeros((n_structures, n_structures))
    parser = MMCIFParser()
    
    # Load all structures
    structures = [parser.get_structure(f'struct_{i}', path) 
                 for i, path in enumerate(structure_paths)]
    
    # Compute pairwise RMSD with alignment
    for i in range(n_structures):
        for j in range(i+1, n_structures):
            # Align structures and get coordinates
            #ref_coords, alt_coords = align_and_get_ca_coords(structures[i], structures[j])
            rmsd = align_and_get_rmsd(structures[i], structures[j])
            # Calculate RMSD after alignment
            #diff = ref_coords - alt_coords
            #rmsd = np.sqrt(np.mean(np.sum(diff * diff, axis=1)))
            rmsd_matrix[i,j] = rmsd_matrix[j,i] = rmsd
    
    # Calculate average RMSD
    mask = np.ones_like(rmsd_matrix, dtype=bool)
    np.fill_diagonal(mask, False)
    average_rmsd = rmsd_matrix[mask].mean()
    
    return average_rmsd, rmsd_matrix


def compute_structure_metrics(structure_paths):
    """Compute comprehensive structure analysis metrics."""
    n_structures = len(structure_paths)
    metrics = {}
    
    # 1. RMSD Calculations
    average_rmsd, rmsd_matrix = compute_rmsd(structure_paths)
    metrics['average_rmsd'] = float(average_rmsd)
    metrics['pairwise_rmsd_matrix'] = rmsd_matrix.tolist()
    
    # 2. Per-structure metrics
    per_structure_metrics = []
    for path in structure_paths:
        structure_metrics = {}
        parser = MMCIFParser()
        structure = parser.get_structure('temp', path)
        
        # 2.1 Per-residue RMSD
        ca_coords = get_ca_coords(path)
        per_res_rmsd = compute_per_residue_rmsd(ca_coords)
        structure_metrics['per_residue_rmsd'] = per_res_rmsd.tolist()
        
        # 2.2 Ramachandran Statistics
        rama_stats = compute_ramachandran_stats(structure[0])
        structure_metrics['ramachandran'] = rama_stats
        
        # 2.3 Distance Matrix
        distance_matrix = compute_distance_matrix(structure[0])
        structure_metrics['distance_matrix'] = distance_matrix.tolist()
        
        per_structure_metrics.append(structure_metrics)
    
    metrics['per_structure_metrics'] = per_structure_metrics
    
    return metrics

def compute_per_residue_rmsd(coords):
    """Compute per-residue RMSD."""
    return np.sqrt(np.sum(coords**2, axis=1))

def compute_ramachandran_stats(model):
    """Compute Ramachandran plot statistics."""
    phi_psi = []
    for chain in model:
        polypeptides = PPBuilder().build_peptides(chain)
        for pp in polypeptides:
            for phi, psi in pp.get_phi_psi_list():
                if phi and psi:  # Both angles are available
                    phi_psi.append((np.degrees(phi), np.degrees(psi)))
    
    if phi_psi:
        phi_psi = np.array(phi_psi)
        return {
            'phi_angles': phi_psi[:, 0].tolist(),
            'psi_angles': phi_psi[:, 1].tolist(),
            'favored_region_percent': compute_ramachandran_regions(phi_psi)
        }
    return {}

def compute_ramachandran_regions(phi_psi):
    """Compute percentage of residues in favored regions."""
    favored = 0
    total = len(phi_psi)
    for phi, psi in phi_psi:
        # Very basic favored region definition
        if (-120 <= phi <= 0) and (-90 <= psi <= 45):  # alpha-helix region
            favored += 1
        elif (-180 <= phi <= -45) and (45 <= psi <= 180):  # beta-sheet region
            favored += 1
    return (favored / total) * 100 if total > 0 else 0

def compute_distance_matrix(model, threshold=8.0):
    """Compute pairwise distances between CA atoms."""
    ca_atoms = []
    for chain in model:
        for residue in chain:
            if 'CA' in residue:
                ca_atoms.append(residue['CA'])
    
    n_residues = len(ca_atoms)
    distance_matrix = np.zeros((n_residues, n_residues))
    
    for i in range(n_residues):
        for j in range(i+1, n_residues):
            distance = ca_atoms[i] - ca_atoms[j]
            distance_matrix[i,j] = distance_matrix[j,i] = distance
            
    return distance_matrix

def plot_rmsd_heatmap(rmsd_matrix, output_dir):
    """Plot RMSD heatmap with fixed range."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(rmsd_matrix, 
                annot=True, 
                fmt=".2f", 
                cmap="viridis",
                vmin=0,
                vmax=10)
    plt.title("Pairwise RMSD Matrix")
    plt.xlabel("Structure Index")
    plt.ylabel("Structure Index")
    plt.savefig(output_dir / "rmsd_heatmap.png")
    plt.close()

def plot_distance_matrix(distance_matrices, output_dir=None):
    """
    Plot average contact map from distance matrices.
    
    Args:
        distance_matrices: List of distance matrices (as lists or numpy arrays)
        threshold: Distance threshold for contacts (Å)
        output_dir: Output directory for saving plot
    """
    # Convert lists to numpy arrays if needed
    distance_matrices = [np.array(dist_mat) if isinstance(dist_mat, list) else dist_mat 
                        for dist_mat in distance_matrices]
    
    # Convert distances to contacts using threshold
    #contact_maps = [dist_mat <= threshold for dist_mat in distance_matrices]
    avg_distance_matrix = np.mean(distance_matrices, axis=0)
    n_structures = len(distance_matrices)
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(avg_distance_matrix, 
                cmap="viridis",
                vmin=0,
                vmax=50,
                square=True)
    plt.title(f"Average Distance Matrix")
    plt.xlabel("Residue Index")
    plt.ylabel("Residue Index")
    
    if output_dir:
        plt.savefig(output_dir / "avg_distance_matrix.png")
    plt.close()

def plot_structure_metrics(metrics, output_dir):
    """Generate and save all visualization plots."""
    output_dir = Path(output_dir)
    
    # Plot RMSD heatmap
    plot_rmsd_heatmap(metrics['pairwise_rmsd_matrix'], output_dir)
    
    # Plot contact map
    distance_matrices = [m['distance_matrix'] for m in metrics['per_structure_metrics']]
    plot_distance_matrix(distance_matrices, output_dir=output_dir)

def get_ca_coords(structure_path):
    """Extract CA coordinates from a structure file."""
    parser = MMCIFParser()
    structure = parser.get_structure('temp', structure_path)
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_coords.append(residue['CA'].get_coord())
    return np.array(ca_coords)

