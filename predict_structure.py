import argparse
import os
import shutil
import yaml
import json
from pathlib import Path
from chai_lab.chai1 import run_inference
from chai_lab import utils
from chai_lab.utils import analysis
from chai_lab.utils.analysis import compute_structure_metrics, plot_structure_metrics
from chai_lab.config import (
    TMP_DIR, 
    OUTPUT_DIR, 
    DOWNLOADS_DIR,
    DEFAULT_NUM_TRUNK_RECYCLES,
    DEFAULT_NUM_DIFFN_TIMESTEPS,
    DEFAULT_SEED
)

def parse_args():
    parser = argparse.ArgumentParser(description='Run structure prediction with Chai-1')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input JSON file')
    parser.add_argument('--output-dir', type=str,
                      help='Base output directory (optional)')
    return parser.parse_args()

def load_input_parameters(input_file='input.yaml'):
    """Load input parameters from a YAML file."""
    with open(input_file, 'r') as f:
        input_data = yaml.safe_load(f)
    proteins = input_data['proteins']
    ligands = input_data['ligands']
    constraints = input_data.get('constraints', [])
    params = input_data['parameters']
    return proteins, ligands, constraints, params

def prepare_fasta_file(proteins, ligands, run_name):
    """Prepare the FASTA file from protein and ligand entries."""
    fasta_content = ""
    for protein in proteins:
        fasta_content += f">protein|name={protein['name']}\n{protein['sequence']}\n"
    for ligand in ligands:
        fasta_content += f">ligand|name={ligand['name']}\n{ligand['smiles']}\n"
    fasta_path = TMP_DIR / f"{run_name}.fasta"
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(fasta_content)
    return fasta_path

def prepare_constraints_file(constraints, run_name):
    """Prepare the constraints file if any constraints are provided."""
    if constraints:
        import csv
        constraints_path = TMP_DIR / f"{run_name}_constraints.csv"
        with constraints_path.open('w', newline='') as csvfile:
            fieldnames = [
                'restraint_id', 'chainA', 'res_idxA', 'chainB', 'res_idxB',
                'connection_type', 'confidence', 'min_distance_angstrom',
                'max_distance_angstrom', 'comment'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for constraint in constraints:
                # Ensure distances are float
                constraint['min_distance_angstrom'] = float(constraint['min_distance_angstrom'])
                constraint['max_distance_angstrom'] = float(constraint['max_distance_angstrom'])
                writer.writerow(constraint)
    else:
        constraints_path = None
    return constraints_path

def setup_output_directory(run_name, base_output_dir=None):
    """Set up the output directory outside of the main codebase."""
    output_dir = Path(base_output_dir or OUTPUT_DIR) / run_name
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def run_model_inference(fasta_path, constraints_path, params, output_dir):
    """Run the inference model."""
    os.environ["CHAI_DOWNLOADS_DIR"] = str(DOWNLOADS_DIR)
    device = f"cuda:{params.get('gpu', 0)}"
    
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        constraint_path=constraints_path,
        num_trunk_recycles=params.get('num_trunk_recycles', DEFAULT_NUM_TRUNK_RECYCLES),
        num_diffn_timesteps=params.get('num_diffn_timesteps', DEFAULT_NUM_DIFFN_TIMESTEPS),
        seed=params.get('seed', DEFAULT_SEED),
        device=device,
        use_esm_embeddings=True,
    )
    return candidates

def process_results(candidates, output_dir):
    """Process the results and save all metrics."""
    structures = [str(path) for path in candidates.cif_paths]
    
    # Compute all metrics
    metrics = compute_structure_metrics(structures)
    
    # Save metrics
    metrics_out_path = output_dir.joinpath("metrics.json")
    with open(metrics_out_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots
    plot_structure_metrics(metrics, output_dir)
    
    # Update scores.json with additional metrics
    scores_out_path = output_dir.joinpath("scores.json")
    with open(scores_out_path, 'r') as f:
        all_scores = json.load(f)
    
    # Add average metrics to each model's scores
    for i, sample_scores in enumerate(all_scores):
        sample_scores.update({
            'ramachandran_favored': metrics['per_structure_metrics'][i]['ramachandran']['favored_region_percent']
        })
    
    with open(scores_out_path, 'w') as f:
        json.dump(all_scores, f, indent=2)

def copy_input_file(input_file, output_dir):
    """Copy the input YAML file to the output directory."""
    shutil.copy(input_file, output_dir.joinpath('input.yaml'))

def main():
    args = parse_args()
    
    # Load input file
    with open(args.input, 'r') as f:
        input_data = json.load(f)
    
    proteins = input_data['proteins']
    ligands = input_data['ligands']
    constraints = input_data.get('constraints', [])
    params = input_data['parameters']
    
    # Set up output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path(OUTPUT_DIR)  # default location
    
    run_name = params['run_name']
    output_dir = base_output_dir / run_name
    
    fasta_path = prepare_fasta_file(proteins, ligands, run_name)
    constraints_path = prepare_constraints_file(constraints, run_name)
    output_dir = setup_output_directory(run_name, base_output_dir=base_output_dir)
    candidates = run_model_inference(fasta_path, constraints_path, params, output_dir)
    process_results(candidates, output_dir)
    copy_input_file(args.input, output_dir)

if __name__ == "__main__":
    main()
