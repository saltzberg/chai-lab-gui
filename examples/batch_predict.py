import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import itertools
from typing import Dict, List
import copy

from predict_structure.py import main as run_prediction

def parse_args():
    parser = argparse.ArgumentParser(description='Batch run structure predictions')
    parser.add_argument('--input-dir', type=str, help='Directory containing input JSON files')
    parser.add_argument('--input-template', type=str, help='Template JSON file for parameter sweep')
    parser.add_argument('--output-dir', type=str, required=True, help='Base output directory')
    parser.add_argument('--max-parallel', type=int, default=1, help='Maximum number of parallel runs')
    parser.add_argument('--param-sweep', type=str, help='JSON file defining parameter sweep ranges')
    return parser.parse_args()

def load_param_sweep(param_file: str) -> Dict:
    """Load parameter sweep configuration."""
    with open(param_file, 'r') as f:
        return json.load(f)

def generate_param_combinations(template_file: str, param_ranges: Dict) -> List[Dict]:
    """Generate all parameter combinations for sweep."""
    with open(template_file, 'r') as f:
        template = json.load(f)
    
    # Extract parameter ranges
    param_names = []
    param_values = []
    for param_path, values in param_ranges.items():
        param_names.append(param_path)
        param_values.append(values)
    
    # Generate all combinations
    configs = []
    for value_combination in itertools.product(*param_values):
        config = copy.deepcopy(template)
        for param_path, value in zip(param_names, value_combination):
            # Handle nested parameters
            keys = param_path.split('.')
            target = config
            for key in keys[:-1]:
                target = target[key]
            target[keys[-1]] = value
            
            # Update run_name to reflect parameters
            param_str = '_'.join(f"{k.split('.')[-1]}{v}" for k, v in zip(param_names, value_combination))
            config['parameters']['run_name'] = f"{config['parameters']['run_name']}_{param_str}"
        
        configs.append(config)
    
    return configs

def run_single_prediction(config: Dict, output_dir: Path):
    """Run a single prediction with given configuration."""
    # Create temporary input file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_input = output_dir / f"input_{timestamp}.json"
    with open(temp_input, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run prediction
    try:
        run_prediction(['--input', str(temp_input), '--output-dir', str(output_dir)])
    finally:
        # Clean up temporary file
        temp_input.unlink()

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configs = []
    
    if args.input_dir:
        # Process all JSON files in input directory
        input_dir = Path(args.input_dir)
        for input_file in input_dir.glob('*.json'):
            with open(input_file, 'r') as f:
                configs.append(json.load(f))
    
    elif args.input_template and args.param_sweep:
        # Generate parameter sweep configurations
        param_ranges = load_param_sweep(args.param_sweep)
        configs = generate_param_combinations(args.input_template, param_ranges)
    
    else:
        raise ValueError("Either --input-dir or both --input-template and --param-sweep must be provided")
    
    # Run predictions in parallel
    with ProcessPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [
            executor.submit(run_single_prediction, config, output_dir)
            for config in configs
        ]
        for future in futures:
            future.result()  # Wait for completion and raise any exceptions

if __name__ == "__main__":
    main() 