#!/usr/bin/env python3

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from omegaconf import OmegaConf, DictConfig


# Dictionary mapping RL algorithms to their scripts and configs
ALGORITHMS = {
    "acegen_molopt": (
        Path(__file__).parent / "acegen" / "ag.py",
        Path(__file__).parent / "acegen" / "config_molopt_denovo.yaml"
    ),
    "acegen_pract": (
        Path(__file__).parent / "acegen" / "ag.py",
        Path(__file__).parent / "acegen" / "config_practical_denovo.yaml"
    ),
    "a2c": (
        Path(__file__).parent / "a2c" / "a2c.py",
        Path(__file__).parent / "a2c" / "config_denovo.yaml"
    ),
    "ahc": (
        Path(__file__).parent / "ahc" / "ahc.py",
        Path(__file__).parent / "ahc" / "config_denovo.yaml"
    ),
    "augmem": (
        Path(__file__).parent / "augmented_memory" / "augmem.py",
        Path(__file__).parent / "augmented_memory" / "config_denovo.yaml"
    ),
    "dpo": (
        Path(__file__).parent / "dpo" / "dpo.py",
        Path(__file__).parent / "dpo" / "config_denovo.yaml"
    ),
    "hill_climb": (
        Path(__file__).parent / "hill_climb" / "hill_climb.py",
        Path(__file__).parent / "hill_climb" / "config_denovo.yaml"
    ),
    "ppo": (
        Path(__file__).parent / "ppo" / "ppo.py",
        Path(__file__).parent / "ppo" / "config_denovo.yaml"
    ),
    "reinforce": (
        Path(__file__).parent / "reinforce" / "reinforce.py",
        Path(__file__).parent / "reinforce" / "config_denovo.yaml"
    ),
    "reinvent": (
        Path(__file__).parent / "reinvent" / "reinvent.py",
        Path(__file__).parent / "reinvent" / "config_denovo.yaml"
    ),
    "screening": (
        Path(__file__).parent / "screening" / "screening.py",
        Path(__file__).parent / "screening" / "config.yaml"
    )
}


def load_config(config_path: Path) -> DictConfig:
    """Load a YAML configuration file and return as OmegaConf DictConfig."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return OmegaConf.create(config_dict)


def update_config_from_args(cfg: DictConfig, overrides: Dict[str, Any]) -> DictConfig:
    """Update configuration with command line overrides."""
    for key, value in overrides.items():
        # Handle nested keys (e.g., "model.lr" -> cfg.model.lr = value)
        keys = key.split('.')
        current = cfg
        
        # Navigate to the parent of the final key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value, converting string representations to appropriate types
        final_key = keys[-1]
        
        # Try to convert to appropriate type
        if isinstance(value, str):
            # Try int first
            try:
                value = int(value)
            except ValueError:
                # Try float
                try:
                    value = float(value)
                except ValueError:
                    # Try boolean
                    if value.lower() in ('true', 'false'):
                        value = value.lower() == 'true'
                    elif value.lower() == 'null':
                        value = None
                    # Otherwise keep as string
        
        current[final_key] = value
    
    return cfg


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run AceGen scripts with configuration files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Usage:
    python run.py <algorithm_name> [--molscore_config <config_name>] [--seed <seed>] [--key=value ...]   

Examples:
  python run.py ahc --molscore_config config.yaml
  python run.py reinforce --molscore_config config.yaml --model=gru_chembl lr=0.001
        """
    )
    
    parser.add_argument(
        "algorithm",
        choices=list(ALGORITHMS.keys()),
        help="Name of the algorithm to run"
    )
    
    parser.add_argument(
        "--molscore_config",
        type=str,
        help="MolScore configuration file path."
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="This is ignored. For compatibility with MolScore baselines."
    )
    
    # Parse known args to separate script selection from config overrides
    args, unknown = parser.parse_known_args()
    
    # Parse config overrides from remaining arguments
    config_overrides = {}
    for arg in unknown:
        if arg.startswith('--') and '=' in arg:
            key, value = arg[2:].split('=', 1)
            config_overrides[key] = value
        elif arg.startswith('--'):
            # Handle boolean flags (--flag means True)
            key = arg[2:]
            config_overrides[key] = True
    
    args.config_overrides = config_overrides
    return args


def main():
    args = parse_arguments()
    
    # Handle list configs request
    script_path, config_path = ALGORITHMS[args.algorithm]
    script_dir = script_path.parent
    
    if not script_path.exists():
        print(f"Script not found: {script_path}")
        return
    
    if not config_path.exists():
        print(f"Config file not found {config_path}")
        return
    
    # Load and update configuration
    print(f"Loading config: {args.molscore_config}")
    molscore_cfg = load_config(Path(args.molscore_config))
    print(f"Loading config: {config_path}")
    default_cfg = load_config(Path(config_path))
    
    # Merge MolScore configs first, specific to acegen
    molscore_cfg["agent_name"] = molscore_cfg.pop("model_name") # rename
    molscore_cfg["log_dir"] = molscore_cfg.pop("output_dir") # rename
    cfg = OmegaConf.merge(default_cfg, molscore_cfg)
    
    # Update experiment_name which determines directory
    cfg["experiment_name"] = molscore_cfg["molscore_task"]
    
    # Update seed
    cfg["seed"] = args.seed
    
    # Update with overrides
    if args.config_overrides:
        print(f"Applying config overrides: {args.config_overrides}")
        cfg = update_config_from_args(cfg, args.config_overrides)
    
    # Load module and run
    sys.path.insert(0, str(script_dir))
    module = importlib.import_module(script_path.stem)
    print("Running script")
    module.main(cfg)


if __name__ == "__main__":
    main()