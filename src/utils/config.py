import yaml
import os
from typing import Dict, Any

def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """
    Loads a YAML configuration file.
    
    Args:
        config_path: Path to the .yaml configuration file.
        
    Returns:
        A dictionary containing the parsed configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config if config is not None else {}
