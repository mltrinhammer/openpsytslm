"""Utility functions for process_data module."""

import yaml
from pathlib import Path
from typing import Optional


def load_config(config_path: str) -> Optional[dict]:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Dictionary with config values, or empty dict if file not found
    """
    path = Path(config_path)
    if not path.exists():
        print(f"[utils] Config file not found: {config_path}, using defaults")
        return {}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        print(f"[utils] Loaded config from: {config_path}")
        return config
    except Exception as e:
        print(f"[utils] Error loading config: {e}")
        return {}
