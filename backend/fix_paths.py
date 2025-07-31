#!/usr/bin/env python3
"""
Fix path configuration issues
"""
import json
import os
from pathlib import Path

def fix_path_config():
    """Fix path issues in path_config.json"""
    config_file = Path(__file__).parent / "path_config.json"
    
    if not config_file.exists():
        print(f"Config file does not exist: {config_file}")
        return
    
    # Read current configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("Current configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Check and fix scripts_dir
    if 'scripts_dir' in config:
        scripts_dir = config['scripts_dir']
        if 'backend/backend' in scripts_dir:
            # Fix duplicate backend path
            fixed_scripts_dir = scripts_dir.replace('backend/backend', 'backend')
            print(f"\nFixing scripts_dir:")
            print(f"  From: {scripts_dir}")
            print(f"  To: {fixed_scripts_dir}")
            config['scripts_dir'] = fixed_scripts_dir
    
    # Check and fix other possible path issues
    for key in ['workspace_dir', 'backend_dir', 'evaluation_dir', 'output_dir', 'logs_dir']:
        if key in config:
            path = config[key]
            if 'backend/backend' in path:
                fixed_path = path.replace('backend/backend', 'backend')
                print(f"\nFixing {key}:")
                print(f"  From: {path}")
                print(f"  To: {fixed_path}")
                config[key] = fixed_path
    
    # Save fixed configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nConfiguration saved to: {config_file}")
    
    # Validate fixed paths
    print("\nValidating fixed paths:")
    for key, value in config.items():
        if key.endswith('_dir') or key.endswith('_path'):
            path = Path(value)
            exists = path.exists()
            is_dir = path.is_dir() if path.exists() else False
            print(f"  {key}: {value}")
            print(f"    exists: {exists}")
            print(f"    is_dir: {is_dir}")

def create_missing_dirs():
    """Create missing directories"""
    config_file = Path(__file__).parent / "path_config.json"
    
    if not config_file.exists():
        print(f"Config file does not exist: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print("\nCreating missing directories:")
    for key, value in config.items():
        if key.endswith('_dir'):
            path = Path(value)
        if not path.exists():
            print(f"Creating directory: {path}")
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            print(f"Warning: {path} exists but is not a directory")

if __name__ == "__main__":
    print("=== Fixing Path Configuration ===")
    fix_path_config()
    create_missing_dirs()
    print("\n=== Fix Complete ===") 