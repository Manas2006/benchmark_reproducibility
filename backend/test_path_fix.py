#!/usr/bin/env python3
"""
Test path fix script
"""
import sys
import os
from pathlib import Path

    # Add app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from path_manager import PathManager

def test_path_fix():
    """Test path fix functionality"""
    print("=== Testing Path Fix ===")
    
    # Create PathManager instance
    pm = PathManager()
    config = pm.get_config()
    
    print(f"workspace_dir: {config.workspace_dir}")
    print(f"backend_dir: {config.backend_dir}")
    print(f"scripts_dir: {config.scripts_dir}")
    print(f"evaluation_dir: {config.evaluation_dir}")
    print(f"output_dir: {config.output_dir}")
    print(f"logs_dir: {config.logs_dir}")
    
    # Check if scripts_dir exists
    scripts_path = Path(config.scripts_dir)
    print(f"\nscripts_dir exists: {scripts_path.exists()}")
    print(f"scripts_dir is_dir: {scripts_path.is_dir()}")
    
    # Check if we can write files
    test_file = scripts_path / "test_write.txt"
    try:
        test_file.write_text("test")
        print(f"Can write file to scripts_dir: {test_file}")
        test_file.unlink()  # Delete test file
        print("Test file deleted")
    except Exception as e:
        print(f"Cannot write file to scripts_dir: {e}")
    
    # Check for duplicate backend paths
    if "backend/backend" in config.scripts_dir:
        print("❌ Found duplicate backend path!")
    else:
        print("✅ scripts_dir path is correct")

if __name__ == "__main__":
    test_path_fix() 