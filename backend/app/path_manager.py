import os
import json
from pathlib import Path
from typing import Optional
from .schemas import PathConfig
import sys

class PathManager:
    def __init__(self):
        self.config_file = Path(__file__).parent.parent / "path_config.json"
        self._config: Optional[PathConfig] = None
        self._load_config()
    
    def _get_default_config(self) -> PathConfig:
        """Generate default path configuration based on current environment"""
        # Get the directory where this file is located
        current_file_dir = Path(__file__).parent
        
        # Determine workspace directory (go up to the parent of backend)
        backend_dir = str(current_file_dir.parent)
        workspace_dir = str(Path(backend_dir).parent)
        
        # Try to detect common paths
        evaluation_dir = str(Path(workspace_dir) / "evaluation")
        
        # Try to detect Python path
        python_path = sys.executable if hasattr(sys, 'executable') else "/usr/bin/python3"
        
        # Try to detect conda environment - use CONDA_PREFIX if available, otherwise None
        # Users can configure this through the path_config.json file
        conda_env = os.environ.get('CONDA_PREFIX', None)
        
        # Default output directories - Fix path construction issues
        output_dir = str(Path(workspace_dir) / "evaluation" / "outputs")
        exports_dir = str(Path(workspace_dir) / "evaluation" / "exports")
        logs_dir = str(Path(backend_dir) / "logs")
        scripts_dir = str(Path(backend_dir) / "scripts")
        job_db_path = str(Path(backend_dir) / "job_db.json")
        
        # Debug logging
        print(f"Path detection debug:")
        print(f"  current_file_dir: {current_file_dir}")
        print(f"  backend_dir: {backend_dir}")
        print(f"  workspace_dir: {workspace_dir}")
        print(f"  evaluation_dir: {evaluation_dir}")
        print(f"  scripts_dir: {scripts_dir}")
        
        return PathConfig(
            workspace_dir=workspace_dir,
            evaluation_dir=evaluation_dir,
            backend_dir=backend_dir,
            python_path=python_path,
            conda_env_path=conda_env,
            output_dir=output_dir,
            exports_dir=exports_dir,
            logs_dir=logs_dir,
            scripts_dir=scripts_dir,
            job_db_path=job_db_path,
            openai_api_key=None,  # Will be set by user through settings
            hf_token=None  # Will be set by user through settings
        )
    
    def _load_config(self):
        """Load configuration from file or create default"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    config_data = json.load(f)
                self._config = PathConfig(**config_data)
                # Validate and fix scripts_dir path
                if self._config.scripts_dir and "backend/backend" in self._config.scripts_dir:
                    # Fix duplicate backend path
                    fixed_scripts_dir = self._config.scripts_dir.replace("backend/backend", "backend")
                    print(f"Fixing scripts_dir path: {self._config.scripts_dir} -> {fixed_scripts_dir}")
                    self._config.scripts_dir = fixed_scripts_dir
            else:
                self._config = self._get_default_config()
                self._save_config()
            
            # Ensure exports directory exists
            if self._config.exports_dir:
                os.makedirs(self._config.exports_dir, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not load path config: {e}")
            self._config = self._get_default_config()
    
    def _save_config(self):
        """Save current configuration to file"""
        try:
            os.makedirs(self.config_file.parent, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self._config.dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save path config: {e}")
    
    def get_config(self) -> PathConfig:
        """Get current path configuration"""
        return self._config
    
    def reload_config(self):
        """Reload configuration from file"""
        self._load_config()
    
    def update_config(self, new_config: PathConfig):
        """Update path configuration"""
        self._config = new_config
        self._save_config()
    
    def reset_to_default(self):
        """Reset configuration to defaults"""
        self._config = self._get_default_config()
        self._save_config()
    
    def validate_paths(self) -> dict:
        """Validate that all configured paths exist and are accessible"""
        errors = []
        warnings = []
        
        config = self._config
        
        # Check if directories exist
        for path_name, path_value in [
            ("evaluation_dir", config.evaluation_dir),
            ("backend_dir", config.backend_dir),
            ("output_dir", config.output_dir),
            ("logs_dir", config.logs_dir),
            ("scripts_dir", config.scripts_dir)
        ]:
            if not os.path.exists(path_value):
                errors.append(f"{path_name}: {path_value} does not exist")
            elif not os.path.isdir(path_value):
                errors.append(f"{path_name}: {path_value} is not a directory")
        
        # Check if files exist
        if not os.path.exists(config.python_path):
            errors.append(f"python_path: {config.python_path} does not exist")
        
        # Check if math_eval.py exists
        math_eval_path = os.path.join(config.evaluation_dir, "math_eval.py")
        if not os.path.exists(math_eval_path):
            errors.append(f"math_eval.py not found in {config.evaluation_dir}")
        
        # Check if conda environment exists (only if configured)
        if config.conda_env_path and not os.path.exists(config.conda_env_path):
            warnings.append(f"conda_env_path: {config.conda_env_path} does not exist")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

# Global path manager instance
path_manager = PathManager() 