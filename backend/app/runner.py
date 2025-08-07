import asyncio
import subprocess
import sys
import os
import time
import uuid
import shlex
import re
from typing import Dict, Any, Optional
from pathlib import Path
from .schemas import EvalRequest, JobStatus, PathConfig
from .enums import Backend
from .path_manager import path_manager
import json

# Global job database
job_db: dict[str, dict] = {}

def save_job_db():
    config = path_manager.get_config()
    serializable_db = {}
    for jid, info in job_db.items():
        d = info.copy()
        d.pop("proc", None)
        d.pop("cli", None)
        serializable_db[jid] = d
    with open(config.job_db_path, "w") as f:
        json.dump(serializable_db, f)

def load_job_db():
    global job_db
    config = path_manager.get_config()
    try:
        with open(config.job_db_path, "r") as f:
            job_db = json.load(f)
    except Exception:
        job_db = {}

# Load job_db on startup
load_job_db()

def is_real_error(error_content: str) -> bool:
    """
    Intelligently determine if an error file contains actual errors that caused task failure.
    
    Args:
        error_content: Content of the error file
        
    Returns:
        True if the error file contains real errors that caused task failure, False otherwise
    """
    # Convert to lowercase for case-insensitive matching
    content_lower = error_content.lower()
    
    # Patterns that indicate real errors (task failure)
    real_error_patterns = [
        r'traceback\s*\(most recent call last\)',
        r'oserror:.*does not appear to have a file named',
        r'filenotfounderror:',
        r'modulenotfounderror:',
        r'importerror:',
        r'keyboardinterrupt',
        r'systemexit',
        r'killed',
        r'signal.*killed',
        r'out of memory',
        r'cuda out of memory',
        r'oom',
        r'segmentation fault',
        r'bus error',
        r'fatal error',
        r'critical error',
        r'failed to load model',
        r'failed to download',
        r'network error',
        r'connection error',
        r'timeout',
        r'job failed',
        r'job cancelled',
        r'job killed',
        r'exit code [1-9]',
        r'return code [1-9]',
        r'error.*exit',
        r'failed.*exit'
    ]
    
    # Patterns that are warnings or non-critical errors (should not cause task failure)
    warning_patterns = [
        r'condaerror: run \'conda init\'',
        r'futurewarning:',
        r'deprecationwarning:',
        r'userwarning:',
        r'warning:',
        r'info:',
        r'note:',
        r'debug:',
        r'verbose:',
        r'loading.*checkpoint.*completed',
        r'processed prompts.*completed',
        r'evaluate.*completed',
        r'saved to.*jsonl',
        r'gsm8k.*avg',
        r'monitor_response:',
        r'monitor_epoch:',
        r'unsolved samples: 0',
        r'num_samples.*num_scores',
        r'acc:.*%',
        r'accuracy:.*%'
    ]
    
    # Check for real error patterns
    for pattern in real_error_patterns:
        if re.search(pattern, content_lower):
            return True
    
    # Check if there are any real error indicators without corresponding success indicators
    has_real_errors = any(re.search(pattern, content_lower) for pattern in real_error_patterns)
    has_success_indicators = any(re.search(pattern, content_lower) for pattern in warning_patterns)
    
    # If we have real errors but no success indicators, it's likely a real failure
    if has_real_errors and not has_success_indicators:
        return True
    
    # If we have success indicators (like completion messages), it's likely not a real failure
    if has_success_indicators:
        return False
    
    # Check for specific error patterns that indicate task completion despite warnings
    completion_indicators = [
        r'processed prompts: 100%',
        r'evaluate: 100%',
        r'saved to.*\.jsonl',
        r'gsm8k.*avg',
        r'accuracy:',
        r'acc:'
    ]
    
    if any(re.search(pattern, content_lower) for pattern in completion_indicators):
        return False
    
    # Default: if we can't determine, assume it's not a real error
    return False

def get_next_local_job_id():
    config = path_manager.get_config()
    os.makedirs(config.logs_dir, exist_ok=True)
    counter_file = os.path.join(config.logs_dir, "job_counter.txt")
    if not os.path.exists(counter_file):
        with open(counter_file, 'w') as f:
            f.write('1')
        return 1
    with open(counter_file, 'r+') as f:
        val = int(f.read().strip() or '1')
        f.seek(0)
        f.write(str(val + 1))
        f.truncate()
    return val

class MathEvalRunner:
    def __init__(self):
        self.config = path_manager.get_config()
        self.evaluation_dir = Path(self.config.evaluation_dir)
        self.scripts_dir = Path(self.config.scripts_dir)
        # Create scripts directory and its parent directories if they don't exist
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        
    def _build_cli_args(self, req: EvalRequest, job_id: str = None) -> list[str]:
        """Build command line arguments for math_eval.py"""
        # Use path config from request if provided, otherwise use default
        path_config = req.path_config if req.path_config else self.config
        
        # Extract repository ID from Hugging Face URL if needed
        model_name = req.model
        if model_name.startswith('https://huggingface.co/'):
            model_name = model_name.replace('https://huggingface.co/', '')
        elif model_name.startswith('http://huggingface.co/'):
            model_name = model_name.replace('http://huggingface.co/', '')
        
        cli = [
            path_config.python_path, "-u", str(self.evaluation_dir / "math_eval.py"),
            "--model_name_or_path", model_name,
            "--data_names", req.dataset,
                    # Use configurable output_dir
        "--output_dir", f"{path_config.output_dir}/{model_name.split('/')[-1]}",
            "--split", "test",
            "--num_test_sample", "-1",  # Full dataset
            "--seed", str(req.seed),
            "--start", "0",
            "--end", "-1",
            "--temperature", str(req.temperature),
            "--n_sampling", str(req.n_sampling),
            "--top_p", str(req.top_p),
            # --max_tokens_per_call handled below
            "--use_vllm",  # Disabled to support models not compatible with vLLM
            "--save_outputs",
            "--overwrite",
            "--use_safetensors"
        ]
        
        # Add prompt-related arguments
        if req.prompt and req.prompt.strip():
            # Use custom prompt template - only add if not empty
            cli.extend(["--prompt", shlex.quote(req.prompt)])
        if req.prompt_type:
            # Use standard prompt type
            cli.extend(["--prompt_type", req.prompt_type])
        else:
            # Default to cot if no prompt_type is provided
            cli.extend(["--prompt_type", "cot"])
        # Add optional parameters
        if req.top_k > 0:
            cli.extend(["--top_k", str(req.top_k)])
        elif req.top_k == 0:
            cli.extend(["--top_k", "-1"])  # Use -1 to disable top_k in vLLM
        # Add max_tokens if present
        max_tokens = getattr(req, 'max_tokens', None)
        if max_tokens is not None:
            cli.extend(["--max_tokens_per_call", str(max_tokens)])
        else:
            cli.extend(["--max_tokens_per_call", "2048"])
        
        # Add job_id if provided
        if job_id:
            cli.extend(["--job_id", job_id])
        
        # Compute result file path
        # This matches the math_eval.py output naming convention
        split = "test"
        num_test_sample = "-1"
        seed = str(req.seed)
        temperature = str(req.temperature)
        start = "0"
        end = "-1"
        model_name = model_name.split("/")[-1]
        dataset = req.dataset.replace(",", "_")
        
        # Determine prompt type for filename
        # This logic should match math_eval.py's filename generation
        if req.prompt and req.prompt.strip() and req.prompt_type == "custom":
            prompt_type_for_file = "custom_custom"  # matches math_eval.py logic
        elif req.prompt and req.prompt.strip():
            prompt_type_for_file = "custom"
        elif req.prompt_type:
            prompt_type_for_file = req.prompt_type
        else:
            prompt_type_for_file = "cot"
        
        # Add job_id to filename if provided to avoid overwrites
        if job_id:
            result_file = f"{path_config.output_dir}/{model_name}/{dataset}/{split}_{prompt_type_for_file}_{num_test_sample}_seed{seed}_t{temperature}_s{start}_e{end}_{job_id}.jsonl"
        else:
            result_file = f"{path_config.output_dir}/{model_name}/{dataset}/{split}_{prompt_type_for_file}_{num_test_sample}_seed{seed}_t{temperature}_s{start}_e{end}.jsonl"
        
        return cli, result_file
    
    def launch_job(self, req: EvalRequest) -> str:
        """Launch a math evaluation job using math_eval.py"""
        uuid_jid = str(uuid.uuid4())
        
        # Use path config from request if provided, otherwise use default
        path_config = req.path_config if req.path_config else self.config
        
        # Debug info: check scripts_dir path
        print(f"Debug: scripts_dir = {self.scripts_dir}")
        print(f"Debug: scripts_dir type = {type(self.scripts_dir)}")
        print(f"Debug: scripts_dir exists = {Path(self.scripts_dir).exists()}")
        
        # Ensure scripts_dir exists
        scripts_path = Path(self.scripts_dir)
        if not scripts_path.exists():
            print(f"Warning: scripts_dir does not exist: {scripts_path}")
            scripts_path.mkdir(parents=True, exist_ok=True)
            print(f"Created scripts_dir: {scripts_path}")
        
        # Build CLI args with job_id to avoid overwrites
        cli, result_file = self._build_cli_args(req, uuid_jid)
        if req.backend == Backend.local:
            local_job_id = get_next_local_job_id()
            out_file = os.path.join(path_config.logs_dir, f"qwen-math-{local_job_id}.out")
            err_file = os.path.join(path_config.logs_dir, f"qwen-math-{local_job_id}.err")
            proc = subprocess.Popen(
                cli, 
                stdout=open(out_file, 'w'),
                stderr=open(err_file, 'w'),
                text=True,
                cwd=self.evaluation_dir
            )
            job_db[uuid_jid] = {
                "status": JobStatus.RUNNING,
                "proc": proc,
                "request": req.dict(),
                "cli": cli,
                "backend": "local",
                "out_file": out_file,
                "err_file": err_file,
                "result_file": result_file,
                "local_job_id": local_job_id
            }
            save_job_db()
            return uuid_jid
        else:
            script_path = self.scripts_dir / f"run_{uuid_jid}.sh"
            sbatch_path = self.scripts_dir / f"job_{uuid_jid}.sbatch"
            
            # Debug info: check script_path
            print(f"Debug: script_path = {script_path}")
            print(f"Debug: script_path parent exists = {script_path.parent.exists()}")
            
            out_file_pattern = os.path.join(path_config.logs_dir, "qwen-math-%j.out")
            err_file_pattern = os.path.join(path_config.logs_dir, "qwen-math-%j.err")
            # Properly escape the command for shell execution
            escaped_cli = []
            for arg in cli:
                if arg == '--prompt':
                    # Skip the --prompt flag, we'll handle it specially
                    continue
                elif arg == shlex.quote(req.prompt):
                    # This is the prompt value, skip it
                    continue
                else:
                    escaped_cli.append(shlex.quote(arg))
            
            # Add the prompt argument properly quoted
            escaped_cli.append('--prompt')
            escaped_cli.append(shlex.quote(req.prompt))
            
            script_content = f"""#!/bin/bash
cd {self.evaluation_dir}

# Set Hugging Face cache to work directory

# Fix MKL threading conflict
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Activate conda environment
source {path_config.conda_env_path}/etc/profile.d/conda.sh
conda activate mathevalUI
{' '.join(escaped_cli)}
"""
            try:
                script_path.write_text(script_content)
                script_path.chmod(0o755)
                print(f"Successfully created script: {script_path}")
            except Exception as e:
                print(f"Error creating script {script_path}: {e}")
                print(f"Script parent directory: {script_path.parent}")
                print(f"Script parent exists: {script_path.parent.exists()}")
                print(f"Script parent is_dir: {script_path.parent.is_dir()}")
                raise
            if req.backend == Backend.slurm:
                sbatch_content = f"""#!/bin/bash
#SBATCH -J qwen-math-{uuid_jid}   # Job name
#SBATCH -o {out_file_pattern}      # Name of stdout output file (uses %j)
#SBATCH -e {err_file_pattern}      # Name of stderr error fifle (uses %j)
#SBATCH -p {path_config.slurm_partition}              # Queue (partition) name
#SBATCH -N 1                    # Total # of nodes
#SBATCH -n 1                    # Total # of tasks (single process for all GPUs)
#SBATCH -t {path_config.slurm_wall_time}              # Run time (hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH -A {path_config.slurm_account}             # Project/Allocation name



# Fix MKL threading conflict
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}}

{script_path}
"""
                sbatch_path.write_text(sbatch_content)
                try:
                    result = subprocess.run(
                        ["sbatch", str(sbatch_path)],
                        capture_output=True,
                        text=True,
                        cwd=self.scripts_dir
                    )
                    if result.returncode == 0:
                        slurm_jid = result.stdout.strip().split()[-1]
                        out_file = os.path.join(path_config.logs_dir, f"qwen-math-{slurm_jid}.out")
                        err_file = os.path.join(path_config.logs_dir, f"qwen-math-{slurm_jid}.err")
                        job_db[uuid_jid] = {
                            "status": JobStatus.QUEUED,
                            "request": req.dict(),
                            "cli": cli,
                            "run_path": str(script_path),
                            "sbatch_path": str(sbatch_path),
                            "slurm_jid": slurm_jid,
                            "backend": "slurm",
                            "out_file": out_file,
                            "err_file": err_file,
                            "result_file": result_file
                        }
                        save_job_db()
                        return uuid_jid
                    else:
                        job_db[uuid_jid] = {
                            "status": JobStatus.ERROR,
                            "request": req.dict(),
                            "cli": cli,
                            "run_path": str(script_path),
                            "sbatch_path": str(sbatch_path),
                            "error": f"SLURM submission failed: {result.stderr}",
                            "backend": "slurm",
                            "result_file": result_file
                        }
                        save_job_db()
                        return uuid_jid
                except Exception as e:
                    job_db[uuid_jid] = {
                        "status": JobStatus.ERROR,
                        "request": req.dict(),
                        "cli": cli,
                        "run_path": str(script_path),
                        "sbatch_path": str(sbatch_path),
                        "error": f"SLURM submission error: {str(e)}",
                        "backend": "slurm",
                        "result_file": result_file
                    }
                    save_job_db()
                    return uuid_jid
            else:
                job_db[uuid_jid] = {
                    "status": JobStatus.READY_FOR_DOWNLOAD,
                    "request": req.dict(),
                    "cli": cli,
                    "run_path": str(script_path),
                    "sbatch_path": None,
                    "backend": "bash",
                    "result_file": result_file
                }
                save_job_db()
                return uuid_jid

    def cancel_job(self, jid: str) -> bool:
        """Cancel a running job (local or SLURM)"""
        if jid not in job_db:
            return False
        job = job_db[jid]
        if job.get("backend") == "slurm" and "slurm_jid" in job:
            try:
                result = subprocess.run(["scancel", job["slurm_jid"]], capture_output=True, text=True)
                job["status"] = JobStatus.ERROR
                save_job_db()
                return result.returncode == 0
            except Exception:
                return False
        elif job.get("backend") == "local" and "proc" in job:
            try:
                proc = job["proc"]
                proc.terminate()
                job["status"] = JobStatus.ERROR
                save_job_db()
                return True
            except Exception:
                return False
        return False

    def delete_job(self, jid: str) -> bool:
        """Delete a job from the job_db, cancel if running"""
        if jid not in job_db:
            return False
        self.cancel_job(jid)
        try:
            del job_db[jid]
            save_job_db()
            return True
        except Exception:
            return False
    
    def get_job_status(self, jid: str) -> Dict[str, Any]:
        """Get the status of a running or completed job"""
        if jid not in job_db:
            return {"status": "NOT_FOUND", "message": "Job not found"}
        
        job_info = job_db[jid]
        original_status = job_info.get("status")
        
        # Check if process is still running (local jobs)
        if "proc" in job_info:
            proc = job_info["proc"]
            if proc.poll() is not None:
                # Process finished
                if proc.returncode == 0:
                    job_info["status"] = JobStatus.DONE
                else:
                    job_info["status"] = JobStatus.ERROR
                job_info["return_code"] = proc.returncode
        
        # Check SLURM job status
        elif job_info.get("backend") == "slurm" and "slurm_jid" in job_info:
            try:
                result = subprocess.run(
                    ["squeue", "-j", job_info["slurm_jid"], "-h", "-o", "%T"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    state = result.stdout.strip()
                    if state == "PENDING":
                        job_info["status"] = JobStatus.QUEUED
                    elif state == "RUNNING":
                        job_info["status"] = JobStatus.RUNNING
                    else:
                        # Job is no longer in queue, check if it completed successfully
                        config = path_manager.get_config()
                        output_file = Path(job_info.get("out_file", f"{config.logs_dir}/qwen-math-{job_info.get('slurm_jid','')}.out"))
                        error_file = Path(job_info.get("err_file", f"{config.logs_dir}/qwen-math-{job_info.get('slurm_jid','')}.err"))
                        result_file = Path(job_info.get("result_file", ""))
                        
                        # Check error file first for failure indicators
                        if error_file.exists():
                            with open(error_file, 'r') as f:
                                error_content = f.read()
                                if is_real_error(error_content):
                                    job_info["status"] = JobStatus.ERROR
                                    job_info["error"] = f"Job failed with errors. Check error log for details."
                                    # Save updated status to file
                                    if original_status != job_info["status"]:
                                        save_job_db()
                                    return job_info
                        
                        # Check if result file exists and is non-empty
                        if result_file.exists() and result_file.stat().st_size > 0:
                            job_info["status"] = JobStatus.DONE
                            job_info.pop("error", None)
                        # Fallback: check if output file exists and has content
                        elif output_file.exists() and output_file.stat().st_size > 0:
                            job_info["status"] = JobStatus.DONE
                            job_info.pop("error", None)
                        else:
                            job_info["status"] = JobStatus.ERROR
                            job_info["error"] = f"Job completed but output/result files missing or empty"
                else:
                    # Failed to check SLURM status
                    config = path_manager.get_config()
                    output_file = Path(job_info.get("out_file", f"{config.logs_dir}/qwen-math-{job_info.get('slurm_jid','')}.out"))
                    error_file = Path(job_info.get("err_file", f"{config.logs_dir}/qwen-math-{job_info.get('slurm_jid','')}.err"))
                    result_file = Path(job_info.get("result_file", ""))
                    
                    # Check error file first for failure indicators
                    if error_file.exists():
                        with open(error_file, 'r') as f:
                            error_content = f.read()
                            if is_real_error(error_content):
                                job_info["status"] = JobStatus.ERROR
                                job_info["error"] = f"Job failed with errors. Check error log for details."
                                # Save updated status to file
                                if original_status != job_info["status"]:
                                    save_job_db()
                                return job_info
                    
                    # If result file exists and is non-empty, mark as DONE
                    if result_file.exists() and result_file.stat().st_size > 0:
                        job_info["status"] = JobStatus.DONE
                        job_info.pop("error", None)
                    # Fallback: check if output file exists and has content
                    elif output_file.exists() and output_file.stat().st_size > 0:
                        job_info["status"] = JobStatus.DONE
                        job_info.pop("error", None)
                    else:
                        job_info["status"] = JobStatus.ERROR
                        job_info["error"] = f"Failed to check SLURM status: {result.stderr}"
            except Exception as e:
                job_info["status"] = JobStatus.ERROR
                job_info["error"] = f"Error checking SLURM status: {str(e)}"
        
        # Save updated status to file if it changed
        if original_status != job_info.get("status"):
            save_job_db()
        
        return job_info
    
    async def stream_job_progress(self, jid: str):
        """Stream real-time progress updates from a job"""
        if jid not in job_db or "proc" not in job_db[jid]:
            yield {"error": "Job not found or not running"}
            return
        
        proc = job_db[jid]["proc"]
        
        while True:
            line = proc.stdout.readline()
            if line:
                yield {"log": line.strip()}
            
            if proc.poll() is not None:
                # Process finished
                if proc.returncode == 0:
                    job_db[jid]["status"] = JobStatus.DONE
                else:
                    job_db[jid]["status"] = JobStatus.ERROR
                # Save updated status to file
                save_job_db()
                yield {"status": job_db[jid]["status"], "return_code": proc.returncode}
                break
            
            await asyncio.sleep(0.1)

# Global runner instance
runner = MathEvalRunner()

def launch_job(req: EvalRequest) -> str:
    """Global function to launch a job"""
    return runner.launch_job(req)

def get_job_status(jid: str) -> Dict[str, Any]:
    """Global function to get job status"""
    return runner.get_job_status(jid) 

def cancel_job(jid: str) -> bool:
    return runner.cancel_job(jid)

def delete_job(jid: str) -> bool:
    return runner.delete_job(jid)

def get_job_raw_data(job_id: str) -> dict:
    """Extract raw answer data from job results for CoT analysis"""
    from fastapi import HTTPException
    import time
    
    if job_id not in job_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_db[job_id]
    result_file = job_info.get("result_file")
    
    if not result_file:
        raise HTTPException(status_code=404, detail="No result file found for this job")
    
    result_path = Path(result_file)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result file not found on disk")
    
    # Check cache first
    cached_data = get_cached_raw_data(job_id)
    if cached_data:
        return cached_data
    
    # Read and parse JSONL file
    raw_data = []
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        record = json.loads(line)
                        raw_data.append({
                            'idx': record.get('idx'),
                            'question': record.get('question', ''),
                            'answer': record.get('answer', ''),
                            'gt': record.get('gt', ''),
                            'gt_cot': record.get('gt_cot', ''),
                            'code': record.get('code', []),
                            'pred': record.get('pred', []),
                            'score': record.get('score', [])
                        })
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
                        continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading result file: {str(e)}")
    
    # Cache the data
    result = {
        "job_id": job_id,
        "total_samples": len(raw_data),
        "data": raw_data,
        "metadata": {
            "result_file": str(result_path),
            "file_size_bytes": result_path.stat().st_size,
            "last_modified": result_path.stat().st_mtime
        }
    }
    
    cache_raw_data(job_id, result)
    return result

def get_cached_raw_data(job_id: str) -> Optional[dict]:
    """Retrieve cached raw data if available and fresh"""
    if job_id in job_db and 'cached_raw_data' in job_db[job_id]:
        cached = job_db[job_id]['cached_raw_data']
        # Cache valid for 1 hour
        if time.time() - cached['timestamp'] < 3600:
            return cached['data']
    return None

def cache_raw_data(job_id: str, raw_data: dict):
    """Cache raw data in job database to avoid repeated file reads"""
    import time
    if job_id in job_db:
        job_db[job_id]['cached_raw_data'] = {
            'data': raw_data,
            'timestamp': time.time()
        }
        # Note: We don't call save_job_db() here to avoid frequent disk writes
        # Cache will be lost on server restart, which is acceptable 