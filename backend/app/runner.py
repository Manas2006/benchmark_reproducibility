import asyncio
import subprocess
import sys
import os
import uuid
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
        
    def _build_cli_args(self, req: EvalRequest) -> list[str]:
        """Build command line arguments for math_eval.py"""
        # Use path config from request if provided, otherwise use default
        path_config = req.path_config if req.path_config else self.config
        
        cli = [
            path_config.python_path, "-u", str(self.evaluation_dir / "math_eval.py"),
            "--model_name_or_path", req.model,
            "--data_names", req.dataset,
            # Use configurable output_dir
            "--output_dir", f"{path_config.output_dir}/{req.model.split('/')[-1]}",
            "--prompt", req.prompt,  # Use custom prompt template
            "--split", "test",
            "--num_test_sample", "-1",  # Full dataset
            "--seed", str(req.seed),
            "--start", "0",
            "--end", "-1",
            "--temperature", str(req.temperature),
            "--n_sampling", str(req.n_sampling),
            "--top_p", str(req.top_p),
            # --max_tokens_per_call handled below
            "--use_vllm",
            "--save_outputs",
            "--overwrite"
        ]
        # Add optional parameters
        if req.top_k > 0:
            cli.extend(["--top_k", str(req.top_k)])
        # Add max_tokens if present
        max_tokens = getattr(req, 'max_tokens', None)
        if max_tokens is not None:
            cli.extend(["--max_tokens_per_call", str(max_tokens)])
        else:
            cli.extend(["--max_tokens_per_call", "2048"])
        # Compute result file path
        # This matches the math_eval.py output naming convention
        # Example: test_custom_-1_seed42_t0.0_s0_e-1.jsonl
        split = "test"
        num_test_sample = "-1"
        seed = str(req.seed)
        temperature = str(req.temperature)
        start = "0"
        end = "-1"
        model_name = req.model.split("/")[-1]
        dataset = req.dataset.replace(",", "_")
        result_file = f"{path_config.output_dir}/{model_name}/{dataset}/{split}_custom_{num_test_sample}_seed{seed}_t{temperature}_s{start}_e{end}.jsonl"
        return cli
    
    def launch_job(self, req: EvalRequest) -> str:
        """Launch a math evaluation job using math_eval.py"""
        uuid_jid = str(uuid.uuid4())
        # Always define result_file at the top
        split = "test"
        num_test_sample = "-1"
        seed = str(req.seed)
        temperature = str(req.temperature)
        start = "0"
        end = "-1"
        model_name = req.model.split("/")[-1]
        dataset = req.dataset.replace(",", "_")
        
        # Use path config from request if provided, otherwise use default
        path_config = req.path_config if req.path_config else self.config
        result_file = f"{path_config.output_dir}/{model_name}/{dataset}/{split}_custom_{num_test_sample}_seed{seed}_t{temperature}_s{start}_e{end}.jsonl"
        
        cli = self._build_cli_args(req)
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
            out_file_pattern = os.path.join(path_config.logs_dir, "qwen-math-%j.out")
            err_file_pattern = os.path.join(path_config.logs_dir, "qwen-math-%j.err")
            script_content = f"""#!/bin/bash
cd {self.evaluation_dir}
{' '.join(cli)}
"""
            script_path.write_text(script_content)
            script_path.chmod(0o755)
            if req.backend == Backend.slurm:
                sbatch_content = f"""#!/bin/bash
#SBATCH -J qwen-math-{uuid_jid}   # Job name
#SBATCH -o {out_file_pattern}      # Name of stdout output file (uses %j)
#SBATCH -e {err_file_pattern}      # Name of stderr error file (uses %j)
#SBATCH -p {path_config.slurm_partition}              # Queue (partition) name
#SBATCH -N 1                    # Total # of nodes
#SBATCH -n 1                    # Total # of tasks (single process for all GPUs)
#SBATCH -t {path_config.slurm_wall_time}              # Run time (hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH -A {path_config.slurm_account}             # Project/Allocation name

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
                        result_file = Path(job_info.get("result_file", ""))
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
                    result_file = Path(job_info.get("result_file", ""))
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