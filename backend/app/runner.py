import asyncio
import subprocess
import sys
import os
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
from .schemas import EvalRequest, JobStatus
from .enums import Backend
import json

# Global job database
job_db: dict[str, dict] = {}
JOB_DB_PATH = "/work/10757/manasp123/qwen-eval-ui/backend/job_db.json"

def save_job_db():
    serializable_db = {}
    for jid, info in job_db.items():
        d = info.copy()
        d.pop("proc", None)
        d.pop("cli", None)
        serializable_db[jid] = d
    with open(JOB_DB_PATH, "w") as f:
        json.dump(serializable_db, f)

def load_job_db():
    global job_db
    try:
        with open(JOB_DB_PATH, "r") as f:
            job_db = json.load(f)
    except Exception:
        job_db = {}

# Load job_db on startup
load_job_db()

class MathEvalRunner:
    def __init__(self):
        self.evaluation_dir = Path(__file__).parent.parent.parent / "evaluation"
        self.scripts_dir = Path("/work/10757/manasp123/qwen-eval-ui/backend/scripts")
        self.scripts_dir.mkdir(exist_ok=True)
        
    def _build_cli_args(self, req: EvalRequest) -> list[str]:
        """Build command line arguments for math_eval.py"""
        cli = [
            "/work/10757/manasp123/ls6/miniconda3/envs/qwen-eval/bin/python", "-u", str(self.evaluation_dir / "math_eval.py"),
            "--model_name_or_path", req.model,
            "--data_names", req.dataset,
            "--output_dir", f"./output/{req.model.split('/')[-1]}",
            "--prompt_type", "tool-integrated",  # Default to TIR
            "--split", "test",
            "--num_test_sample", "-1",  # Full dataset
            "--seed", str(req.seed),
            "--start", "0",
            "--end", "-1",
            "--temperature", str(req.temperature),
            "--n_sampling", str(req.n_sampling),
            "--top_p", str(req.top_p),
            "--max_tokens_per_call", "2048",
            "--use_vllm",
            "--save_outputs",
            "--overwrite"
        ]
        
        # Add optional parameters
        if req.top_k > 0:
            cli.extend(["--top_k", str(req.top_k)])
            
        return cli
    
    def launch_job(self, req: EvalRequest) -> str:
        """Launch a math evaluation job using math_eval.py"""
        # Generate unique job ID
        jid = str(uuid.uuid4())
        
        # Build CLI arguments
        cli = self._build_cli_args(req)
        
        if req.backend == Backend.local:
            # Run locally with subprocess
            proc = subprocess.Popen(
                cli, 
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, 
                text=True,
                cwd=self.evaluation_dir
            )
            job_db[jid] = {
                "status": JobStatus.RUNNING,
                "proc": proc,
                "request": req.dict(),
                "cli": cli
            }
            save_job_db()
        else:
            # Create script files for bash/slurm execution
            script_path = self.scripts_dir / f"run_{jid}.sh"
            sbatch_path = self.scripts_dir / f"job_{jid}.sbatch"
            
            # Write run script
            script_content = f"""#!/bin/bash
cd {self.evaluation_dir}
{' '.join(cli)}
"""
            script_path.write_text(script_content)
            script_path.chmod(0o755)
            
            # Write SLURM batch file if needed
            if req.backend == Backend.slurm:
                sbatch_content = f"""#!/bin/bash
#SBATCH -J qwen-math-{jid}   # Job name
#SBATCH -o /work/10757/manasp123/qwen-eval-ui/logs/qwen-math-{jid}.out      # Name of stdout output file
#SBATCH -e /work/10757/manasp123/qwen-eval-ui/logs/qwen-math-{jid}.err      # Name of stderr error file
#SBATCH -p gpu-a100-small              # Queue (partition) name
#SBATCH -N 1                    # Total # of nodes
#SBATCH -n 1                    # Total # of tasks (single process for all GPUs)
#SBATCH -t 6:00:00              # Run time (hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH -A CCR24036             # Project/Allocation name

export CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}}

{script_path}
"""
                sbatch_path.write_text(sbatch_content)
                
                # Actually submit to SLURM
                try:
                    result = subprocess.run(
                        ["sbatch", str(sbatch_path)],
                        capture_output=True,
                        text=True,
                        cwd=self.scripts_dir
                    )
                    if result.returncode == 0:
                        # Extract SLURM job ID from output (e.g., "Submitted batch job 12345")
                        slurm_jid = result.stdout.strip().split()[-1]
                        job_db[jid] = {
                            "status": JobStatus.RUNNING,
                            "request": req.dict(),
                            "cli": cli,
                            "run_path": str(script_path),
                            "sbatch_path": str(sbatch_path),
                            "slurm_jid": slurm_jid
                        }
                        save_job_db()
                    else:
                        job_db[jid] = {
                            "status": JobStatus.ERROR,
                            "request": req.dict(),
                            "cli": cli,
                            "run_path": str(script_path),
                            "sbatch_path": str(sbatch_path),
                            "error": f"SLURM submission failed: {result.stderr}"
                        }
                        save_job_db()
                except Exception as e:
                    job_db[jid] = {
                        "status": JobStatus.ERROR,
                        "request": req.dict(),
                        "cli": cli,
                        "run_path": str(script_path),
                        "sbatch_path": str(sbatch_path),
                        "error": f"SLURM submission error: {str(e)}"
                    }
                    save_job_db()
            else:
                # For bash backend, just mark as ready
                job_db[jid] = {
                    "status": JobStatus.READY_FOR_DOWNLOAD,
                    "request": req.dict(),
                    "cli": cli,
                    "run_path": str(script_path),
                    "sbatch_path": None
                }
                save_job_db()
        
        return jid
    
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
        elif "slurm_jid" in job_info:
            try:
                result = subprocess.run(
                    ["squeue", "-j", job_info["slurm_jid"]],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    if job_info["slurm_jid"] in result.stdout:
                        # Job is still in queue or running
                        job_info["status"] = JobStatus.RUNNING
                    else:
                        # Job is no longer in queue, check if it completed successfully
                        # Check for output files to determine success/failure
                        output_file = Path(f"/work/10757/manasp123/qwen-eval-ui/logs/qwen-math-{jid}.out")
                        if output_file.exists():
                            job_info["status"] = JobStatus.DONE
                        else:
                            job_info["status"] = JobStatus.ERROR
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