from fastapi import FastAPI, BackgroundTasks, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import json
import asyncio
import os
from pathlib import Path

from .schemas import EvalRequest, JobStatus, PathConfig, PathConfigResponse
from .runner import launch_job, job_db, get_job_status, cancel_job, delete_job
from .path_manager import path_manager

app = FastAPI(title="Qwen Math Evaluation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend
app.mount("/static", StaticFiles(directory="../frontend"), name="static")

@app.get("/")
async def root():
    """Serve the main frontend page"""
    return FileResponse("../frontend/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Path configuration endpoints
@app.get("/config/paths")
async def get_path_config():
    """Get current path configuration"""
    current_config = path_manager.get_config()
    default_config = path_manager._get_default_config()
    return PathConfigResponse(
        current_config=current_config,
        default_config=default_config
    )

@app.post("/config/paths")
async def update_path_config(config: PathConfig):
    """Update path configuration"""
    path_manager.update_config(config)
    return {"message": "Path configuration updated successfully", "config": config}

@app.post("/config/paths/reset")
async def reset_path_config():
    """Reset path configuration to defaults"""
    path_manager.reset_to_default()
    return {"message": "Path configuration reset to defaults", "config": path_manager.get_config()}

@app.get("/config/paths/validate")
async def validate_paths():
    """Validate current path configuration"""
    validation = path_manager.validate_paths()
    return validation

@app.post("/jobs")
async def create_job(req: EvalRequest):
    """Create a new evaluation job"""
    jid = launch_job(req)
    # Return only serializable data
    job_info = job_db[jid].copy()
    job_info.pop("proc", None)
    job_info.pop("cli", None)
    # Add slurm_jid if present
    slurm_jid = job_info.get("slurm_jid")
    return {"job_id": jid, "slurm_jid": slurm_jid, **job_info}

@app.post("/jobs/{jid}/cancel")
async def cancel_job_endpoint(jid: str):
    """Cancel a running job (local or SLURM)"""
    success = cancel_job(jid)
    return {"job_id": jid, "cancelled": success}

@app.delete("/jobs/{jid}")
async def delete_job_endpoint(jid: str):
    """Delete a job (and cancel if running)"""
    success = delete_job(jid)
    return {"job_id": jid, "deleted": success}

@app.get("/jobs/{jid}")
async def job_status(jid: str):
    """Get the status of a job"""
    status_info = get_job_status(jid)
    if isinstance(status_info, dict):
        status_info = status_info.copy()
        status_info.pop("proc", None)
        status_info.pop("cli", None)
    slurm_jid = status_info.get("slurm_jid")
    result_file = status_info.get("result_file")
    return {"job_id": jid, "slurm_jid": slurm_jid, **status_info, "result_file": result_file}

@app.websocket("/stream/{jid}")
async def stream(jid: str, ws: WebSocket):
    await ws.accept()
    info = job_db.get(jid)
    if not info:
        try:
            await ws.send_text(json.dumps({"error": f"Job {jid} not found (UUID or SLURM job number)"}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        return

    import os
    import asyncio
    out_path = info.get("out_file")
    err_path = info.get("err_file")
    # For SLURM jobs, if the files do not exist, try to parse the SBATCH script for the real paths
    if info.get("backend") == "slurm":
        sbatch_path = info.get("sbatch_path")
        if sbatch_path and (not out_path or not err_path):
            try:
                with open(sbatch_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    if line.startswith("#SBATCH -o"):
                        out_path = line.split(None, 2)[-1].strip()
                    if line.startswith("#SBATCH -e"):
                        err_path = line.split(None, 2)[-1].strip()
            except Exception:
                pass
    # Wait for the files to appear (up to 60 seconds)
    wait_time = 0
    while (not out_path or not os.path.exists(out_path) or not err_path or not os.path.exists(err_path)) and wait_time < 60:
        try:
            await ws.send_text(json.dumps({"waiting": f"Waiting for job output files to appear... ({wait_time}s)"}))
        except Exception:
            # WebSocket connection lost during waiting
            return
        await asyncio.sleep(2)
        wait_time += 2
    if not out_path or not os.path.exists(out_path) or not err_path or not os.path.exists(err_path):
        try:
            await ws.send_text(json.dumps({"error": f"Output or error file not found for this job after waiting. (out: {out_path}, err: {err_path})"}))
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
        return
    try:
        with open(out_path, "r") as outf, open(err_path, "r") as errf:
            outf.seek(0, os.SEEK_END)
            errf.seek(0, os.SEEK_END)
            while True:
                out_line = outf.readline()
                err_line = errf.readline()
                sent = False
                if out_line:
                    try:
                        await ws.send_text(json.dumps({"out": out_line.rstrip()}))
                        sent = True
                    except Exception:
                        # WebSocket connection lost
                        break
                if err_line:
                    try:
                        await ws.send_text(json.dumps({"err": err_line.rstrip()}))
                        sent = True
                    except Exception:
                        # WebSocket connection lost
                        break
                # Check if job is still running
                status = info.get("status")
                if status not in ["RUNNING", "QUEUED"]:
                    # Drain any remaining lines
                    try:
                        for line in outf:
                            await ws.send_text(json.dumps({"out": line.rstrip()}))
                        for line in errf:
                            await ws.send_text(json.dumps({"err": line.rstrip()}))
                    except Exception:
                        # WebSocket connection lost during drain
                        pass
                    break
                if not sent:
                    await asyncio.sleep(1)
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            # WebSocket connection already closed
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            # WebSocket already closed
            pass
    return

@app.get("/file")
async def serve_file(path: str):
    """Serve a file from the filesystem"""
    try:
        # Decode the URL-encoded path
        import urllib.parse
        decoded_path = urllib.parse.unquote(path)
        
        # Convert to Path object for better handling
        file_path = Path(decoded_path)
        
        # Security check: ensure the path is within allowed directories
        config = path_manager.get_config()
        allowed_dirs = [
            Path(config.output_dir),
            Path(config.logs_dir),
            Path(config.evaluation_dir),
        ]
        
        is_allowed = False
        for allowed_dir in allowed_dirs:
            try:
                file_path.relative_to(allowed_dir)
                is_allowed = True
                break
            except ValueError:
                continue
        
        if not is_allowed:
            raise HTTPException(status_code=403, detail="Access denied to this file path")
        
        # Check if file exists
        if not file_path.exists():
            # Check if this might be a job result file that hasn't been generated yet
            if "test_" in file_path.name and file_path.suffix in [".jsonl", ".json"]:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Result file not found: {file_path.name}. The job may still be running or may have failed."
                )
            else:
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        # Check if it's actually a file
        if not file_path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {file_path}")
        
        # Return the file
        return FileResponse(
            path=str(file_path),
            filename=file_path.name,
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")

@app.get("/metrics/{job_id}")
async def get_metrics_file(job_id: str):
    """Find and serve metrics file for a specific job"""
    try:
        if job_id not in job_db:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_info = job_db[job_id]
        result_file = job_info.get("result_file")
        
        if not result_file:
            raise HTTPException(status_code=404, detail="No result file found for this job")
        
        # Convert result file path to Path object
        result_path = Path(result_file)
        
        # Security check
        config = path_manager.get_config()
        allowed_dirs = [Path(config.output_dir), Path(config.logs_dir), Path(config.evaluation_dir)]
        
        is_allowed = False
        for allowed_dir in allowed_dirs:
            try:
                result_path.relative_to(allowed_dir)
                is_allowed = True
                break
            except ValueError:
                continue
        
        if not is_allowed:
            raise HTTPException(status_code=403, detail="Access denied to this file path")
        
        # Try different patterns for metrics file
        base_name = result_path.stem  # Remove extension
        prompt_type = job_info.get("request", {}).get("prompt_type", "cot")
        
        possible_metrics_files = [
            result_path.parent / f"{base_name}_{prompt_type}_metrics.json",
            result_path.parent / f"{base_name}_metrics.json",
            result_path.parent / f"{result_path.stem}_{prompt_type}_metrics.json",
            result_path.parent / f"{result_path.stem}_metrics.json"
        ]
        
        # Find the first existing metrics file
        metrics_file = None
        for possible_file in possible_metrics_files:
            if possible_file.exists():
                metrics_file = possible_file
                break
        
        if not metrics_file:
            raise HTTPException(
                status_code=404, 
                detail=f"Metrics file not found for job {job_id}. The job may still be running or may have failed."
            )
        
        # Return the metrics file
        return FileResponse(
            path=str(metrics_file),
            filename=metrics_file.name,
            media_type='application/json'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding metrics file: {str(e)}")

@app.get("/jobs")
async def list_jobs():
    jobs = []
    try:
        for jid, info in job_db.items():
            # Get real-time status for each job
            job_info = get_job_status(jid)
            if isinstance(job_info, dict):
                job_info = job_info.copy()
                job_info.pop("proc", None)
                job_info.pop("cli", None)
            slurm_jid = job_info.get("slurm_jid")
            result_file = job_info.get("result_file")
            jobs.append({"job_id": jid, "slurm_jid": slurm_jid, **job_info, "result_file": result_file})
        return {"jobs": jobs}
    except Exception as e:
        print(f"Error in /jobs: {e}")
        return {"jobs": [], "error": str(e)} 