from fastapi import FastAPI, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio

from .schemas import EvalRequest, JobStatus
from .runner import launch_job, job_db, get_job_status, cancel_job, delete_job

app = FastAPI(title="Qwen Math Evaluation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Qwen Math Evaluation API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

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
        await ws.send_text(json.dumps({"error": f"Job {jid} not found (UUID or SLURM job number)"}))
        await ws.close()
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
        await ws.send_text(json.dumps({"waiting": f"Waiting for job output files to appear... ({wait_time}s)"}))
        await asyncio.sleep(2)
        wait_time += 2
    if not out_path or not os.path.exists(out_path) or not err_path or not os.path.exists(err_path):
        await ws.send_text(json.dumps({"error": f"Output or error file not found for this job after waiting. (out: {out_path}, err: {err_path})"}))
        await ws.close()
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
                    await ws.send_text(json.dumps({"out": out_line.rstrip()}))
                    sent = True
                if err_line:
                    await ws.send_text(json.dumps({"err": err_line.rstrip()}))
                    sent = True
                # Check if job is still running
                status = info.get("status")
                if status not in ["RUNNING", "QUEUED"]:
                    # Drain any remaining lines
                    for line in outf:
                        await ws.send_text(json.dumps({"out": line.rstrip()}))
                    for line in errf:
                        await ws.send_text(json.dumps({"err": line.rstrip()}))
                    break
                if not sent:
                    await asyncio.sleep(1)
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
    finally:
        await ws.close()
    return

@app.get("/jobs")
async def list_jobs():
    jobs = []
    try:
        for jid, info in job_db.items():
            job_info = info.copy()
            job_info.pop("proc", None)
            job_info.pop("cli", None)
            slurm_jid = job_info.get("slurm_jid")
            result_file = job_info.get("result_file")
            jobs.append({"job_id": jid, "slurm_jid": slurm_jid, **job_info, "result_file": result_file})
        return {"jobs": jobs}
    except Exception as e:
        print(f"Error in /jobs: {e}")
        return {"jobs": [], "error": str(e)} 