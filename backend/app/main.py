from fastapi import FastAPI, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio

from .schemas import EvalRequest, JobStatus
from .runner import launch_job, job_db, get_job_status

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
    # Remove non-serializable objects
    job_info.pop("proc", None)
    job_info.pop("cli", None)
    
    return {"job_id": jid, **job_info}

@app.get("/jobs/{jid}")
async def job_status(jid: str):
    """Get the status of a job"""
    status_info = get_job_status(jid)
    
    # Return only serializable data
    if isinstance(status_info, dict):
        status_info = status_info.copy()
        status_info.pop("proc", None)
        status_info.pop("cli", None)
    
    return {"job_id": jid, **status_info}

@app.websocket("/stream/{jid}")
async def stream(jid: str, ws: WebSocket):
    await ws.accept()
    # Allow jid to be either backend UUID or SLURM job number
    info = job_db.get(jid)
    if not info:
        # Try to find by SLURM job number
        for uuid, job in job_db.items():
            if str(job.get("slurm_jid")) == jid:
                info = job
                jid = uuid
                break
    if not info:
        await ws.send_text(json.dumps({"error": f"Job {jid} not found (UUID or SLURM job number)"}))
        await ws.close()
        return

    # Local job: stream process output
    if "proc" in info:
        proc = info["proc"]
        try:
            # Try to import pynvml for GPU monitoring
            try:
                import pynvml
                pynvml.nvmlInit()
                gpu_available = True
            except ImportError:
                gpu_available = False
            while True:
                line = proc.stdout.readline()
                if line:
                    await ws.send_text(json.dumps({"log": line.strip()}))
                if gpu_available:
                    try:
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        await ws.send_text(json.dumps({
                            "gpu": {
                                "mem_used": mem.used,
                                "mem_total": mem.total,
                                "utilization": util.gpu
                            }
                        }))
                    except Exception as e:
                        await ws.send_text(json.dumps({"gpu_error": str(e)}))
                if proc.poll() is not None:
                    if proc.returncode == 0:
                        job_db[jid]["status"] = JobStatus.DONE
                    else:
                        job_db[jid]["status"] = JobStatus.ERROR
                    await ws.send_text(json.dumps({
                        "status": job_db[jid]["status"],
                        "return_code": proc.returncode
                    }))
                    break
                await asyncio.sleep(2)
        except Exception as e:
            await ws.send_text(json.dumps({"error": str(e)}))
        finally:
            await ws.close()
        return

    # SLURM job: tail output file
    if "slurm_jid" in info:
        import os
        log_path = f"/work/10757/manasp123/qwen-eval-ui/logs/qwen-math-{jid}.out"
        try:
            # Wait for the file to appear
            for _ in range(30):  # Wait up to 30*2=60 seconds
                if os.path.exists(log_path):
                    break
                await asyncio.sleep(2)
            if not os.path.exists(log_path):
                await ws.send_text(json.dumps({"error": f"Log file {log_path} not found."}))
                await ws.close()
                return
            with open(log_path, "r") as f:
                f.seek(0, os.SEEK_END)  # Start at end of file
                while True:
                    line = f.readline()
                    if line:
                        await ws.send_text(json.dumps({"log": line.strip()}))
                    else:
                        # Check if job is still running
                        status = get_job_status(jid)
                        if status.get("status") not in ["RUNNING", "QUEUED"]:
                            break
                        await asyncio.sleep(2)
        except Exception as e:
            await ws.send_text(json.dumps({"error": str(e)}))
        finally:
            await ws.close()
        return

    # If neither, just close
    await ws.close()

@app.get("/jobs")
async def list_jobs():
    jobs = []
    try:
        for jid, info in job_db.items():
            job_info = info.copy()
            job_info.pop("proc", None)
            job_info.pop("cli", None)
            jobs.append({"job_id": jid, **job_info})
        return {"jobs": jobs}
    except Exception as e:
        print(f"Error in /jobs: {e}")
        return {"jobs": [], "error": str(e)} 