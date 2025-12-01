from fastapi import FastAPI, BackgroundTasks, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Dict, Any, Optional
import json
import asyncio
import os
import re
import sys
import time
from pathlib import Path
import pandas as pd

from .schemas import (
    EvalRequest, JobStatus, PathConfig, PathConfigResponse, 
    CoTAnalysisResponse, CoTAnalysisResponseV2, OpenAITestRequest, 
    OpenAITestResponse, QuestionPreview, HeatmapDataResponse,
    PromptPreviewRequest, PromptPreviewResponse,
    TruncationAnalysisRequest, TruncationAnalysisResponse,
    CoTAnalysisQueueRequest, CoTAnalysisProgressResponse, CoTAnalysisQueueStatus
)
from .runner import launch_job, job_db, get_job_status, cancel_job, delete_job, get_job_raw_data, save_job_db, reload_job_db
from .path_manager import path_manager
from .cot_queue import cot_queue
import subprocess
import shlex
import requests

def _build_truncation_response(job_id: str, request: TruncationAnalysisRequest, output_dir: Path, computation_time: float) -> TruncationAnalysisResponse:
    """Helper function to build truncation analysis response"""
    model_stub = os.path.basename(request.model_name_or_path.rstrip('/'))
    raw_curves_path = output_dir / f"{request.dataset_name}_truncation_curves_{model_stub}.json"
    correct_plot_path = output_dir / f"{request.dataset_name}_correct_{model_stub}.png"
    incorrect_plot_path = output_dir / f"{request.dataset_name}_incorrect_{model_stub}.png"
    
    return TruncationAnalysisResponse(
        job_id=job_id,
        status="completed",
        message="Truncation analysis completed successfully",
        raw_curves_path=str(raw_curves_path) if raw_curves_path.exists() else None,
        correct_plot_path=str(correct_plot_path) if correct_plot_path.exists() else None,
        incorrect_plot_path=str(incorrect_plot_path) if incorrect_plot_path.exists() else None,
        computation_time=computation_time
    )

app = FastAPI(title="Qwen Math Evaluation API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get workspace directory from path_manager for relative paths
_workspace_dir = Path(path_manager.get_config().workspace_dir)
_frontend_dir = _workspace_dir / "frontend"

# Mount static files for frontend
app.mount("/static", StaticFiles(directory=str(_frontend_dir)), name="static")

@app.get("/")
async def root():
    """Serve the main frontend page"""
    return FileResponse(str(_frontend_dir / "index.html"))

@app.get("/debug/html")
async def debug_html():
    """Debug endpoint to check HTML content"""
    try:
        with open("../frontend/index.html", "r") as f:
            content = f.read()
        return {"content_length": len(content), "has_cot_analysis": "cot-analysis" in content}
    except Exception as e:
        return {"error": str(e)}

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
    try:
        path_manager.update_config(config)
        updated_config = path_manager.get_config()
        print(f"Config updated: slurm_partition={updated_config.slurm_partition}, wall_time={updated_config.slurm_wall_time}")
        return {"message": "Path configuration updated successfully", "config": updated_config}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Configuration update error: {error_details}")
        raise HTTPException(status_code=422, detail=f"Failed to update configuration: {str(e)}")

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
    prob_file = status_info.get("prob_file")
    summary_file = status_info.get("summary_file")
    request_cfg = status_info.get("request", {})
    is_ece = status_info.get("is_ece") or request_cfg.get("enable_ece_eval")

    if is_ece:
        status_info["is_ece"] = True
        if summary_file:
            try:
                summary_path = Path(summary_file)
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        summary_json = json.load(f)
                    datasets = summary_json.get("datasets", {})
                    averaged_prob = None
                    for dataset_data in datasets.values():
                        averaged_prob = dataset_data.get("averaged_prob_file")
                        if averaged_prob:
                            break
                    if averaged_prob and Path(averaged_prob).exists():
                        prob_file = averaged_prob
                        status_info["prob_file"] = prob_file
                        job_db[jid]["prob_file"] = prob_file
                        save_job_db()
            except Exception as e:
                print(f"Warning: failed to parse ECE summary for job {jid}: {e}")

    return {"job_id": jid, "slurm_jid": slurm_jid, **status_info, "result_file": result_file, "prob_file": prob_file, "summary_file": summary_file}

@app.get("/jobs/{jid}/prob-file")
async def get_prob_file(jid: str):
    if jid not in job_db:
        raise HTTPException(status_code=404, detail="Job not found")
    job_info = job_db[jid]
    prob_file = job_info.get("prob_file")
    if not prob_file:
        raise HTTPException(status_code=404, detail="Probability file not available for this job")
    file_path = Path(prob_file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Probability file not found on disk")
    # Security check
    config = path_manager.get_config()
    allowed_dirs = [Path(config.output_dir), Path(config.logs_dir), Path(config.evaluation_dir)]
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
    return FileResponse(path=str(file_path), filename=file_path.name, media_type='application/json')

@app.get("/jobs/{jid}/error-log")
async def get_error_log(jid: str):
    """Get error log file content for a job"""
    if jid not in job_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_db[jid]
    err_file = job_info.get("err_file")
    
    if not err_file:
        raise HTTPException(status_code=404, detail="Error log file not available for this job")
    
    file_path = Path(err_file)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Error log file not found on disk")
    
    # Security check
    config = path_manager.get_config()
    allowed_dirs = [Path(config.output_dir), Path(config.logs_dir), Path(config.evaluation_dir)]
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
    
    try:
        with open(file_path, 'r') as f:
            error_content = f.read()
        
        return {
            "job_id": jid,
            "error_file": str(file_path),
            "content": error_content,
            "size_bytes": file_path.stat().st_size,
            "lines": error_content.count('\n') + 1
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading error log file: {str(e)}")

@app.get("/jobs/{jid}/prob-plot")
async def generate_prob_plot(jid: str, plot_type: str, sample_id: int | None = None, math_level: str | None = None):
    valid_plot_types = ("aggregate", "single", "path_aggregate", "path_single", "correct_aggregate", "incorrect_aggregate", 
                       "level_single", "level_aggregate", "correct_vs_incorrect",
                       "first_token_prob", "last_token_prob", 
                       "first_token_prob_correct", "first_token_prob_incorrect",
                       "last_token_prob_correct", "last_token_prob_incorrect")
    if plot_type not in valid_plot_types:
        raise HTTPException(status_code=400, detail=f"plot_type must be one of: {', '.join(valid_plot_types)}")
    if jid not in job_db:
        raise HTTPException(status_code=404, detail="Job not found")
    job_info = job_db[jid]
    prob_file = job_info.get("prob_file")
    if not prob_file:
        raise HTTPException(status_code=404, detail="Probability file not available for this job")
    prob_path = Path(prob_file)
    if not prob_path.exists():
        raise HTTPException(status_code=404, detail="Probability file not found on disk")

    # Build plotting command
    config = path_manager.get_config()
    python_bin = config.python_path
    eval_dir = Path(config.evaluation_dir)
    plot_script = eval_dir / "prob_plot.py"
    if not plot_script.exists():
        raise HTTPException(status_code=500, detail=f"Plot script not found: {plot_script}")

    # Names
    dataset_name = job_info.get("request", {}).get("dataset", "dataset")
    model_name = job_info.get("request", {}).get("model", "model")
    prompt_type = job_info.get("request", {}).get("prompt_type", "") or "custom"
    method_name = f"{Path(model_name).name}-{prompt_type}"

    output_dir = prob_path.parent / "prob_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expected output path
    if plot_type == "aggregate":
        expected_png = output_dir / f"{dataset_name}_aggregate_{method_name}.png"
    elif plot_type == "correct_aggregate":
        expected_png = output_dir / f"{dataset_name}_correct_aggregate_{method_name}.png"
    elif plot_type == "incorrect_aggregate":
        expected_png = output_dir / f"{dataset_name}_incorrect_aggregate_{method_name}.png"
    elif plot_type == "correct_vs_incorrect":
        expected_png = output_dir / f"{dataset_name}_correct_vs_incorrect_{method_name}.png"
    elif plot_type == "path_aggregate":
        expected_png = output_dir / f"{dataset_name}_path_aggregate_{method_name}.png"
    elif plot_type == "path_single":
        if sample_id is None:
            raise HTTPException(status_code=400, detail="sample_id is required for path_single plot")
        expected_png = output_dir / f"{dataset_name}_path_single_{method_name}_id_{sample_id}.png"
    elif plot_type == "level_single":
        if math_level is None:
            raise HTTPException(status_code=400, detail="math_level is required for level_single plot")
        expected_png = output_dir / f"{dataset_name}_level_{math_level}_{method_name}.png"
    elif plot_type == "level_aggregate":
        # For level_aggregate, we'll return multiple images as a zip file
        expected_png = output_dir / f"{dataset_name}_level_aggregate_{method_name}.zip"
    elif plot_type == "first_token_prob":
        expected_png = output_dir / f"{dataset_name}_first_token_prob_{method_name}.png"
    elif plot_type == "first_token_prob_correct":
        expected_png = output_dir / f"{dataset_name}_first_token_prob_correct_{method_name}.png"
    elif plot_type == "first_token_prob_incorrect":
        expected_png = output_dir / f"{dataset_name}_first_token_prob_incorrect_{method_name}.png"
    elif plot_type == "last_token_prob":
        expected_png = output_dir / f"{dataset_name}_last_token_prob_{method_name}.png"
    elif plot_type == "last_token_prob_correct":
        expected_png = output_dir / f"{dataset_name}_last_token_prob_correct_{method_name}.png"
    elif plot_type == "last_token_prob_incorrect":
        expected_png = output_dir / f"{dataset_name}_last_token_prob_incorrect_{method_name}.png"
    else:  # single
        if sample_id is None:
            raise HTTPException(status_code=400, detail="sample_id is required for single plot")
        expected_png = output_dir / f"{dataset_name}_single_{method_name}_id_{sample_id}.png"

    cmd = [
        python_bin,
        str(plot_script),
        str(prob_path),
        "--dataset_name", str(dataset_name),
        "--method_name", str(method_name),
        "--output_dir", str(output_dir),
        "--plot_type", plot_type,
    ]
    if plot_type in ("single", "path_single"):
        cmd.extend(["--sample_id", str(sample_id)])
    elif plot_type == "level_single":
        cmd.extend(["--math_level", str(math_level)])
    
    # Add model_name and data_name for ECE calculation (especially for correct_vs_incorrect)
    # This enables automatic ECE calculation when tokenizer is available
    if model_name and model_name != "model":
        cmd.extend(["--model_name", str(model_name)])
    # Use dataset_name as data_name for answer extraction
    cmd.extend(["--data_name", str(dataset_name)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=eval_dir)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Plotting failed: {result.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running plot script: {str(e)}")

    if not expected_png.exists():
        # Fallback: try to find any recent file in output_dir
        if plot_type == "level_aggregate":
            # Look for zip files for level_aggregate
            files = sorted(output_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                raise HTTPException(status_code=500, detail="Level aggregate plot archive not generated")
            expected_png = files[0]
        else:
            # Look for PNG files for other plot types
            files = sorted(output_dir.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                raise HTTPException(status_code=500, detail="Plot image not generated")
            expected_png = files[0]

    # Return appropriate media type based on file extension
    if expected_png.suffix.lower() == '.zip':
        return FileResponse(path=str(expected_png), filename=expected_png.name, media_type='application/zip')
    else:
        return FileResponse(path=str(expected_png), filename=expected_png.name, media_type='image/png')

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
        slurm_jid = info.get("slurm_jid")
        if sbatch_path and (not out_path or not err_path):
            try:
                with open(sbatch_path, "r") as f:
                    lines = f.readlines()
                for line in lines:
                    if line.startswith("#SBATCH -o"):
                        out_path = line.split(None, 2)[-1].strip()
                    if line.startswith("#SBATCH -e"):
                        err_path = line.split(None, 2)[-1].strip()
                
                # Replace %j placeholder with actual SLURM job ID
                if slurm_jid and out_path and "%j" in out_path:
                    out_path = out_path.replace("%j", str(slurm_jid))
                if slurm_jid and err_path and "%j" in err_path:
                    err_path = err_path.replace("%j", str(slurm_jid))
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
                        line = out_line.rstrip()
                        # Check for structured monitoring information
                        if line.startswith("MONITOR_PROMPT:"):
                            try:
                                monitor_data = json.loads(line[15:])  # Remove "MONITOR_PROMPT: "
                                await ws.send_text(json.dumps({"monitor_prompt": monitor_data}))
                            except:
                                await ws.send_text(json.dumps({"out": line}))
                        elif line.startswith("MONITOR_EPOCH:"):
                            try:
                                monitor_data = json.loads(line[14:])  # Remove "MONITOR_EPOCH: "
                                await ws.send_text(json.dumps({"monitor_epoch": monitor_data}))
                            except:
                                await ws.send_text(json.dumps({"out": line}))
                        elif line.startswith("MONITOR_RESPONSE:"):
                            try:
                                monitor_data = json.loads(line[17:])  # Remove "MONITOR_RESPONSE: "
                                await ws.send_text(json.dumps({"monitor_response": monitor_data}))
                            except:
                                await ws.send_text(json.dumps({"out": line}))
                        else:
                            await ws.send_text(json.dumps({"out": line}))
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
        summary_file = job_info.get("summary_file")
        
        if summary_file:
            summary_path = Path(summary_file)
            config = path_manager.get_config()
            allowed_dirs = [Path(config.output_dir), Path(config.logs_dir), Path(config.evaluation_dir)]
            if summary_path.exists():
                is_allowed = False
                for allowed_dir in allowed_dirs:
                    try:
                        summary_path.relative_to(allowed_dir)
                        is_allowed = True
                        break
                    except ValueError:
                        continue
                if is_allowed:
                    return FileResponse(
                        path=str(summary_path),
                        filename=summary_path.name,
                        media_type='application/json'
                    )
        
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

@app.post("/jobs/reload")
async def reload_jobs():
    """Reload job database from disk"""
    try:
        count = reload_job_db()
        return {"message": "Job database reloaded successfully", "job_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reloading job database: {str(e)}")

@app.get("/jobs")
async def list_jobs():
    jobs = []
    try:
        import asyncio
        # Use cached status from job_db to avoid blocking on slow status checks
        # Only check status for jobs that might have changed (running/queued jobs)
        for jid, info in job_db.items():
            # Copy job info
            job_info = info.copy()
            job_info.pop("proc", None)
            job_info.pop("cli", None)
            
            # Only do expensive status checks for jobs that are still running/queued
            current_status = job_info.get("status", "")
            if current_status in ["RUNNING", "QUEUED"]:
                # Do async status check with timeout
                try:
                    # Run status check in executor to avoid blocking
                    loop = asyncio.get_event_loop()
                    updated_info = await asyncio.wait_for(
                        loop.run_in_executor(None, get_job_status, jid),
                        timeout=2.0  # 2 second timeout per job
                    )
                    if isinstance(updated_info, dict):
                        updated_info = updated_info.copy()
                        updated_info.pop("proc", None)
                        updated_info.pop("cli", None)
                        job_info.update(updated_info)
                except asyncio.TimeoutError:
                    # If status check times out, use cached status
                    print(f"Status check timeout for job {jid}, using cached status")
                except Exception as e:
                    print(f"Error checking status for job {jid}: {e}")
                    # Continue with cached status
            
            slurm_jid = job_info.get("slurm_jid")
            result_file = job_info.get("result_file")
            prob_file = job_info.get("prob_file")
            jobs.append({"job_id": jid, "slurm_jid": slurm_jid, **job_info, "result_file": result_file, "prob_file": prob_file})
        return {"jobs": jobs}
    except Exception as e:
        print(f"Error in /jobs: {e}")
        import traceback
        traceback.print_exc()
        return {"jobs": [], "error": str(e)}

@app.get("/jobs/{job_id}/questions", response_model=List[QuestionPreview])
async def get_job_questions(job_id: str):
    """Get list of questions/samples for a job"""
    if job_id not in job_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_db[job_id]
    result_file = job_info.get("result_file")
    prob_file = job_info.get("prob_file")
    
    if not result_file or not Path(result_file).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        questions = []
        with open(result_file, 'r') as f:
            for line in f:
                if line.strip():
                    sample = json.loads(line)
                    idx = sample.get("idx", 0)
                    question = sample.get("question", sample.get("problem", ""))
                    preview = question[:100] + "..." if len(question) > 100 else question
                    has_prob_data = prob_file is not None and Path(prob_file).exists()
                    questions.append(QuestionPreview(
                        idx=idx,
                        preview=preview,
                        has_prob_data=has_prob_data
                    ))
        return questions
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading questions: {str(e)}")

@app.post("/jobs/{job_id}/generate-plot")
async def generate_plot(job_id: str, plot_type: str, dataset_name: str, method_name: str):
    """Generate a specific plot type for a job"""
    if job_id not in job_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_db[job_id]
    result_file = job_info.get("result_file")
    
    if not result_file or not Path(result_file).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    try:
        import subprocess
        import os
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(result_file), "prob_plots")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get model name from job info for ECE calculation
        model_name = job_info.get("request", {}).get("model", None)
        
        # Clean up model name (remove URLs, keep HuggingFace format like "org/model")
        if model_name:
            if model_name.startswith('https://huggingface.co/'):
                model_name = model_name.replace('https://huggingface.co/', '')
            elif model_name.startswith('http://huggingface.co/'):
                model_name = model_name.replace('http://huggingface.co/', '')
            # Keep the full model identifier (e.g., "Qwen/Qwen2.5-Math-7B-Instruct")
            # Don't split - AutoTokenizer needs the full path
        
        # Build command
        cmd = [
            "python3", "prob_plot.py",
            result_file,
            "--dataset_name", dataset_name,
            "--method_name", method_name,
            "--output_dir", output_dir,
            "--plot_type", plot_type
        ]
        
        # Add model_name and data_name for ECE calculation (especially for correct_vs_incorrect)
        if model_name and model_name != "model" and model_name.strip():
            cmd.extend(["--model_name", str(model_name)])
            print(f"Adding model_name for ECE calculation: {model_name}")
        else:
            print(f"Warning: No valid model_name found. ECE calculation will be skipped.")
            print(f"  model_name from job_info: {job_info.get('request', {}).get('model', None)}")
        cmd.extend(["--data_name", str(dataset_name)])
        
        # Run the plotting command in the evaluation directory
        eval_dir = path_manager.get_config().evaluation_dir
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=eval_dir)
        
        if result.returncode == 0:
            # Determine the output filename
            if plot_type in ['first_token_prob_correct', 'first_token_prob_incorrect']:
                filename = f"{dataset_name}_first_token_prob_{plot_type.split('_')[-1]}_{method_name}.png"
            elif plot_type in ['last_token_prob_correct', 'last_token_prob_incorrect']:
                filename = f"{dataset_name}_last_token_prob_{plot_type.split('_')[-1]}_{method_name}.png"
            else:
                filename = f"{dataset_name}_{plot_type}_{method_name}.png"
            
            plot_path = os.path.join(output_dir, filename)
            
            if os.path.exists(plot_path):
                return {
                    "success": True,
                    "message": f"Plot generated successfully",
                    "plot_path": plot_path,
                    "filename": filename
                }
            else:
                return {
                    "success": False,
                    "message": f"Plot file not found: {plot_path}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        else:
            return {
                "success": False,
                "message": f"Plot generation failed",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")


@app.get("/jobs/{job_id}/plot-image/{filename}")
async def get_plot_image(job_id: str, filename: str):
    """Serve generated plot images"""
    if job_id not in job_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_db[job_id]
    result_file = job_info.get("result_file")
    
    if not result_file:
        raise HTTPException(status_code=404, detail="Result file not found")
    
    # Construct plot path
    output_dir = os.path.join(os.path.dirname(result_file), "prob_plots")
    plot_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(plot_path):
        raise HTTPException(status_code=404, detail="Plot file not found")
    
    # Security check - ensure the file is within allowed directories
    config = path_manager.get_config()
    allowed_dirs = [Path(config.output_dir), Path(config.logs_dir), Path(config.evaluation_dir)]
    
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            Path(plot_path).relative_to(allowed_dir)
            is_allowed = True
            break
        except ValueError:
            continue
    
    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access denied to this file path")
    
    # Return the image file
    from fastapi.responses import FileResponse
    return FileResponse(plot_path, media_type="image/png")


@app.get("/jobs/{job_id}/heatmap-data/{question_idx}", response_model=HeatmapDataResponse)
async def get_heatmap_data(job_id: str, question_idx: int):
    """Get token-level probability data for heatmap visualization"""
    if job_id not in job_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_db[job_id]
    result_file = job_info.get("result_file")
    prob_file = job_info.get("prob_file")
    
    if not result_file or not Path(result_file).exists():
        raise HTTPException(status_code=404, detail="Result file not found")
    
    if not prob_file or not Path(prob_file).exists():
        raise HTTPException(status_code=404, detail="Probability data not available for this job")
    
    try:
        # Load the specific sample from result file
        sample = None
        with open(result_file, 'r') as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    if s.get("idx") == question_idx:
                        sample = s
                        break
        
        if sample is None:
            raise HTTPException(status_code=404, detail=f"Question {question_idx} not found")
        
        # Load probability data
        prob_data = None
        with open(prob_file, 'r') as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    if p.get("idx") == question_idx:
                        prob_data = p
                        break
        
        if prob_data is None:
            raise HTTPException(status_code=404, detail=f"Probability data for question {question_idx} not found")
        
        # Extract data
        question_text = sample.get("question", sample.get("problem", ""))
        ground_truth = sample.get("gt", sample.get("answer", ""))
        
        # Get the full model output (reasoning + answer)
        # Try different fields in order of preference
        model_output_raw = None
        for field in ["code", "pred", "output", "model_output"]:
            if field in sample and sample[field]:
                model_output_raw = sample[field]
                break
        
        if model_output_raw is None:
            model_output = ""
        elif isinstance(model_output_raw, list):
            # Join list elements with newlines for better readability
            model_output = "\n".join(str(x) for x in model_output_raw)
        else:
            model_output = str(model_output_raw)
        
        # Extract predicted answer (final answer from model output)
        predicted_answer = None
        if model_output:
            # Try to extract the final answer from the model output
            # Look for patterns like "#### 42" or "The answer is 42" or just the last number
            import re
            # Pattern 1: #### followed by answer
            match = re.search(r'####\s*([^\n]+)', model_output)
            if match:
                predicted_answer = match.group(1).strip()
            else:
                # Pattern 2: "The answer is" or similar
                match = re.search(r'(?:answer is|final answer is|result is)\s*:?\s*([^\n.]+)', model_output, re.IGNORECASE)
                if match:
                    predicted_answer = match.group(1).strip()
                else:
                    # Pattern 3: Last number in the text
                    numbers = re.findall(r'-?\d+\.?\d*', model_output)
                    if numbers:
                        predicted_answer = numbers[-1]
        
        # Determine correctness
        is_correct = False
        if predicted_answer and ground_truth:
            # Normalize both answers for comparison
            def normalize_answer(ans):
                if not ans:
                    return ""
                # Remove extra whitespace and convert to lowercase
                ans = str(ans).strip().lower()
                # Remove common prefixes/suffixes
                ans = re.sub(r'^(the answer is|answer:|final answer:|result:)\s*', '', ans)
                ans = re.sub(r'[^\w\d.-]', '', ans)  # Keep only alphanumeric, dots, and hyphens
                return ans
            
            norm_pred = normalize_answer(predicted_answer)
            norm_gt = normalize_answer(ground_truth)
            is_correct = norm_pred == norm_gt
        
        # If we couldn't determine from extracted answer, check the score field
        if predicted_answer is None:
            score = sample.get("score", [])
            if isinstance(score, list) and score:
                is_correct = bool(score[0])
            elif isinstance(score, (bool, int)):
                is_correct = bool(score)
        
        # Get probability arrays and token IDs
        chosen_probs_dict = prob_data.get("chosen_token_probs", {})
        correct_probs_dict = prob_data.get("probability_log", {})
        chosen_token_ids_dict = prob_data.get("chosen_token_ids", {})
        correct_token_ids_dict = prob_data.get("correct_token_ids", {})
        
        # Get epoch_0 data (main generation pass)
        chosen_probs = chosen_probs_dict.get("epoch_0", [])
        chosen_token_ids = chosen_token_ids_dict.get("epoch_0", [])
        correct_token_ids = correct_token_ids_dict.get("epoch_0", [])
        
        # Use actual ground truth token probabilities from probability_log
        correct_probs = correct_probs_dict.get("epoch_0", [])
        
        # If no probability_log data available, create minimal fallback
        if not correct_probs and chosen_probs:
            correct_probs = [0.01] * len(chosen_probs)
        
        # Apply log scaling to make small probabilities more visible
        if correct_probs:
            import math
            # Add small epsilon to avoid log(0)
            epsilon = 1e-15
            correct_probs = [math.log(max(p, epsilon)) for p in correct_probs]
            # Normalize to 0-1 range
            min_log = min(correct_probs)
            max_log = max(correct_probs)
            if max_log > min_log:
                correct_probs = [(p - min_log) / (max_log - min_log) for p in correct_probs]
            else:
                correct_probs = [0.0] * len(correct_probs)
        
        # Use actual model tokenization if available, otherwise fallback to word-based
        if chosen_token_ids and len(chosen_token_ids) > 0:
            # We have actual token IDs from the model
            # Decode them using the model's tokenizer
            tokens = []
            tokenizer = None
            
            # Try to get model name from job info
            model_name = job_info.get("request", {}).get("model", None)
            if model_name:
                # Clean up model name (remove URLs, keep HuggingFace format like "org/model")
                if model_name.startswith('https://huggingface.co/'):
                    model_name = model_name.replace('https://huggingface.co/', '')
                elif model_name.startswith('http://huggingface.co/'):
                    model_name = model_name.replace('http://huggingface.co/', '')
                # Keep the full model identifier (e.g., "Qwen/Qwen2.5-Math-1.5B")
                # Don't split - AutoTokenizer needs the full path
            
            # Try to load tokenizer (optional - doesn't require GPU, but needs internet/cache)
            if model_name:
                try:
                    from transformers import AutoTokenizer
                    # Try loading tokenizer - this runs on CPU, no GPU needed
                    # But it may fail if no internet access or model not cached
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name, 
                        trust_remote_code=True,
                        local_files_only=False  # Allow download if not cached
                    )
                except Exception as e:
                    # Tokenizer loading failed - this is OK, we'll use fallback
                    print(f"Info: Could not load tokenizer for {model_name}: {e}")
                    print("   Using fallback token representation (no tokenizer needed)")
                    tokenizer = None
            
            # Decode tokens if tokenizer is available
            if tokenizer:
                for token_id in chosen_token_ids:
                    if token_id is not None:
                        try:
                            # Decode the token ID to get the actual token
                            decoded_token = tokenizer.decode([token_id])
                            # For display purposes, convert whitespace to visible characters
                            if decoded_token == '\n':
                                decoded_token = '↵'
                            elif decoded_token == ' ':
                                decoded_token = '␣'
                            elif decoded_token == '\t':
                                decoded_token = '⇥'
                            elif decoded_token == '\n\n':
                                decoded_token = '↵↵'
                            # Only replace truly empty tokens with token IDs
                            elif decoded_token.strip() == '' and decoded_token:
                                # This is whitespace-only, keep it as is
                                pass
                            elif not decoded_token:
                                decoded_token = f"<{token_id}>"
                            tokens.append(decoded_token)
                        except Exception as e:
                            # If decoding fails, use the token ID
                            tokens.append(f"<{token_id}>")
                    else:
                        tokens.append("<unknown>")
            else:
                # Fallback: Use simple token ID representation (no tokenizer needed)
                # This works fine for visualization - shows token_123 instead of actual text
                for token_id in chosen_token_ids:
                    if token_id is not None:
                        tokens.append(f"token_{token_id}")
                    else:
                        tokens.append("unknown")
        else:
            # Fallback to word-based tokenization if no token IDs available
            import re
            tokens = re.findall(r'\S+', model_output)
        
        # Align arrays - truncate to shortest length
        min_len = min(len(tokens), len(chosen_probs), len(correct_probs))
        tokens = tokens[:min_len]
        chosen_probs = chosen_probs[:min_len]
        correct_probs = correct_probs[:min_len]
        chosen_token_ids = chosen_token_ids[:min_len] if chosen_token_ids else list(range(min_len))
        correct_token_ids = correct_token_ids[:min_len] if correct_token_ids else list(range(min_len))
        
        # Normalize probabilities per-array using min-max to [0,1]
        def normalize_probs(probs):
            if not probs or len(probs) == 0:
                return probs
            lo = min(probs)
            hi = max(probs)
            if hi <= lo:
                # All values equal; return mid intensity
                return [0.5] * len(probs)
            return [(p - lo) / (hi - lo) for p in probs]
        
        # Store original probabilities for display in tooltips
        chosen_probs_original = list(chosen_probs)
        correct_probs_original = list(correct_probs)
        
        # Normalize for color mapping (per-heatmap)
        chosen_probs_normalized = normalize_probs(chosen_probs)
        correct_probs_normalized = normalize_probs(correct_probs)
        
        return HeatmapDataResponse(
            job_id=job_id,
            question_idx=question_idx,
            question_text=question_text,
            model_output=model_output,
            output_tokens=tokens,
            chosen_probs=chosen_probs_normalized,
            correct_probs=correct_probs_normalized,
            chosen_probs_original=chosen_probs_original,
            correct_probs_original=correct_probs_original,
            chosen_token_ids=chosen_token_ids,
            correct_token_ids=correct_token_ids,
            is_correct=is_correct,
            predicted_answer=predicted_answer,
            ground_truth=ground_truth
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting heatmap data: {str(e)}")

def _get_model_and_dataset_from_job(job_id: str) -> tuple[str, str]:
    """
    Extract model name and dataset from job info.
    
    Returns:
        Tuple of (model_name, dataset)
    """
    job_info = job_db.get(job_id, {})
    result_file = job_info.get("result_file", "")
    request = job_info.get("request", {})
    
    model_name = request.get("model", "Unknown")
    dataset = request.get("dataset", "unknown")
    
    # Try to extract from result_file path if not in request
    if model_name == "Unknown" and result_file:
            path_parts = Path(result_file).parts
            for part in path_parts:
                if any(model in part.lower() for model in ['qwen', 'gpt', 'claude', 'llama', 'mistral', 'gemini', 'mathstral']):
                    model_name = part
                if part in ['gsm8k', 'math', 'mmlu', 'humaneval']:
                    dataset = part
    
    # Sanitize model name for filesystem (replace / with _)
    model_name = model_name.replace('/', '_').replace('\\', '_')
    
    # Extract base dataset name if comma-separated
    if ',' in dataset:
        dataset = dataset.split(',')[0].strip()
    
    return model_name, dataset

def export_cot_analysis_to_excel(job_id: str, analysis_data: dict, output_dir: Optional[Path] = None) -> str:
    """Export CoT analysis results to Excel file using Pillars v2 data"""
    try:
        config = path_manager.get_config()
        
        # Use provided output_dir or create organized folder structure
        if output_dir is None:
            model_name, dataset = _get_model_and_dataset_from_job(job_id)
            # Extract timestamp and judge mode from analysis_data or create default
            timestamp = analysis_data.get('timestamp', time.strftime("%Y%m%d_%H%M%S"))
            judge_mode = analysis_data.get('config', {}).get('judge_mode', 'ALWAYS')
            
            # Create organized folder structure: cot_analysis/{model}/{dataset}/{job_id}_{timestamp}_{judge_mode}/
            exports_dir = Path(config.exports_dir)
            organized_dir = exports_dir / "cot_analysis" / model_name / dataset / f"{job_id}_{timestamp.replace(' ', '_').replace(':', '')}_{judge_mode}"
            organized_dir.mkdir(parents=True, exist_ok=True)
            output_dir = organized_dir
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename
        filename = f"cot_analysis_{job_id}.xlsx"
        excel_path = output_dir / filename
        
        # Create workbook with pandas and openpyxl
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 1. Summary Sheet
            summary_data = [
                ['CoT Analysis Summary'],
                [''],
                ['Job ID', analysis_data.get('job_id', 'N/A')],
                ['Analysis Method', analysis_data.get('analysis_method', 'N/A')],
                ['Timestamp', analysis_data.get('timestamp', 'N/A')],
                ['Total Samples', analysis_data.get('summary', {}).get('total_samples', 0)],
                [''],
                ['Overall Scores'],
                ['Overall Score', analysis_data.get('summary', {}).get('avg_overall', 0)],
                ['Faithfulness', analysis_data.get('summary', {}).get('avg_faithfulness', 0)],
                ['Utility', analysis_data.get('summary', {}).get('avg_utility', 0)],
                ['Coherence', analysis_data.get('summary', {}).get('avg_coherence', 0)],
                ['Factuality', analysis_data.get('summary', {}).get('avg_factuality', 0)],
                [''],
                ['Flag Statistics'],
                ['Total Flags', analysis_data.get('summary', {}).get('total_flags', 0)],
                ['Faithfulness Flags', analysis_data.get('summary', {}).get('flags_by_pillar', {}).get('faithfulness', 0)],
                ['Utility Flags', analysis_data.get('summary', {}).get('flags_by_pillar', {}).get('utility', 0)],
                ['Coherence Flags', analysis_data.get('summary', {}).get('flags_by_pillar', {}).get('coherence', 0)],
                ['Factuality Flags', analysis_data.get('summary', {}).get('flags_by_pillar', {}).get('factuality', 0)],
                [''],
                ['Judge Statistics'],
                ['Judge Call Rate', f"{analysis_data.get('summary', {}).get('judge_call_rate', 0) * 100:.1f}%"],
                ['Budget Used', f"{analysis_data.get('summary', {}).get('judge_budget_used', 0)}/{analysis_data.get('summary', {}).get('judge_budget_total', 0)}"]
            ]
            
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, index=False, sheet_name='Summary')
            
            # 2. Detailed Analysis Sheet
            detailed_data = []
            detailed_data.append([
                'Sample #', 'Problem', 'Model Output', 'Final Answer Correct',
                'Overall Score', 'Faithfulness', 'Utility', 'Coherence', 'Factuality',
                'Flag Count', 'Flags', 'Evidence Summary', 'Judge Scores', 'Arithmetic Errors'
            ])
            
            per_sample = analysis_data.get('per_sample', [])
            for idx, sample in enumerate(per_sample):
                flags = sample.get('flags', [])
                flag_descriptions = [f"{f.get('pillar', 'Unknown')}: {f.get('issue', 'Unknown')}" for f in flags]
                arith_errors = sample.get('evidence', {}).get('arith_bad_examples', [])
                
                detailed_data.append([
                    idx + 1,
                    (sample.get('problem', 'N/A')[:1000]),
                    (sample.get('model_output', 'N/A')[:2000]),
                    'Yes' if sample.get('evidence', {}).get('final_correct', False) else 'No',
                    sample.get('scores', {}).get('overall', 0),
                    sample.get('scores', {}).get('faithfulness', 0),
                    sample.get('scores', {}).get('utility', 0),
                    sample.get('scores', {}).get('coherence', 0),
                    sample.get('scores', {}).get('factuality', 0),
                    len(flags),
                    '; '.join(flag_descriptions) or 'None',
                    f"Final: {'Correct' if sample.get('evidence', {}).get('final_correct', False) else 'Incorrect'}, Intermediate OK: {sample.get('evidence', {}).get('intermediate_ok_rate', 0):.2f}",
                    json.dumps(sample.get('judge_raw', {}))[:500] if sample.get('judge_raw') else 'N/A',
                    len(arith_errors) if arith_errors else 0
                ])
            
            detailed_df = pd.DataFrame(detailed_data[1:], columns=detailed_data[0])
            detailed_df.to_excel(writer, index=False, sheet_name='Detailed Analysis')
            
            # 3. Flag Summary Sheet
            flag_counts = {}
            for sample in per_sample:
                for flag in sample.get('flags', []):
                    key = f"{flag.get('pillar', 'Unknown')}: {flag.get('issue', 'Unknown')}"
                    flag_counts[key] = flag_counts.get(key, 0) + 1
            
            flag_data = [['Pillar', 'Issue Type', 'Count']]
            for key, count in sorted(flag_counts.items()):
                pillar, issue = key.split(': ', 1)
                flag_data.append([pillar, issue, count])
            
            flag_df = pd.DataFrame(flag_data[1:], columns=flag_data[0])
            flag_df.to_excel(writer, index=False, sheet_name='Flag Summary')
            
            # 4. Raw Data Sheet
            raw_data = [['Raw JSON Data'], [json.dumps(analysis_data, indent=2)]]
            raw_df = pd.DataFrame(raw_data)
            raw_df.to_excel(writer, index=False, sheet_name='Raw Data')
        
        print(f"✅ Excel export created: {excel_path}")
        return str(excel_path)
        
    except Exception as e:
        print(f"❌ Error creating Excel export: {e}")
        raise

def export_results_to_excel(job_id: str, job_info: dict) -> str:
    """Export job results to Excel file"""
    try:
        config = path_manager.get_config()
        exports_dir = Path(config.exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        result_file = job_info.get("result_file", "")
        if not result_file or not Path(result_file).exists():
            raise HTTPException(status_code=404, detail="No results available. This job may not have completed successfully or was cancelled.")
        
        # Load the result file
        try:
            with open(result_file, 'r') as f:
                results = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading result file: {str(e)}")
        
        if not results:
            raise HTTPException(status_code=404, detail="Result file is empty")
        
        # Get model and dataset info
        model_name = job_info.get("request", {}).get("model", "Unknown")
        dataset = job_info.get("request", {}).get("dataset", "unknown")
        
        # Sanitize model name for filename (remove slashes and special chars)
        safe_model_name = model_name.replace("/", "_").replace("\\", "_")
        
        # Create filename
        filename = f"{safe_model_name}_{dataset}_{job_id}_results.xlsx"
        excel_path = exports_dir / filename
        
        # Create workbook with pandas
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = [
                ['Job Summary'],
                [''],
                ['Job ID', job_id],
                ['Model', model_name],
                ['Dataset', dataset],
                ['Status', job_info.get("status", "N/A")],
                ['Total Samples', len(results)],
                [''],
                ['Metrics'],
            ]
            
            # Calculate accuracy
            correct = sum(1 for r in results if r.get('score', [False])[0])
            accuracy = (correct / len(results) * 100) if results else 0
            summary_data.append(['Accuracy', f"{accuracy:.2f}%"])
            summary_data.append(['Correct', correct])
            summary_data.append(['Incorrect', len(results) - correct])
            
            summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value'])
            summary_df.to_excel(writer, index=False, sheet_name='Summary')
            
            # Results sheet
            results_data = []
            for idx, result in enumerate(results):
                results_data.append({
                    'Sample #': idx + 1,
                    'Question': result.get('question', 'N/A')[:1000],
                    'Ground Truth': result.get('gt', 'N/A'),
                    'Prediction': result.get('pred', ['N/A'])[0] if result.get('pred') else 'N/A',
                    'Correct': 'Yes' if result.get('score', [False])[0] else 'No',
                    'Answer': (result.get('answer', 'N/A')[:2000] if isinstance(result.get('answer'), str) else str(result.get('answer', 'N/A'))[:2000]),
                })
            
            results_df = pd.DataFrame(results_data)
            results_df.to_excel(writer, index=False, sheet_name='Results')
        
        print(f"✅ Excel export created: {excel_path}")
        return str(excel_path)
        
    except Exception as e:
        print(f"❌ Error creating Excel export: {e}")
        raise

@app.get('/downloads/all-jobs-archive')
async def download_all_jobs_archive():
    """Download zip archive of all completed job outputs"""
    try:
        config = path_manager.get_config()
        exports_dir = Path(config.exports_dir)
        
        # Find the most recent zip file
        zip_files = sorted(exports_dir.glob('all_completed_jobs_*.zip'), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not zip_files:
            raise HTTPException(status_code=404, detail="No job archive found. Please create one first.")
        
        latest_zip = zip_files[0]
        
        return FileResponse(
            path=str(latest_zip),
            filename=latest_zip.name,
            media_type='application/zip'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving archive: {str(e)}")

@app.get('/jobs/{job_id}/export')
async def export_job_excel(job_id: str):
    """Export job results to Excel file"""
    try:
        # Check if Excel export path exists in job_db
        job_info = job_db.get(job_id, {})
        
        if not job_info:
            raise HTTPException(status_code=404, detail="Job not found")
        
        excel_export_path = job_info.get("excel_export_path")
        
        if excel_export_path and Path(excel_export_path).exists():
            # Return pre-generated Excel file (from CoT analysis)
            excel_file = Path(excel_export_path)
            return FileResponse(
                path=str(excel_file),
                filename=excel_file.name,
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        else:
            # Generate Excel file on the fly
            excel_path = export_results_to_excel(job_id, job_info)
            excel_file = Path(excel_path)
        return FileResponse(
            path=str(excel_file),
            filename=excel_file.name,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving Excel file: {str(e)}")

@app.get('/jobs/{job_id}/raw-answers')
async def get_job_raw_answers(job_id: str):
    """Get raw answer data for CoT analysis"""
    try:
        raw_data = get_job_raw_data(job_id)
        return raw_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching raw data: {str(e)}")

@app.get("/jobs/{job_id}/data-health")
async def check_job_data_health(job_id: str):
    """Health check for job data availability"""
    try:
        raw_data = get_job_raw_data(job_id)
        data_list = raw_data.get('data', [])
        return {
            "job_id": job_id,
            "data_available": True,
            "sample_count": len(data_list),
            "has_answers": sum(1 for r in data_list if r.get('answer')),
            "avg_answer_length": sum(len(r.get('answer', '')) for r in data_list) / len(data_list) if data_list else 0,
            "metadata": raw_data.get('metadata', {})
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "data_available": False,
            "error": str(e)
        }

async def run_cot_analysis_async(cot_job_id: str, job_id: str, config_dict: Dict[str, Any]):
    """
    Async function to run CoT analysis with progress tracking.
    
    Args:
        cot_job_id: The CoT analysis job ID
        job_id: The parent job ID
        config_dict: Configuration dict with judge_mode, diagnostic, etc.
    """
    start_time = time.time()
    judge_mode = config_dict.get("judge_mode", "ALWAYS")
    diagnostic = config_dict.get("diagnostic", False)
    
    try:
        # Get raw data
        raw_data = get_job_raw_data(job_id)
        data_list = raw_data.get('data', [])
        
        if not data_list:
            raise HTTPException(status_code=404, detail="No data found for analysis")
        
        total_samples = len(data_list)
        
        # Update progress: starting
        cot_queue.update_progress(cot_job_id, 0, total_samples, "Initializing analysis...")
        
        # Initialize new Pillars v2 evaluator
        from .cot_eval_v2.evaluator import PillarsEvaluator
        from .cot_eval_v2.judge import Judge
        
        # Get configuration
        config = path_manager.get_config()
        
        # Create judge if OpenAI key available
        judge = None
        if config.openai_api_key:
            # Set the API key as environment variable for OpenAI client
            import os
            os.environ['OPENAI_API_KEY'] = config.openai_api_key
            judge = Judge(mode=judge_mode, diagnostic=diagnostic)
        
        # Initialize evaluator
        evaluator = PillarsEvaluator(judge=judge)
        
        # Update progress: initialized
        cot_queue.update_progress(cot_job_id, 0, total_samples, "Analysis initialized, processing samples...")
        
        # Process each sample
        results = []
        
        print(f"🔍 Starting CoT analysis for {total_samples} samples...", flush=True)
        
        for i, sample in enumerate(data_list):
            # Update progress before processing each sample
            cot_queue.update_progress(
                cot_job_id,
                i,
                total_samples,
                f"Analyzing sample {i+1}/{total_samples}..."
            )
            try:
                print(f"📊 Analyzing sample {i+1}/{total_samples}...", flush=True)
                sys.stdout.flush()
                
                # Extract model reasoning and answer
                # Try multiple fields in order of preference (different datasets use different field names)
                model_reasoning_text = ""
                predicted_answer_text = ""
                
                # Try to get reasoning from 'code', 'solution', or 'output' fields
                if sample.get('code'):
                    code_val = sample.get('code', [''])
                    if isinstance(code_val, list):
                        model_reasoning_text = code_val[0] if code_val and code_val[0] else ""
                    else:
                        model_reasoning_text = str(code_val) if code_val else ""
                
                if not model_reasoning_text and sample.get('solution'):
                    model_reasoning_text = str(sample.get('solution', ''))
                
                if not model_reasoning_text and sample.get('output'):
                    output_val = sample.get('output', '')
                    if isinstance(output_val, list):
                        model_reasoning_text = output_val[0] if output_val and output_val[0] else ""
                    else:
                        model_reasoning_text = str(output_val) if output_val else ""
                
                # Try to get predicted answer from 'pred', 'answer', or extract from reasoning
                if sample.get('pred'):
                    pred_val = sample.get('pred', [''])
                    if isinstance(pred_val, list):
                        predicted_answer_text = str(pred_val[0]) if pred_val and pred_val[0] is not None else ""
                    else:
                        predicted_answer_text = str(pred_val) if pred_val else ""
                
                if not predicted_answer_text and sample.get('answer'):
                    predicted_answer_text = str(sample.get('answer', ''))
                
                # If still no answer, try to extract from reasoning text
                if not predicted_answer_text and model_reasoning_text:
                    # Look for boxed answer
                    boxed_match = re.search(r'\\boxed\{([^}]+)\}', model_reasoning_text)
                    if boxed_match:
                        predicted_answer_text = boxed_match.group(1).strip()
                    else:
                        # Look for framebox
                        framebox_match = re.search(r'\\framebox\{([^}]+)\}', model_reasoning_text)
                        if framebox_match:
                            predicted_answer_text = framebox_match.group(1).strip()
                
                # Construct full CoT text
                if predicted_answer_text:
                    full_cot_text = f"{model_reasoning_text}\n#### {predicted_answer_text}"
                else:
                    full_cot_text = model_reasoning_text
                
                # Get existing correctness from job data
                existing_scores = sample.get('score', [])
                is_correct = existing_scores[0] if existing_scores else False
                
                print(f"   Question: {sample.get('question', '')[:100]}{'...' if len(sample.get('question', '')) > 100 else ''}", flush=True)
                print(f"   Ground Truth: {sample.get('gt', '')}", flush=True)
                print(f"   Reasoning length: {len(model_reasoning_text)} chars", flush=True)
                print(f"   Predicted: {predicted_answer_text}", flush=True)
                print(f"   Correct: {is_correct}", flush=True)
                if not model_reasoning_text:
                    print(f"   ⚠️ Warning: No model reasoning found in 'code', 'solution', or 'output' fields", flush=True)
                sys.stdout.flush()
                
                # Run analysis
                print(f"   🔍 Running Pillars v2 evaluation...", flush=True)
                sys.stdout.flush()
                flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(
                    problem=sample.get('question', ''),
                    cot_text=full_cot_text,
                    gold=sample.get('gt', '')
                )
                
                # Override the final_correct with the existing job data
                evidence['final_correct'] = is_correct
                
                # Recalculate rule scores with the correct final_correct value
                from .cot_eval_v2.scoring import rule_scores
                rule_scores = rule_scores(evidence)
                
                # Recalculate fused scores
                from .cot_eval_v2.scoring import fuse_with_judge
                fused_scores = fuse_with_judge(rule_scores, judge_scores, evidence)
                
                # Convert flags to summary format
                flags_summary = flags.summarize_for_prompt()
                
                # Convert flags to dictionary format for API response
                flags_dict = {}
                for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
                    pillar_flags = flags.get_flags_by_pillar(pillar)
                    flags_dict[pillar] = [flag.to_dict() for flag in pillar_flags]
                
                # Log analysis results
                print(f"   ✅ Analysis complete!", flush=True)
                print(f"   📊 Rule scores: {rule_scores}", flush=True)
                print(f"   🤖 Judge scores: {judge_scores}", flush=True)
                print(f"   🔗 Fused scores: {fused_scores}", flush=True)
                total_flags = sum(len(flags_dict[pillar]) for pillar in flags_dict)
                print(f"   🚩 Flags found: {total_flags}", flush=True)
                if total_flags > 0:
                    for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
                        pillar_flags = flags_dict[pillar]
                        if pillar_flags:
                            print(f"      - {pillar}: {len(pillar_flags)} flags", flush=True)
                            for flag in pillar_flags[:2]:  # Show first 2 flags per pillar
                                print(f"        * {flag['issue']} (step: {flag['step']})", flush=True)
                print(flush=True)
                sys.stdout.flush()
                
                # Build result for this sample
                sample_result = {
                    "sample_id": i,
                    "problem": sample.get('question', ''),
                    "cot_text": full_cot_text,
                    "gold_answer": sample.get('gt', ''),
                    "flags": flags_dict,
                    "evidence": evidence,
                    "rule_scores": rule_scores,
                    "judge_scores": judge_scores,
                    "fused_scores": fused_scores,
                    "final_answer": predicted_answer_text
                }
                
                results.append(sample_result)
                
            except Exception as e:
                print(f"Error processing sample {i}: {e}", flush=True)
                sys.stdout.flush()
                # Add error result
                results.append({
                    "sample_id": i,
                    "problem": sample.get('question', ''),
                    "cot_text": "",
                    "gold_answer": sample.get('gt', ''),
                    "error": str(e),
                    "flags": {},
                    "evidence": {},
                    "rule_scores": {},
                    "judge_scores": {},
                    "fused_scores": {},
                    "final_answer": ""
                })
        
        # Calculate summary statistics
        valid_results = [r for r in results if "error" not in r]
        if valid_results:
            # Calculate average scores
            avg_scores = {}
            for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
                scores = [r["fused_scores"].get(pillar, 0) for r in valid_results]
                avg_scores[pillar] = sum(scores) / len(scores) if scores else 0
            
            avg_scores["overall"] = sum(avg_scores.values()) / len(avg_scores)
            
            # Count flags
            flag_counts = {}
            for result in valid_results:
                for flag_type, flags in result["flags"].items():
                    if flag_type not in flag_counts:
                        flag_counts[flag_type] = 0
                    flag_counts[flag_type] += len(flags)
        else:
            avg_scores = {"faithfulness": 0, "utility": 0, "coherence": 0, "factuality": 0, "overall": 0}
            flag_counts = {}
        
        # Convert results to proper schema format
        per_sample = []
        for result in results:
            if "error" not in result:
                # Convert flags to PillarsFlag format
                flags_list = []
                for pillar, flags in result["flags"].items():
                    for flag in flags:
                        flags_list.append({
                            "pillar": pillar,
                            "step": flag.get("step", "unknown"),
                            "issue": flag.get("issue", "unknown"),
                            "details": flag.get("details", {})
                        })
                
                # Create PillarsScores
                scores = {
                    "faithfulness": result["fused_scores"].get("faithfulness", 0.0),
                    "utility": result["fused_scores"].get("utility", 0.0),
                    "coherence": result["fused_scores"].get("coherence", 0.0),
                    "factuality": result["fused_scores"].get("factuality", 0.0),
                    "overall": result["fused_scores"].get("overall", 0.0)
                }
                
                per_sample.append({
                    "scores": scores,
                    "flags": flags_list,
                    "evidence": result["evidence"],
                    "rules_raw": result["rule_scores"],
                    "judge_raw": result["judge_scores"],
                    "config_snapshot": {"judge_available": judge is not None},
                    "problem": result["problem"],
                    "model_output": result["cot_text"],
                    "gold": result["gold_answer"]
                })
        
        # Create summary
        summary = {
            "avg_faithfulness": avg_scores.get("faithfulness", 0.0),
            "avg_utility": avg_scores.get("utility", 0.0),
            "avg_coherence": avg_scores.get("coherence", 0.0),
            "avg_factuality": avg_scores.get("factuality", 0.0),
            "avg_overall": avg_scores.get("overall", 0.0),
            "total_flags": sum(flag_counts.values()),
            "flags_by_pillar": flag_counts,
            "judge_call_rate": 1.0 if judge is not None else 0.0,
            "judge_budget_used": 0,
            "judge_budget_total": 0,
            "total_samples": total_samples,
            "analysis_time": time.time() - start_time,
            "avg_time_per_sample": (time.time() - start_time) / total_samples if total_samples > 0 else 0.0
        }
        
        # Build final response
        analysis_result = {
            "job_id": job_id,
            "per_sample": per_sample,
            "summary": summary,
            "analysis_method": "pillars_v2",
            "config": {"judge_available": judge is not None},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Print final summary
        print(f"🎉 CoT analysis complete! Processed {len(results)} samples", flush=True)
        successful_samples = len([r for r in results if 'error' not in r])
        print(f"✅ Successfully analyzed: {successful_samples}/{len(results)} samples", flush=True)
        if valid_results:
            print(f"📊 Average scores: {avg_scores}", flush=True)
            print(f"🚩 Total flags found: {sum(flag_counts.values())}", flush=True)
        print(flush=True)
        sys.stdout.flush()
        
        # Update progress: saving files
        cot_queue.update_progress(cot_job_id, total_samples, total_samples, "Saving results...")
        
        # Create organized output directory
        model_name, dataset = _get_model_and_dataset_from_job(job_id)
        timestamp_str = analysis_result.get('timestamp', time.strftime("%Y%m%d_%H%M%S")).replace(' ', '_').replace(':', '')
        judge_mode_str = judge_mode
        
        config_path = path_manager.get_config()
        exports_dir = Path(config_path.exports_dir)
        organized_dir = exports_dir / "cot_analysis" / model_name / dataset / f"{job_id}_{timestamp_str}_{judge_mode_str}"
        organized_dir.mkdir(parents=True, exist_ok=True)
        
        # Save CoT analysis results to organized folder
        try:
            cot_analysis_file = organized_dir / f"cot_analysis_{job_id}.json"
            
            # Save the analysis result
            with open(cot_analysis_file, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            
            print(f"💾 CoT analysis saved to: {cot_analysis_file}", flush=True)
            sys.stdout.flush()
            
            # Store JSON path in queue entry
            queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
            if queue_entry:
                queue_entry["json_path"] = str(cot_analysis_file)
                job_db[f"cot_analysis_{cot_job_id}"] = queue_entry
                save_job_db()
        except Exception as e:
            print(f"⚠️ Warning: Could not save CoT analysis to file: {e}", flush=True)
            sys.stdout.flush()
        
        # Generate Excel export in organized folder
        try:
            excel_path = export_cot_analysis_to_excel(job_id, analysis_result, organized_dir)
            
            # Store Excel path in queue entry and parent job_db
            queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
            if queue_entry:
                queue_entry["excel_export_path"] = excel_path
                job_db[f"cot_analysis_{cot_job_id}"] = queue_entry
            
            # Also update parent job_db
            job_info = job_db.get(job_id, {})
            job_info["excel_export_path"] = excel_path
            job_db[job_id] = job_info
            save_job_db()
            
            print(f"📊 Excel export saved to: {excel_path}", flush=True)
            sys.stdout.flush()
        except Exception as e:
            print(f"⚠️ Warning: Could not generate Excel export: {e}", flush=True)
            sys.stdout.flush()
        
        # Update progress: complete
        cot_queue.update_progress(cot_job_id, total_samples, total_samples, "Analysis complete!")
        
    except HTTPException:
        raise
    except Exception as e:
        # Update progress: error
        queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
        if queue_entry:
            queue_entry["status"] = "ERROR"
            queue_entry["error"] = str(e)
            queue_entry["progress"]["current_activity"] = f"Error: {str(e)}"
            job_db[f"cot_analysis_{cot_job_id}"] = queue_entry
            save_job_db()
        raise

@app.get("/jobs/{job_id}/cot-analysis", response_model=CoTAnalysisResponseV2)
async def get_cot_analysis(job_id: str, queue: bool = True):
    """
    Get Chain-of-Thought analysis for a job using new four-pillar evaluation.
    
    Args:
        job_id: The job ID to analyze
        queue: If True (default), queue the analysis for async processing.
               If False, run synchronously (backward compatible).
    """
    if queue:
        # Queue mode: submit to queue and return immediately
        config = path_manager.get_config()
        config_dict = {
            "judge_mode": "ALWAYS",  # Default, can be made configurable
            "diagnostic": False
        }
        
        cot_job_id = await cot_queue.submit(job_id, config_dict, run_cot_analysis_async)
        
        # Return a minimal response indicating the job was queued
        # The actual analysis result will be available via the progress endpoint
        queue_entry = cot_queue.get_status(cot_job_id)
        
        # Return a response that matches the schema (minimal valid response)
        return {
            "job_id": job_id,
            "per_sample": [],
            "summary": {
                "avg_faithfulness": 0.0,
                "avg_utility": 0.0,
                "avg_coherence": 0.0,
                "avg_factuality": 0.0,
                "avg_overall": 0.0,
                "total_flags": 0,
                "flags_by_pillar": {},
                "judge_call_rate": 0.0,
                "judge_budget_used": 0,
                "judge_budget_total": 0,
                "total_samples": 0,
                "analysis_time": 0.0,
                "avg_time_per_sample": 0.0
            },
            "analysis_method": "pillars_v2",
            "config": {"judge_available": config.openai_api_key is not None, "queued": True, "cot_job_id": cot_job_id},
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    else:
        # Synchronous mode (backward compatible)
        start_time = time.time()
        
        try:
            # Get raw data
            raw_data = get_job_raw_data(job_id)
            data_list = raw_data.get('data', [])
            
            if not data_list:
                raise HTTPException(status_code=404, detail="No data found for analysis")
            
            # Initialize new Pillars v2 evaluator
            from .cot_eval_v2.evaluator import PillarsEvaluator
            from .cot_eval_v2.judge import Judge
            
            # Get configuration
            config = path_manager.get_config()
            
            # Create judge if OpenAI key available
            judge = None
            if config.openai_api_key:
                # Set the API key as environment variable for OpenAI client
                import os
                os.environ['OPENAI_API_KEY'] = config.openai_api_key
                judge = Judge(mode="ALWAYS")
            
            # Initialize evaluator
            evaluator = PillarsEvaluator(judge=judge)
            
            # Process each sample (same logic as async function but without progress tracking)
            results = []
            total_samples = len(data_list)
            
            print(f"🔍 Starting CoT analysis for {total_samples} samples...", flush=True)
            
            for i, sample in enumerate(data_list):
                try:
                    print(f"📊 Analyzing sample {i+1}/{total_samples}...", flush=True)
                    sys.stdout.flush()
                    
                    # Extract model reasoning and answer (same as async function)
                    model_reasoning_text = ""
                    predicted_answer_text = ""
                    
                    if sample.get('code'):
                        code_val = sample.get('code', [''])
                        if isinstance(code_val, list):
                            model_reasoning_text = code_val[0] if code_val and code_val[0] else ""
                        else:
                            model_reasoning_text = str(code_val) if code_val else ""
                    
                    if not model_reasoning_text and sample.get('solution'):
                        model_reasoning_text = str(sample.get('solution', ''))
                    
                    if not model_reasoning_text and sample.get('output'):
                        output_val = sample.get('output', '')
                        if isinstance(output_val, list):
                            model_reasoning_text = output_val[0] if output_val and output_val[0] else ""
                        else:
                            model_reasoning_text = str(output_val) if output_val else ""
                    
                    if sample.get('pred'):
                        pred_val = sample.get('pred', [''])
                        if isinstance(pred_val, list):
                            predicted_answer_text = str(pred_val[0]) if pred_val and pred_val[0] is not None else ""
                        else:
                            predicted_answer_text = str(pred_val) if pred_val else ""
                    
                    if not predicted_answer_text and sample.get('answer'):
                        predicted_answer_text = str(sample.get('answer', ''))
                    
                    if not predicted_answer_text and model_reasoning_text:
                        boxed_match = re.search(r'\\boxed\{([^}]+)\}', model_reasoning_text)
                        if boxed_match:
                            predicted_answer_text = boxed_match.group(1).strip()
                        else:
                            framebox_match = re.search(r'\\framebox\{([^}]+)\}', model_reasoning_text)
                            if framebox_match:
                                predicted_answer_text = framebox_match.group(1).strip()
                    
                    if predicted_answer_text:
                        full_cot_text = f"{model_reasoning_text}\n#### {predicted_answer_text}"
                    else:
                        full_cot_text = model_reasoning_text
                    
                    existing_scores = sample.get('score', [])
                    is_correct = existing_scores[0] if existing_scores else False
                    
                    # Run analysis
                    flags, evidence, rule_scores, judge_scores, fused_scores = evaluator.analyze(
                        problem=sample.get('question', ''),
                        cot_text=full_cot_text,
                        gold=sample.get('gt', '')
                    )
                    
                    evidence['final_correct'] = is_correct
                    
                    from .cot_eval_v2.scoring import rule_scores as calc_rule_scores
                    rule_scores = calc_rule_scores(evidence)
                    
                    from .cot_eval_v2.scoring import fuse_with_judge
                    fused_scores = fuse_with_judge(rule_scores, judge_scores, evidence)
                    
                    flags_summary = flags.summarize_for_prompt()
                    
                    flags_dict = {}
                    for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
                        pillar_flags = flags.get_flags_by_pillar(pillar)
                        flags_dict[pillar] = [flag.to_dict() for flag in pillar_flags]
                    
                    sample_result = {
                        "sample_id": i,
                        "problem": sample.get('question', ''),
                        "cot_text": full_cot_text,
                        "gold_answer": sample.get('gt', ''),
                        "flags": flags_dict,
                        "evidence": evidence,
                        "rule_scores": rule_scores,
                        "judge_scores": judge_scores,
                        "fused_scores": fused_scores,
                        "final_answer": predicted_answer_text
                    }
                    
                    results.append(sample_result)
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}", flush=True)
                    sys.stdout.flush()
                    results.append({
                        "sample_id": i,
                        "problem": sample.get('question', ''),
                        "cot_text": "",
                        "gold_answer": sample.get('gt', ''),
                        "error": str(e),
                        "flags": {},
                        "evidence": {},
                        "rule_scores": {},
                        "judge_scores": {},
                        "fused_scores": {},
                        "final_answer": ""
                    })
            
            # Calculate summary statistics (same as async function)
            valid_results = [r for r in results if "error" not in r]
            if valid_results:
                avg_scores = {}
                for pillar in ["faithfulness", "utility", "coherence", "factuality"]:
                    scores = [r["fused_scores"].get(pillar, 0) for r in valid_results]
                    avg_scores[pillar] = sum(scores) / len(scores) if scores else 0
                avg_scores["overall"] = sum(avg_scores.values()) / len(avg_scores)
                
                flag_counts = {}
                for result in valid_results:
                    for flag_type, flags in result["flags"].items():
                        if flag_type not in flag_counts:
                            flag_counts[flag_type] = 0
                        flag_counts[flag_type] += len(flags)
            else:
                avg_scores = {"faithfulness": 0, "utility": 0, "coherence": 0, "factuality": 0, "overall": 0}
                flag_counts = {}
            
            # Convert results to proper schema format
            per_sample = []
            for result in results:
                if "error" not in result:
                    flags_list = []
                    for pillar, flags in result["flags"].items():
                        for flag in flags:
                            flags_list.append({
                                "pillar": pillar,
                                "step": flag.get("step", "unknown"),
                                "issue": flag.get("issue", "unknown"),
                                "details": flag.get("details", {})
                            })
                    
                    scores = {
                        "faithfulness": result["fused_scores"].get("faithfulness", 0.0),
                        "utility": result["fused_scores"].get("utility", 0.0),
                        "coherence": result["fused_scores"].get("coherence", 0.0),
                        "factuality": result["fused_scores"].get("factuality", 0.0),
                        "overall": result["fused_scores"].get("overall", 0.0)
                    }
                    
                    per_sample.append({
                        "scores": scores,
                        "flags": flags_list,
                        "evidence": result["evidence"],
                        "rules_raw": result["rule_scores"],
                        "judge_raw": result["judge_scores"],
                        "config_snapshot": {"judge_available": judge is not None},
                        "problem": result["problem"],
                        "model_output": result["cot_text"],
                        "gold": result["gold_answer"]
                    })
            
            # Create summary
            summary = {
                "avg_faithfulness": avg_scores.get("faithfulness", 0.0),
                "avg_utility": avg_scores.get("utility", 0.0),
                "avg_coherence": avg_scores.get("coherence", 0.0),
                "avg_factuality": avg_scores.get("factuality", 0.0),
                "avg_overall": avg_scores.get("overall", 0.0),
                "total_flags": sum(flag_counts.values()),
                "flags_by_pillar": flag_counts,
                "judge_call_rate": 1.0 if judge is not None else 0.0,
                "judge_budget_used": 0,
                "judge_budget_total": 0,
                "total_samples": total_samples,
                "analysis_time": time.time() - start_time,
                "avg_time_per_sample": (time.time() - start_time) / total_samples if total_samples > 0 else 0.0
            }
            
            # Build final response
            analysis_result = {
                "job_id": job_id,
                "per_sample": per_sample,
                "summary": summary,
                "analysis_method": "pillars_v2",
                "config": {"judge_available": judge is not None},
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to organized folder (same as async function)
            model_name, dataset = _get_model_and_dataset_from_job(job_id)
            timestamp_str = analysis_result.get('timestamp', time.strftime("%Y%m%d_%H%M%S")).replace(' ', '_').replace(':', '')
            judge_mode_str = "ALWAYS"
            
            config_path = path_manager.get_config()
            exports_dir = Path(config_path.exports_dir)
            organized_dir = exports_dir / "cot_analysis" / model_name / dataset / f"{job_id}_{timestamp_str}_{judge_mode_str}"
            organized_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON
            try:
                cot_analysis_file = organized_dir / f"cot_analysis_{job_id}.json"
                with open(cot_analysis_file, 'w') as f:
                    json.dump(analysis_result, f, indent=2)
                print(f"💾 CoT analysis saved to: {cot_analysis_file}", flush=True)
            except Exception as e:
                print(f"⚠️ Warning: Could not save CoT analysis to file: {e}", flush=True)
            
            # Generate Excel
            try:
                excel_path = export_cot_analysis_to_excel(job_id, analysis_result, organized_dir)
                job_info = job_db.get(job_id, {})
                job_info["excel_export_path"] = excel_path
                job_db[job_id] = job_info
                save_job_db()
                print(f"📊 Excel export saved to: {excel_path}", flush=True)
            except Exception as e:
                print(f"⚠️ Warning: Could not generate Excel export: {e}", flush=True)
        
            return analysis_result
        
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error performing CoT analysis: {str(e)}")

@app.post("/jobs/{job_id}/cot-analysis/queue")
async def queue_cot_analysis(job_id: str, request: CoTAnalysisQueueRequest):
    """Explicitly queue a CoT analysis job"""
    try:
        config_dict = {
            "judge_mode": request.judge_mode,
            "diagnostic": request.diagnostic
        }
        
        cot_job_id = await cot_queue.submit(job_id, config_dict, run_cot_analysis_async)
        
        # Return status
        queue_entry = cot_queue.get_status(cot_job_id)
        return {
            "cot_job_id": cot_job_id,
            "parent_job_id": job_id,
            "status": queue_entry.get("status", "QUEUED"),
            "queue_position": queue_entry.get("queue_position"),
            "message": "CoT analysis queued for processing"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error queuing CoT analysis: {str(e)}")

@app.get("/cot-analyses/queue")
async def list_cot_analyses_queue():
    """List all queued/running CoT analysis jobs"""
    try:
        jobs = cot_queue.list_jobs()
        
        # Convert to response format
        response_jobs = []
        for job in jobs:
            progress = job.get("progress", {})
            response_jobs.append({
                "cot_job_id": job.get("cot_job_id"),
                "parent_job_id": job.get("parent_job_id"),
                "status": job.get("status", "UNKNOWN"),
                "queue_position": job.get("queue_position"),
                "created_at": job.get("created_at"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at"),
                "model_name": job.get("model_name"),
                "dataset": job.get("dataset"),
                "judge_mode": job.get("judge_mode"),
                "progress": {
                    "current_sample": progress.get("current_sample", 0),
                    "total_samples": progress.get("total_samples", 0),
                    "processed_samples": progress.get("processed_samples", 0),
                    "percentage": progress.get("percentage", 0.0),
                    "estimated_time_remaining": progress.get("estimated_time_remaining"),
                    "current_activity": progress.get("current_activity", "")
                },
                "error": job.get("error")
            })
        
        return {"jobs": response_jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing CoT analyses: {str(e)}")

@app.get("/cot-analyses/{cot_job_id}/status", response_model=CoTAnalysisQueueStatus)
async def get_cot_analysis_status(cot_job_id: str):
    """Get status of a CoT analysis job"""
    try:
        queue_entry = cot_queue.get_status(cot_job_id)
        if not queue_entry:
            raise HTTPException(status_code=404, detail="CoT analysis job not found")
        
        progress = queue_entry.get("progress", {})
        progress_response = CoTAnalysisProgressResponse(
            cot_job_id=cot_job_id,
            status=queue_entry.get("status", "UNKNOWN"),
            current_sample=progress.get("current_sample", 0),
            total_samples=progress.get("total_samples", 0),
            processed_samples=progress.get("processed_samples", 0),
            percentage=progress.get("percentage", 0.0),
            estimated_time_remaining=progress.get("estimated_time_remaining"),
            start_time=progress.get("start_time"),
            current_activity=progress.get("current_activity"),
            queue_position=queue_entry.get("queue_position")
        )
        
        return CoTAnalysisQueueStatus(
            cot_job_id=cot_job_id,
            parent_job_id=queue_entry.get("parent_job_id"),
            status=queue_entry.get("status", "UNKNOWN"),
            queue_position=queue_entry.get("queue_position"),
            created_at=queue_entry.get("created_at", time.time()),
            started_at=queue_entry.get("started_at"),
            completed_at=queue_entry.get("completed_at"),
            model_name=queue_entry.get("model_name"),
            dataset=queue_entry.get("dataset"),
            judge_mode=queue_entry.get("judge_mode"),
            progress=progress_response
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CoT analysis status: {str(e)}")

@app.get("/cot-analyses/{cot_job_id}/progress", response_model=CoTAnalysisProgressResponse)
async def get_cot_analysis_progress(cot_job_id: str):
    """Get detailed progress info for a running CoT analysis"""
    try:
        queue_entry = cot_queue.get_status(cot_job_id)
        if not queue_entry:
            raise HTTPException(status_code=404, detail="CoT analysis job not found")
        
        progress = queue_entry.get("progress", {})
        return CoTAnalysisProgressResponse(
            cot_job_id=cot_job_id,
            status=queue_entry.get("status", "UNKNOWN"),
            current_sample=progress.get("current_sample", 0),
            total_samples=progress.get("total_samples", 0),
            processed_samples=progress.get("processed_samples", 0),
            percentage=progress.get("percentage", 0.0),
            estimated_time_remaining=progress.get("estimated_time_remaining"),
            start_time=progress.get("start_time"),
            current_activity=progress.get("current_activity", ""),
            queue_position=queue_entry.get("queue_position")
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CoT analysis progress: {str(e)}")

@app.post("/jobs/{job_id}/cot-analysis/compute")
async def compute_cot_analysis(job_id: str):
    """Trigger CoT analysis computation and cache results"""
    try:
        # This could be enhanced to run analysis in background and cache results
        # For now, just redirect to the GET endpoint
        return await get_cot_analysis(job_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing CoT analysis: {str(e)}")

@app.get("/cot-analyses")
async def list_cot_analyses():
    """List all jobs that have completed CoT analysis results"""
    try:
        analyses = []
        
        for job_id, job_info in job_db.items():
            # Check if job has CoT analysis (either Excel export or JSON file)
            excel_export_path = job_info.get("excel_export_path")
            result_file = job_info.get("result_file")
            
            # Check if Excel file exists
            has_excel = excel_export_path and Path(excel_export_path).exists()
            
            # Check if JSON analysis file exists
            has_json = False
            cot_analysis_data = None
            if result_file:
                result_path = Path(result_file)
                output_dir = result_path.parent
                cot_analysis_file = output_dir / f"cot_analysis_{job_id}.json"
                has_json = cot_analysis_file.exists()
                
                # Try to load summary data if JSON exists
                if has_json:
                    try:
                        with open(cot_analysis_file, 'r') as f:
                            cot_analysis_data = json.load(f)
                    except Exception:
                        pass
            
            # Only include jobs that have CoT analysis
            if has_excel or has_json:
                # Extract model and dataset from job info
                request = job_info.get("request", {})
                model = request.get("model", "Unknown")
                dataset = request.get("dataset", "Unknown")
                
                # Get summary scores if available
                summary = {}
                timestamp = None
                if cot_analysis_data:
                    summary = cot_analysis_data.get("summary", {})
                    timestamp = cot_analysis_data.get("timestamp") or summary.get("timestamp")
                
                analysis_entry = {
                    "job_id": job_id,
                    "slurm_jid": job_info.get("slurm_jid"),
                    "model": model,
                    "dataset": dataset,
                    "status": job_info.get("status", "UNKNOWN"),
                    "overall_score": summary.get("avg_overall", 0.0),
                    "faithfulness": summary.get("avg_faithfulness", 0.0),
                    "utility": summary.get("avg_utility", 0.0),
                    "coherence": summary.get("avg_coherence", 0.0),
                    "factuality": summary.get("avg_factuality", 0.0),
                    "total_flags": summary.get("total_flags", 0),
                    "total_samples": summary.get("total_samples", 0),
                    "timestamp": timestamp,
                    "has_excel": has_excel,
                    "excel_path": excel_export_path if has_excel else None,
                    "has_json": has_json
                }
                
                analyses.append(analysis_entry)
        
        # Sort by timestamp (most recent first) or job_id
        analyses.sort(key=lambda x: x.get("timestamp") or "", reverse=True)
        
        return {"analyses": analyses}
        
    except Exception as e:
        print(f"Error in /cot-analyses: {e}")
        import traceback
        traceback.print_exc()
        return {"analyses": [], "error": str(e)}

@app.post("/config")
async def save_configuration(request: dict):
    """Save configuration settings"""
    try:
        # Update the path manager configuration
        if 'openai_api_key' in request:
            path_manager._config.openai_api_key = request['openai_api_key']
            path_manager._save_config()
            print(f"✅ OpenAI API key updated in configuration")
        
        return {"message": "Configuration saved successfully"}
        
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save configuration: {str(e)}")

@app.post("/config/test-openai-key", response_model=OpenAITestResponse)
async def test_openai_key(request: OpenAITestRequest):
    """Test OpenAI API key validity"""
    try:
        # Import OpenAI here to avoid import errors if not available
        try:
            from openai import OpenAI
        except ImportError:
            return OpenAITestResponse(
                valid=False,
                error="OpenAI library not installed. Please install with: pip install openai"
            )
        
        # Test the API key by making a simple request
        client = OpenAI(api_key=request.api_key)
        
        try:
            # Make a simple test request to list models
            response = client.models.list()
            
            # Get basic info about available models
            model_info = {
                "total_models": len(response.data),
                "gpt_models": [m.id for m in response.data if "gpt" in m.id.lower()],
                "available_for_chat": [m.id for m in response.data if "gpt-4" in m.id or "gpt-3.5" in m.id]
            }
            
            return OpenAITestResponse(
                valid=True,
                model_info=model_info
            )
            
        except Exception as api_error:
            return OpenAITestResponse(
                valid=False,
                error=f"API key test failed: {str(api_error)}"
            )
            
    except Exception as e:
        return OpenAITestResponse(
            valid=False,
            error=f"Error testing API key: {str(e)}"
        ) 
@app.post("/jobs/{job_id}/truncation-analysis", response_model=TruncationAnalysisResponse)
async def run_truncation_analysis(job_id: str, request: TruncationAnalysisRequest):
    """Run CoT truncation analysis for a completed job"""
    start_time = time.time()
    
    try:
        if job_id not in job_db:
            raise HTTPException(status_code=404, detail="Job not found")
        
        job_info = job_db[job_id]
        result_file = job_info.get("result_file")
        
        if not result_file:
            raise HTTPException(status_code=404, detail="No result file found for this job")
        
        result_path = Path(result_file)
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="Result file not found on disk")
        
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
        
        # Load the JSONL results
        samples = []
        with open(result_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        samples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        
        if not samples:
            raise HTTPException(status_code=400, detail="No valid samples found in result file")
        
        # Build truncation analysis command
        python_bin = config.python_path
        eval_dir = Path(config.evaluation_dir)
        truncation_script = eval_dir / "run_truncation_analysis_with_logs.py"
        
        if not truncation_script.exists():
            raise HTTPException(status_code=500, detail=f"Truncation analysis script not found: {truncation_script}")
        
        # Create output directory for this analysis
        output_dir = result_path.parent / "truncation_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create temporary input file for the analysis
        temp_input_file = output_dir / f"temp_input_{job_id}.jsonl"
        with open(temp_input_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        # Handle different backends
        if request.backend == "local":
            # Run locally
            cmd = [
                python_bin,
                str(truncation_script),
                "--input_file", str(temp_input_file),
                "--model_name_or_path", request.model_name_or_path,
                "--dataset_name", request.dataset_name,
                "--output_dir", str(output_dir),
                "--temperature", str(request.temperature),
                "--top_p", str(request.top_p),
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=eval_dir, timeout=3600)  # 1 hour timeout
                if result.returncode != 0:
                    raise HTTPException(status_code=500, detail=f"Truncation analysis failed: {result.stderr}")
                
                computation_time = time.time() - start_time
                return _build_truncation_response(job_id, request, output_dir, computation_time)
                
            except subprocess.TimeoutExpired:
                raise HTTPException(status_code=500, detail="Truncation analysis timed out after 1 hour")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error running truncation analysis: {str(e)}")
        
        elif request.backend == "slurm":
            # Submit to SLURM
            truncation_job_id = f"trunc_{job_id}_{int(time.time())}"
            
            # Create SLURM script
            scripts_dir = Path(config.scripts_dir)
            scripts_dir.mkdir(parents=True, exist_ok=True)
            
            script_path = scripts_dir / f"run_truncation_{truncation_job_id}.sh"
            sbatch_path = scripts_dir / f"job_truncation_{truncation_job_id}.sbatch"
            
            # Build the command for the script
            cmd = [
                python_bin,
                str(truncation_script),
                "--input_file", str(temp_input_file),
                "--model_name_or_path", request.model_name_or_path,
                "--dataset_name", request.dataset_name,
                "--output_dir", str(output_dir),
                "--temperature", str(request.temperature),
                "--top_p", str(request.top_p),
                "--job_id", truncation_job_id,  # Add job ID for unique filenames
            ]
            
            # Create the shell script
            escaped_cli = []
            for arg in cmd:
                escaped_cli.append(shlex.quote(str(arg)))
            
            # Build conda activation section if conda_env_path is configured
            conda_activation = ""
            if config.conda_env_path:
                # Extract environment name from path (e.g., /path/to/envs/mathevalUI -> mathevalUI)
                conda_env_name = os.path.basename(config.conda_env_path)
                # If path ends with /envs/env_name, extract env_name
                if '/envs/' in config.conda_env_path:
                    conda_env_name = config.conda_env_path.split('/envs/')[-1]
                conda_activation = f"""# Activate conda environment
source {config.conda_env_path}/etc/profile.d/conda.sh
conda activate {conda_env_name}
"""
            else:
                conda_activation = "# Note: No conda environment configured. Using system Python.\n"
            
            script_content = f"""#!/bin/bash
cd {eval_dir}

# Set Hugging Face cache to work directory

# Fix MKL threading conflict
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

{conda_activation}
# Run the truncation analysis
{' '.join(escaped_cli)}
"""
            
            script_path.write_text(script_content)
            script_path.chmod(0o755)
            
            # Create SLURM sbatch script with GPU allocation
            out_file_pattern = os.path.join(config.logs_dir, f"truncation-{truncation_job_id}-%j.out")
            err_file_pattern = os.path.join(config.logs_dir, f"truncation-{truncation_job_id}-%j.err")
            
            sbatch_content = f"""#!/bin/bash
#SBATCH -J truncation-{truncation_job_id}   # Job name
#SBATCH -o {out_file_pattern}      # Name of stdout output file (uses %j)
#SBATCH -e {err_file_pattern}      # Name of stderr error file (uses %j)
#SBATCH -p {config.slurm_partition}              # Queue (partition) name
#SBATCH -N 1                    # Total # of nodes
#SBATCH -n 1                    # Total # of tasks (single process for all GPUs)
#SBATCH -t {config.slurm_wall_time}              # Run time (hh:mm:ss)
#SBATCH --mail-type=all         # Send email at begin and end of job
#SBATCH -A {config.slurm_account}             # Project/Allocation name

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
                    cwd=scripts_dir
                )
                
                if result.returncode == 0:
                    slurm_jid = result.stdout.strip().split()[-1]
                    
                    # Create actual file paths by replacing %j with SLURM job ID
                    actual_out_file = out_file_pattern.replace("%j", str(slurm_jid))
                    actual_err_file = err_file_pattern.replace("%j", str(slurm_jid))
                    
                    # Store the truncation job info
                    truncation_job_id_full = f"truncation_{job_id}_{slurm_jid}"
                    job_db[truncation_job_id_full] = {
                        "status": "queued",
                        "request": request.dict(),
                        "parent_job_id": job_id,
                        "run_path": str(script_path),
                        "sbatch_path": str(sbatch_path),
                        "backend": "slurm",
                        "slurm_jid": slurm_jid,
                        "out_file": actual_out_file,
                        "err_file": actual_err_file,
                        "output_dir": str(output_dir),
                        "temp_input_file": str(temp_input_file),
                        "dataset_name": request.dataset_name,
                        "model_name_or_path": request.model_name_or_path,
                        "temperature": request.temperature,
                        "top_p": request.top_p,
                        "start_time": start_time
                    }
                    save_job_db()
                    
                    return TruncationAnalysisResponse(
                        job_id=job_id,
                        status="queued",
                        message=f"Truncation analysis queued on SLURM with job ID {slurm_jid}",
                        raw_curves_path=None,
                        correct_plot_path=None,
                        incorrect_plot_path=None,
                        computation_time=None
                    )
                else:
                    raise HTTPException(status_code=500, detail=f"Failed to submit SLURM job: {result.stderr}")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error submitting to SLURM: {str(e)}")
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported backend: {request.backend}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error performing truncation analysis: {str(e)}")

@app.get("/jobs/{job_id}/truncation-analysis/plot")
async def get_truncation_plot(job_id: str, plot_type: str = "correct"):
    """Get truncation analysis plot files"""
    if plot_type not in ["correct", "incorrect"]:
        raise HTTPException(status_code=400, detail="plot_type must be 'correct' or 'incorrect'")
    
    if job_id not in job_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = job_db[job_id]
    result_file = job_info.get("result_file")
    
    if not result_file:
        raise HTTPException(status_code=404, detail="No result file found for this job")
    
    result_path = Path(result_file)
    output_dir = result_path.parent / "truncation_analysis"
    
    # Look for plot files
    plot_files = list(output_dir.glob(f"*_{plot_type}_*.png"))
    if not plot_files:
        raise HTTPException(status_code=404, detail=f"No {plot_type} plot found for this job")
    
    # Return the most recent plot file
    latest_plot = max(plot_files, key=lambda p: p.stat().st_mtime)
    
    # Security check
    config = path_manager.get_config()
    allowed_dirs = [Path(config.output_dir), Path(config.logs_dir), Path(config.evaluation_dir)]
    
    is_allowed = False
    for allowed_dir in allowed_dirs:
        try:
            latest_plot.relative_to(allowed_dir)
            is_allowed = True
            break
        except ValueError:
            continue
    
    if not is_allowed:
        raise HTTPException(status_code=403, detail="Access denied to this file path")
    
    return FileResponse(path=str(latest_plot), filename=latest_plot.name, media_type='image/png') 

@app.post("/prompt/preview", response_model=PromptPreviewResponse)
async def get_prompt_preview(request: PromptPreviewRequest):
    """Get the full prompt preview for a given prompt type and sample question"""
    try:
        # Import evaluation code
        import sys
        config = path_manager.get_config()
        evaluation_dir = Path(config.evaluation_dir)
        
        # Add evaluation directory to path
        if str(evaluation_dir) not in sys.path:
            sys.path.insert(0, str(evaluation_dir))
        
        # Import evaluation utilities
        from utils import construct_prompt
        from examples import get_examples
        
        # Create a mock args object with the prompt type
        class MockArgs:
            def __init__(self, prompt_type, custom_prompt, num_shots):
                self.prompt_type = prompt_type
                self.prompt = custom_prompt if custom_prompt else None
                self.num_shots = num_shots
                self.adapt_few_shot = False
        
        # Use the original prompt_type for args (not the mapped one)
        args = MockArgs(request.prompt_type, request.custom_prompt, request.num_shots)
        
        # Create a mock example with the sample question
        example = {
            "question": request.sample_question,
            "gt_ans": "42"  # Dummy answer for construction
        }
        
        # Map dataset names (similar to load_prompt logic)
        # Handle comma-separated datasets by using the first one
        data_name = request.dataset.split(',')[0].strip() if ',' in request.dataset else request.dataset.strip()
        if data_name in ["gsm_hard", "svamp", "tabmwp", "asdiv", "mawps"]:
            data_name = "gsm8k"
        elif data_name in ["math_oai", "hungarian_exam", "math-oai", "aime24", "amc23"]:
            data_name = "math"
        elif data_name in ["sat_math"]:
            data_name = "mmlu_stem"
        elif data_name in ["gaokao2024_I", "gaokao2024_II", "gaokao_math_qa", "gaokao2024_mix", "cn_middle_school"]:
            data_name = "gaokao"
        
        # Handle prompt type mapping (similar to construct_prompt logic)
        prompt_type = request.prompt_type
        if prompt_type == "platypus_fs":
            prompt_type_for_load = "cot"  # Use cot examples for platypus_fs
        elif prompt_type == "tool-integrated":
            prompt_type_for_load = "tora"  # Use tora examples for tool-integrated
        else:
            prompt_type_for_load = prompt_type
        
        # Construct the prompt
        try:
            # Import evaluation utilities
            from examples import get_examples
            EXAMPLES = get_examples()
            
            # Check if dataset has examples available (use base dataset name)
            # The load_prompt function maps datasets to base names, so we just need to ensure
            # the base dataset exists after mapping
            if data_name not in EXAMPLES:
                # Default to gsm8k if dataset not found (most common dataset)
                if "gsm8k" in EXAMPLES:
                    data_name = "gsm8k"
                elif "math" in EXAMPLES:
                    data_name = "math"
                else:
                    # Get available base datasets (exclude prompt-specific ones like "gsm8k-pal")
                    available_datasets = [d for d in EXAMPLES.keys() 
                                        if d in ['gsm8k', 'math', 'mmlu_stem', 'gaokao', 'carp_en', 'minerva_math', 'aqua', 'sat_math', 'mmlu_mathematics', 'mmlu_physics', 'mmlu_chemistry', 'mmlu_biology', 'mmlu_computer']]
                    if available_datasets:
                        data_name = available_datasets[0]
                    else:
                        raise ValueError(f"Dataset '{request.dataset}' not found in EXAMPLES and no fallback available.")
            
            # Check if prompt type exists in PROMPT_TEMPLATES (for non-custom prompts)
            from utils import PROMPT_TEMPLATES
            if request.prompt_type != "custom" and request.prompt_type not in PROMPT_TEMPLATES:
                available_types = ', '.join(sorted(PROMPT_TEMPLATES.keys()))
                raise ValueError(f"Prompt type '{request.prompt_type}' not found in PROMPT_TEMPLATES. Available types: {available_types}")
            
            # Construct the prompt using the evaluation code
            # This will handle all the formatting, few-shot examples, and special cases
            full_prompt = construct_prompt(example, data_name, args)
            
        except KeyError as e:
            # Handle missing key errors (e.g., dataset not in EXAMPLES)
            from utils import PROMPT_TEMPLATES
            try:
                from examples import get_examples
                EXAMPLES = get_examples()
                available_datasets = [d for d in EXAMPLES.keys() 
                                    if d in ['gsm8k', 'math', 'mmlu_stem', 'gaokao', 'carp_en', 'minerva_math', 'aqua', 'sat_math']]
            except:
                available_datasets = ['gsm8k', 'math']  # Fallback
            
            available_types = ', '.join(sorted(PROMPT_TEMPLATES.keys())) if 'PROMPT_TEMPLATES' in locals() else 'Unknown'
            full_prompt = f"Error: {str(e)}\n\nAvailable datasets: {', '.join(available_datasets)}\nAvailable prompt types: {available_types}\nPrompt type requested: {request.prompt_type}\nDataset mapped to: {data_name}\nSample question: {request.sample_question}"
        except ValueError as e:
            # Handle value errors (e.g., invalid prompt type or dataset)
            full_prompt = f"Error: {str(e)}\n\nPrompt type: {request.prompt_type}\nDataset: {data_name}\nSample question: {request.sample_question}"
        except Exception as e:
            # If construction fails, return a detailed error message
            import traceback
            error_details = traceback.format_exc()
            full_prompt = f"Error constructing prompt: {str(e)}\n\nPrompt type: {request.prompt_type}\nDataset: {data_name}\nSample question: {request.sample_question}\n\nTraceback:\n{error_details}"
        
        return PromptPreviewResponse(
            sample_question=request.sample_question,
            full_prompt=full_prompt,
            prompt_type=request.prompt_type
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating prompt preview: {str(e)}") 
