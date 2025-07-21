from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import subprocess
import random
import os
from typing import List
from .models import (
    RunRequest,
    RunResponse,
    JobSummary,
    LogResponse,
    UnitTestRequest,
    UnitTestResponse,
)
from .job_manager import job_manager


app = FastAPI(
    title="Reproducibility Dashboard API",
    description="API for managing hyperparameter sweeps and reproducibility experiments",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/run", response_model=RunResponse)
async def run_experiment(request: RunRequest):
    """Start a new experiment job."""
    try:
        run_id = job_manager.start_job(request)
        total_experiments = job_manager.calculate_total_experiments(request)

        return RunResponse(
            run_id=str(run_id),
            message=f"Experiment started successfully. Running {total_experiments} experiments.",
            script_path=f"run-{run_id}.sbat",
            total_experiments=total_experiments,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs", response_model=List[JobSummary])
async def get_jobs():
    """Get list of all jobs."""
    return job_manager.get_jobs()


@app.get("/jobs/{run_id}/logs", response_model=LogResponse)
async def get_job_logs(run_id: str):
    """Get logs for a specific job."""
    from uuid import UUID

    try:
        run_uuid = UUID(run_id)
        logs = job_manager.get_job_logs(run_uuid)
        status = job_manager.get_job_status(run_uuid)

        if logs is None:
            raise HTTPException(status_code=404, detail="Job not found")

        return LogResponse(run_id=run_id, logs=logs, status=status)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/jobs/{run_id}")
async def cancel_job(run_id: str):
    """Cancel a running job."""
    from uuid import UUID

    try:
        run_uuid = UUID(run_id)
        success = job_manager.cancel_job(run_uuid)

        if not success:
            raise HTTPException(
                status_code=404, detail="Job not found or cannot be cancelled"
            )

        return {"message": "Job cancelled successfully"}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results")
async def get_results():
    """Get all experiment results."""
    try:
        results = job_manager.get_results()
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/results/download")
async def download_results():
    """Download results as CSV file."""
    try:
        if not os.path.exists(job_manager.results_file):
            raise HTTPException(status_code=404, detail="No results file found")

        return FileResponse(
            job_manager.results_file,
            media_type="text/csv",
            filename="experiment_results.csv",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/unit_test", response_model=UnitTestResponse)
async def run_unit_test(request: UnitTestRequest):
    """Run a unit test with a single example."""
    try:
        # Create a simple test script
        test_script = f"""#!/bin/bash
cd {request.local_dir}
source ~/anaconda3/etc/profile.d/conda.sh
conda activate reproducibility-env

# Run a single test
bash run.sh \\
  --model "{request.model}" \\
  --dataset "{request.dataset}" \\
  --split "{request.split or "test"}" \\
  --temperature {request.temperature} \\
  --top_p {request.top_p} \\
  --top_k {request.top_k} \\
  --seed {request.seed} \\
  --max_length {request.max_length or 2048} \\
  --max_new_tokens {request.max_new_tokens or 512} \\
  --prompt "{request.prompt}" \\
  --output_dir ./unit_test_output
"""

        # Write test script
        test_script_path = "unit_test.sh"
        with open(test_script_path, "w") as f:
            f.write(test_script)

        # Make executable
        os.chmod(test_script_path, 0o755)

        # Run test
        result = subprocess.run(
            ["bash", test_script_path],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Clean up
        os.remove(test_script_path)

        if result.returncode == 0:
            return UnitTestResponse(success=True, output=result.stdout, error=None)
        else:
            return UnitTestResponse(
                success=False, output=result.stdout, error=result.stderr
            )

    except subprocess.TimeoutExpired:
        return UnitTestResponse(
            success=False, output="", error="Unit test timed out after 5 minutes"
        )
    except Exception as e:
        return UnitTestResponse(success=False, output="", error=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
