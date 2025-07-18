import os
import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from uuid import UUID, uuid4
from typing import List, Optional, Dict, Any
from jinja2 import Environment, FileSystemLoader
from .models import RunRequest, JobInfo, JobStatus, JobSummary


class JobManager:
    def __init__(self, results_file: str = "results.csv"):
        self.jobs: Dict[UUID, JobInfo] = {}
        self.results_file = results_file
        self.template_env = Environment(
            loader=FileSystemLoader(
                os.path.join(os.path.dirname(__file__), "templates")
            )
        )
        self._initialize_results_csv()

    def _initialize_results_csv(self):
        """Initialize the results CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.results_file):
            headers = [
                "run_id",
                "model",
                "dataset",
                "split",
                "temp",
                "top_p",
                "top_k",
                "seed",
                "max_length",
                "max_new_tokens",
                "accuracy",
                "loss",
                "runtime",
                "timestamp",
                "custom_metrics",
                "evaluation_metric",
                "at_k_value",
                "evaluation_prompt",
                "evaluation_tool",
                "judge_model",
            ]
            with open(self.results_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def generate_run_id(self) -> UUID:
        """Generate a new unique run ID."""
        return uuid4()

    def create_experiment_script(self, request: RunRequest) -> str:
        """Generate the experiment script from template and save it."""
        if not request.run_id:
            request.run_id = self.generate_run_id()

        # Ensure output directory exists
        os.makedirs(request.output_dir, exist_ok=True)

        # Render the template
        template = self.template_env.get_template("run_script.j2")
        script_content = template.render(
            raw_sbatch_directives=request.raw_sbatch_directives,
            local_dir=request.local_dir,
            output_dir=request.output_dir,
            models=request.models,
            datasets=request.datasets,
            temps=request.temps,
            top_ps=request.top_ps,
            top_ks=request.top_ks,
            max_lengths=request.max_lengths,
            max_new_tokens=request.max_new_tokens,
            seeds=request.seeds,
            prompt=request.prompt,
            evaluation_metric=request.evaluation_metric,
            at_k_value=request.at_k_value,
            evaluation_prompt=request.evaluation_prompt,
            evaluation_tool=request.evaluation_tool,
            judge_model=request.judge_model,
            judge_api_key=request.judge_api_key,
            local_llm_path=request.local_llm_path,
            custom_extractor_code=request.custom_extractor_code,
            existing_result_path=request.existing_result_path,
        )

        # Save script to file
        script_path = os.path.join(request.output_dir, f"run-{request.run_id}.sbat")
        with open(script_path, "w") as f:
            f.write(script_content)

        # Make script executable
        os.chmod(script_path, 0o755)

        return script_path

    def calculate_total_experiments(self, request: RunRequest) -> int:
        """Calculate total number of experiments that will be run."""
        if request.existing_result_path:
            return 1  # Only evaluation, no generation
        return (
            len(request.models)
            * len(request.datasets)
            * len(request.temps)
            * len(request.top_ps)
            * len(request.top_ks)
            * len(request.seeds)
        )

    def start_job(self, request: RunRequest) -> UUID:
        """Start a new experiment job."""
        if not request.run_id:
            request.run_id = self.generate_run_id()

        # Generate script
        script_path = self.create_experiment_script(request)

        # Create job info
        job_info = JobInfo(
            run_id=request.run_id,
            status=JobStatus.PENDING,
            started_at=datetime.now(),
            params=request,
            script_path=script_path,
        )

        self.jobs[request.run_id] = job_info

        # Submit to SLURM
        try:
            result = subprocess.run(
                ["sbatch", script_path], capture_output=True, text=True, check=True
            )

            # Extract SLURM job ID from output
            slurm_job_id = None
            for line in result.stdout.split("\n"):
                if "Submitted batch job" in line:
                    slurm_job_id = line.split()[-1]
                    break

            job_info.slurm_job_id = slurm_job_id
            job_info.status = JobStatus.RUNNING

            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_slurm_job,
                args=(request.run_id, slurm_job_id),
                daemon=True,
            )
            monitor_thread.start()

        except subprocess.CalledProcessError as e:
            job_info.status = JobStatus.FAILED
            job_info.log_buffer.append(f"Failed to submit job: {e.stderr}")

        return request.run_id

    def _monitor_slurm_job(self, run_id: UUID, slurm_job_id: str):
        """Monitor SLURM job status and logs."""
        job_info = self.jobs[run_id]

        try:
            while True:
                # Check job status
                status_result = subprocess.run(
                    ["squeue", "-j", slurm_job_id, "--noheader", "--format=%T"],
                    capture_output=True,
                    text=True,
                )

                if status_result.returncode != 0:
                    # Job no longer in queue, check if completed
                    sacct_result = subprocess.run(
                        ["sacct", "-j", slurm_job_id, "--format=State", "--noheader"],
                        capture_output=True,
                        text=True,
                    )

                    if sacct_result.returncode == 0:
                        state = sacct_result.stdout.strip()
                        if "COMPLETED" in state:
                            job_info.status = JobStatus.COMPLETED
                        elif "FAILED" in state or "CANCELLED" in state:
                            job_info.status = JobStatus.FAILED
                        else:
                            job_info.status = JobStatus.FAILED
                    else:
                        job_info.status = JobStatus.FAILED

                    break

                time.sleep(5)  # Poll every 5 seconds

            # Job completed, read output file
            job_info.completed_at = datetime.now()
            job_info.duration = (
                job_info.completed_at - job_info.started_at
            ).total_seconds()

            # Try to read SLURM output file
            output_file = f"slurm-{slurm_job_id}.out"
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    for line in f:
                        job_info.log_buffer.append(line.rstrip())

                        # Parse RESULT lines
                        if line.strip().startswith("RESULT:"):
                            try:
                                result_json = line.strip()[
                                    7:
                                ]  # Remove "RESULT:" prefix
                                result_data = json.loads(result_json)
                                self._save_result(run_id, result_data)
                            except json.JSONDecodeError as e:
                                job_info.log_buffer.append(f"Error parsing result: {e}")

        except Exception as e:
            job_info.status = JobStatus.FAILED
            job_info.log_buffer.append(f"Error monitoring job: {str(e)}")

    def _save_result(self, run_id: UUID, result_data: dict):
        """Save a single experiment result to CSV."""
        try:
            job_info = self.jobs[run_id]

            # Prepare row data
            row = [
                str(run_id),
                result_data.get("model", ""),
                result_data.get("dataset", ""),
                result_data.get("split", ""),
                result_data.get("temp", ""),
                result_data.get("top_p", ""),
                result_data.get("top_k", ""),
                result_data.get("seed", ""),
                result_data.get("max_length", ""),
                result_data.get("max_new_tokens", ""),
                result_data.get("accuracy", ""),
                result_data.get("loss", ""),
                result_data.get("runtime", ""),
                datetime.now().isoformat(),
                json.dumps(result_data.get("custom_metrics", {})),
                # Evaluation settings
                job_info.params.evaluation_metric,
                job_info.params.at_k_value,
                job_info.params.evaluation_prompt,
                job_info.params.evaluation_tool,
                job_info.params.judge_model or "",
            ]

            # Append to CSV
            with open(self.results_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

        except Exception as e:
            job_info = self.jobs[run_id]
            job_info.log_buffer.append(f"Error saving result: {str(e)}")

    def get_jobs(self) -> List[JobSummary]:
        """Get list of all jobs with summary information."""
        summaries = []
        for job_info in self.jobs.values():
            total_experiments = self.calculate_total_experiments(job_info.params)
            summary = JobSummary(
                run_id=str(job_info.run_id),
                slurm_job_id=job_info.slurm_job_id,
                status=job_info.status,
                started_at=job_info.started_at,
                duration=job_info.duration,
                job_name=job_info.params.job_name_template,
                total_experiments=total_experiments,
            )
            summaries.append(summary)

        return sorted(summaries, key=lambda x: x.started_at, reverse=True)

    def get_job_logs(self, run_id: UUID) -> Optional[str]:
        """Get logs for a specific job."""
        if run_id in self.jobs:
            return "\n".join(self.jobs[run_id].log_buffer)
        return None

    def get_job_status(self, run_id: UUID) -> Optional[JobStatus]:
        """Get status for a specific job."""
        if run_id in self.jobs:
            return self.jobs[run_id].status
        return None

    def cancel_job(self, run_id: UUID) -> bool:
        """Cancel a running job."""
        if run_id not in self.jobs:
            return False

        job_info = self.jobs[run_id]
        if job_info.status == JobStatus.RUNNING and job_info.slurm_job_id:
            try:
                subprocess.run(["scancel", job_info.slurm_job_id], check=True)
                job_info.status = JobStatus.CANCELLED
                job_info.completed_at = datetime.now()
                job_info.duration = (
                    job_info.completed_at - job_info.started_at
                ).total_seconds()
                return True
            except subprocess.CalledProcessError:
                # Job already completed or failed
                job_info.status = JobStatus.COMPLETED
                return True

        return False

    def get_results(self) -> List[dict]:
        """Read and return all results from CSV."""
        results = []

        if not os.path.exists(self.results_file):
            return results

        try:
            with open(self.results_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Parse custom_metrics JSON
                    try:
                        row["custom_metrics"] = (
                            json.loads(row["custom_metrics"])
                            if row["custom_metrics"]
                            else {}
                        )
                    except json.JSONDecodeError:
                        row["custom_metrics"] = {}

                    # Convert numeric fields
                    for field in [
                        "temp",
                        "top_p",
                        "top_k",
                        "seed",
                        "accuracy",
                        "loss",
                        "runtime",
                        "at_k_value",
                    ]:
                        if row.get(field):
                            try:
                                row[field] = float(row[field])
                            except (ValueError, TypeError):
                                row[field] = None

                    for field in ["max_length", "max_new_tokens"]:
                        if row.get(field):
                            try:
                                row[field] = int(row[field])
                            except (ValueError, TypeError):
                                row[field] = None

                    results.append(row)

        except Exception as e:
            print(f"Error reading results: {e}")

        return results


# Global job manager instance
job_manager = JobManager()
