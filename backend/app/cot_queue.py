"""
CoT Analysis Queue Management

This module provides queue management for CoT analysis jobs with parallel execution
and progress tracking.
"""

import asyncio
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from .runner import job_db, save_job_db

# Maximum concurrent CoT analysis jobs
MAX_CONCURRENT_JOBS = 2

# Global semaphore for controlling concurrency
_concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

class CoTAnalysisQueue:
    """
    Queue manager for CoT analysis jobs.
    
    Manages job submission, status tracking, and parallel execution with concurrency control.
    """
    
    def __init__(self):
        """Initialize the queue manager"""
        self._queue_lock = asyncio.Lock()
        self._running_jobs: Dict[str, asyncio.Task] = {}
    
    def generate_cot_job_id(self, parent_job_id: str) -> str:
        """
        Generate a unique CoT analysis job ID.
        
        Args:
            parent_job_id: The parent job ID
            
        Returns:
            Unique CoT job ID in format: cot_{parent_job_id}_{timestamp}
        """
        timestamp = int(time.time())
        return f"cot_{parent_job_id}_{timestamp}"
    
    async def submit(
        self,
        parent_job_id: str,
        config: Dict[str, Any],
        execute_fn: Callable[[str, str, Dict[str, Any]], None]
    ) -> str:
        """
        Submit a CoT analysis job to the queue.
        
        Args:
            parent_job_id: The parent job ID
            config: Configuration dict with judge_mode, diagnostic, etc.
            execute_fn: Async function to execute the analysis
            
        Returns:
            CoT job ID
        """
        cot_job_id = self.generate_cot_job_id(parent_job_id)
        
        # Get parent job info
        parent_job = job_db.get(parent_job_id, {})
        
        # Create queue entry
        queue_entry = {
            "cot_job_id": cot_job_id,
            "parent_job_id": parent_job_id,
            "status": "QUEUED",
            "config": config,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None,
            "model_name": parent_job.get("request", {}).get("model", "Unknown"),
            "dataset": parent_job.get("request", {}).get("dataset", "unknown"),
            "judge_mode": config.get("judge_mode", "ALWAYS"),
            "progress": {
                "current_sample": 0,
                "total_samples": 0,
                "processed_samples": 0,
                "percentage": 0.0,
                "estimated_time_remaining": None,
                "start_time": None,
                "current_activity": "Waiting in queue..."
            },
            "queue_position": None,
            "error": None
        }
        
        async with self._queue_lock:
            # Calculate queue position (jobs in QUEUED or RUNNING state)
            queued_jobs = [
                (jid, info) for jid, info in job_db.items()
                if jid.startswith("cot_analysis_") and info.get("status") in ["QUEUED", "RUNNING"]
            ]
            queue_entry["queue_position"] = len(queued_jobs)
            
            # Store in job_db with prefix
            job_db[f"cot_analysis_{cot_job_id}"] = queue_entry
            save_job_db()
            
            # Start processing if we're under the limit
            await self._try_start_next_job(cot_job_id, execute_fn)
        
        return cot_job_id
    
    async def _try_start_next_job(self, cot_job_id: str, execute_fn: Callable[[str, str, Dict[str, Any]], None]):
        """
        Try to start the next job in the queue if there's capacity.
        
        Args:
            cot_job_id: The CoT job ID to potentially start
            execute_fn: Async function to execute the analysis
        """
        # Count running jobs
        running_count = sum(
            1 for jid, info in job_db.items()
            if jid.startswith("cot_analysis_") and info.get("status") == "RUNNING"
        )
        
        # Check if we can start this job
        if running_count < MAX_CONCURRENT_JOBS:
            queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
            if queue_entry and queue_entry.get("status") == "QUEUED":
                # Start the job
                async with self._queue_lock:
                    # Double-check status hasn't changed
                    queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
                    if queue_entry and queue_entry.get("status") == "QUEUED":
                        # Update status to RUNNING
                        queue_entry["status"] = "RUNNING"
                        queue_entry["started_at"] = time.time()
                        queue_entry["queue_position"] = None
                        queue_entry["progress"]["start_time"] = time.time()
                        queue_entry["progress"]["current_activity"] = "Starting analysis..."
                        job_db[f"cot_analysis_{cot_job_id}"] = queue_entry
                        save_job_db()
                        
                        # Get parent_job_id and config from queue_entry
                        parent_job_id = queue_entry.get("parent_job_id")
                        config = queue_entry.get("config", {})
                        
                        # Start the task
                        task = asyncio.create_task(
                            self._run_with_semaphore(cot_job_id, parent_job_id, config, execute_fn)
                        )
                        self._running_jobs[cot_job_id] = task
    
    async def _run_with_semaphore(self, cot_job_id: str, parent_job_id: str, config: Dict[str, Any], execute_fn: Callable[[str, str, Dict[str, Any]], None]):
        """
        Run a CoT analysis job with semaphore control.
        
        Args:
            cot_job_id: The CoT job ID
            parent_job_id: The parent job ID
            config: Configuration dict
            execute_fn: Async function to execute the analysis
        """
        async with _concurrency_semaphore:
            try:
                # Execute the analysis
                await execute_fn(cot_job_id, parent_job_id, config)
                
                # Mark as done
                queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
                if queue_entry:
                    queue_entry["status"] = "DONE"
                    queue_entry["completed_at"] = time.time()
                    queue_entry["progress"]["percentage"] = 100.0
                    queue_entry["progress"]["current_activity"] = "Analysis complete!"
                    queue_entry["progress"]["estimated_time_remaining"] = 0.0
                    job_db[f"cot_analysis_{cot_job_id}"] = queue_entry
                    save_job_db()
            except Exception as e:
                # Mark as error
                queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
                if queue_entry:
                    queue_entry["status"] = "ERROR"
                    queue_entry["completed_at"] = time.time()
                    queue_entry["error"] = str(e)
                    queue_entry["progress"]["current_activity"] = f"Error: {str(e)}"
                    job_db[f"cot_analysis_{cot_job_id}"] = queue_entry
                    save_job_db()
            finally:
                # Remove from running jobs
                if cot_job_id in self._running_jobs:
                    del self._running_jobs[cot_job_id]
                
                # Note: Queue processing for next jobs is handled by submit() 
                # when new jobs are added, so we don't need to process queue here
    
    async def _process_queue(self, execute_fn: Callable[[str, str, Dict[str, Any]], None]):
        """
        Process the queue and start next jobs if capacity is available.
        
        Args:
            execute_fn: Async function to execute the analysis
        """
        # Find next queued job
        queued_jobs = [
            (jid.replace("cot_analysis_", ""), info)
            for jid, info in job_db.items()
            if jid.startswith("cot_analysis_") and info.get("status") == "QUEUED"
        ]
        
        # Sort by creation time (FIFO)
        queued_jobs.sort(key=lambda x: x[1].get("created_at", 0))
        
        # Try to start jobs until we hit the limit
        for cot_job_id, job_info in queued_jobs:
            running_count = sum(
                1 for jid, info in job_db.items()
                if jid.startswith("cot_analysis_") and info.get("status") == "RUNNING"
            )
            
            if running_count >= MAX_CONCURRENT_JOBS:
                break
            
            await self._try_start_next_job(cot_job_id, execute_fn)
    
    def get_status(self, cot_job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a CoT analysis job.
        
        Args:
            cot_job_id: The CoT job ID
            
        Returns:
            Status dictionary or None if not found
        """
        queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
        if not queue_entry:
            return None
        
        # Update queue position if queued
        if queue_entry.get("status") == "QUEUED":
            queued_jobs = [
                (jid, info) for jid, info in job_db.items()
                if jid.startswith("cot_analysis_") and info.get("status") in ["QUEUED", "RUNNING"]
                and info.get("created_at", 0) < queue_entry.get("created_at", float('inf'))
            ]
            queue_entry["queue_position"] = len(queued_jobs)
        
        return queue_entry
    
    def list_jobs(self) -> List[Dict[str, Any]]:
        """
        List all CoT analysis jobs.
        
        Returns:
            List of job status dictionaries
        """
        jobs = []
        for jid, info in job_db.items():
            if jid.startswith("cot_analysis_"):
                jobs.append(info)
        
        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        return jobs
    
    def update_progress(
        self,
        cot_job_id: str,
        current_sample: int,
        total_samples: int,
        current_activity: Optional[str] = None
    ):
        """
        Update progress for a running CoT analysis job.
        
        Args:
            cot_job_id: The CoT job ID
            current_sample: Currently processing sample index (0-based)
            total_samples: Total samples to process
            current_activity: Optional activity description
        """
        queue_entry = job_db.get(f"cot_analysis_{cot_job_id}")
        if not queue_entry:
            return
        
        progress = queue_entry.get("progress", {})
        progress["current_sample"] = current_sample
        progress["total_samples"] = total_samples
        progress["processed_samples"] = current_sample + 1
        
        # Calculate percentage
        if total_samples > 0:
            progress["percentage"] = (progress["processed_samples"] / total_samples) * 100.0
        else:
            progress["percentage"] = 0.0
        
        # Calculate estimated time remaining
        start_time = progress.get("start_time")
        if start_time and current_sample > 0:
            elapsed = time.time() - start_time
            avg_time_per_sample = elapsed / (current_sample + 1)
            remaining_samples = total_samples - progress["processed_samples"]
            progress["estimated_time_remaining"] = avg_time_per_sample * remaining_samples
        else:
            progress["estimated_time_remaining"] = None
        
        # Update current activity
        if current_activity:
            progress["current_activity"] = current_activity
        elif total_samples > 0:
            progress["current_activity"] = f"Analyzing sample {progress['processed_samples']}/{total_samples}..."
        else:
            progress["current_activity"] = "Processing..."
        
        queue_entry["progress"] = progress
        job_db[f"cot_analysis_{cot_job_id}"] = queue_entry
        
        # Save periodically (not on every update to avoid excessive writes)
        # In production, you might want to add debouncing here
        save_job_db()

# Global queue instance
cot_queue = CoTAnalysisQueue()

