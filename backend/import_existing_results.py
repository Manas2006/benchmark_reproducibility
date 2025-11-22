#!/usr/bin/env python3
"""
Utility script to import existing result files from evaluation/outputs/ into the job database.

This script scans the outputs directory for result files and registers them in job_db.json
so they appear in the evaluation system UI.

Usage:
    python import_existing_results.py [--output-dir PATH] [--dry-run]
"""

import json
import uuid
import re
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import argparse

# Add the app directory to the path so we can import modules
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(Path(__file__).parent))

# Import with proper module path
from app.path_manager import path_manager
from app.schemas import JobStatus

def parse_result_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse a result filename to extract job metadata.
    
    Expected format: test_{prompt_type}_{num_test_sample}_seed{seed}_t{temperature}_s{start}_e{end}[_{job_id}].jsonl
    
    Returns dict with parsed information or None if format doesn't match.
    """
    # Pattern: test_{prompt_type}_{num_test_sample}_seed{seed}_t{temperature}_s{start}_e{end}[_{job_id}].jsonl
    pattern = r'test_([^_]+)_(-?\d+)_seed(\d+)_t([\d.]+)_s(\d+)_e(-?\d+)(?:_([a-f0-9-]+))?\.jsonl$'
    match = re.match(pattern, filename)
    
    if not match:
        return None
    
    prompt_type, num_test_sample, seed, temperature, start, end, job_id = match.groups()
    
    return {
        'prompt_type': prompt_type,
        'num_test_sample': int(num_test_sample),
        'seed': int(seed),
        'temperature': float(temperature),
        'start': int(start),
        'end': int(end),
        'job_id': job_id  # May be None
    }

def find_result_files(output_dir: Path) -> list[tuple[Path, Dict[str, Any]]]:
    """
    Scan output directory for result files and return them with parsed metadata.
    
    Returns list of (file_path, metadata) tuples.
    """
    result_files = []
    
    # Walk through model directories
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        
        # Walk through dataset directories
        for dataset_dir in model_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            
            # Look for .jsonl files in this directory
            for file_path in dataset_dir.glob("*.jsonl"):
                # Skip probability files and processed files
                if "_prob.jsonl" in file_path.name or "_processed.jsonl" in file_path.name:
                    continue
                
                # Parse filename
                metadata = parse_result_filename(file_path.name)
                if metadata:
                    metadata['model'] = model_name
                    metadata['dataset'] = dataset_name
                    metadata['file_path'] = file_path
                    result_files.append((file_path, metadata))
    
    return result_files

def find_prob_file(result_file: Path, job_id: str) -> Optional[Path]:
    """Find the corresponding probability file for a result file."""
    # Probability files have format: {base}_{prompt_type}_prob.jsonl
    base_name = result_file.stem
    # Remove job_id suffix if present
    if f"_{job_id}" in base_name:
        base_name = base_name.replace(f"_{job_id}", "")
    
    # Extract prompt_type from base_name
    metadata = parse_result_filename(result_file.name)
    if not metadata:
        return None
    
    prompt_type = metadata['prompt_type']
    prob_file = result_file.parent / f"{base_name}_{prompt_type}_prob.jsonl"
    
    if prob_file.exists():
        return prob_file
    
    # Try alternative: with job_id
    if job_id:
        alt_base = result_file.stem.replace(f"_{job_id}", "")
        alt_prob = result_file.parent / f"{alt_base}_{prompt_type}_prob.jsonl"
        if alt_prob.exists():
            return alt_prob
    
    return None

def create_job_entry(result_file: Path, metadata: Dict[str, Any], job_db: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a job database entry for an existing result file.
    
    Returns the job_id and job entry dict.
    """
    # Use existing job_id from filename if present, otherwise generate new one
    job_id = metadata.get('job_id')
    if not job_id or job_id in job_db:
        # Generate new UUID
        job_id = str(uuid.uuid4())
    
    # Find probability file if it exists
    prob_file = find_prob_file(result_file, metadata.get('job_id', ''))
    
    # Build request dict (matching the format in job_db.json)
    request = {
        'model': metadata['model'],
        'dataset': metadata['dataset'],
        'prompt_type': metadata['prompt_type'],
        'seed': metadata['seed'],
        'temperature': metadata['temperature'],
        'backend': 'imported',  # Mark as imported
        'eval_method': 'pass@k',  # Default, may need adjustment
        'n_sampling': 1,  # Default, may need adjustment
        'top_p': 1.0,  # Default
        'top_k': -1  # Default
    }
    
    # Create job entry
    job_entry = {
        'status': JobStatus.DONE,  # Assume completed if file exists
        'request': request,
        'backend': 'imported',
        'result_file': str(result_file),
    }
    
    if prob_file:
        job_entry['prob_file'] = str(prob_file)
    
    return job_id, job_entry

def import_results(output_dir: Optional[Path] = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Import existing result files into the job database.
    
    Args:
        output_dir: Directory to scan (defaults to configured output_dir)
        dry_run: If True, don't actually modify job_db.json
    
    Returns:
        Dict with import statistics
    """
    config = path_manager.get_config()
    
    if output_dir is None:
        output_dir = Path(config.output_dir)
    
    if not output_dir.exists():
        return {
            'error': f'Output directory does not exist: {output_dir}',
            'imported': 0,
            'skipped': 0,
            'errors': []
        }
    
    # Load existing job database
    job_db_path = Path(config.job_db_path)
    if job_db_path.exists():
        with open(job_db_path, 'r') as f:
            job_db = json.load(f)
    else:
        job_db = {}
    
    # Find all result files
    result_files = find_result_files(output_dir)
    
    imported = []
    skipped = []
    errors = []
    
    for result_file, metadata in result_files:
        try:
            # Check if this file is already registered
            existing_job_id = metadata.get('job_id')
            if existing_job_id and existing_job_id in job_db:
                # Check if result_file matches
                existing_entry = job_db[existing_job_id]
                if existing_entry.get('result_file') == str(result_file):
                    skipped.append({
                        'file': str(result_file),
                        'reason': 'Already registered',
                        'job_id': existing_job_id
                    })
                    continue
            
            # Create job entry
            job_id, job_entry = create_job_entry(result_file, metadata, job_db)
            
            # Check if we're creating a duplicate (same file path)
            duplicate = False
            for existing_id, existing_entry in job_db.items():
                if existing_entry.get('result_file') == str(result_file):
                    skipped.append({
                        'file': str(result_file),
                        'reason': 'Duplicate file path',
                        'existing_job_id': existing_id
                    })
                    duplicate = True
                    break
            
            if duplicate:
                continue
            
            if not dry_run:
                job_db[job_id] = job_entry
            
            imported.append({
                'file': str(result_file),
                'job_id': job_id,
                'model': metadata['model'],
                'dataset': metadata['dataset']
            })
            
        except Exception as e:
            errors.append({
                'file': str(result_file),
                'error': str(e)
            })
    
    # Save job database
    if not dry_run and imported:
        with open(job_db_path, 'w') as f:
            json.dump(job_db, f, indent=2)
    
    return {
        'imported': len(imported),
        'skipped': len(skipped),
        'errors': len(errors),
        'imported_files': imported,
        'skipped_files': skipped,
        'error_files': errors
    }

def main():
    parser = argparse.ArgumentParser(
        description='Import existing result files into the job database'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory to scan (defaults to configured output_dir)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be imported without actually importing'
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    print(f"Scanning for result files...")
    if args.dry_run:
        print("DRY RUN MODE - No changes will be made")
    
    result = import_results(output_dir, dry_run=args.dry_run)
    
    print(f"\nImport Summary:")
    print(f"  Imported: {result['imported']}")
    print(f"  Skipped: {result['skipped']}")
    print(f"  Errors: {result['errors']}")
    
    if result['imported_files']:
        print(f"\nImported files:")
        for item in result['imported_files']:
            print(f"  - {item['job_id']}: {item['file']}")
    
    if result['skipped_files']:
        print(f"\nSkipped files:")
        for item in result['skipped_files'][:10]:  # Show first 10
            print(f"  - {item['file']}: {item['reason']}")
        if len(result['skipped_files']) > 10:
            print(f"  ... and {len(result['skipped_files']) - 10} more")
    
    if result['error_files']:
        print(f"\nErrors:")
        for item in result['error_files']:
            print(f"  - {item['file']}: {item['error']}")
    
    if 'error' in result:
        print(f"\nError: {result['error']}")
        sys.exit(1)

if __name__ == '__main__':
    main()

