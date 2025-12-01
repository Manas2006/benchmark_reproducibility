#!/usr/bin/env python3
"""
Re-run evaluation on all completed HumanEval jobs with the updated evaluation code.
"""
import subprocess
import glob
import os
import sys

def main():
    # Find all HumanEval result files
    files = []
    for file in glob.glob('evaluation/outputs/*/humaneval/*.jsonl'):
        if '_processed' not in file and '_prob' not in file and '_metrics' not in file:
            files.append(file)
    
    print(f'Found {len(files)} HumanEval result files to re-evaluate\n')
    
    for i, file_path in enumerate(files, 1):
        print(f'[{i}/{len(files)}] Re-evaluating: {file_path}')
        print('-' * 80)
        
        # Run evaluation - use conda environment
        eval_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluation')
        abs_file_path = os.path.abspath(file_path)
        
        # Use bash to activate conda and run the evaluation
        cmd = f"""source /work/10757/manasp123/ls6/miniconda3/etc/profile.d/conda.sh && \
conda activate math_eval && \
cd {eval_dir} && \
python3 evaluate.py --data_name humaneval --prompt_type humaneval --file_path {abs_file_path}"""
        
        try:
            result = subprocess.run(cmd, shell=True, executable='/bin/bash',
                                  capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                print(f'✅ Successfully re-evaluated: {file_path}\n')
                # Print last few lines of output to see results
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    print('\n'.join(lines[-10:]))
                    print()
            else:
                print(f'❌ Error re-evaluating {file_path}:')
                print(result.stderr)
                print()
        except subprocess.TimeoutExpired:
            print(f'⏱️  Timeout re-evaluating {file_path} (exceeded 1 hour)\n')
        except Exception as e:
            print(f'❌ Exception re-evaluating {file_path}: {e}\n')
    
    print('=' * 80)
    print(f'Completed re-evaluation of {len(files)} files')

if __name__ == '__main__':
    main()

