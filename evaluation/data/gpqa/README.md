# GPQA Dataset

GPQA is a challenging multiple-choice Q&A dataset of very hard questions written and validated by experts in biology, physics, and chemistry.

**Dataset**: [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa)  
**Paper**: [arXiv:2311.12022](https://arxiv.org/abs/2311.12022)  
**Repository**: https://github.com/idavidrein/gpqa

## Download Instructions

GPQA is a **gated dataset** on HuggingFace, which means you need to accept the terms and conditions before accessing it.

### Steps to Download:

1. **Accept the terms on HuggingFace:**
   - Visit: https://huggingface.co/datasets/Idavidrein/gpqa
   - Click "Agree and access repository" to accept the terms
   - You must agree NOT to reveal examples from this dataset in plain text or images online

2. **Authenticate with HuggingFace:**
   ```bash
   huggingface-cli login
   ```
   Or set your token as an environment variable:
   ```bash
   export HF_TOKEN=your_token_here
   ```

3. **Run the download script:**
   ```bash
   cd evaluation
   python3 download_gpqa.py
   ```

The script will download the dataset and convert it to JSONL format compatible with the evaluation framework.

## Dataset Details

- **Size**: 448 multiple-choice questions
- **Format**: Multiple choice (A, B, C, D, or E)
- **Domains**: Biology, Physics, Chemistry
- **Difficulty**: Very hard - experts get 65% accuracy, non-experts get 34% accuracy

## Dataset Structure

After downloading, the dataset will be in:
- `evaluation/data/gpqa/test.jsonl` (or other split names)

Each example contains:
- `idx`: Index number
- `question`: The question text
- `target`: The correct answer (A, B, C, D, or E)
- `gt`: Ground truth (same as target)
- `gt_cot`: None (no chain-of-thought provided)


