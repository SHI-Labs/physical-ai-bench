# PAI-Bench -- Reason

[![Python Version](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-orange)](https://huggingface.co/datasets/shi-labs/physical-ai-bench-reason)

---

## Recommended: Using lmms-eval

We recommend using the benchmark through [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for standardized evaluation.

## Setup

Install dependencies using uv:

```bash
uv pip install -e .
```

## Usage

Run inference on a Hugging Face dataset using tensor parallelism:

```bash
uv run python run.py --dataset_name shi-labs/pai-reason --local_dir ./dataset --output_dir ./output --tensor_parallel_size 4
```

The script will show progress bars for both inference processing and evaluation.

### Arguments

- `--dataset_name`: Hugging Face dataset name (required)
- `--local_dir`: Local directory to store dataset (required)
- `--output_dir`: Output directory for results (default: ./output)
- `--tensor_parallel_size`: Number of tensor parallel processes for model (default: 8)
- `--eval_only`: Only run evaluation on existing results (optional)

## Components

- `run.py` - Main inference and evaluation script using tensor parallelism
- `models.py` - Model implementations including Cosmos-Reason1-7B with tensor parallel support
- `pyproject.toml` - Project configuration and dependencies

## Output

The system generates:

- Results: `{output_dir}/results.json`
- Evaluation results printed to console with accuracy metrics by category/subcategory
