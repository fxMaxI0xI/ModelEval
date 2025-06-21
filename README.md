# LLM Benchmark Evaluation

This repository provides a simple setup to evaluate HuggingFace models on benchmarks such as CodalBench.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the CodalBench evaluation for a given model:

```bash
python eval/eval_codalbench.py --model_name_or_path <model-id>
```

Add `--preference` to evaluate a single category or `--debug` to run a short debug evaluation.

## Requirements

The minimal dependencies are listed in `requirements.txt` and include:

- transformers
- datasets
- torch
- inspect_ai
- openai
