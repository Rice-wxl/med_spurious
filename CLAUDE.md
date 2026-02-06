# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a medical AI evaluation and spurious correlation analysis project for benchmarking LLMs/LMMs on medical knowledge tasks. It evaluates models on medical exam datasets (MedQA, MedMCQA, MMLU-Medical, MedXpertQA) using various prompting strategies.

## Key Commands

### Installation
```bash
cd data/MedXpertQA/eval
pip install -r requirements.txt
```

### Running Evaluations
```bash
# General evaluation script
bash scripts/run.sh [models] [datasets] [tasks] [output_dir] [method] [prompting_type] [temperature]

# Model-specific scripts
bash scripts/run_o1.sh   # OpenAI o1 models
bash scripts/run_o3.sh   # OpenAI o3 models
bash scripts/run_qvq.sh  # QVQ models
bash scripts/run_r1.sh   # R1 models
```

## Architecture

```
main.py                    # Orchestrator: runs zero_shot_ao(), zero_shot_cot(), few_shot()
├── setup.py               # Configuration loading
├── utils.py               # Preprocessing, metrics, answer parsing
├── model/
│   ├── base_agent.py      # Abstract LLMAgent base class
│   └── api_agent.py       # API implementations (OpenAI, Claude, Gemini, DeepSeek, etc.)
├── config/
│   ├── dataset_info.json  # Dataset configurations
│   ├── model_info.json    # Supported models list
│   └── prompt_templates.py # Prompt formatting for different strategies
└── data/                  # Test datasets (JSONL format)
```

### Data Format
JSONL files with structure:
```json
{
  "question": "...",
  "answer": "...",
  "answer_idx": "C",
  "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "meta_info": "step1|step2&3",
  "images": []  // optional for multimodal
}
```

## Supported Models

20+ models via API: GPT-4o, o1, o3-mini, Claude 3 (Sonnet/Haiku), Gemini (1.5 Pro, 2.0 Flash/Pro), DeepSeek (chat/reasoner), QWQ, QVQ, Llama Vision.

## Prompting Strategies

- **zero_shot_ao**: Zero-shot answer-only
- **zero_shot_cot**: Zero-shot chain-of-thought reasoning
- **few_shot**: Few-shot with demonstrations

## Notebooks

`notebooks/medQA+X.ipynb` - Analysis notebook for model inference, CoT parsing, and results analysis using HuggingFace models (e.g., OLMo-2-0325-32B-Instruct).

## Planned Features

`spurious_detect/` and `spurious_inject/` directories are placeholders for spurious correlation detection and injection modules.
