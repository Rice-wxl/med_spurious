# Baseline Inference Results

**Models:** `meta-llama/Llama-3.1-8B-Instruct` · `allenai/OLMo-3-7B-Instruct` · `Qwen/Qwen3-8B`
**Evaluation:** 100 samples total (25 per dataset), zero-shot MCQ

---

## Overall Accuracy

| Model | Correct | Total | Accuracy | Unparseable |
|-------|---------|-------|----------|-------------|
| **LLaMA-3.1-8B** | 51 | 100 | **51.0%** | 0 |
| **OLMo-3-7B** | 38 | 100 | **38.0%** | 0 |
| **Qwen3-8B** | 55 | 100 | **55.0%** | 0 |

---

## Per-Dataset Accuracy

| Dataset | LLaMA-3.1-8B | OLMo-3-7B | Qwen3-8B |
|---------|-------------|-----------|----------|
| US QBank | 14/25 (56.0%) | 7/25 (28.0%) | 15/25 (60.0%) |
| MedBullets | 15/25 (60.0%) | 13/25 (52.0%) | 17/25 (68.0%) |
| MedXpertQA | 3/25 (12.0%) | 6/25 (24.0%) | 3/25 (12.0%) |
| MMLU Prof. Med. | 19/25 (76.0%) | 12/25 (48.0%) | 20/25 (80.0%) |
