# Ziwei Doushu Fortune-Telling with LLM Fine-Tuning
 
> Supervised fine-tuning of Qwen3-4B on Chinese astrological chart generation and career interpretation, with optional RAG-enhanced inference.
 
A course project for **10-423/623 Generative AI (CMU, Dec 2025)**.  
Authors: Johnathan Tang · Tsung-Yeh Hsieh · Chi-Yeh Chen
 
---
 
## Overview
 
This project explores whether a small open-source LLM can learn to:
 
1. **Reconstruct** a complete Ziwei Doushu (紫微斗數) natal chart in a fixed JSON schema directly from birth date, time, and gender.
2. **Generate** a natural-language career interpretation grounded in the reconstructed chart.
We fine-tune **Qwen3-4B** with LoRA on ~3,000 synthetically generated chart–interpretation pairs, then optionally augment inference with a classical-text RAG corpus. Our fine-tuned model matches or outperforms Gemini-2.5-Flash on structural chart metrics.
 
---
 
## Repository Structure
 
```
.
├── data/
│   ├── train.jsonl          # Training split
│   ├── val.jsonl            # Validation split
│   ├── test.jsonl           # Test split
│   └── sft_prompt.txt       # System prompt used in SFT and inference
│
├── rag/
│   ├── rag_corpus/
│   │   └── rag.jsonl        # Chunked RAG knowledge corpus
│   └── rag_index/
│       ├── rag_index.faiss  # FAISS vector index
│       └── rag_index_meta.json
│
├── helper.py                # Prompt builders and custom data collator
├── ft.py                    # LoRA supervised fine-tuning
├── inference.py             # Vanilla (SFT-only) inference
├── build_rag.py             # Build RAG corpus from raw text
├── rag_build_index.py       # Embed corpus and build FAISS index
├── rag_retriever.py         # FAISS-based retriever class
└── rag_inference.py         # RAG-augmented inference
```
 
---
 
## Setup
 
### Requirements
 
```bash
pip install torch transformers peft trl datasets sentence-transformers faiss-cpu tqdm
```
 
> Training was performed on an AWS g6.2xlarge (NVIDIA L4 GPU, 24 GiB VRAM). A GPU with ≥16 GiB VRAM is recommended for fine-tuning.
 
### Data Format
 
Each `.jsonl` record contains three fields:
 
```json
{
  "出生資料": { "生日": "2001-01-02", "性別": "女", "時辰_index": 7 },
  "命盤": { "命盤": { "命宮": { ... }, ... } },
  "解讀": "（一）事業運 ..."
}
```
 
---
 
## Training
 
Fine-tune Qwen3-4B with LoRA:
 
```bash
python ft.py
```
 
Key hyperparameters (see `ft.py`):
 
| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen3-4b` |
| LoRA rank (r) | 64 |
| LoRA alpha | 32 |
| Epochs | 5 |
| Batch size | 2 |
| Gradient accumulation | 5 |
| Learning rate | 2e-4 |
| Target modules | `q_proj`, `v_proj` |
 
The LoRA adapter is saved to `./qwen_lora_output`.
 
---
 
## Inference
 
### Vanilla (SFT only)
 
```bash
python inference.py \
  --data_file data/test.jsonl \
  --output_file outputs/test_predictions.jsonl \
  --use_lora \
  --max_new_tokens 2048
```
 
To run the raw base model (no LoRA):
 
```bash
python inference.py \
  --data_file data/test.jsonl \
  --output_file outputs/test_predictions.jsonl \
  --max_new_tokens 2048
```
 
### RAG-Augmented Inference
 
**Step 1 — Build the corpus** (only needed once, from your own source text):
 
```bash
python build_rag.py
```
 
**Step 2 — Build the FAISS index:**
 
```bash
python rag_build_index.py
```
 
**Step 3 — Run RAG inference:**
 
```bash
python rag_inference.py \
  --data_file data/test.jsonl \
  --knowledge_file rag/rag_corpus/rag.jsonl \
  --output_file outputs/prediction_sft_rag.jsonl \
  --rag_log_file outputs/sft_rag_log.jsonl \
  --max_new_tokens 2048 \
  --top_k 3
```
 
Use `--use_raw_model` to run RAG with the base Qwen3-4B instead of the fine-tuned adapter.
 
---
 
## Results
 
Three models were evaluated against a Gemini-2.5-Flash commercial baseline:
 
| Model | Structural | Interpretation | Text Similarity |
|---|---|---|---|
| Gemini-2.5-Flash | ✦ baseline | ✦ baseline | ✦ baseline |
| Qwen3-4B (raw) | lower | lower | lower |
| **Ziwei (SFT)** | **≈ Gemini** | **≈ Gemini** | comparable |
| Ziwei-RAG | slightly lower | ≈ Gemini | **highest** |
 
Key findings:
- Fine-tuned Qwen3-4B matches Gemini on chart reconstruction (Chart Accuracy, Star Jaccard).
- RAG improves text alignment (Cosine Similarity, BERTScore) but slightly reduces structural precision.
- Da-xian range prediction (Range IoU, Gan-Zhi Accuracy) remains challenging for all models.
---
 
## Project Notes
 
- **Prompt template** is defined in `data/sft_prompt.txt` and shared between training and inference via `helper.py`.
- **Custom collator** (`CompletionOnlyCollator` in `helper.py`) masks all prompt tokens so the model only trains on the JSON answer portion.
- **RAG retriever** (`rag_retriever.py`) uses sentence-transformers + FAISS for semantic retrieval; `rag_inference.py` uses a lightweight character-overlap fallback retriever for speed.
- Ziwei charts were generated using the open-source [iztro](https://github.com/SylarLong/iztro) engine. Interpretations were labeled by Qwen3-8B with structured prompting.
---
 
## Citation
 
```
Tang, J., Hsieh, T.-Y., & Chen, C.-Y. (2025).
Supervised Fine-tuning on Chinese Fortune-telling.
10-423/623 Generative AI Course Project, Carnegie Mellon University.
