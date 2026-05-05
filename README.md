# Ziwei-Doushu-Fortune-Telling-with-LLM-Fine-Tuning
Supervised fine-tuning of Qwen3-4B on Chinese astrological chart generation and career interpretation, with optional RAG-enhanced inference.
A course project for 10-423/623 Generative AI (CMU, Dec 2025).
Authors: Johnathan Tang · Tsung-Yeh Hsieh · Chi-Yeh Chen

Overview
This project explores whether a small open-source LLM can learn to:

Reconstruct a complete Ziwei Doushu (紫微斗數) natal chart in a fixed JSON schema directly from birth date, time, and gender.
Generate a natural-language career interpretation grounded in the reconstructed chart.

We fine-tune Qwen3-4B with LoRA on ~3,000 synthetically generated chart–interpretation pairs, then optionally augment inference with a classical-text RAG corpus. Our fine-tuned model matches or outperforms Gemini-2.5-Flash on structural chart metrics.

Repository Structure
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

Setup
Requirements
bashpip install torch transformers peft trl datasets sentence-transformers faiss-cpu tqdm

Training was performed on an AWS g6.2xlarge (NVIDIA L4 GPU, 24 GiB VRAM). A GPU with ≥16 GiB VRAM is recommended for fine-tuning.

Data Format
Each .jsonl record contains three fields:
json{
  "出生資料": { "生日": "2001-01-02", "性別": "女", "時辰_index": 7 },
  "命盤": { "命盤": { "命宮": { ... }, ... } },
  "解讀": "（一）事業運 ..."
}

Training
Fine-tune Qwen3-4B with LoRA:
bashpython ft.py
Key hyperparameters (see ft.py):
ParameterValueBase modelQwen/Qwen3-4bLoRA rank (r)64LoRA alpha32Epochs5Batch size2Gradient accumulation5Learning rate2e-4Target modulesq_proj, v_proj
The LoRA adapter is saved to ./qwen_lora_output.

Inference
Vanilla (SFT only)
bashpython inference.py \
  --data_file data/test.jsonl \
  --output_file outputs/test_predictions.jsonl \
  --use_lora \
  --max_new_tokens 2048
To run the raw base model (no LoRA):
bashpython inference.py \
  --data_file data/test.jsonl \
  --output_file outputs/test_predictions.jsonl \
  --max_new_tokens 2048
RAG-Augmented Inference
Step 1 — Build the corpus (only needed once, from your own source text):
bashpython build_rag.py
Step 2 — Build the FAISS index:
bashpython rag_build_index.py
Step 3 — Run RAG inference:
bashpython rag_inference.py \
  --data_file data/test.jsonl \
  --knowledge_file rag/rag_corpus/rag.jsonl \
  --output_file outputs/prediction_sft_rag.jsonl \
  --rag_log_file outputs/sft_rag_log.jsonl \
  --max_new_tokens 2048 \
  --top_k 3
Use --use_raw_model to run RAG with the base Qwen3-4B instead of the fine-tuned adapter.

Results
Three models were evaluated against a Gemini-2.5-Flash commercial baseline:
ModelStructuralInterpretationText SimilarityGemini-2.5-Flash✦ baseline✦ baseline✦ baselineQwen3-4B (raw)lowerlowerlowerZiwei (SFT)≈ Gemini≈ GeminicomparableZiwei-RAGslightly lower≈ Geminihighest
Key findings:

Fine-tuned Qwen3-4B matches Gemini on chart reconstruction (Chart Accuracy, Star Jaccard).
RAG improves text alignment (Cosine Similarity, BERTScore) but slightly reduces structural precision.
Da-xian range prediction (Range IoU, Gan-Zhi Accuracy) remains challenging for all models.


Project Notes

Prompt template is defined in data/sft_prompt.txt and shared between training and inference via helper.py.
Custom collator (CompletionOnlyCollator in helper.py) masks all prompt tokens so the model only trains on the JSON answer portion.
RAG retriever (rag_retriever.py) uses sentence-transformers + FAISS for semantic retrieval; rag_inference.py uses a lightweight character-overlap fallback retriever for speed.
Ziwei charts were generated using the open-source iztro engine. Interpretations were labeled by Qwen3-8B with structured prompting.


Citation
Tang, J., Hsieh, T.-Y., & Chen, C.-Y. (2025).
Supervised Fine-tuning on Chinese Fortune-telling.
10-423/623 Generative AI Course Project, Carnegie Mellon University.
