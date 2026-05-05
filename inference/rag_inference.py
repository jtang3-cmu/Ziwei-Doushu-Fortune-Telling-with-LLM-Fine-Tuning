# python rag_inference.py --data_file data/test.jsonl --knowledge_file rag/rag.jsonl --output_file outputs/prediction_sft_rag.jsonl --max_new_tokens 2048 --rag_log_file outputs/sft_rag_log.jsonl

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from helper import build_prompt, load_system_intro 


BASE_MODEL_NAME = "Qwen/Qwen3-4b"
LORA_PATH = "./qwen_lora_output"


# -------------------- 0. implementation of RAG -------------------- #
@dataclass
class RagDocument:
    text: str
    tokens: Counter


def simple_tokenize(text: str) -> Counter:
    """
    split characters and ignore space
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    tokens = [ch for ch in text if not ch.isspace()]
    return Counter(tokens)


def load_knowledge_corpus(path: str) -> List[RagDocument]:
    """
    read RAG knowledge base
    - if it is .jsonl file, use "text" or "content (內容)" as key
    - otherwise, regard it as text format
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    docs: List[RagDocument] = []

    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = (
                    obj.get("text")
                    or obj.get("content")
                    or obj.get("內容")
                    or json.dumps(obj, ensure_ascii=False)
                )
                docs.append(RagDocument(text=text, tokens=simple_tokenize(text)))
    else:
        # regard as text. see \n as another paragraph
        with open(path, "r", encoding="utf-8") as f:
            full = f.read()
        # split paragraph
        chunks = [c.strip() for c in full.split("\n\n") if c.strip()]
        for c in chunks:
            docs.append(RagDocument(text=c, tokens=simple_tokenize(c)))

    print(f"[RAG] Loaded corpus from {path}, num_docs = {len(docs)}")
    return docs


def build_retrieval_query(birth_info: Dict[str, Any]) -> str:
    """
    Accroding to birth info to form a query for retriver.
    """
    parts = []
    for k, v in birth_info.items():
        parts.append(f"{k}={v}")
    return " ".join(parts)

def retrieve_topk(
    query: str,
    corpus: List[RagDocument],
    top_k: int = 3,
) -> List[Tuple[RagDocument, float]]:
    """
    Simple retrieval: use token overlap score
    """
    if not corpus:
        return []

    q_tokens = simple_tokenize(query)
    if not q_tokens:
        return []

    results: List[Tuple[RagDocument, float]] = []
    for doc in corpus:
        score = 0.0
        for t, q_cnt in q_tokens.items():
            d_cnt = doc.tokens.get(t, 0)
            if d_cnt > 0:
                score += min(q_cnt, d_cnt)
        # not filtering score==0
        results.append((doc, score))

    # sort by score in decending order
    results.sort(key=lambda x: x[1], reverse=True)
    # return top_k
    return results[:top_k]


def build_rag_context(query: str, corpus: List[RagDocument], top_k: int = 3) -> Tuple[str, float]:
    """
    group top-k retrieved results into a context to model, and also return the max_score
    """
    hits = retrieve_topk(query, corpus, top_k=top_k)
    if not hits:
        return "", 0.0

    max_score = hits[0][1]

    blocks = []
    for idx, (doc, score) in enumerate(hits, start=1):
        blocks.append(f"[DOC {idx} | score={score:.2f}]\n{doc.text}")

    rag_context = "\n\n".join(blocks)
    return rag_context, max_score


# -------------------- 1. load model & tokenizer -------------------- #
def load_model_and_tokenizer(
    base_model_name: str = BASE_MODEL_NAME,
    lora_path: str = LORA_PATH,
    use_lora: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    if use_lora:
        # load lora weights
        try:
            print("[INFO] Loading LoRA model with AutoPeftModelForCausalLM...")
            model = AutoPeftModelForCausalLM.from_pretrained(
                lora_path,
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as e:
            print(f"[WARN] AutoPeftModelForCausalLM failed: {e}")
            print("[INFO] Fallback: load base model + PeftModel...")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.bfloat16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(base_model, lora_path)
    else:
        # use raw model: Qwen3-4B
        print("[INFO] Loading RAW base model (no LoRA):", base_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    # pad_token setting
    pad_id = None
    if getattr(model, "generation_config", None) and model.generation_config.pad_token_id is not None:
        pad_id = model.generation_config.pad_token_id
        print("[PAD] use model.generation_config.pad_token_id =", pad_id)
    elif tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
        print("[PAD] use tokenizer.pad_token_id =", pad_id)
    else:
        pad_id = tokenizer.eos_token_id
        print("[PAD] use tokenizer.eos_token_id =", pad_id)

    tokenizer.pad_token_id = pad_id
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)

    model.config.pad_token_id = pad_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = pad_id

    model.eval()
    return model, tokenizer


# -------------------- 2. read jsonl dataset -------------------- #
def load_jsonl_dataset(path: str):
    """
    read test data (same format as training data)
    example:
      {"出生資料": {...}, "命盤": {...}, "解讀": "..."}
      {"birth_info": {...}, "chart": {...}, "interpretation": "..."}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ds = load_dataset("json", data_files=path, split="train")
    print(f"[INFO] Loaded dataset from {path}, size = {len(ds)}")
    return ds


# -------------------- 3. extract birth_info from sample -------------------- #
def extract_birth_info(sample):
    if "出生資料" in sample:
        return sample["出生資料"]
    raise KeyError("can not find 'birth info' key, please modify your extract_birth_info()")


# -------------------- 3.5 RAG version prompt builder -------------------- #
def build_rag_prompt(birth_info: dict, rag_context: str) -> str:
    """
    structure:
    1) system: origin SYSTEM_INTRO
    2) reference: RAG context
    3) user: birth info
    4) assistant: <think>...</think>
    """
    birth_date = birth_info.get("生日")
    gender = birth_info.get("性別")
    time_index = birth_info.get("時辰_index")

    SYSTEM_INTRO = load_system_intro()

    # system
    system_intro_block = (
        "<|im_start|>system\n"
        f"{SYSTEM_INTRO}\n"
        "<|im_end|>\n"
    )

    # rag content
    if rag_context:
        system_rag_block = (
            "<|im_start|>system\n"
            "以下是本次命盤解讀可參考的排盤規則與步驟（由系統檢索而來）。\n"
            "請你在解讀時優先遵守上一段 system 的核心原則，並在不衝突的情況下參考這些內容：\n"
            "<context>\n"
            f"{rag_context}\n"
            "</context>\n"
            "<|im_end|>\n"
        )
    else:
        system_rag_block = ""

    # user
    user_block = (
        "<|im_start|>user\n"
        f"出生: {birth_date}\n"
        f"性別: {gender}\n"
        f"時辰_index: {time_index}\n"
        "<|im_end|>\n"
    )

    # assistant prefix
    assistant_prefix = "<|im_start|>assistant\n<think>\n\n</think>\n"

    return system_intro_block + system_rag_block + user_block + assistant_prefix

# -------------------- 4. inference one with RAG -------------------- #
def run_inference_one_rag(
    model,
    tokenizer,
    birth_info: dict,
    rag_context: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """
    almost same as the original run_inference_one.
    The difference is we add rag_context in build_rag_prompt
    """
    if rag_context:
        # enable rag
        prompt = build_rag_prompt(birth_info, rag_context)
    else:
        # without rag
        prompt = build_prompt(birth_info)
    # print("-----------overall------------")
    # print(prompt)
    # print("-----------end------------")
    # tokenization
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    # generating
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)

    # crop prompt, completion: pure model output
    if full_text.startswith(prompt):
        completion = full_text[len(prompt):]
    else:
        idx = full_text.rfind(prompt)
        if idx != -1:
            completion = full_text[idx + len(prompt):]
        else:
            completion = full_text

    completion = completion.strip()

    # remove end token <|im_end|> 
    end_token = "<|im_end|>"
    if completion.endswith(end_token):
        completion = completion[: -len(end_token)].rstrip()

    # parse into json file
    parsed = None
    try:
        parsed = json.loads(completion)
    except Exception:
        pass

    return completion, parsed, full_text


# -------------------- 5. main function: read jsonl and run RAG inference -------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        required=True,
        help="input file path (test.jsonl)",
    )
    parser.add_argument(
        "--knowledge_file",
        required=True,
        help="RAG .jsonl file path",
    )
    parser.add_argument(
        "--output_file",
        default="predictions_rag.jsonl",
        help="output file (.jsonl）",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="token length of model output")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=3, help="choose top-k retrieved results")

    parser.add_argument(
        "--use_raw_model",
        action="store_true",
        help="use raw model: Qwen3-4B",
    )

    parser.add_argument(
        "--base_model_name",
        type=str,
        default=BASE_MODEL_NAME,
        help="HF base model name (default: Qwen/Qwen3-4b)",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=LORA_PATH,
        help="LoRA checkpoint",
    )

    parser.add_argument(
        "--rag_log_file",
        default=None,
        help="rag log file path, record using rag or not",
    )


    args = parser.parse_args()

    # 1) load model
    model, tokenizer = load_model_and_tokenizer(
        base_model_name=args.base_model_name,
        lora_path=args.lora_path,
        use_lora=not args.use_raw_model,
    )

    # 2) read jsonl data
    dataset = load_jsonl_dataset(args.data_file)

    # 3) load RAG knowledge base
    corpus = load_knowledge_corpus(args.knowledge_file)

    # 4) output file path
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    fout = open(args.output_file, "w", encoding="utf-8")

    # if enable rag_log_file, build its path
    rag_log_fout = None
    if args.rag_log_file is not None:
        os.makedirs(os.path.dirname(args.rag_log_file) or ".", exist_ok=True)
        rag_log_fout = open(args.rag_log_file, "w", encoding="utf-8")

    # 5) run RAG inference, and write into output dir
    for i, sample in enumerate(tqdm(dataset)):
        try:
            birth_info = extract_birth_info(sample)
        except Exception as e:
            print(f"[WARN] Skipped {i} th data, can't find birth_info key: {e}")
            continue

        query = build_retrieval_query(birth_info)
        rag_context, max_score = build_rag_context(query, corpus, top_k=args.top_k)

        # Bool: use RAG or not
        used_rag = bool(rag_context.strip())

        # log into rag_log_file
        if rag_log_fout is not None:
            log_obj = {
            "index": i,
            "used_rag": used_rag,
            "query": query,
            "max_score": max_score,
        }
            # if there is sample numbers, we can also log them
            if isinstance(sample, dict) and "id" in sample:
                log_obj["id"] = sample["id"]
            rag_log_fout.write(json.dumps(log_obj, ensure_ascii=False) + "\n")

        completion, parsed, full_text = run_inference_one_rag(
            model,
            tokenizer,
            birth_info,
            rag_context,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # save model output as string output
        # with open(args.output_file, "a", encoding="utf-8") as fout:
        #     fout.write(completion + "\n")
        # ensure to be saved as valid string 
        output_record = {
            # if want to keep input birth info, you can add birth_info
            # "input": birth_info, 
            "prediction": completion  # save model output as string
        }

        with open(args.output_file, "a", encoding="utf-8") as fout:
            # force to be one line
            fout.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    fout.close()
    if rag_log_fout is not None:
        rag_log_fout.close()

    print(f"[DONE] saved RAG inference into {args.output_file}")
    if args.rag_log_file is not None:
        print(f"[DONE] save RAG log file into {args.rag_log_file}")


if __name__ == "__main__":
    main()
