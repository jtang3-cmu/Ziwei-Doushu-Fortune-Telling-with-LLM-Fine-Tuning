# python inference.py --data_file data/test.jsonl --output_file outputs/test_predictions.jsonl --max_new_tokens 2048
import argparse
import json
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, PeftModel
from datasets import load_dataset

from helper import build_prompt 
from tqdm import tqdm


BASE_MODEL_NAME = "Qwen/Qwen3-4b"
LORA_PATH = "./qwen_lora_output"


# -------------------- 1. load model & tokenizer -------------------- #
def load_model_and_tokenizer(
    base_model_name: str = BASE_MODEL_NAME,
    lora_path: str = LORA_PATH,
    use_lora: bool = True,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # loading LoRA weights if --use_lora
    if use_lora:
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
        print("[INFO] Loading base model without LoRA...")
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


# -------------------- 2. read test data (.jsonl) -------------------- #
def load_jsonl_dataset(path: str):
    """
    read test data (same format as training data)
    example:
      {"出生資料": {...}, "命盤": {...}, "解讀": "..."}
      {"birth_info": {...}, "chart": {...}, "interpretation": "..."}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # datasets would automatically process json / jsonl
    ds = load_dataset("json", data_files=path, split="train")
    print(f"[INFO] Loaded dataset from {path}, size = {len(ds)}")
    return ds


# -------------------- 3. extract birth info -------------------- #
def extract_birth_info(sample):
    """
    extract birth info
    """
    if "出生資料" in sample:
        return sample["出生資料"]

    raise KeyError("can not find 'birth info' key, please modify your extract_birth_info()")


# -------------------- 4. inference one sample: build_prompt + generate -------------------- #
def run_inference_one(
    model,
    tokenizer,
    birth_info: dict,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    # Use Input prompt: system + user + assistant + <think>...</think>
    prompt = build_prompt(birth_info)
    print(prompt)

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

    # crop prompt, only keep the model output: completion
    if full_text.startswith(prompt):
        completion = full_text[len(prompt):]
    else:
        idx = full_text.rfind(prompt)
        if idx != -1:
            completion = full_text[idx + len(prompt):]
        else:
            completion = full_text

    completion = completion.strip()

    # remove the end token <|im_end|> 
    end_token = "<|im_end|>"
    if completion.endswith(end_token):
        completion = completion[: -len(end_token)].rstrip()

    # parse the model output into jsonl or json file
    parsed = None
    try:
        parsed = json.loads(completion)
    except Exception:
        pass

    return completion, parsed, full_text


# -------------------- 5. main function: run inference -------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        required=True,
        help="the jsonl file path to do do inference. should be same as training data format",
    )
    parser.add_argument(
        "--output_file",
        default="predictions.jsonl",
        help="output file path, would be jsonl file",
    )
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="token length of output length")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--use_lora", action="store_true", help="use lora weights or not")

    args = parser.parse_args()

    # 1) load model
    model, tokenizer = load_model_and_tokenizer(use_lora=args.use_lora)

    # 2) read jsonl
    dataset = load_jsonl_dataset(args.data_file)

    # 3) run inference, and write into output_file each run
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    for i, sample in enumerate(tqdm(dataset)):
        try:
            birth_info = extract_birth_info(sample)
        except Exception as e:
            print(f"[WARN] Skipped {i} th data (can not find birth info): {e}")
            continue

        completion, parsed, full_text = run_inference_one(
            model,
            tokenizer,
            birth_info,
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

    print(f"[DONE] saved inference as {args.output_file}")


if __name__ == "__main__":
    main()