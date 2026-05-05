import json

def load_system_intro():
    """
    Read system intro from sft_prompt.txt in the current directory.
    SFT and evaluation share the same prefix description.
    """
    path = "data/sft_prompt.txt"
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        # Append <INPUT> later for readability, add two more newlines
        SYSTEM_INTRO = text + "\n\n"
        # print(f"[System Intro] Loaded from {path}, length={len(SYSTEM_INTRO)} chars.")
    except FileNotFoundError:
        SYSTEM_INTRO = ""
        print(f"[System Intro] File not found at {path}, proceed without system intro.")
    except Exception as e:
        SYSTEM_INTRO = ""
        print(f"[System Intro] Failed to load from {path}: {e}")
    return SYSTEM_INTRO

def build_prompt(birth_info):
    """
    Return the prefix up to </think>:
    system + user + <|im_start|>assistant + <think>...</think>
    """
    birth_date = birth_info.get("生日")
    gender = birth_info.get("性別")
    time_index = birth_info.get("時辰_index")

    SYSTEM_INTRO = load_system_intro()

    system_block = (
        "<|im_start|>system\n"
        f"{SYSTEM_INTRO}\n"
        "<|im_end|>\n"
    )

    user_block = (
        "<|im_start|>user\n"
        f"出生: {birth_date}\n"
        f"性別: {gender}\n"
        f"時辰_index: {time_index}\n"
        "<|im_end|>\n"
    )

    # This prefix ends at </think>; the actual answer to be learned follows after it
    assistant_prefix = "<|im_start|>assistant\n<think>\n\n</think>\n"

    return system_block + user_block + assistant_prefix


def build_training_text(sample):
    """
    sample is the dict parsed from one jsonl line:
    {
      "出生資料": {...},
      "命盤": {...},
      "解讀": "..."
    }
    Return a dict containing "text", used by SFTTrainer.
    """
    birth_info = sample["出生資料"]
    answer = build_answer(sample)

    prefix = build_prompt(birth_info)  # Prefix up to </think>
    full_text = prefix + answer        # Final training text = prefix + answer

    # If you want to also close assistant, add <|im_end|>
    # full_text = prefix + answer + "<|im_end|>\n"

    return {"text": full_text}

def build_answer(sample) -> str:

    obj = {
        "命盤": sample["命盤"],
        "解讀": sample["解讀"],
    }
    return json.dumps(obj, ensure_ascii=False)


from dataclasses import dataclass
from typing import List, Dict, Any
import torch

@dataclass
class CompletionOnlyCollator:
    tokenizer: Any
    response_template: str
    max_length: int = 2048

    def __post_init__(self):
        # Convert template into token id sequence for later boundary detection
        self.template_ids = self.tokenizer(
            self.response_template,
            add_special_tokens=False,
        )["input_ids"]

    def _find_template_end(self, ids: List[int]) -> int:
        """
        Find the end index (exclusive) of template_ids inside ids.
        If not found, return 0 (meaning no masking of prefix).
        """
        t = self.template_ids
        t_len = len(t)
        if t_len == 0:
            return 0

        for i in range(len(ids) - t_len + 1):
            if ids[i : i + t_len] == t:
                return i + t_len  # Return the index marking the end of template
        return 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        features may have two forms:
        1) Raw: {"text": "..."} (we handle tokenization)
        2) Pre-processed: {"input_ids": [...], "attention_mask": [...]} (SFTTrainer already processed)
        """
        if "text" in features[0]:
            # Case 1: dataset is raw text; perform tokenize + padding here
            texts = [f["text"] for f in features]
            batch = self.tokenizer(
                texts,
                padding="max_length",
                # padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        else:
            # Case 2: SFTTrainer already converted text into input_ids, etc.
            # Here we only handle padding + label masking
            batch = self.tokenizer.pad(
                features,
                padding=True,
                return_tensors="pt",
            )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Mask labels for each sample
        for i in range(labels.size(0)):
            ids = labels[i].tolist()
            end = self._find_template_end(ids)
            if end > 0:
                labels[i, :end] = -100  # Tokens before template end do not contribute to loss
        
        pad_token_id = self.tokenizer.pad_token_id
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        elif pad_token_id is not None:
            labels[input_ids == pad_token_id] = -100

        batch["labels"] = labels
        return batch
