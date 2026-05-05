from helper import build_training_text
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
from datasets import load_dataset
# from trl.trainer.utils import DataCollatorForCompletionOnlyLM
from helper import CompletionOnlyCollator

# 1. Set model name (using Qwen3-4B)
model_name = "Qwen/Qwen3-4b"

# 2. Prepare a minimal test dataset (Format: Prompt -> Response)
# Replace with reading your own JSON/CSV file in real use

raw_dataset = load_dataset(
    "json",
    data_files="data/train.jsonl",
    split="train"
)

# Map to {"text": "..."}
dataset = raw_dataset.map(
    build_training_text,
    remove_columns=raw_dataset.column_names,
)

print("dataset columns:", dataset.column_names)

# for i in range(2):  # Only check first two samples
#     print("\n================ SAMPLE", i, "================")
#     print(dataset[i]["text"])
#     print("================ END SAMPLE ================\n")

# 3. Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)



# tokenizer.pad_token = tokenizer.eos_token # Qwen requires this setting

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16, # Use torch.float16 if your GPU is older
    device_map="auto"
)

# TMP
# Prefer using pad_token_id already defined in model
pad_id = None

if model.generation_config.pad_token_id is not None:
    print(f"PAD using model.generation_config")
    pad_id = model.generation_config.pad_token_id
elif tokenizer.pad_token_id is not None:
    print(f"PAD using tokenizer")
    pad_id = tokenizer.pad_token_id
else:
    # If nothing is found, fallback to eos as pad
    print("PAD using tokenizer.eos_token_id")
    pad_id = tokenizer.eos_token_id

# Set tokenizer pad_token_id
tokenizer.pad_token_id = pad_id
# Optionally set pad_token to help debugging
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_id)

# Sync back to model.config / generation_config
model.config.pad_token_id = pad_id
model.generation_config.pad_token_id = pad_id
# TMP

# 4. Configure LoRA parameters (the key part)
peft_config = LoraConfig(
    r=64,                       # LoRA rank; larger means more parameters
    lora_alpha=32,              # Scaling factor; usually twice of r
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"] # Specify the layers to fine-tune
)

# 5. Configure training arguments
training_args = SFTConfig(
    output_dir="./qwen_lora_output",   # Model save directory
    # max_steps=3,                     # Recommended to remove; use num_train_epochs
    num_train_epochs=5,                # Train for 5 epochs
    per_device_train_batch_size=2,
    gradient_accumulation_steps=5,     # Accumulate gradients 5 times
    learning_rate=2e-4,
    logging_steps=1,
    dataset_text_field="text",
    packing=False,
)

# 6. Start training

response_template = "<|im_start|>assistant\n<think>\n\n</think>\n"

data_collator = CompletionOnlyCollator(
    tokenizer=tokenizer,
    response_template=response_template,
    max_length=4096,  # Depends on your model context limit
)

# ! debug
debug = True
if debug is True:
    # Take only one sample to test
    sample = dataset[0]           # {'text': '......'}
    batch = data_collator([sample])  # Simulate a batch with batch_size=1

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)

    # Find the first token that is not -100 (should be first token after template)
    first_non_mask = (labels != -100).nonzero(as_tuple=True)[0][0].item()
    print("first non-masked index:", first_non_mask)

    print("\n=== Tokens around boundary ===")
    start = max(0, first_non_mask - 20)
    end = first_non_mask + 20

    for idx in range(start, end):
        tid = input_ids[idx].item()
        lid = labels[idx].item()
        tok = tokenizer.decode([tid])
        flag = "L" if lid == -100 else "T"  # L=masked(label -100), T=trained
        print(f"{idx:4d} | {flag} | {repr(tok)} | label={lid}")

    print("\n=== Decoded full text ===")
    print(tokenizer.decode(input_ids, skip_special_tokens=False))

    print("\n=== Last 40 tokens (including padding) ===")
    end = len(input_ids)
    start = max(0, end - 40)

    # ===== Debug: Check whether padding is masked properly =====
    sample = dataset[0]                    # Take first sample again
    batch = data_collator([sample])        # Simulate batch (size=1)

    input_ids = batch["input_ids"][0]
    labels = batch["labels"][0]
    attn = batch["attention_mask"][0]

    print("input_ids shape:", input_ids.shape)
    print("labels shape:", labels.shape)

    # Find indices where attention_mask = 0 (i.e., padding positions)
    pad_positions = (attn == 0).nonzero(as_tuple=True)[0].tolist()
    print("pad positions:", pad_positions)

    if pad_positions:
        print("\n=== Check each padding token label ===")
        for idx in pad_positions[:20]:  # Print first 20 only
            tid = input_ids[idx].item()
            lid = labels[idx].item()
            tok = tokenizer.decode([tid])
            print(f"idx={idx:4d} | token={repr(tok)} | label={lid}")
    else:
        print("This sample has no padding (all real tokens).")

    # Quick sanity check: are all padding labels = -100?
    if pad_positions:
        ok = (labels[attn == 0] == -100).all().item()
        print("\nAre all padding labels -100? ", ok)

    for idx in range(start, end):
        tid = input_ids[idx].item()
        lid = labels[idx].item()
        am = attn[idx].item()
        tok = tokenizer.decode([tid])
        flag = "PAD" if am == 0 else "TOK"
        print(f"{idx:4d} | {flag} | token={repr(tok)} | label={lid}")
    
    print("\n=== Debug pad alignment ===")
    print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
    print("model.config.pad_token_id:", model.config.pad_token_id)
    print("model.generation_config.pad_token_id:", model.generation_config.pad_token_id)

    print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
    print("model.config.eos_token_id:", model.config.eos_token_id)
    print("model.generation_config.eos_token_id:", model.generation_config.eos_token_id)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_args,
    data_collator=data_collator,
)

print("Start training...")
trainer.train()

# 7. Save model (save only LoRA adapter, small file size)
trainer.save_model("./qwen_lora_output")
print("Model saved to ./qwen_lora_output")
