from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name = "unsloth/Meta-Llama-3.1-8B",
    model_name = "unsloth/Qwen2.5-7B",
    max_seq_length = max_seq_length,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                    "gate_proj", "down_proj", "up_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state=42,
    use_rslora = False,
    loftq_config = None
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Company database: {}

### Input:
SQL Prompt: {}

### Response:
SQL: {}

Explanation: {}
"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    company_databases = examples["sql_context"]
    prompts = examples["sql_prompt"]
    sqls = examples["sql"]
    explanations = examples["sql_explanation"]
    texts = []
    for company_database, prompt, sql, explanation in zip(company_databases, prompts, sqls, explanations):
        # add EOD token, otherwise generation loop stuck
        text = alpaca_prompt.format(company_database, prompt, sql, explanation) + EOS_TOKEN
        texts.append(text)
    return { "text": texts }
pass

from datasets import load_dataset   
dataset = load_dataset("gretelai/synthetic_text_to_sql", split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

dataset['text']

from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from trl import SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, 
        warmup_steps=5, 
        max_steps = 60,
        learning_rate=2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type="linear",
        seed=42,
        output_dir="outputs"
    )
)

trainer_stats = trainer.train()

model.save_pretrained_gguf("model", tokenizer, quantization_method="f16")