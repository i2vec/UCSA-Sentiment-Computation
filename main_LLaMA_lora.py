import argparse

import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset, concatenate_datasets
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict
)
 


def generate_prompt(data_point):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result
 
def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


def main():
    # model and dataset
    model = LlamaForCausalLM.from_pretrained(
        BASE_MODEL, 
        torch_dtype=torch.float16, 
        device_map='auto'
    )
    
    data_files = {
        'train': 'data_lora/train.json', 
        'dev': 'data_lora/dev.json', 
        'test': 'data_lora/test.json'
    }
    goemotions = load_dataset('json', data_files=data_files)
    goemotions = goemotions.map(generate_and_tokenize_prompt)
    
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT= 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "v_proj",
    ]
    
    BATCH_SIZE = 128
    MICRO_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    LEARNING_RATE = 3e-4
    TRAIN_STEPS = 300
    OUTPUT_DIR = "experiments"
    
    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    
    training_arguments = transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        max_steps=TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=10,
        optim="adamw_torch",
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=50,
        save_steps=50,
        output_dir=OUTPUT_DIR,
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="tensorboard"
    )
    
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=goemotions['train'],
        eval_dataset=goemotions['dev'],
        args=training_arguments,
        data_collator=data_collator
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))
    
    model = torch.compile(model)
    
    # trainer.train()
    # model.save_pretrained(OUTPUT_DIR)
    # model.predict
    

def get_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    CUTOFF_LEN = 80    
    BASE_MODEL = "decapoda-research/llama-7b-hf"             
    tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"
    main()