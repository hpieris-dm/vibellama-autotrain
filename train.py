#!/usr/bin/env python3
"""
Script to fine-tune Llama-3.2-1B with QLoRA on the IMDB sentiment dataset.
"""
import os
import argparse
import logging
import json
from datetime import datetime

import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer, TrainerCallback
import wandb
from huggingface_hub import HfApi


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 1B with QLoRA")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))
    parser.add_argument("--wandb-project", type=str, default="essex-sentiment-analysis")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-hub-id", type=str, default=None)
    return parser.parse_args()


class BestEpochCallback(TrainerCallback):
    def __init__(self):
        self.best_loss = float('inf')
        self.best_epoch = -1

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and metrics.get('eval_loss', None) < self.best_loss:
            self.best_loss = metrics['eval_loss']
            self.best_epoch = state.epoch


def format_prompt(review, sentiment):
    # Chat-format prompt
    system_prompt = "You are a helpful assistant that analyzes sentiment."
    sentiment_text = "positive" if sentiment == 1 else "negative"
    if len(review) > 1000:
        review = review[:1000] + "..."
    return (
        f"<|begin_of_text|><|system|>\n{system_prompt}<|end_of_text|>"
        f"<|assistant|>\n{sentiment_text}<|end_of_text|>"
    )


def process_dataset(batch):
    prompts = [format_prompt(r, s) for r, s in zip(batch['text'], batch['label'])]
    tokenized = tokenizer(
        prompts,
        truncation=True,
        max_length=args.max_seq_len,
        padding='max_length',
    )
    tokenized['labels'] = tokenized['input_ids'].copy()
    return tokenized


def main():
    args = parse_args()

    # Set random seed
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Tokens and directories
    if not args.hf_token:
        logger.error("HF token not provided; set HF_TOKEN env var or pass --hf-token.")
        return
    os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY', '')

    # Decide output_dir and model_hub_id if not set
    args.output_dir = args.output_dir or f"llama_3_2_1b_qlora_seed{args.seed}"
    args.model_hub_id = args.model_hub_id or f"yourusername/VibeLlama-1b-seed-{args.seed}"

    # Initialize W&B
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.output_dir}-{timestamp}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )

    # Load dataset
    dataset = load_dataset("imdb")
    train_split = dataset['train'].train_test_split(test_size=0.1, seed=args.seed)
    train_ds = train_split['train']
    eval_ds = train_split['test']

    # Load quantized model & tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map='auto',
        token=args.hf_token
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=['q_proj','v_proj'],
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, peft_config)

    # Process datasets
    processed_train = train_ds.map(process_dataset, batched=True, remove_columns=train_ds.column_names)
    processed_eval = eval_ds.map(process_dataset, batched=True, remove_columns=eval_ds.column_names)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=50,
        load_best_model_at_end=True,
        fp16=True,
        push_to_hub=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        data_collator=data_collator,
        callbacks=[BestEpochCallback()]
    )

    # Train
    start = datetime.now()
    trainer.train()
    elapsed = datetime.now() - start
    logger.info(f"Training completed in {elapsed}")

    # Evaluate
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

    # Save model and tokenizer
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metadata
    metadata = {
        'eval_results': {k: float(v) for k, v in eval_results.items()},
        'date': datetime.now().isoformat()
    }
    with open(os.path.join(args.output_dir, 'training_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Push to Hugging Face Hub
    api = HfApi()
    api.create_repo(args.model_hub_id, exist_ok=True)
    api.upload_folder(folder_path=args.output_dir, repo_id=args.model_hub_id)

    logger.info(f"Model pushed to {args.model_hub_id}")


if __name__ == '__main__':
    main()

