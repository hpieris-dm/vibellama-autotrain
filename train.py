#!/usr/bin/env python3
"""
Script to fine-tune Llama-3.2-1B with QLoRA on the IMDB sentiment dataset.
"""
import os
import argparse
import logging
import json
import textwrap
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
)
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from transformers import TrainerCallback
from trl import SFTTrainer

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
    """
    Format the input for Llama 3.2 using the chat template with system and user messages.
    """
    system_prompt = "You are a helpful assistant that analyzes the sentiment of provided inputs."
    
    # Convert the binary label to text
    sentiment_text = "positive" if sentiment == 1 else "negative"
    
    # Truncate review if it's too long (for efficiency)
    if len(review) > 1000:
        review = review[:1000] + "..."
    
    # Format using Llama 3.2 chat template
    prompt = f"<|begin_of_text|><|system|>\n{system_prompt}<|end_of_text|>\n<|user|>\nDetermine if the following input has a positive or negative sentiment. Reply with only 'positive' or 'negative'.\n\nReview: {review}<|end_of_text|>\n<|assistant|>\n{sentiment_text}<|end_of_text|>"
    return prompt


def process_dataset(examples):
    formatted_prompts = [format_prompt(review, label) for review, label in zip(examples["text"], examples["label"])]
    return {"formatted_text": formatted_prompts}


def generate_readme_content(model_name: str, output_dir: str):
    """
    Overwrite (or create) README.md with valid front-matter
    so Hugging Face Hub will accept the upload.
    """
    readme_path = os.path.join(output_dir, "README.md")
    content = textwrap.dedent(f"""\
        ---
        base_model: {model_name}
        tags:
          - text-generation
        license: mit
        ---

        This LoRA adapter was fine-tuned from `{model_name}` on the IMDb dataset using QLoRA.
        """)
    with open(readme_path, "w") as f:
        f.write(content)


def main():
    args = parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    if not args.hf_token:
        logger.error("HF token not provided. Set HF_TOKEN env var or pass --hf-token.")
        return
    if not args.output_dir:
        logger.error("Output directory not provided.")
        return
    if not args.model_hub_id:
        logger.error("Model hub ID not provided.")
        return
    
    os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY', '')

    # Initialize W&B
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.output_dir}-{timestamp}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )

    # Load dataset
    imdb_dataset = load_dataset("imdb")
    
    # Use the full training set
    train_dataset = imdb_dataset["train"]
    print(f"Total training examples: {len(train_dataset)}")

    # Create a 90/10 split for training and evaluation
    train_eval_split = train_dataset.train_test_split(test_size=0.1, seed=args.seed)
    train_dataset = train_eval_split["train"]
    eval_dataset = train_eval_split["test"]
    print(f"Training examples after split: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")

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
    processed_train_dataset = train_dataset.map(
    process_dataset,
    batched=True,
    remove_columns=train_dataset.column_names,
    desc="Formatting training examples"
)

    # Apply formatting to the evaluation dataset
    processed_eval_dataset = eval_dataset.map(
        process_dataset,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Formatting evaluation examples"
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_steps=50,
        save_total_limit=3,
        remove_unused_columns=True,
        push_to_hub=False,
        report_to="none",
        bf16=True,                      # Use bfloat16 precision
        weight_decay=0.01,              # Standard weight decay
        max_grad_norm=1.0,              # Gradient clipping
        seed=args.seed,                 # Use the same seed from above
        data_seed=args.seed,            # Use the same seed for data shuffling
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=processed_train_dataset,
        eval_dataset=processed_eval_dataset,  # Add evaluation dataset
        args=training_args,
        tokenizer=tokenizer,
        dataset_text_field="formatted_text",
        max_seq_length=args.max_seq_len,      # Maximum sequence length reduced for memory efficiency
    )

    # Train
    start = datetime.now()
    best_epoch_callback = BestEpochCallback()
    trainer.add_callback(best_epoch_callback)
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
    generate_readme_content(args.model_name, args.output_dir)
    api = HfApi()
    api.create_repo(args.model_hub_id, exist_ok=True)
    api.upload_folder(folder_path=args.output_dir, repo_id=args.model_hub_id)

    logger.info(f"Model pushed to {args.model_hub_id}")


if __name__ == '__main__':
    main()

