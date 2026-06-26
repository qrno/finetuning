import sys

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer
from transformers.training_args import TrainingArguments
from accelerate import PartialState

from config import Config
from collator import DiffusionCollator
from callbacks import StratifiedEvalCallback

# Prevent duplicated prints in multi-GPU
print = PartialState().print


def print_device_info() -> None:
    """Log PyTorch device and platform details."""
    print("\n" + "=" * 60)
    print("PYTORCH DEVICE & PLATFORM INFO")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA/ROCm Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device Count: {torch.cuda.device_count()}")
        print(f"  Device Name: {torch.cuda.get_device_name(0)}")
        print(f"  Current Device: {torch.cuda.current_device()}")

    is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
    print(f"ROCm / HIP Active: {is_rocm}")
    if is_rocm:
        print(f"  ROCm version: {torch.version.hip}")
    print("=" * 60 + "\n")


def main() -> None:
    config = Config()
    print_device_info()

    print(f"[INFO] Loading tokenizer from {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.model_max_length = config.MAX_LEN

    print("[INFO] Loading GigaVerbo-v2 dataset (streaming)...")
    dataset = load_dataset("Polygl0t/gigaverbo-v2", streaming=True)

    train_dataset = dataset["train"].skip(config.EVAL_SAMPLES)
    eval_dataset = dataset["train"].take(config.EVAL_SAMPLES)

    print("[INFO] Creating diffusion data collator...")
    collator = DiffusionCollator(tokenizer, config)

    print(f"[INFO] Loading model from {config.MODEL_NAME}...")
    model = AutoModelForMaskedLM.from_pretrained(config.MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        max_steps=config.MAX_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        remove_unused_columns=False,
    )

    print("\n" + "=" * 60)
    print("ACCELERATE RESOLVED DEVICES")
    print("=" * 60)
    print(f"Device: {training_args.device}")
    print(f"GPUs: {training_args.n_gpu}")
    print(f"Local Rank: {training_args.local_rank}")
    print("=" * 60 + "\n")

    print("[INFO] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.add_callback(StratifiedEvalCallback(trainer))

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    trainer.train()

    print(f"\n[INFO] Saving model to {config.OUTPUT_DIR}...")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    print("[SUCCESS] Training complete!")


if __name__ == "__main__":
    main()
    # os._exit skips GC cleanup — avoids stale HTTP connection errors
    # from the streaming dataset's background file handles.
    import os
    os._exit(0)
