import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
)
from transformers.training_args import TrainingArguments

from config import Config


class DiffusionCollator:
    def __init__(self, tokenizer: RobertaTokenizerFast, config: Config) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.special_ids = set(tokenizer.all_special_ids)
        self.mask_probs = [i / config.MAX_LEN for i in range(1, config.MAX_LEN + 1)]

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        texts = []
        for f in features:
            if isinstance(f, dict):
                texts.append(f.get("text", f.get("content", "")))
            else:
                texts.append(f["text"] if "text" in f else str(f))

        encoded = self.tokenizer(
            texts,
            max_length=self.config.MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        batch_input_ids = encoded["input_ids"]
        labels = batch_input_ids.clone()
        batch_attention = encoded["attention_mask"]

        is_special = torch.zeros_like(batch_input_ids, dtype=torch.bool)
        for sid in self.special_ids:
            is_special |= batch_input_ids == sid

        B, L = batch_input_ids.shape
        pos_idxs = torch.arange(L).unsqueeze(0).expand(B, L)
        is_prefix = pos_idxs < self.config.PREFIX_LEN

        mask_candidate = (batch_attention == 1) & (~is_special) & (~is_prefix)

        p = float(self.mask_probs[torch.randint(0, len(self.mask_probs), (1,))])
        rand = torch.rand_like(batch_input_ids, dtype=torch.float)
        mask_positions = (rand < p) & mask_candidate

        batch_input_ids[mask_positions] = self.tokenizer.mask_token_id
        labels[~mask_positions] = -100

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": labels,
        }


def run_inference_test(
    model: RobertaForMaskedLM,
    tokenizer: RobertaTokenizerFast,
    dataset: DatasetDict,
    diffusion_collator: DiffusionCollator,
) -> None:
    """Get a small sample"""
    print("\n[INFO] Running inference test...")

    sample = next(iter(dataset["train"]))
    batch = diffusion_collator([sample])
    input_ids_masked = batch["input_ids"].to(model.device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids_masked).logits
        pred_ids = logits.argmax(dim=-1)

    masked_str = tokenizer.decode(
        input_ids_masked[0],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).replace(tokenizer.mask_token, "█")

    pred_str = tokenizer.decode(
        pred_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print("\n" + "=" * 60)
    print("Sample inference")
    print("=" * 60)
    print("\nMasked Input:")
    print(masked_str)
    print("\nModel Output:")
    print(pred_str)
    print("=" * 60 + "\n")


def main() -> None:
    config = Config()

    print(f"[INFO] Loading tokenizer from {config.MODEL_NAME}...")
    tokenizer = RobertaTokenizerFast.from_pretrained(config.MODEL_NAME)
    tokenizer.model_max_length = config.MAX_LEN

    print("[INFO] Loading OpenWebText dataset...")
    dataset = load_dataset("Skylion007/openwebtext", streaming=True)

    print("[INFO] Creating diffusion data collator...")
    diffusion_collator = DiffusionCollator(tokenizer, config)

    print(f"[INFO] Loadin gmodel from {config.MODEL_NAME}...")
    model = RobertaForMaskedLM.from_pretrained(config.MODEL_NAME)

    training_args = TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        max_steps=config.MAX_STEPS,
        per_device_train_batch_size=config.BATCH_SIZE,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        logging_steps=config.LOGGING_STEPS,
        remove_unused_columns=False,
    )

    print("[INFO] Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["train"],
        data_collator=diffusion_collator,
    )

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60 + "\n")
    trainer.train()

    print(f"\n[INFO] Saving model to {config.OUTPUT_DIR}...")
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    print("[SUCCESS] Training complete!")
    run_inference_test(model, tokenizer, dataset, diffusion_collator)


if __name__ == "__main__":
    main()
