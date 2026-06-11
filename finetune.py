import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForMaskedLM,
    Trainer,
)
from transformers.training_args import TrainingArguments
from accelerate import PartialState

from config import Config
import math
import sys

# This is so that things are not printed 4x
print = PartialState().print

class DiffusionCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: Config) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.special_ids = set(tokenizer.all_special_ids)
        T = 100
        self.mask_probs = [1.0 - math.cos((i / T) * (math.pi / 2)) for i in range(1, T + 1)]
        #self.mask_probs = [0.1, 0.2, 0.3, 0.4, 0.5]


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

        p_indices = torch.randint(0, len(self.mask_probs), (B,))
        p = torch.tensor(self.mask_probs, dtype=torch.float32)[p_indices].unsqueeze(1) # shape (B, 1)
        
        rand = torch.rand_like(batch_input_ids, dtype=torch.float)
        mask_positions = (rand < p) & mask_candidate

        batch_input_ids[mask_positions] = self.tokenizer.mask_token_id
        labels[~mask_positions] = -100

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": labels,
            "p": p.squeeze(1),
        }


def diffusion_generate(
    model: AutoModelForMaskedLM,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    inference_steps: int = 100,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Iterative mask-predict decoding.
    """
    model.eval()
    B, L = input_ids.shape
    
    current_ids = input_ids.clone()
    original_mask = (input_ids == tokenizer.mask_token_id)
    M_0 = original_mask.sum(dim=-1) # (B,)
    
    for step in range(1, inference_steps + 1):
        mask_positions = (current_ids == tokenizer.mask_token_id)
        if mask_positions.sum() == 0:
            break
            
        if show_progress:
            # We skip special tokens but leave the [MASK] tokens visible
            # We have to do this carefully since skip_special_tokens=True would hide masks too
            decoded = tokenizer.decode(current_ids[0], skip_special_tokens=False)
            decoded = decoded.replace(tokenizer.mask_token, "█")
            for special in tokenizer.all_special_tokens:
                if special != tokenizer.mask_token:
                    decoded = decoded.replace(special, "")
            
            # Clear the screen and move cursor to top left
            print("\033[2J\033[H", end="")
            print(f"[Step {step:03d}/{inference_steps:03d}]\n\n{decoded.strip()}")
            sys.stdout.flush()
            
            import time
            time.sleep(0.1)
            
        with torch.no_grad():
            outputs = model(current_ids)
            logits = outputs.logits
            
        probs = logits.softmax(dim=-1)
        confidences, pred_ids = probs.max(dim=-1)
        
        # Unmask and update all non-prefix tokens with their current predictions
        # This allows the model to correct tokens it unmasked in previous steps
        current_ids[original_mask] = pred_ids[original_mask]

        if step == inference_steps:
            break
            
        # Target number of masks for the end of this step
        M_t = (M_0 * (inference_steps - step) / inference_steps).long()
        
        # We want to re-mask tokens from the entire original mask set (i.e. anything not in the prefix).
        # This means previously unmasked tokens can be re-masked if their confidence drops.
        confidences[~original_mask] = float('inf')
        
        for i in range(B):
            k = M_t[i].item()
            if k > 0:
                _, lowest_conf_indices = torch.topk(confidences[i], k, largest=False)
                current_ids[i, lowest_conf_indices] = tokenizer.mask_token_id
                
    if show_progress:
        decoded = tokenizer.decode(current_ids[0], skip_special_tokens=True)
        print("\033[2J\033[H", end="")
        print(f"[Step {inference_steps:03d}/{inference_steps:03d}]\n\n{decoded.strip()}")
        sys.stdout.flush()
        print("\n")
        
    return current_ids


def generate_from_prefix(
    model: AutoModelForMaskedLM,
    tokenizer: PreTrainedTokenizer,
    prefix_text: str,
    max_len: int = 256,
    inference_steps: int = 100,
    show_progress: bool = False,
) -> str:
    """
    Given a prefix string, tokenizes it, pads the rest of the sequence with [MASK]
    tokens up to `max_len`, and uses diffusion generation to fill them in.
    """
    model.eval()
    
    # Tokenize the prefix
    encoded = tokenizer(
        prefix_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt"
    )
    
    prefix_ids = encoded["input_ids"][0]
    # Remove the trailing EOS/SEP token if the tokenizer added one,
    # so we can append masks properly.
    if prefix_ids[-1] in tokenizer.all_special_ids and len(prefix_ids) > 1:
        # We assume the last token is SEP or EOS, we strip it so we can generate
        # continuous text, but it depends on the exact model.
        # RoBERTa format: <s> prefix </s>
        # If we remove </s>, we can add masks and then append </s> at the end.
        prefix_ids = prefix_ids[:-1]
        
    num_prefix_tokens = len(prefix_ids)
    
    if num_prefix_tokens >= max_len - 1:
        # Prefix is already too long
        return tokenizer.decode(prefix_ids, skip_special_tokens=True)
        
    # How many masks do we need?
    num_masks = max_len - num_prefix_tokens - 1 # -1 for the final EOS token
    
    # Construct the input sequence
    # [PREFIX_TOKENS] + [MASK]*num_masks + [EOS]
    input_ids = torch.cat([
        prefix_ids,
        torch.tensor([tokenizer.mask_token_id] * num_masks),
        torch.tensor([tokenizer.sep_token_id]) # RoBERTa uses sep_token_id for end of sequence
    ]).unsqueeze(0).to(model.device)
    
    # Run diffusion generation
    generated_ids = diffusion_generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        inference_steps=inference_steps,
        show_progress=show_progress
    )
    
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def run_inference_test(
    model: AutoModelForMaskedLM,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict,
    diffusion_collator: DiffusionCollator,
    reps: int = 0
) -> None:
    """Get a small sample"""
    print(f"\n[INFO] Running inference test... {reps} reps")

    for _ in range(reps):
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

        p_val = batch["p"]
        if isinstance(p_val, torch.Tensor):
            p_val = p_val[0].item()

        print("\n" + "=" * 60)
        print("Sample inference")
        print("=" * 60)
        print("\nOriginal Input:")
        print(sample["text"][:256])
        print(f"\nMasked Input: p={p_val:.4f}")
        print(masked_str)
        print("\nModel Output:")
        print(pred_str)
        print("=" * 60 + "\n")


def interactive_loop(config: Config) -> None:
    print("\n[INFO] Starting interactive generation mode...")
    
    # Try loading the finetuned model, otherwise load base
    try:
        print(f"[INFO] Loading model from {config.OUTPUT_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)
        model = AutoModelForMaskedLM.from_pretrained(config.OUTPUT_DIR)
    except Exception as e:
        print(f"[WARNING] Could not load from {config.OUTPUT_DIR} ({e}). Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForMaskedLM.from_pretrained(config.MODEL_NAME)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    model.to(device)
    
    print("\n" + "=" * 60)
    print("Interactive Diffusion Generation")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")
    
    while True:
        try:
            prefix = input("\nEnter prefix >> ")
            if prefix.strip().lower() in ["quit", "exit"]:
                break
            if not prefix.strip():
                continue
                
            print("Generating...")
            # Use max_len from config
            generated_text = generate_from_prefix(
                model=model,
                tokenizer=tokenizer,
                prefix_text=prefix,
                max_len=config.MAX_LEN,
                inference_steps=100, # Updated to 100 for smoother diffusion
                show_progress=True
            )
            
            print("\nGenerated Text:")
            print("-" * 60)
            print(generated_text)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}")


def main() -> None:
    config = Config()

    print("\n" + "=" * 60)
    print("PYTORCH DEVICE & PLATFORM INFO")
    print("=" * 60)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA/ROCm Available (torch.cuda.is_available()): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA/ROCm Device Count: {torch.cuda.device_count()}")
        print(f"CUDA/ROCm Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Current Device Index: {torch.cuda.current_device()}")
    else:
        print("CUDA/ROCm is NOT available to PyTorch.")
    
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    print(f"ROCm / HIP Active: {is_rocm}")
    if is_rocm:
        print(f"ROCm version (torch.version.hip): {torch.version.hip}")
    print("=" * 60 + "\n")

    print(f"[INFO] Loading tokenizer from {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.model_max_length = config.MAX_LEN

    print("[INFO] Loading OpenWebText dataset...")
    dataset = load_dataset("Skylion007/openwebtext", streaming=True)

    print("[INFO] Creating diffusion data collator...")
    diffusion_collator = DiffusionCollator(tokenizer, config)

    print(f"[INFO] Loadin gmodel from {config.MODEL_NAME}...")
    model = AutoModelForMaskedLM.from_pretrained(config.MODEL_NAME)

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

    print("\n" + "=" * 60)
    print("HUGGING FACE / ACCELERATE RESOLVED DEVICES")
    print("=" * 60)
    print(f"Resolved Device (training_args.device): {training_args.device}")
    print(f"Number of GPUs (training_args.n_gpu): {training_args.n_gpu}")
    print(f"Use MPS (Apple Silicon): {getattr(training_args, 'use_mps_device', False)}")
    print(f"Local Rank: {training_args.local_rank}")
    print("=" * 60 + "\n")

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
    if len(sys.argv) > 1 and sys.argv[1] in ["interact", "--interact"]:
        config = Config()
        interactive_loop(config)
    else:
        main()
