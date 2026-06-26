import torch
from datasets import IterableDatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, PreTrainedTokenizer

from config import Config
from collator import DiffusionCollator
from generate import generate_from_prefix


def run_inference_test(
    model: AutoModelForMaskedLM,
    tokenizer: PreTrainedTokenizer,
    dataset: IterableDatasetDict,
    collator: DiffusionCollator,
    reps: int = 3,
) -> None:
    """Run a few inference samples to sanity-check the model."""
    print(f"\n[INFO] Running inference test... {reps} reps")

    sample_iter = iter(dataset["train"].shuffle(seed=42))

    for _ in range(reps):
        sample = next(sample_iter)
        batch = collator([sample])
        input_ids_masked = batch["input_ids"].to(model.device)

        model.eval()
        with torch.no_grad():
            pred_ids = model(input_ids_masked).logits.argmax(dim=-1)

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

        p_val = batch["p"][0].item()

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
    """Interactive REPL for diffusion generation from prefix text."""
    print("\n[INFO] Starting interactive generation mode...")

    try:
        print(f"[INFO] Loading model from {config.OUTPUT_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(config.OUTPUT_DIR)
        model = AutoModelForMaskedLM.from_pretrained(config.OUTPUT_DIR)
    except Exception as e:
        print(
            f"[WARNING] Could not load from {config.OUTPUT_DIR} ({e}). "
            "Loading base model..."
        )
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
            if prefix.strip().lower() in ("quit", "exit"):
                break
            if not prefix.strip():
                continue

            print("Generating...")
            generated_text = generate_from_prefix(
                model=model,
                tokenizer=tokenizer,
                prefix_text=prefix,
                max_len=config.MAX_LEN,
                inference_steps=100,
                show_progress=True,
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


if __name__ == "__main__":
    config = Config()
    interactive_loop(config)
