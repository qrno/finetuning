import sys
import time

import torch
from transformers import AutoModelForMaskedLM, PreTrainedTokenizer


def diffusion_generate(
    model: AutoModelForMaskedLM,
    tokenizer: PreTrainedTokenizer,
    input_ids: torch.Tensor,
    inference_steps: int = 100,
    show_progress: bool = False,
) -> torch.Tensor:
    """Iterative mask-predict decoding for discrete diffusion.

    Starting from a partially-masked sequence, iteratively:
      1. Predict all masked positions.
      2. Re-mask the least confident predictions according to a linear schedule.

    All originally-masked positions are re-predicted every step (allows error correction).
    """
    model.eval()
    B, L = input_ids.shape

    current_ids = input_ids.clone()
    original_mask = input_ids == tokenizer.mask_token_id
    M_0 = original_mask.sum(dim=-1)  # (B,) total masks per example

    for step in range(1, inference_steps + 1):
        if (current_ids == tokenizer.mask_token_id).sum() == 0:
            break

        if show_progress:
            _print_progress(tokenizer, current_ids[0], step, inference_steps)

        with torch.no_grad():
            logits = model(current_ids).logits

        probs = logits.softmax(dim=-1)
        confidences, pred_ids = probs.max(dim=-1)

        # Re-predict all originally-masked positions (allows error correction)
        current_ids[original_mask] = pred_ids[original_mask]

        if step == inference_steps:
            break

        # Linear un-masking schedule: target number of masks remaining after this step
        M_t = (M_0 * (inference_steps - step) / inference_steps).long()

        # Re-mask the least confident among originally-masked positions
        confidences[~original_mask] = float("inf")

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
    """Tokenize a prefix, pad with [MASK] tokens, and run diffusion generation.

    Constructs: [BOS] prefix_tokens [MASK]*N [EOS]
    where N fills up to `max_len` total tokens.
    """
    model.eval()

    encoded = tokenizer(
        prefix_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )

    prefix_ids = encoded["input_ids"][0]

    # Strip trailing special token (e.g. </s> for RoBERTa) to append masks
    if prefix_ids[-1] in tokenizer.all_special_ids and len(prefix_ids) > 1:
        prefix_ids = prefix_ids[:-1]

    num_prefix_tokens = len(prefix_ids)

    if num_prefix_tokens >= max_len - 1:
        return tokenizer.decode(prefix_ids, skip_special_tokens=True)

    num_masks = max_len - num_prefix_tokens - 1  # -1 for the final EOS token

    # [PREFIX_TOKENS] + [MASK]*num_masks + [EOS]
    input_ids = torch.cat([
        prefix_ids,
        torch.full((num_masks,), tokenizer.mask_token_id),
        torch.tensor([tokenizer.sep_token_id]),
    ]).unsqueeze(0).to(model.device)

    generated_ids = diffusion_generate(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        inference_steps=inference_steps,
        show_progress=show_progress,
    )

    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


def _print_progress(
    tokenizer: PreTrainedTokenizer,
    token_ids: torch.Tensor,
    step: int,
    total_steps: int,
) -> None:
    """Display the current generation state with masks shown as █ blocks."""
    decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
    decoded = decoded.replace(tokenizer.mask_token, "█")
    for special in tokenizer.all_special_tokens:
        if special != tokenizer.mask_token:
            decoded = decoded.replace(special, "")

    print("\033[2J\033[H", end="")
    print(f"[Step {step:03d}/{total_steps:03d}]\n\n{decoded.strip()}")
    sys.stdout.flush()
    time.sleep(0.02)
