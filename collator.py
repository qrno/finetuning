import math

import torch
from transformers import PreTrainedTokenizer

from config import Config


class DiffusionCollator:
    """Data collator for discrete diffusion training.

    Applies random masking at varying noise levels sampled from a cosine schedule.
    Tokens in the prefix (first PREFIX_LEN positions) and special tokens are never masked.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config: Config) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.special_ids = set(tokenizer.all_special_ids)
        self._call_count = 0
        self.fixed_mask_rate: float | None = None

        T = 100
        self.mask_probs = [
            1.0 - math.cos((i / T) * (math.pi / 2)) for i in range(1, T + 1)
        ]

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        texts = []
        metadata = []
        for f in features:
            if isinstance(f, dict):
                texts.append(f.get("text", f.get("content", "")))
                metadata.append({
                    k: f[k] for k in ("source", "subset", "edu_int_score", "toxic_int_score")
                    if k in f
                })
            else:
                texts.append(str(f))
                metadata.append({})

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

        # Build mask for positions that should never be masked
        is_special = torch.zeros_like(batch_input_ids, dtype=torch.bool)
        for sid in self.special_ids:
            is_special |= batch_input_ids == sid

        B, L = batch_input_ids.shape
        pos_idxs = torch.arange(L).unsqueeze(0).expand(B, L)
        is_prefix = pos_idxs < self.config.PREFIX_LEN

        mask_candidate = (batch_attention == 1) & (~is_special) & (~is_prefix)

        # Use fixed mask rate (stratified eval) or sample from cosine schedule
        if self.fixed_mask_rate is not None:
            p = torch.full((B, 1), self.fixed_mask_rate)
        else:
            p_indices = torch.randint(0, len(self.mask_probs), (B,))
            p = torch.tensor(self.mask_probs, dtype=torch.float32)[p_indices].unsqueeze(1)

        rand = torch.rand_like(batch_input_ids, dtype=torch.float)
        mask_positions = (rand < p) & mask_candidate

        batch_input_ids[mask_positions] = self.tokenizer.mask_token_id
        labels[~mask_positions] = -100

        # --- Sample exploration logging ---
        self._call_count += 1
        if (
            self.config.SAMPLE_LOG_INTERVAL > 0
            and self._call_count % self.config.SAMPLE_LOG_INTERVAL == 1
        ):
            self._log_sample(texts[0], metadata[0], batch_input_ids[0], p[0].item())

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention,
            "labels": labels,
            "p": p.squeeze(1),
        }

    def _log_sample(
        self,
        original_text: str,
        meta: dict,
        masked_ids: torch.Tensor,
        noise_level: float,
    ) -> None:
        """Print a single sample for dataset exploration / debugging."""
        masked_decoded = self.tokenizer.decode(masked_ids, skip_special_tokens=False)
        # Replace [MASK] with a visible block character for readability
        mask_token = self.tokenizer.mask_token
        masked_display = masked_decoded.replace(mask_token, "█")
        # Strip other special tokens for cleaner display
        for tok in self.tokenizer.all_special_tokens:
            if tok != mask_token:
                masked_display = masked_display.replace(tok, "")

        n_masks = (masked_ids == self.tokenizer.mask_token_id).sum().item()
        n_real = (masked_ids != self.tokenizer.pad_token_id).sum().item()

        sep = "─" * 72
        print(f"\n┌{sep}┐")
        print(f"│ 📋 SAMPLE #{self._call_count}  "
              f"noise={noise_level:.3f}  masks={n_masks}/{n_real} tokens")
        if meta:
            meta_str = "  ".join(f"{k}={v}" for k, v in meta.items())
            print(f"│ 🏷️  {meta_str}")
        print(f"├{sep}┤")
        print(f"│ ORIGINAL:")
        # Truncate for display (first 500 chars)
        display_text = original_text[:500]
        if len(original_text) > 500:
            display_text += "…"
        for line in display_text.split("\n"):
            print(f"│   {line}")
        print(f"├{sep}┤")
        print(f"│ AFTER MASKING:")
        display_masked = masked_display.strip()[:500]
        if len(masked_display.strip()) > 500:
            display_masked += "…"
        for line in display_masked.split("\n"):
            print(f"│   {line}")
        print(f"└{sep}┘\n")
