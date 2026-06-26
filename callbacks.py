import json
from pathlib import Path

from transformers import TrainerCallback

NOISE_BUCKETS = [0.1, 0.3, 0.5, 0.7, 0.9]
EVAL_LOG_FILE = Path("eval_log.jsonl")


class StratifiedEvalCallback(TrainerCallback):
    """Re-runs eval at fixed noise levels after each standard eval.

    Logs separate metrics like `eval_mask10_loss`, `eval_mask50_loss`, etc.
    so you can track whether the model is improving at high masking rates
    (diffusion capability) vs. just fitting the dataset (low masking rates).

    Appends all eval results to eval_log.jsonl for later plotting.
    """

    def __init__(self, trainer) -> None:
        self.trainer = trainer
        self._running = False

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        if self._running:
            return

        self._running = True
        try:
            collator = self.trainer.data_collator
            original_rate = collator.fixed_mask_rate

            # Collect standard eval loss
            record = {
                "step": state.global_step,
                "eval_loss": metrics.get("eval_loss") if metrics else None,
            }

            # Run stratified evals
            for rate in NOISE_BUCKETS:
                collator.fixed_mask_rate = rate
                prefix = f"eval_mask{int(rate * 100)}"
                bucket_metrics = self.trainer.evaluate(metric_key_prefix=prefix)
                record[f"mask{int(rate * 100)}_loss"] = bucket_metrics.get(
                    f"{prefix}_loss"
                )

            collator.fixed_mask_rate = original_rate

            self._print_summary(state.global_step, record)
            self._save_record(record)
        finally:
            self._running = False

    @staticmethod
    def _print_summary(step: int, record: dict) -> None:
        losses = {
            k: v for k, v in record.items() if k.endswith("_loss") and v is not None
        }
        max_loss = max(losses.values()) if losses else 1.0

        print(f"\n{'=' * 50}")
        print(f"  📊 STRATIFIED EVAL @ step {step}")
        print(f"{'─' * 50}")
        for key, loss in losses.items():
            bar_len = int((loss / max_loss) * 20) if max_loss > 0 else 0
            bar = "█" * bar_len + "░" * (20 - bar_len)
            label = key.replace("_loss", "").replace("_", " ")
            print(f"  {label:>10}  │ {loss:.3f}  {bar}")
        print(f"{'=' * 50}\n")

    @staticmethod
    def _save_record(record: dict) -> None:
        with open(EVAL_LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")
