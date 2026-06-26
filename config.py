from dataclasses import dataclass


@dataclass
class Config:
    """Configuration for discrete diffusion MLM finetuning."""

    # Model
    MODEL_NAME: str = "unb-labia/BERTomelo-ModernBERT-Base-v1"
    PREFIX_LEN: int = 32
    MAX_LEN: int = 256

    # Training
    BATCH_SIZE: int = 16
    MAX_STEPS: int = 20
    SAVE_STEPS: int = 50
    LOGGING_STEPS: int = 5
    SAVE_TOTAL_LIMIT: int = 1
    OUTPUT_DIR: str = "weights"

    # Evaluation
    EVAL_SAMPLES: int = 500
    EVAL_STEPS: int = 5

    # Sample exploration: print a sample every N collator calls (0 = off)
    SAMPLE_LOG_INTERVAL: int = 0
