class Config:
    """Config for RoBERTa model, copied by hand from main project"""

    MODEL_NAME: str = "roberta-base"
    PREFIX_LEN: int = 32
    MAX_LEN: int = 256
    CONFIDENCE_THRESHOLD: float = 0.9
    TEMPERATURE: float = 0.8

    BATCH_SIZE: int = 16
    OUTPUT_DIR: str = "weights"
    MAX_STEPS: int = 100
    SAVE_STEPS: int = 500
    LOGGING_STEPS: int = 5
    SAVE_TOTAL_LIMIT: int = 1
