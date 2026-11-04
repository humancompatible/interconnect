from pathlib import Path

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE = "Write a short sentence about public transport."
INSTRUCTIONS = [
    "Be extremely enthusiastic and positive.",
    "Be strictly neutral and impersonal; use no emotional, positive or negative language.",
    "Be extremely critical and negative.",
]
STOP_TOKENS = ["<|im_end|>", "<|endoftext|>"]
DATA = Path(__file__).parent / "data"
FEAT_DIR = DATA / "saved_features/amazon_polarity_content_mean" / MODEL.split("/")[-1].replace("-", "").replace(".", "")
