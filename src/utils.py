import os
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "msmarco"
RESULTS_DIR  = PROJECT_ROOT / "results"
MODELS_DIR   = PROJECT_ROOT / "models"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# Ensure standard dirs exist at import time
ensure_dir(RESULTS_DIR)
ensure_dir(MODELS_DIR)



