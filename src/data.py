import pandas as pd
from pathlib import Path
from .utils import DATA_DIR

def load_documents(csv_path: str | Path = DATA_DIR / "documents.csv", limit: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)
    # Expect columns: id, text (MS MARCO passages already chunked)
    if "id" not in df.columns or "text" not in df.columns:
        raise ValueError("documents.csv must have columns: id,text")
    return df
