from pydantic import BaseModel, Field
from typing import Dict, Any
import pandas as pd
import json
import os

class ColumnProfile(BaseModel):
    dtype: str
    missing_pct: float
    unique_values: int
    memory_mb: float

class DatasetMetadata(BaseModel):
    file_name: str
    num_rows: int
    num_cols: int
    total_memory_mb: float
    columns: Dict[str, ColumnProfile]

def generate_metadata(df: pd.DataFrame, file_name: str) -> DatasetMetadata:
    """Extracts schema-level metadata from a DataFrame."""
    cols = {}
    for col in df.columns:
        missing_pct = df[col].isna().mean() * 100
        unique_values = df[col].nunique(dropna=True)
        memory_mb = df[col].memory_usage(deep=True) / (1024 ** 2)
        cols[col] = ColumnProfile(
            dtype=str(df[col].dtype),
            missing_pct=round(missing_pct, 3),
            unique_values=int(unique_values),
            memory_mb=round(memory_mb, 3)
        )
    meta = DatasetMetadata(
        file_name=file_name,
        num_rows=len(df),
        num_cols=len(df.columns),
        total_memory_mb=round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        columns=cols
    )
    return meta

def save_metadata(metadata: DatasetMetadata, path: str = "data/metadata.json"):
    """Save metadata as JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metadata.model_dump(), f, indent=4)

def load_metadata(path: str = "data/metadata.json") -> DatasetMetadata | None:
    """Load metadata from JSON."""
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return DatasetMetadata(**data)