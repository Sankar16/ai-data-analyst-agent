import pandas as pd

def basic_stats(df: pd.DataFrame) -> str:
    return df.describe(include="all").transpose().to_string()

def missing_values(df: pd.DataFrame) -> pd.DataFrame:
    miss = df.isna().sum().to_frame(name="missing")
    miss["missing_pct"] = (miss["missing"] / len(df)).round(4)
    return miss.sort_values("missing", ascending=False)

def memory_usage(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / (1024 ** 2)