import io
import pandas as pd

def load_csv(source: io.BytesIO | str, infer_datetime: bool = True) -> pd.DataFrame:
    if isinstance(source, io.BytesIO):
        source.seek(0)
        df = pd.read_csv(source, low_memory=False)
    else:
        df = pd.read_csv(source, low_memory=False)

    if infer_datetime:
        for col in df.columns:
            try:
                parsed = pd.to_datetime(df[col], errors="raise")
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
            except Exception:
                pass
    return df