import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import streamlit as st
import pandas as pd
from agent.tools.data_loader import load_csv
from agent.tools.eda_tool import basic_stats, missing_values, memory_usage
from agent.tools.metadata_tool import generate_metadata, save_metadata

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
st.title("🔎 AI Data Analyst Agent — M1")


uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded is not None:
    df = load_csv(io.BytesIO(uploaded.getvalue()))

    st.subheader("Preview")
    st.dataframe(df.head(20))

    meta = generate_metadata(df, uploaded.name)
    save_metadata(meta)

    st.subheader("📘 Dataset Metadata")
    st.json(meta.model_dump())

    st.subheader("Schema & Basics")
    st.json({
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "memory_mb": round(memory_usage(df), 3),
    })

    st.subheader("Descriptive Stats (numeric)")
    st.text(basic_stats(df))

    st.subheader("Missing Values")
    st.dataframe(missing_values(df))
else:
    st.info("Upload a CSV to begin.")