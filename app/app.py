import os
import sys
import io
import streamlit as st
import pandas as pd

# Ensure 'agent' package is visible when running via Streamlit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom imports
from agent.tools.data_loader import load_csv
from agent.tools.eda_tool import basic_stats, missing_values, memory_usage
from agent.tools.metadata_tool import (
    generate_metadata,
    save_metadata,
    load_metadata,
    compare_metadata,
)

# ───────────────────────────────────────────────
# Streamlit page setup
# ───────────────────────────────────────────────
st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
st.title("🧠 AI Data Analyst Agent")

uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded is not None:
    # ───────────────────────────────────────────────
    # 1️⃣ Load CSV
    # ───────────────────────────────────────────────
    df = load_csv(io.BytesIO(uploaded.getvalue()))
    st.success(f"✅ Loaded `{uploaded.name}` successfully!")

    # ───────────────────────────────────────────────
    # 2️⃣ Show basic preview
    # ───────────────────────────────────────────────
    st.write("### 📊 Data Preview")
    st.dataframe(df.head())
    st.write(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")
    st.write(f"**Memory Usage:** {memory_usage(df):.3f} MB")

    # ───────────────────────────────────────────────
    # 3️⃣ Generate & compare metadata (Schema Drift)
    # ───────────────────────────────────────────────
    new_metadata = generate_metadata(df, uploaded.name)
    old_metadata = load_metadata("data/metadata.json")

    if old_metadata:
        st.subheader("🧩 Schema Drift Detection")
        drift_report = compare_metadata(old_metadata, new_metadata)

        if drift_report:
            st.warning("⚠️ Schema drift detected!")
            st.json(drift_report)
        else:
            st.success("✅ No schema drift detected — dataset is consistent.")

    # Save latest metadata
    save_metadata(new_metadata)

    # ───────────────────────────────────────────────
    # 4️⃣ Display metadata
    # ───────────────────────────────────────────────
    st.subheader("📘 Dataset Metadata")
    st.json(new_metadata.model_dump())

    # ───────────────────────────────────────────────
    # 5️⃣ Existing EDA
    # ───────────────────────────────────────────────
    st.write("### 📈 Summary Statistics")
    st.write(basic_stats(df))

    st.write("### 🧩 Missing Values")
    st.write(missing_values(df))