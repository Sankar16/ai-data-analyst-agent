import io
import streamlit as st
import pandas as pd
from agent.tools.data_loader import load_csv
from agent.tools.eda_tool import basic_stats, missing_values, memory_usage
from agent.tools.metadata_tool import generate_metadata, save_metadata  # 👈 new import

st.set_page_config(page_title="AI Data Analyst Agent", layout="wide")
st.title("🧠 AI Data Analyst Agent")

uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded is not None:
    df = load_csv(io.BytesIO(uploaded.getvalue()))
    st.success(f"✅ Loaded `{uploaded.name}` successfully!")

    st.write("### 📊 Data Preview")
    st.dataframe(df.head())

    # Generate metadata
    metadata = generate_metadata(df, uploaded.name)
    save_metadata(metadata)

    st.subheader("📘 Dataset Metadata")
    st.json(metadata.model_dump())

    # Existing analysis
    st.write("### 📈 Summary Statistics")
    st.write(basic_stats(df))

    st.write("### 🧩 Missing Values")
    st.write(missing_values(df))