import pandas as pd
import plotly.express as px
import streamlit as st


def plot_numeric_distribution(df: pd.DataFrame):
    """Plot distribution histograms for numeric columns."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        st.info("No numeric columns found for visualization.")
        return

    st.subheader("📊 Numeric Column Distributions")
    for col in numeric_cols:
        fig = px.histogram(
            df, x=col, nbins=30, title=f"Distribution of {col}", template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_categorical_distribution(df: pd.DataFrame):
    """Plot bar charts for categorical columns."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.info("No categorical columns found for visualization.")
        return

    st.subheader("🏷️ Categorical Column Distributions")
    for col in cat_cols:
        if df[col].nunique() > 20:
            continue  # Skip high-cardinality columns
        value_counts = df[col].value_counts().reset_index()
        value_counts.columns = [col, "count"]
        fig = px.bar(
            value_counts,
            x=col,
            y="count",
            title=f"Frequency of {col}",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)