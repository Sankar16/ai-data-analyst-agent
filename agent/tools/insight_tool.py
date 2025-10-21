import pandas as pd
import streamlit as st


def generate_data_insights(df: pd.DataFrame):
    """Generate simple rule-based insights about dataset quality."""
    st.subheader("🧠 Automated Insights")

    total_rows = len(df)
    st.write(f"- The dataset contains **{total_rows} rows** and **{len(df.columns)} columns**.")

    # Missing values
    missing_summary = df.isna().mean() * 100
    missing_cols = missing_summary[missing_summary > 0]
    if not missing_cols.empty:
        st.warning("⚠️ Missing values detected:")
        for col, pct in missing_cols.items():
            st.write(f"  - `{col}` has **{pct:.1f}%** missing values.")
    else:
        st.success("✅ No missing values found!")

    # Unique value analysis
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count == total_rows:
            st.info(f"🔹 Column `{col}` has all unique values — might be an identifier.")
        elif unique_count < 5:
            st.info(f"🔹 Column `{col}` has low diversity ({unique_count} unique values).")

    # Numeric outlier check (basic)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        outlier_mask = (df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))
        outlier_pct = (outlier_mask.sum() / len(df)) * 100
        if outlier_pct > 5:
            st.warning(f"📈 `{col}` has about **{outlier_pct:.1f}%** outliers.")