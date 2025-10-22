import json
import pandas as pd
import torch
from transformers import pipeline


class AIAgent:
    """Simple local LLM agent for generating insights using Hugging Face models."""

    def __init__(self, model_name: str = "distilgpt2"):
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=200,
        )

    def generate(self, prompt: str) -> str:
        """Generate text response from model."""
        response = self.generator(prompt, max_new_tokens=200, do_sample=True, temperature=0.7)
        return response[0]["generated_text"]


def generate_ai_insights(df: pd.DataFrame, metadata: dict) -> str:
    """Generate contextual insights from metadata using a local model."""
    agent = AIAgent()

    # Build dataset summary as context
    context = f"Dataset has {len(df)} rows and {len(df.columns)} columns.\n"
    context += "Column overview:\n"
    for col, meta in metadata["columns"].items():
        context += (
            f"- {col}: type={meta['dtype']}, missing={meta['missing_pct']}%, "
            f"unique={meta['unique_values']}\n"
        )

    prompt = (
        "You are an intelligent data analyst assistant. "
        "Analyze this dataset metadata and describe 3-5 key insights about its structure, "
        "missing values, and potential preprocessing steps.\n"
        f"### Dataset Metadata ###\n{context}\n"
        "### Insights:\n"
    )

    return agent.generate(prompt)