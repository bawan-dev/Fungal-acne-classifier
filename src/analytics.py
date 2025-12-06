import os
from collections import Counter

import pandas as pd
import streamlit as st

LOG_PATH = os.path.join("logs", "analysis_log.csv")

UNSAFE_KEYWORDS = [
    "lauric acid",
    "myristic acid",
    "stearic acid",
    "oleic acid",
    "isopropyl myristate",
    "cetyl alcohol",
    "glyceryl stearate",
    "polysorbate",
    "sorbitan",
]


def load_logs():
    """Load the log file or return an empty frame."""
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame(columns=["timestamp", "raw_text", "pred_label", "score"])
    return pd.read_csv(LOG_PATH)


def main():
    st.set_page_config(
        page_title="DermaLens Analytics",
        page_icon="📊",
        layout="centered",
    )

    st.title("DermaLens Analytics Dashboard")
    st.write("Insights based on past fungal acne safety analyses.")

    df = load_logs()

    if df.empty:
        st.info("No analytics yet. Run some analyses in the main app.")
        return

    st.subheader("Summary Statistics")
    st.write(f"Total analyses: **{len(df)}**")
    st.write(f"Average fungal acne score: **{df['score'].mean():.2f}**")

    st.subheader("Label Distribution")
    st.bar_chart(df["pred_label"].value_counts())

    st.subheader("Score Distribution")
    st.bar_chart(df["score"])

    st.subheader("Most Common Unsafe Ingredients")
    unsafe_counts = Counter()
    for text in df["raw_text"].astype(str):
        lower = text.lower()
        for word in UNSAFE_KEYWORDS:
            if word in lower:
                unsafe_counts[word] += 1

    if unsafe_counts:
        st.write(dict(unsafe_counts))
    else:
        st.info("No unsafe ingredients found in logs yet.")

    st.subheader("Raw Log Data")
    st.dataframe(df, use_container_width=True)


if __name__ == "__main__":
    main()
