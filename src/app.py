import os
import sys
import io
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from lime.lime_text import LimeTextExplainer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_engine import (
    AnalysisEngine,
    UNSAFE_KEYWORDS,
    NEUTRAL_RISK,
    fetch_product_ingredients,
    extract_ingredients_from_image,
)
from src.preprocessing import join_ingredients_for_model

st.set_page_config(
    page_title="Fungal Acne Ingredient Classifier",
    page_icon="🧴",
    layout="wide",
)


@st.cache_resource
def get_engine() -> AnalysisEngine:
    return AnalysisEngine()


def ensure_array(probs):
    if isinstance(probs, np.ndarray):
        return probs
    return np.array(probs)


def render_chips(groups):
    palette = {
        "safe": ("#dcfce7", "#22c55e"),
        "mild": ("#fef9c3", "#eab308"),
        "unsafe": ("#fee2e2", "#ef4444"),
    }
    for category, items in groups.items():
        if not items:
            continue
        bg, border = palette[category]
        st.markdown(f"**{category.title()}**")
        chip_html = ""
        for ing in items:
            chip_html += f"""
            <div class="chip" style="
                background:{bg};
                border:1px solid {border};
                animation: fadeIn 0.4s ease;
            ">{ing}</div>
            """
        st.markdown(
            f"""
            <div style="
                display:flex;
                flex-wrap:wrap;
                gap:6px;
                margin-bottom:12px;
            ">
                {chip_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_similar_products(similar_products):
    if not similar_products:
        st.write("No similar products found.")
        return
    for item in similar_products:
        st.write(f"**{item['product_name']}** — similarity `{item['score']:.2f}`")


def render_ingredient_insights(insights):
    if not insights:
        st.write("No ingredient-level matches found.")
        return
    for ing, sim in insights.items():
        st.write(f"**{ing}** → **{sim['closest_ingredient']}** (`{sim['score']:.2f}`)")


def render_previous_products(engine: AnalysisEngine):
    entries = engine.get_previous_results()
    if not entries:
        st.write("No previously analyzed products yet.")
        return None

    options = [f"{e.get('product_name','Untitled')} ({e.get('timestamp','')})" for e in entries]
    choice = st.selectbox("Previously Analyzed Products", ["Select..."] + options)
    if choice == "Select...":
        return None
    idx = options.index(choice)
    entry = entries[idx]
    cached = engine.load_cached_analysis(entry)
    return cached


def run_lime_if_requested(engine: AnalysisEngine, text: str, classes, model):
    explainer = LimeTextExplainer(class_names=list(classes))

    def predict_proba_lime(text_list):
        processed = [join_ingredients_for_model(t) for t in text_list]
        return model.predict_proba(processed)

    exp = explainer.explain_instance(
        text,
        predict_proba_lime,
        num_features=8,
        top_labels=1,
    )
    top_idx = exp.top_labels[0]
    top_label_name = classes[top_idx]
    lime_df = pd.DataFrame(exp.as_list(label=top_idx), columns=["feature", "weight"])

    # Try to render a plot for PDF embedding
    lime_image_bytes: Optional[bytes] = None
    try:
        import matplotlib.pyplot as plt

        fig = exp.as_pyplot_figure(label=top_idx)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        lime_image_bytes = buf.read()
    except Exception:
        lime_image_bytes = None

    return top_label_name, lime_df, lime_image_bytes


def main():
    st.markdown(
        """
        <style>
        .chip {
            padding:6px 10px;
            border-radius:999px;
            display:inline-flex;
            align-items:center;
            font-size:0.9rem;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(4px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .card {
            background:#ffffff;
            border-radius:16px;
            padding:16px 18px;
            box-shadow:0 10px 25px rgba(15,23,42,0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    engine = get_engine()

    st.title("Fungal Acne Ingredient Classifier")
    st.caption("Paste ingredients or upload a label to get safety ratings, similarity, and expert insights.")

    col_input, col_meta = st.columns([3, 2])
    with col_input:
        product_name = st.text_input("Product name (optional)", "")
        ingredients_text = st.text_area(
            "Ingredients list or product name",
            height=140,
            placeholder="e.g. Aqua, Glycerin, Niacinamide, Panthenol, ...",
        )
        upload = st.file_uploader("Upload ingredient label (image)", type=["png", "jpg", "jpeg"])
        extracted_text = None
        if upload is not None:
            extracted_text = extract_ingredients_from_image(upload)
            st.info(f"OCR detected ingredients: {extracted_text}")
            if not ingredients_text:
                ingredients_text = extracted_text
        auto_fetch = st.button("Auto-detect ingredients from product name")
        if auto_fetch and ingredients_text and "," not in ingredients_text:
            fetched = fetch_product_ingredients(ingredients_text)
            st.warning(fetched.get("message"))
            if fetched.get("ingredients"):
                ingredients_text = fetched["ingredients"]
        analyze_clicked = st.button("Analyze", type="primary")

    with col_meta:
        st.markdown("### Quick Tips")
        st.write("- Provide a clear comma-separated list for best results.")
        st.write("- Use Expert Mode to view LIME and probability charts.")
        expert_mode = st.toggle("Expert Mode", value=False)

    # Session state to persist results
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "lime_image" not in st.session_state:
        st.session_state.lime_image = None

    # Load from history if chosen
    with st.expander("Previously Analyzed Products", expanded=False):
        cached = render_previous_products(engine)
        if cached:
            st.success("Loaded from history without reprocessing.")
            st.session_state.analysis_result = cached

    # Run new analysis
    if analyze_clicked:
        if not ingredients_text.strip():
            st.warning("Please enter an ingredients list first.")
            return
        result = engine.analyze(ingredients_text, product_name=product_name or None)
        st.session_state.analysis_result = result
        st.session_state.lime_image = None

    result = st.session_state.analysis_result
    if result is None:
        return

    # Prepare data
    probs = ensure_array(result["tfidf"]["probs"])
    classes = result["tfidf"]["classes"]
    sorted_idx = probs.argsort()[::-1]
    max_prob = float(np.max(probs)) if len(probs) else 0.0

    tabs = st.tabs(["Overview", "Ingredients", "Similar Products", "Expert Mode"])

    with tabs[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction")
        st.markdown(f"**Result:** `{result['tfidf']['label']}`")
        st.caption(f"Highest confidence: {max_prob:.2f}")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Fungal Acne Score")
        st.markdown(f"### {result['safety_score']} / 10")
        st.write(result["explanation"])
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Ingredient Breakdown")
        render_chips(result.get("highlight_groups", {}))
        st.subheader("Ingredient Insights")
        render_ingredient_insights(result.get("ingredient_similarities", {}))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Similar Products")
        render_similar_products(result.get("similar_products", []))
        st.markdown("</div>", unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Probability distribution")
        prob_df = pd.DataFrame(
            {
                "label": np.array(classes)[sorted_idx],
                "probability": probs[sorted_idx],
            }
        ).set_index("label")
        st.bar_chart(prob_df)

        if expert_mode:
            st.subheader("LIME explanation")
            top_label_name, lime_df, lime_image = run_lime_if_requested(engine, result["ingredients_raw"], classes, engine.model)
            st.write(f"Top label explained: **{top_label_name}**")
            st.dataframe(lime_df)
            st.session_state.lime_image = lime_image
        st.markdown("</div>", unsafe_allow_html=True)

    # PDF export
    pdf_buffer = engine.generate_pdf_report(result, lime_image=st.session_state.lime_image)
    if pdf_buffer:
        st.download_button(
            label="Download Full Report as PDF",
            data=pdf_buffer,
            file_name="fungal_acne_report.pdf",
            mime="application/pdf",
        )


if __name__ == "__main__":
    main()
