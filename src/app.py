import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer

from src.preprocessing import join_ingredients_for_model, split_ingredients
from src.safety_score import calculate_safety_score

MODEL_PATH = "models/tfidf_multiclass_model.joblib"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)

    # Safety check to prevent Streamlit Cloud from loading the wrong model
    if not hasattr(model, "classes_") or len(model.classes_) != 10:
        raise ValueError(
            f"Incorrect model loaded from {MODEL_PATH}. "
            f"Expected 10-class model but got {len(getattr(model, 'classes_', []))} classes.\n"
            "Make sure ONLY tfidf_multiclass_model.joblib exists in /models."
        )

    return model

# -------------------------------------------------------------
# 🔴 KEYWORD LISTS
# -------------------------------------------------------------
UNSAFE_KEYWORDS = [
    "lauric acid", "myristic acid", "stearic acid", "oleic acid",
    "isopropyl myristate", "cetyl alcohol", "glyceryl stearate",
    "polysorbate", "sorbitan","cetearyl alcohol",
]

NEUTRAL_RISK = [
    "dimethicone", "caprylic/capric triglyceride", "fragrance",
]

# -------------------------------------------------------------
# 🏷️ LABEL DESCRIPTIONS
# -------------------------------------------------------------
LABEL_INFO = {
    "safe": {"name": "Safe", "desc": "Low-risk, non-comedogenic, suitable for all skin types."},
    "neutral": {"name": "Neutral", "desc": "Does not strongly help or harm skin — minimal impact."},
    "malassezia_trigger": {"name": "Malassezia Trigger", "desc": "May worsen fungal acne due to certain fatty acids/esters."},
    "comedogenic": {"name": "Comedogenic", "desc": "Likely to clog pores — acne-prone skin may react."},
    "irritant": {"name": "Irritant", "desc": "May cause skin irritation or redness in sensitive users."},
    "fragrance_heavy": {"name": "Fragrance Heavy", "desc": "Contains strong fragrance; may irritate sensitive skin."},
    "fatty_acid": {"name": "Fatty-Acid Rich", "desc": "Moisturising but may worsen fungal acne."},
    "emollient_heavy": {"name": "Emollient Heavy", "desc": "Thick, moisturising formula; great for dry skin, heavy for oily skin."},
    "surfactant": {"name": "Surfactant Based", "desc": "Contains cleansing agents typically used in washes."},
    "preservative": {"name": "Preservative Focused", "desc": "Formula dominated by essential preservatives."},
}


# -------------------------------------------------------------
# 📘 EXPLANATION GENERATOR
# -------------------------------------------------------------
def generate_explanation(ingredients, score):
    ingredients_lower = [i.lower() for i in ingredients]

    detected_strong = [b for b in UNSAFE_KEYWORDS if any(b in ing for ing in ingredients_lower)]
    detected_mild = [m for m in NEUTRAL_RISK if any(m in ing for ing in ingredients_lower)]

    if score >= 8:
        explanation = "This product appears **low risk** for fungal acne. No major triggers detected."
    elif score >= 5:
        explanation = "This product has a **moderate risk** rating. Some ingredients may trigger fungal acne."
        if detected_mild:
            explanation += f" Mild-risk ingredients: {', '.join(detected_mild)}."
    else:
        explanation = "This product is **high risk** for fungal acne. Contains known Malassezia-feeding ingredients."
        if detected_strong:
            explanation += f" High-risk ingredients: {', '.join(detected_strong)}."

    return explanation


# -------------------------------------------------------------
# 🎨 INGREDIENT HIGHLIGHTING
# -------------------------------------------------------------
def highlight_ingredients_html(ingredients):
    blocks = []
    for ing in ingredients:
        lower = ing.lower()

        if any(bad in lower for bad in UNSAFE_KEYWORDS):
            colour, icon = "#fecaca", "❌"
        elif any(mid in lower for mid in NEUTRAL_RISK):
            colour, icon = "#fef3c7", "⚠️"
        else:
            colour, icon = "#dcfce7", "✅"

        blocks.append(f"""
        <div style="
            background-color:{colour};
            padding:6px 10px;
            border-radius:6px;
            margin-bottom:4px;
            font-size:0.95rem;
        ">
            {icon} {ing}
        </div>
        """)
    return blocks


# -------------------------------------------------------------
# 🧠 MODEL CONFIDENCE BADGE
# -------------------------------------------------------------
def get_confidence_badge(max_prob: float):
    if max_prob >= 0.80:
        colour, label = "#22c55e", "High model confidence"
    elif max_prob >= 0.60:
        colour, label = "#eab308", "Moderate model confidence"
    else:
        colour, label = "#ef4444", "Low model confidence"

    return f"""
    <div style="
        background-color:{colour};
        padding:6px 12px;
        border-radius:999px;
        display:inline-block;
        font-weight:600;
        margin-top:6px;
    ">
        {label} ({max_prob:.2f})
    </div>
    """


# -------------------------------------------------------------
# 🚀 MAIN APP WITH HUMAN MODE / EXPERT MODE TOGGLE
# -------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Fungal Acne Classifier (10-Class)",
        page_icon="🧴",
        layout="centered",
    )

    st.title("🧴 Fungal Acne Ingredient Classifier")
    st.write("Paste your ingredients below.")

    text = st.text_area(
        "Ingredients",
        height=160,
        placeholder="e.g. Aqua, Glycerin, Niacinamide, Panthenol ..."
    )

    # 🔥 HUMAN MODE / EXPERT MODE SWITCH
    expert_mode = st.toggle("Enable Expert Mode (advanced model details)", value=False)

    analyze_clicked = st.button("Analyze")

    if not analyze_clicked:
        return

    if text.strip() == "":
        st.warning("Please enter an ingredient list.")
        return

    # Load model + compute prediction
    model = load_model()
    clean_text = join_ingredients_for_model(text)

    pred_label = model.predict([clean_text])[0]
    pred_probs = model.predict_proba([clean_text])[0]
    classes = model.classes_

    ingredients = split_ingredients(text)
    score = calculate_safety_score(text)
    highlighted_html = highlight_ingredients_html(ingredients)

    info = LABEL_INFO[pred_label]
    display_name = info["name"]
    desc = info["desc"]

    # ---------------------------------------------------------
    # HUMAN MODE OUTPUT
    # ---------------------------------------------------------
    st.subheader("🌟 Result Summary (Human Mode)")
    st.markdown(f"### **Result: {display_name}**")
    st.caption(desc)

    st.markdown(f"### Fungal Acne Score: **{score}/10**")
    st.write(generate_explanation(ingredients, score))

    st.subheader("🧬 Ingredient Breakdown")
    for block in highlighted_html:
        st.markdown(block, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # EXPERT MODE OUTPUT
    # ---------------------------------------------------------
    if expert_mode:
        st.divider()
        st.subheader("🧠 Expert Mode — Model Details")

        # Confidence badge
        max_prob = float(np.max(pred_probs))
        st.markdown(get_confidence_badge(max_prob), unsafe_allow_html=True)

        # Probabilities
        st.write("### Class Probabilities")
        sorted_idx = pred_probs.argsort()[::-1]
        for idx in sorted_idx:
            st.write(f"- **{classes[idx]}**: {pred_probs[idx]:.3f}")

        prob_df = pd.DataFrame({
            "label": classes[sorted_idx],
            "probability": pred_probs[sorted_idx]
        }).set_index("label")
        st.bar_chart(prob_df)

        # LIME explanation
        with st.expander("🔎 LIME Explanation (Why the model predicted this)"):
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

            top_label_idx = exp.top_labels[0]
            top_label_name = classes[top_label_idx]
            st.write(f"Top label explained: **{top_label_name}**")

            lime_df = pd.DataFrame(exp.as_list(label=top_label_idx),
                                   columns=["feature", "weight"])
            st.dataframe(lime_df)


# -------------------------------------------------------------
if __name__ == "__main__":
    main()
