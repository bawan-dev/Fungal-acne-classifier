import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
from lime.lime_text import LimeTextExplainer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis_engine import (  # noqa: E402
    AnalysisEngine,
    NEUTRAL_RISK,
    UNSAFE_KEYWORDS,
    fetch_product_ingredients,
    extract_ingredients_from_image,
)
from src.preprocessing import join_ingredients_for_model  # noqa: E402
from src import barcode_scanner, store_availability, user_favourites  # noqa: E402

st.set_page_config(
    page_title="DermaLens | Skincare Intelligence",
    page_icon="DL",
    layout="wide",
    initial_sidebar_state="expanded",
)

EXECUTOR = ThreadPoolExecutor(max_workers=2)


@st.cache_resource
def get_engine() -> AnalysisEngine:
    return AnalysisEngine()


def ensure_array(probs):
    if isinstance(probs, np.ndarray):
        return probs
    return np.array(probs)


def safe_rerun():
    """
    Streamlit rerun helper for compatibility across versions.
    """
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if callable(rerun_fn):
        rerun_fn()


def make_placeholder_image(label: str) -> bytes:
    """
    Create a small placeholder image with initials for favourites.
    """
    initials = "".join([part[0].upper() for part in label.split()[:2]]) or "DL"
    base_val = sum(ord(c) for c in label) % 255
    color = (80 + base_val % 120, 140, 120 + (base_val * 2) % 135)
    img = Image.new("RGB", (140, 140), color=color)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), initials, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(((140 - text_w) / 2, (140 - text_h) / 2), initials, fill="white", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


def init_state():
    defaults = {
        "analysis_result": None,
        "lime_image": None,
        "share_payload": "",
        "availability_future": None,
        "availability_result": None,
        "ingredients_input": "",
        "product_name_input": "",
        "dark_mode": True,
        "show_share": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_brand_styles(dark_mode: bool):
    bg = "#0f172a" if dark_mode else "#f7f8fb"
    surface = "#111827" if dark_mode else "#ffffff"
    text = "#e5e7eb" if dark_mode else "#111827"
    muted = "#9ca3af" if dark_mode else "#4b5563"
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap');
        html, body, [class*="css"]  {{
            font-family: 'Manrope', sans-serif;
            background: {bg};
            color: {text};
        }}
        .dermalens-hero {{
            background: linear-gradient(120deg, #0ea5e9, #20c997);
            border-radius: 16px;
            padding: 18px 20px;
            color: #f8fafc;
            box-shadow: 0 18px 36px rgba(0,0,0,0.15);
        }}
        .dl-card {{
            background: {surface};
            border: 1px solid rgba(255,255,255,0.04);
            border-radius: 16px;
            padding: 18px;
            box-shadow: 0 16px 32px rgba(15,23,42,0.15);
        }}
        .chip {{
            padding:6px 10px;
            border-radius:999px;
            display:inline-flex;
            align-items:center;
            font-size:0.9rem;
            background: rgba(255,255,255,0.06);
            border:1px solid rgba(255,255,255,0.06);
        }}
        .mini-badge {{
            display:inline-flex;
            align-items:center;
            gap:6px;
            padding:6px 10px;
            border-radius:10px;
            background: rgba(255,255,255,0.08);
            color:{text};
            font-size:0.9rem;
        }}
        .logo-mark {{
            width: 44px;
            height: 44px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: radial-gradient(circle at 20% 20%, #67e8f9, #0ea5e9, #075985);
            font-weight: 700;
            color: #0b1021;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }}
        .fade-card {{
            background: {surface};
            border-radius:14px;
            padding:14px;
            border:1px solid rgba(255,255,255,0.05);
        }}
        .muted {{
            color:{muted};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header():
    st.markdown(
        """
        <div class="dermalens-hero">
            <div style="display:flex; align-items:center; justify-content:space-between;">
                <div style="display:flex; gap:14px; align-items:center;">
                    <div class="logo-mark">DL</div>
                    <div>
                        <div style="font-size:24px; font-weight:700;">DermaLens</div>
                        <div style="opacity:0.9;">Full-stack skincare intelligence for safer routines.</div>
                    </div>
                </div>
                <div class="mini-badge">💎 Powered by ML + Embeddings</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chips(groups):
    palette = {
        "safe": ("#0f766e", "#22c55e"),
        "mild": ("#b45309", "#eab308"),
        "unsafe": ("#b91c1c", "#ef4444"),
    }
    for category, items in groups.items():
        if not items:
            continue
        _, border = palette[category]
        st.markdown(f"**{category.title()}**")
        chip_html = ""
        for ing in items:
            chip_html += f"""
            <div class="chip" style="
                border:1px solid {border};
            ">{ing}</div>
            """
        st.markdown(
            f"""
            <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px;">
                {chip_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_similar_products(similar_products):
    if not similar_products:
        st.info("No similar products found yet.")
        return
    for item in similar_products:
        with st.container():
            st.markdown(
                f"**{item['product_name']}** · similarity `{item['score']:.2f}`",
            )
            preview = item.get("ingredients", "")
            if preview:
                st.caption(preview[:140] + ("..." if len(preview) > 140 else ""))


def render_ingredient_insights(insights):
    if not insights:
        st.info("No ingredient-level matches found.")
        return
    for ing, sim in insights.items():
        st.markdown(f"**{ing}** ↔ **{sim['closest_ingredient']}** (`{sim['score']:.2f}`)")


def render_previous_products(engine: AnalysisEngine, use_sidebar: bool = False):
    entries = engine.get_previous_results()
    target = st.sidebar if use_sidebar else st
    if not entries:
        target.write("No previously analyzed products yet.")
        return None

    options = [f"{e.get('product_name','Untitled')} ({e.get('timestamp','')})" for e in entries]
    choice = target.selectbox("History", ["Select..."] + options, key="history_select")
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


def build_share_payload(result: Dict) -> str:
    return (
        f"DermaLens analysis — {result.get('product_name', 'Product')}\n"
        f"Fungal Acne Score: {result.get('safety_score', '?')}/10\n"
        f"Model Prediction: {result.get('tfidf', {}).get('label', '?')}\n"
        f"Top risks: {', '.join(result.get('highlight_groups', {}).get('unsafe', []) or ['none flagged'])}\n"
        f"Ingredients: {result.get('ingredients_raw', '')}"
    )


def trigger_availability_check(product_name: str):
    if not product_name:
        st.session_state.availability_result = []
        st.session_state.availability_future = None
        return
    st.session_state.availability_result = None
    st.session_state.availability_future = EXECUTOR.submit(
        store_availability.check_store_availability, product_name
    )


def perform_analysis(engine: AnalysisEngine, ingredients_text: str, product_name: str):
    placeholder = st.empty()
    with placeholder, st.spinner("Running DermaLens pipelines..."):
        progress = st.progress(0)
        for pct in [0.25, 0.6, 1.0]:
            time.sleep(0.08)
            progress.progress(int(pct * 100))
        result = engine.analyze(ingredients_text, product_name=product_name or None)
    st.session_state.analysis_result = result
    st.session_state.lime_image = None
    st.session_state.share_payload = build_share_payload(result)
    st.session_state.show_share = False
    trigger_availability_check(result.get("product_name", ""))


def render_availability():
    availability = st.session_state.get("availability_result")
    future = st.session_state.get("availability_future")
    if availability is None and future is not None:
        if future.done():
            try:
                st.session_state.availability_result = future.result()
                availability = st.session_state.availability_result
            except Exception as exc:
                availability = [
                    {"store": "Boots", "available": False, "error": str(exc), "price": None, "link": None}
                ]
            safe_rerun()
        else:
            st.info("Checking Boots and Superdrug stock in the background...")
            return

    if availability is None:
        st.info("Availability will show here once a product is analyzed.")
        return

    if availability == []:
        st.info("No availability data yet.")
        return

    for item in availability:
        status = "In stock" if item.get("available") else "Unavailable"
        price = item.get("price") or "-"
        link = item.get("link") or ""
        error = item.get("error")
        st.markdown(f"**{item.get('store', 'Store')}** - {status} - {price}")
        if link:
            st.caption(f"[Open store link]({link})")
        if error:
            st.caption(f"Note: {error}")


def render_actions(engine: AnalysisEngine, result: Dict, key_prefix: str = "analysis"):
    cols = st.columns(3)
    with cols[0]:
        if st.button("⭐ Add to favourites", key=f"fav_{key_prefix}"):
            payload = {**result, "fa_risk": result.get("tfidf", {}).get("label")}
            user_favourites.add_favourite(payload)
            st.success("Saved to favourites.")
    with cols[1]:
        if st.button("Share this analysis", key=f"share_{key_prefix}"):
            st.session_state["show_share"] = True
    with cols[2]:
        pdf_buffer = engine.generate_pdf_report(result, lime_image=st.session_state.lime_image)
        if pdf_buffer:
            st.download_button(
                label="Download PDF report",
                data=pdf_buffer,
                file_name="dermalens_report.pdf",
                mime="application/pdf",
                key=f"pdf_{key_prefix}",
            )


def render_analysis(engine: AnalysisEngine, result: Dict, expert_mode: bool, key_prefix: str = "analysis"):
    probs = ensure_array(result["tfidf"]["probs"])
    classes = result["tfidf"]["classes"]
    sorted_idx = probs.argsort()[::-1]
    max_prob = float(np.max(probs)) if len(probs) else 0.0

    summary_tab, ingredients_tab, expert_tab = st.tabs(
        ["Summary", "Ingredients", "Expert Mode"]
    )

    with summary_tab:
        col_pred, col_score, col_avail = st.columns([1.2, 1, 1])
        with col_pred:
            st.markdown("#### Prediction", unsafe_allow_html=True)
            st.markdown(
                f"<div class='fade-card'><div style='font-size:32px;font-weight:700;'>{result['tfidf']['label']}</div><div class='muted'>Confidence {max_prob:.2f}</div></div>",
                unsafe_allow_html=True,
            )
        with col_score:
            st.markdown("#### Fungal Acne Score")
            score_value = min(max(float(result["safety_score"]), 0), 10)
            st.progress(int(score_value * 10))
            st.caption(result["explanation"])
        with col_avail:
            st.markdown("#### Availability")
            render_availability()

        st.markdown("#### Similar Products")
        render_similar_products(result.get("similar_products", []))
        render_actions(engine, result, key_prefix=key_prefix)

    with ingredients_tab:
        st.markdown("#### Ingredient Chips")
        render_chips(result.get("highlight_groups", {}))
        st.markdown("#### Ingredient Insights")
        render_ingredient_insights(result.get("ingredient_similarities", {}))

    with expert_tab:
        st.markdown("#### Probability Distribution")
        prob_df = pd.DataFrame(
            {
                "label": np.array(classes)[sorted_idx],
                "probability": probs[sorted_idx],
            }
        ).set_index("label")
        st.bar_chart(prob_df)

        if expert_mode:
            st.markdown("#### LIME Explanation")
            top_label_name, lime_df, lime_image = run_lime_if_requested(
                engine, result["ingredients_raw"], classes, engine.model
            )
            st.write(f"Top label explained: **{top_label_name}**")
            st.dataframe(lime_df)
            st.session_state.lime_image = lime_image
        else:
            st.info("Toggle Expert Mode to run LIME explanations.")


def render_scan_tab(engine: AnalysisEngine):
    st.markdown("Capture a product barcode to auto-fill a search.")
    camera_image = st.camera_input("Scan Product (Camera)")
    if camera_image is not None:
        codes = barcode_scanner.decode_barcodes(camera_image)
        if not codes:
            st.warning("No barcode detected. Try again with better lighting.")
        else:
            code = codes[0]
            st.success(f"Detected barcode: {code}")
            lookup = barcode_scanner.lookup_product_from_barcode(code)
            product_guess = lookup.get("product_name", "")
            st.info(lookup.get("message", ""))
            if product_guess:
                st.markdown(f"**Guessed product:** {product_guess}")
                fetched = fetch_product_ingredients(product_guess)
                if fetched.get("ingredients"):
                    st.session_state.ingredients_input = fetched["ingredients"]
                    st.session_state.product_name_input = product_guess
                    st.success("Ingredients pulled from web search. Ready to analyze.")
                    if st.button("Analyze scanned product"):
                        perform_analysis(engine, fetched["ingredients"], product_guess)

    st.markdown("---")
    st.markdown("Manual fallback if scanning is not available or fails.")
    manual_product = st.text_input(
        "Manual product name (Scan tab)",
        key="scan_manual_product",
    )
    manual_ingredients = st.text_area(
        "Manual ingredients (Scan tab)",
        height=120,
        key="scan_manual_ingredients",
    )
    if st.button("Analyze manual entry", key="scan_manual_button"):
        text = manual_ingredients.strip()
        if not text:
            st.warning("Provide ingredients to analyze.")
        else:
            st.session_state.ingredients_input = text
            st.session_state.product_name_input = manual_product
            perform_analysis(engine, text, manual_product or "Scanned product")


def render_favourites_tab(engine: AnalysisEngine):
    favourites = user_favourites.load_favourites()
    if not favourites:
        st.info("No favourites saved yet. Analyze a product and star it to save.")
        return

    for fav in favourites:
        with st.container():
            cols = st.columns([1, 3, 1])
            with cols[0]:
                img_bytes = fav.get("image") or make_placeholder_image(fav.get("product_name", "DL"))
                st.image(img_bytes, width=80)
            with cols[1]:
                st.markdown(f"**{fav.get('product_name','Untitled')}**")
                st.caption(f"Score: {fav.get('safety_score','?')}/10 · Prediction: {fav.get('tfidf', {}).get('label', fav.get('fa_risk','?'))}")
            with cols[2]:
                if st.button("View again", key=f"view_{fav.get('timestamp','')}_{fav.get('product_name','')}"):
                    analysis = fav.get("analysis") or fav
                    st.session_state.analysis_result = analysis
                    st.session_state.share_payload = build_share_payload(analysis)
                    trigger_availability_check(analysis.get("product_name", ""))
                    st.success("Loaded favourite into the analyzer.")


def render_sidebar(engine: AnalysisEngine):
    st.sidebar.markdown("### Controls")
    st.sidebar.toggle("Dark mode", key="dark_mode")
    st.sidebar.markdown("#### Expert Mode")
    st.sidebar.toggle("Enable LIME explanations", key="expert_mode", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Quick History")
    cached = render_previous_products(engine, use_sidebar=True)
    if cached:
        st.session_state.analysis_result = cached
        st.session_state.share_payload = build_share_payload(cached)
        trigger_availability_check(cached.get("product_name", ""))
        st.sidebar.success("Loaded from history.")


def render_share_section():
    if st.session_state.get("show_share") and st.session_state.get("share_payload"):
        st.code(st.session_state.share_payload, language="text")
        st.caption("Copy and share this summary anywhere.")


def render_input_panel(engine: AnalysisEngine):
    st.markdown("#### Analyze a product")
    st.text_input(
        "Product name",
        placeholder="e.g. CeraVe Moisturizing Lotion",
        key="product_name_input",
    )
    st.text_area(
        "Ingredients",
        placeholder="Aqua, Glycerin, Niacinamide, Panthenol...",
        height=180,
        key="ingredients_input",
    )
    upload = st.file_uploader("Upload ingredient label (image)", type=["png", "jpg", "jpeg"])
    if upload is not None:
        extracted = extract_ingredients_from_image(upload)
        if extracted:
            st.session_state.ingredients_input = extracted
            st.info(f"OCR detected ingredients: {extracted}")

    cols = st.columns([1, 1, 1])
    with cols[0]:
        if st.button("Auto-fetch ingredients"):
            name = st.session_state.get("product_name_input") or st.session_state.get("ingredients_input")
            fetched = fetch_product_ingredients(name or "")
            st.toast(fetched.get("message", "Done"))
            if fetched.get("ingredients"):
                st.session_state.ingredients_input = fetched["ingredients"]
    with cols[1]:
        if st.button("Analyze with DermaLens", type="primary"):
            text = st.session_state.get("ingredients_input", "")
            if not text.strip():
                st.warning("Please provide an ingredient list first.")
            else:
                perform_analysis(
                    engine,
                    text,
                    st.session_state.get("product_name_input", "Untitled Product"),
                )
    with cols[2]:
        st.markdown(" ")
        st.markdown(" ")
        st.caption("Add a name to unlock store availability checks.")


def main():
    init_state()
    engine = get_engine()
    apply_brand_styles(st.session_state.get("dark_mode", True))
    render_sidebar(engine)
    render_brand_header()

    tab_analyze, tab_scan, tab_favourites = st.tabs(
        ["Analyze", "Scan Product (Camera)", "Favourites"]
    )

    with tab_analyze:
        render_input_panel(engine)
        result = st.session_state.get("analysis_result")
        if result:
            render_analysis(engine, result, st.session_state.get("expert_mode", False), key_prefix="analysis_main")
            render_share_section()

    with tab_scan:
        render_scan_tab(engine)
        result = st.session_state.get("analysis_result")
        if result:
            st.markdown("---")
            st.markdown("### Latest Analysis")
            render_analysis(engine, result, st.session_state.get("expert_mode", False), key_prefix="analysis_scan")

    with tab_favourites:
        render_favourites_tab(engine)


if __name__ == "__main__":
    main()
