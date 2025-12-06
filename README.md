# DermaLens — Skincare Intelligence Platform

DermaLens is a premium Streamlit dashboard for ingredient intelligence, fungal-acne risk scoring, product similarity, and real-world shopping signals. It blends ML classification, sentence-embedding search, explainability, barcode scanning, and lightweight web scraping into one clean experience.

## Highlights
- Modern dual-theme UI with DermaLens branding, hero header, and dashboard cards.
- TF-IDF multiclass classifier plus a fungal-acne risk engine with unsafe/mild ingredient highlighting.
- BERT sentence embeddings for product and ingredient similarity.
- Auto ingredient lookup by product name (DuckDuckGo HTML scrape).
- Barcode scanning via the camera (`pyzbar` + Pillow) to detect products and fetch ingredients automatically.
- Store availability checker (Boots UK + Superdrug) running in the background and cached.
- Favourites system (`data/user_favourites.jsonl`) with quick “view again” replays.
- Expert Mode with LIME explanations and probability charts.
- PDF export and shareable text summary for any analysis.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Optional: plug in a webcam for the Scan Product tab.

## Project Structure
- `src/app.py` — Streamlit UI (tabs, dark/light themes, favourites, sharing, barcode flow).
- `src/analysis_engine.py` — ML pipeline, scoring, similarity, PDF generation.
- `src/embeddings_utils.py` — sentence-transformer loading, similarity helpers, user memory.
- `src/ingredient_similarity.py` — compatibility wrapper for embedding helpers.
- `src/ingredient_lookup.py` — product-name web search → ingredient extraction.
- `src/store_availability.py` — Boots/Superdrug HTML scraping with caching.
- `src/barcode_scanner.py` — barcode decoding and quick product lookup.
- `src/user_favourites.py` — JSONL persistence for favourites.
- `data/` — seed memory and user history (`user_product_memory.jsonl`, `user_favourites.jsonl`).
- `models/` — TF-IDF model + ingredient embeddings.

## Usage
1. Go to the **Analyze** tab, enter a product name and ingredients, or upload a label image.
2. Click **Auto-fetch ingredients** to pull a list from web search when you only know the product name.
3. Hit **Analyze with DermaLens** to get the prediction, fungal-acne score, ingredient chips, similar products, and availability.
4. Toggle **Expert Mode** to view LIME explanations and probability breakdowns.
5. Save results with **⭐ Add to favourites**; revisit them in the **Favourites** tab (with a “View again” action).
6. Use **Scan Product (Camera)** to decode barcodes and auto-fetch ingredients before analyzing.

## Testing
```bash
pytest
```

## Notes
- Scraping is best-effort and network-dependent; results are cached per product name.
- Barcode lookups rely on publicly searchable metadata; manual input is still available via the Analyze tab.
- Favourites and user memory are stored locally in `data/` for privacy and portability.
