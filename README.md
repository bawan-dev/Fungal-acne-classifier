# Fungal Acne Ingredient Classifier

A Streamlit app that classifies skincare ingredient lists, highlights fungal-acne risk, and surfaces similar products with BERT embeddings. Now includes offline-friendly stubs for brand auto-detection and OCR, PDF export, and a refreshed UI with history.

## Features
- TF-IDF + Logistic Regression (10 classes) with LIME explainability.
- Tabs: Overview, Ingredients, Similar Products, Expert Mode.
- Grouped ingredient chips (safe, mild, unsafe) with subtle animations.
- Product Memory 2.0: stores analyzed products (name, ingredients, embedding, timestamp) and lets you reload instantly.
- Similar products and ingredient-level similarity using sentence-transformers.
- PDF export (in-memory) with summary, scores, similarities, and optional LIME image.
- Brand auto-detection stub (`fetch_product_ingredients`) ready for future networked implementation.
- Screenshot/OCR stub (`extract_ingredients_from_image`) so the UI flow works without Tesseract.

## Quickstart
```bash
git clone https://github.com/yourname/fungal-acne-classifier.git
cd fungal-acne-classifier
pip install -r requirements.txt
streamlit run src/app.py
```

## Requirements
- Python 3.9+
- See `requirements.txt` (includes `torch`, `sentence-transformers`, `fpdf2`, `lime`, `streamlit`).
- Optional (not installed here): Tesseract OCR and a search backend for brand auto-detection.

## Usage
1. Enter a product name (optional) and paste ingredients (comma-separated).  
2. Optionally upload an ingredient label image (uses OCR stub) or click auto-detect (brand fetch stub).  
3. Click **Analyze**. Results populate across the tabs:
   - **Overview**: prediction, fungal acne score, explanation.
   - **Ingredients**: grouped chips and ingredient-level similarity.
   - **Similar Products**: nearest neighbors via embeddings.
   - **Expert Mode**: probability chart + LIME table (when toggled).
4. Download the PDF report.  
5. Access **Previously Analyzed Products** to reload past runs instantly (no reprocessing).

## Architecture
- `src/app.py` — Streamlit UI, tabs, history UI, PDF download hook.
- `src/analysis_engine.py` — orchestrates predictions, safety scoring, similarity, PDF generation, stubs for brand fetch/OCR.
- `src/embeddings_utils.py` — embedding loading, similarity utilities, and product memory storage.
- `src/ingredient_similarity.py` — thin wrapper to stay backward-compatible.
- `src/ingredient_embeddings.py` — script to regenerate base ingredient embeddings.
- `data/product_memory.csv` — seeded product memory; `data/user_product_memory.jsonl` stores new analyses.

## Stubs and Offline Behavior
- `fetch_product_ingredients(name)` returns an offline placeholder; wire it to DuckDuckGo/requests-html later.
- `extract_ingredients_from_image(image)` returns a mocked ingredient string; swap in Tesseract once installed.

## Tests
Run the lightweight test suite:
```bash
pytest
```
Tests cover parsing, similarity helper behavior, predict pipeline with fakes, and OCR stub.

## Regenerating Embeddings
If you update `data/product_memory.csv`, regenerate embeddings:
```bash
python src/ingredient_embeddings.py
```
This writes `models/ingredient_embeddings.pt` for the similarity search.

## Notes
- Keep `models/tfidf_multiclass_model.joblib` in place; the app expects a 10-class model.
- PDF export uses `fpdf2` and embeds LIME images when available.
