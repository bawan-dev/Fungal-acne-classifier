🧴 Fungal Acne Ingredient Classifier
A full ML/NLP + Embedding-powered system for skincare ingredient analysis, similarity search & risk scoring.

This project is a Streamlit-based machine learning app that analyzes skincare ingredient lists, predicts fungal-acne safety, highlights risky ingredients, and recommends similar products using BERT embeddings.
It also supports OCR input, PDF report generation, product memory storage, and expert-mode explainability via LIME.

The classifier is designed to feel like a modern, user-friendly skincare analysis tool — while showcasing advanced ML engineering under the hood.

🚀 Key Features
🔬 ML Classification (10 classes)

TF-IDF + Logistic Regression model (10-label classifier).

Expert mode unlocks:

Probability breakdowns

LIME explanations

Interpretable feature influences

🧠 BERT Ingredient Embeddings

Uses sentence-transformers (MiniLM-L6-v2) to embed:

Ingredient lists

Individual ingredients

Full products

Supports:

Similar product recommendations

Ingredient-level nearest-neighbor insights

Per-ingredient replacements & similarity scores

🧪 Fungal Acne Risk Engine

Detects fatty acids, esters, polysorbates & known triggers.

Generates a 0–10 fungal-acne safety score.

Produces a user-friendly explanation.

💾 Product Memory 2.0

Each processed product is saved with:

Name

Ingredient list

Embeddings

Timestamp

Users can reload past analyses instantly without recomputation.

📄 PDF Export

Generate a clean report that includes:

Scores

Predictions

Breakdown

Similar product recommendations

Ingredient insights

Optional LIME figures

🖼️ OCR Image Input (Stubbed)

Upload a product label image.

OCR pipeline is stubbed for local/offline usage.

Ready for Tesseract integration.

🔍 Brand Auto-Detection (Stub)

Stub function for future web search integration.

Plug in DuckDuckGo API / scrape flow later.

🎨 Modern UI

Multi-tab interface:

Overview

Ingredients

Similar Products

Expert Mode

Ingredient chips grouped by safety categories.

Responsive, mobile-friendly layout.

⚙️ Installation
git clone https://github.com/yourname/fungal-acne-classifier.git
cd fungal-acne-classifier
pip install -r requirements.txt
streamlit run src/app.py

📁 Project Structure
src/
│── app.py                  # Streamlit UI
│── analysis_engine.py      # Core logic + scoring + PDF + stubs
│── embeddings_utils.py     # Embedding loading, similarity search
│── ingredient_similarity.py# Thin wrapper for backward compatibility
│── ingredient_embeddings.py# Script to regenerate embeddings
│── preprocessing.py        # Ingredient cleaning utilities
│── safety_score.py         # Fungal acne scoring logic
data/
│── product_memory.csv      # Seed database
│── user_product_memory.jsonl # Stored past analyses
models/
│── tfidf_multiclass_model.joblib
│── ingredient_embeddings.pt

🧬 Rebuilding the Embeddings

If you update data/product_memory.csv:

python src/ingredient_embeddings.py


This regenerates models/ingredient_embeddings.pt.

🧪 Running Tests
pytest


Tests cover:

Parsing

Similarity helpers

Fake prediction pipelines

OCR stub behavior

🧱 Architecture Overview
1. Ingredient → Model Input Pipeline

Cleans and normalizes text.

Joins multi-ingredient lists.

Predicts ML class + probabilities.

Computes fungal acne safety score.

2. BERT Embedding Engine

Loads embeddings safely under PyTorch 2.6+.

Handles 1D/2D tensors reliably.

Supports both product-level and ingredient-level similarity.

3. Streamlit Interface

Multi-tab design

Ingredient chip renderer

History loader

PDF exporter

Expert mode LIME renderer

4. Future-Ready Stubs

Search-based ingredient auto-fetch

OCR via Tesseract

Expandable memory system

Real-time product scanning

🧠 Why This Project Is Impressive (For Recruiters)

This project demonstrates skills in:

Machine Learning
TF-IDF, multiclass classification, explainability.

NLP & Embeddings
BERT similarity search, cosine distance ranking.

Data Engineering
Product memory persistence, embedding caching.

Software Engineering
Clean module structure, safe PyTorch loading, test suite.

Full-stack ML App Development
Streamlit frontend, PDF export, OCR input handling.

Your repository now legitimately looks like something a junior ML engineer or even mid-level would ship.

📌 Author

Built with ❤️ by Bawan, for educational, skincare analysis, and ML demonstration purposes.