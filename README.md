<<<<<<< HEAD
🧴 Fungal Acne Ingredient Classifier
A 10-Class Machine Learning Model for Skincare Safety Analysis

This project is a machine learning-powered web app that:

Classifies skincare formulas into 10 ingredient categories

Evaluates fungal acne (Malassezia) safety

Provides ingredient-by-ingredient breakdowns

Includes Human Mode (simple explanations) + Expert Mode (LIME, probabilities, model confidence)

Runs completely in Streamlit

🚀 Features
🧠 Multi-Class ML Model

Trained on 1,000+ curated examples with 10 labels:

safe

neutral

malassezia_trigger

comedogenic

irritant

fragrance_heavy

fatty_acid

emollient_heavy

surfactant

preservative

🔍 Human Mode

Clear skincare explanation

Fungal acne score (0–10)

Ingredient breakdown (safe / mild / high-risk)

🧪 Expert Mode

Class probabilities

Confidence badge

LIME interpretability

Bar charts & feature weights

ML explainability for recruiters

🌐 Deployable Anywhere

Streamlit Cloud

HuggingFace Spaces

Local usage (streamlit run src/app.py)

🗂 Project Structure
src/
│   app.py                  # Streamlit UI
│   preprocessing.py        # Ingredient cleaning and parsing
│   safety_score.py         # Fungal acne scoring logic
│   train_multiclass.py     # Model training script
│
models/
│   tfidf_multiclass_model.joblib
│
data/
│   ingredients_multilabel.csv

🛠 Installation
git clone https://github.com/bawans-dev/fungal-acne-classifier.git
cd fungal-acne-classifier
pip install -r requirements.txt
streamlit run src/app.py

🧴 Example Output

(Add screenshots here after deployment)

📦 Deployment
Streamlit Cloud

Push to GitHub → Create new app → Select src/app.py

HuggingFace Spaces

Create Space → Select “Streamlit” → Upload repo →

📄 License

MIT License (recommended)

✨ Author

Bawan Sabah – Machine Learning & Applied AI
=======
# Fungal-acne-classifier
Machine learning app that classifies skincare ingredients into 10 categories and evaluates fungal acne safety.
>>>>>>> 0b1c32e1d3b59f0f7b6b7ac1d6052b637b11b2e7
