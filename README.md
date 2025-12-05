# 🧴 Fungal Acne Ingredient Classifier (10-Class ML Model)

A machine learning web app that analyzes skincare product ingredient lists,  
classifies them into **10 dermatological categories**, and generates a  
**fungal acne (Malassezia) safety score** with full explanations.

🔗 **Live Demo:** https://fungal-acne-classifier-6jla5qnhtv8itpkkb5nfhw.streamlit.app/  
_(replace with your actual app link)_

---

## 🚀 Features

### 🔍 Ingredient Classification (10 Categories)
The model classifies ingredients into:

- `safe`
- `neutral`
- `malassezia_trigger`
- `comedogenic`
- `irritant`
- `fragrance_heavy`
- `fatty_acid`
- `emollient_heavy`
- `surfactant`
- `preservative`

Model trained on **1,000+ synthetic labeled samples** using TF-IDF + Logistic Regression (multi-class).

---

## 🧪 Fungal Acne Safety Score (0–10)

The app computes a safety score using domain knowledge:
- Detects fatty acids, esters, polysorbates, sorbitan compounds
- Flags questionable ingredients like dimethicone or caprylic triglyceride
- Highlights each ingredient in **green (safe)**, **yellow (neutral)** or **red (trigger)**

---

## 🧠 Explainability (LIME)

The app includes an **Expert Mode** that shows:
- Model probabilities for all 10 classes  
- A bar chart of confidence levels  
- LIME explanation table for the predicted label  
This gives transparency into *why* the model predicted a certain category.

---

## 🖼 Screenshots

### 🏠 Home Screen
*(Screenshot 1 goes here)*

### 🧴 Example Ingredient Analysis
*(Screenshot 2 goes here)*

### 🔎 LIME Explainability View
*(Screenshot 3 goes here)*

---

## 📦 Tech Stack

- Python 3.10  
- Streamlit (UI)  
- Scikit-learn (ML model)  
- LIME (explainability)  
- Pandas / NumPy  
- Joblib (model saving/loading)

---

## 🏗 How It Works

### 1️⃣ Ingredient Preprocessing  
Ingredients are split, normalized, and cleaned.

### 2️⃣ TF-IDF Vectorization  
Text converted into numerical features using word n-grams.

### 3️⃣ Multiclass Logistic Regression  
Model predicts the most likely ingredient category.

### 4️⃣ Safety Score Algorithm  
Domain-based heuristic adds up:
- High-risk fatty acids
- Esters
- Polysorbates
- Sorbitan compounds
- Comedogenic ingredients

### 5️⃣ Streamlit Front-End  
Interactive web UI for live predictions.

---

## 🔧 Installation (Local)
git clone https://github.com/bawan-dev/Fungal-acne-classifier.git

cd Fungal-acne-classifier
pip install -r requirements.txt
streamlit run src/app.py

---

## ✨ Future Improvements

- Live product database scanning  
- Barcode image detection  
- Ingredient embedding models (BERT or BioBERT)  
- Real skincare dataset training  
- User accounts + saved scans  

---

## 👤 Author

**Bawan Sabah**  
Machine Learning & Robotics Student  
GitHub: https://github.com/bawan-dev  

