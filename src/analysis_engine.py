import datetime
import io
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch

try:
    from fpdf import FPDF  # type: ignore
except ImportError:  # pragma: no cover
    FPDF = None

from src import embeddings_utils
from src import ingredient_lookup
from src.preprocessing import join_ingredients_for_model, split_ingredients
from src.safety_score import calculate_safety_score

# Shared keyword lists for categorisation
UNSAFE_KEYWORDS = [
    "lauric acid",
    "myristic acid",
    "stearic acid",
    "oleic acid",
    "isopropyl myristate",
    "cetyl alcohol",
    "cetearyl alcohol",
    "glyceryl stearate",
    "polysorbate",
    "sorbitan",
]

NEUTRAL_RISK = [
    "dimethicone",
    "caprylic/capric triglyceride",
    "fragrance",
]


class AnalysisEngine:
    def __init__(self, model_path: str = "models/tfidf_multiclass_model.joblib"):
        self.model_path = model_path
        self.model = self._load_model()
        self.sentence_model = embeddings_utils.load_sentence_model()
        self.product_names: List[str] = []
        self.ingredient_lists: List[str] = []
        self.embeddings: torch.Tensor = torch.empty((0, 0))
        self.flat_ingredients: List[str] = []
        self.flat_embeddings: torch.Tensor = torch.empty((0, 0))
        self.user_entries: List[Dict] = []
        self.refresh_memory()

    def _load_model(self):
        model = joblib.load(self.model_path)
        if not hasattr(model, "classes_") or len(model.classes_) != 10:
            raise ValueError(
                f"Incorrect model loaded from {self.model_path}. "
                f"Expected 10-class model but got {len(getattr(model, 'classes_', []))} classes.\n"
                "Make sure ONLY tfidf_multiclass_model.joblib exists in /models."
            )
        return model

    def refresh_memory(self):
        base_names, base_ing, base_embeds = embeddings_utils.load_base_product_memory(self.sentence_model)
        user_names, user_ing, user_embeds, user_entries = embeddings_utils.load_user_memory()

        self.product_names = [*base_names, *user_names]
        self.ingredient_lists = [*base_ing, *user_ing]
        if base_embeds.numel() == 0 and user_embeds.numel() == 0:
            self.embeddings = torch.empty((0, self.sentence_model.get_sentence_embedding_dimension()), dtype=torch.float32)
        elif base_embeds.numel() == 0:
            self.embeddings = user_embeds
        elif user_embeds.numel() == 0:
            self.embeddings = base_embeds
        else:
            self.embeddings = torch.cat([base_embeds, user_embeds], dim=0)

        self.flat_ingredients, self.flat_embeddings = embeddings_utils.build_flat_ingredient_embeddings(
            self.ingredient_lists,
            self.sentence_model,
            device=self.embeddings.device if self.embeddings.numel() > 0 else None,
        )
        self.user_entries = user_entries

    def generate_explanation(self, ingredients: List[str], score: int) -> str:
        ingredients_lower = [i.lower() for i in ingredients]

        detected_strong = [
            bad for bad in UNSAFE_KEYWORDS if any(bad in ing for ing in ingredients_lower)
        ]
        detected_mild = [
            mid for mid in NEUTRAL_RISK if any(mid in ing for ing in ingredients_lower)
        ]

        if score >= 8:
            explanation = (
                "This product appears low risk for fungal acne. "
                "No major fungal acne triggers were detected."
            )
        elif score >= 5:
            explanation = (
                "This product has a moderate risk rating. "
                "Some ingredients may cause issues for sensitive or acne-prone skin."
            )
            if detected_mild:
                explanation += f" Mild-risk ingredients: {', '.join(detected_mild)}."
        else:
            explanation = (
                "This product is rated high risk for fungal acne. "
                "It contains fatty acids, esters or other compounds known to feed Malassezia."
            )
            if detected_strong:
                explanation += f" High-risk ingredients: {', '.join(detected_strong)}."
        return explanation

    def _categorise_ingredients(self, ingredients: List[str]) -> Dict[str, List[str]]:
        safe, mild, unsafe = [], [], []
        for ing in ingredients:
            lower = ing.lower()
            if any(bad in lower for bad in UNSAFE_KEYWORDS):
                unsafe.append(ing)
            elif any(mid in lower for mid in NEUTRAL_RISK):
                mild.append(ing)
            else:
                safe.append(ing)
        return {"safe": safe, "mild": mild, "unsafe": unsafe}

    def analyze(self, ingredients_text: str, product_name: Optional[str] = None, skip_store: bool = False) -> Dict:
        clean_text = join_ingredients_for_model(ingredients_text)
        ingredients_list = split_ingredients(ingredients_text)

        pred_label = self.model.predict([clean_text])[0]
        pred_probs = self.model.predict_proba([clean_text])[0]
        classes = list(self.model.classes_)

        score = calculate_safety_score(ingredients_text)
        highlight_groups = self._categorise_ingredients(ingredients_list)
        explanation = self.generate_explanation(ingredients_list, score)

        embedding = embeddings_utils.embed_text(clean_text, self.sentence_model, device=self.embeddings.device if self.embeddings.numel() > 0 else None)

        similar_products = embeddings_utils.find_similar_products(
            embedding,
            self.product_names,
            self.ingredient_lists,
            self.embeddings,
            top_k=5,
        )

        ingredient_similarities = embeddings_utils.most_similar_ingredients(
            ingredients_text,
            self.flat_ingredients,
            self.flat_embeddings,
            self.sentence_model,
            top_k=1,
        )

        result = {
            "product_name": product_name or "Untitled Product",
            "ingredients_raw": ingredients_text,
            "clean_text": clean_text,
            "ingredients_list": ingredients_list,
            "tfidf": {
                "label": pred_label,
                "probs": pred_probs.tolist(),
                "classes": classes,
            },
            "safety_score": score,
            "highlight_groups": highlight_groups,
            "explanation": explanation,
            "embedding": embedding.cpu().tolist(),
            "similar_products": similar_products,
            "ingredient_similarities": ingredient_similarities,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        }

        if not skip_store:
            self._store_result(result)

        return result

    def _store_result(self, result: Dict):
        entry = {
            "product_name": result.get("product_name", "Untitled Product"),
            "ingredients": result.get("ingredients_raw", ""),
            "embedding": result.get("embedding", []),
            "timestamp": result.get("timestamp"),
            "analysis": result,
        }
        embeddings_utils.append_user_memory(entry)
        self.refresh_memory()

    def get_previous_results(self) -> List[Dict]:
        return list(self.user_entries)

    def load_cached_analysis(self, entry: Dict) -> Optional[Dict]:
        """
        Return a stored analysis result if available, normalising arrays.
        """
        analysis = entry.get("analysis")
        if not analysis:
            return None
        tfidf = analysis.get("tfidf", {})
        if "probs" in tfidf:
            tfidf["probs"] = np.array(tfidf["probs"])
        analysis["tfidf"] = tfidf
        return analysis

    def generate_pdf_report(self, result: Dict, lime_image: Optional[bytes] = None) -> Optional[io.BytesIO]:
        """
        Build an in-memory PDF summarising the analysis. Returns BytesIO or None.
        If fpdf is not installed, gracefully return None.
        """
        if FPDF is None:
            return None

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "DermaLens Skincare Analysis", ln=True)

        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, f"Product: {result.get('product_name', 'N/A')}", ln=True)
        pdf.cell(0, 8, f"Timestamp: {result.get('timestamp', '')}", ln=True)
        pdf.ln(4)

        # Prediction
        tfidf = result.get("tfidf", {})
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Model Prediction", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, f"Label: {tfidf.get('label', 'N/A')}", ln=True)

        probs = tfidf.get("probs")
        classes = tfidf.get("classes", [])
        if probs is not None:
            probs_arr = np.array(probs)
            top_idx = np.argsort(probs_arr)[::-1][:3]
            for idx in top_idx:
                pdf.cell(0, 8, f"{classes[idx]}: {probs_arr[idx]:.3f}", ln=True)
        pdf.ln(4)

        # Safety
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Fungal Acne Score", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.cell(0, 8, f"Score: {result.get('safety_score', 'N/A')}/10", ln=True)
        pdf.multi_cell(0, 8, f"Explanation: {result.get('explanation', '')}")
        pdf.ln(4)

        # Ingredients
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Ingredients", ln=True)
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 8, result.get("ingredients_raw", ""))
        pdf.ln(4)

        # Similar products
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Similar Products", ln=True)
        pdf.set_font("Arial", "", 11)
        for item in result.get("similar_products", [])[:5]:
            pdf.cell(0, 8, f"{item['product_name']} ({item['score']:.2f})", ln=True)
        pdf.ln(4)

        # Ingredient similarities
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Ingredient Insights", ln=True)
        pdf.set_font("Arial", "", 11)
        for ing, sim in result.get("ingredient_similarities", {}).items():
            pdf.cell(0, 8, f"{ing} -> {sim['closest_ingredient']} ({sim['score']:.2f})", ln=True)
        pdf.ln(4)

        # LIME image if provided
        if lime_image:
            try:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, "LIME Explanation", ln=True)
                img_stream = io.BytesIO(lime_image)
                pdf.image(img_stream, x=None, y=None, w=170)
            except Exception:
                pass

        buffer = io.BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        return buffer


def fetch_product_ingredients(product_name: str) -> Dict[str, str]:
    """
    Fetch ingredients for a product name via lightweight web search.
    Returns a dict with status, message, and optional ingredients text.
    """
    return ingredient_lookup.search_ingredients_by_product_name(product_name)


def extract_ingredients_from_image(upload) -> str:
    """
    Placeholder OCR extraction. Returns a mocked ingredient string.
    """
    _ = upload  # unused placeholder
    return "water, glycerin, niacinamide, panthenol"
