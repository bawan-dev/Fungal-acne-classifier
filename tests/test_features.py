import torch

from src.preprocessing import split_ingredients, clean_ingredients_text
from src.analysis_engine import AnalysisEngine, extract_ingredients_from_image
from src import embeddings_utils


def test_split_and_clean():
    text = "Aqua,  Glycerin ,Niacinamide / Tocopherol"
    cleaned = clean_ingredients_text(text)
    assert "aqua" in cleaned
    parts = split_ingredients(text)
    assert parts == ["aqua", "glycerin", "niacinamide / tocopherol"]


class DummyModel:
    classes_ = [f"class_{i}" for i in range(10)]

    def predict(self, X):
        return ["safe"]

    def predict_proba(self, X):
        return torch.full((1, 10), 0.1).numpy()


class FakeSentenceModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, text, convert_to_tensor=True):
        # simple deterministic embedding
        import numpy as np

        if isinstance(text, list):
            return torch.tensor([[1.0, 0.0, 0.0] for _ in text])
        return torch.tensor([1.0, 0.0, 0.0])


def build_fake_engine():
    engine = AnalysisEngine.__new__(AnalysisEngine)
    engine.model_path = ""
    engine.model = DummyModel()
    engine.sentence_model = FakeSentenceModel()
    engine.product_names = []
    engine.ingredient_lists = []
    engine.embeddings = torch.empty((0, engine.sentence_model.get_sentence_embedding_dimension()))
    engine.flat_ingredients = []
    engine.flat_embeddings = torch.empty((0, engine.sentence_model.get_sentence_embedding_dimension()))
    engine.user_entries = []
    return engine


def test_analyze_basic_flow():
    engine = build_fake_engine()
    result = engine.analyze("water, glycerin, niacinamide", product_name="Test", skip_store=True)
    assert result["tfidf"]["label"] == "safe"
    assert result["safety_score"] >= 0
    assert result["highlight_groups"]["safe"]


def test_similarity_helpers():
    names = ["Product A", "Product B"]
    ingredients = ["a, b", "c, d"]
    embeddings = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    query = torch.tensor([1.0, 0.0, 0.0])
    similar = embeddings_utils.find_similar_products(query, names, ingredients, embeddings, top_k=1)
    assert similar[0]["product_name"] == "Product A"


def test_ocr_placeholder():
    text = extract_ingredients_from_image(None)
    assert isinstance(text, str)
    assert text
