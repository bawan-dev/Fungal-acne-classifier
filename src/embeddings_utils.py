import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

BASE_EMBEDDINGS_PATH = "models/ingredient_embeddings.pt"
BASE_PRODUCT_MEMORY_PATH = "data/product_memory.csv"
USER_MEMORY_PATH = "data/user_product_memory.jsonl"


def _safe_load_torch(path: str) -> torch.Tensor:
    """
    Load a tensor or list safely with weights_only when available.
    Always returns a float32 2D tensor on CPU.
    """
    if not os.path.exists(path):
        return torch.empty((0, 0), dtype=torch.float32)

    try:
        raw = torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, ValueError, RuntimeError):
        raw = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(raw, torch.Tensor):
        tensor = raw
    elif isinstance(raw, (list, tuple)):
        try:
            tensor = torch.as_tensor(raw, dtype=torch.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Embeddings in {path} are not numeric.") from exc
    else:
        raise ValueError(f"Unexpected embeddings format in {path}: {type(raw)}")

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim > 2:
        tensor = tensor.view(tensor.size(0), -1)

    return tensor.float().contiguous()


def load_sentence_model() -> SentenceTransformer:
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text, model: SentenceTransformer, device=None) -> torch.Tensor:
    """
    Embed text (string or list of strings) and return a 1D float tensor.
    """
    if isinstance(text, (list, tuple)):
        text = ", ".join([str(t).strip() for t in text if str(t).strip()])
    tensor = model.encode(text, convert_to_tensor=True)
    tensor = tensor.float()
    if device:
        tensor = tensor.to(device)
    if tensor.ndim > 1:
        tensor = tensor.squeeze(0)
    return tensor


def load_base_product_memory(model: SentenceTransformer) -> Tuple[List[str], List[str], torch.Tensor]:
    """
    Load the baked-in product memory CSV and embeddings file.
    Returns product_names, ingredient_lists, and embeddings tensor.
    """
    df = (
        pd.read_csv(BASE_PRODUCT_MEMORY_PATH)
        .dropna(subset=["product_names", "ingredients"])
    )
    df["product_names"] = df["product_names"].astype(str).str.strip()
    df["ingredients"] = df["ingredients"].astype(str).str.strip()

    product_names = df["product_names"].tolist()
    ingredient_lists = df["ingredients"].tolist()

    embeddings = _safe_load_torch(BASE_EMBEDDINGS_PATH)
    expected_dim = model.get_sentence_embedding_dimension()

    if embeddings.numel() == 0:
        embeddings = torch.empty((0, expected_dim), dtype=torch.float32)

    if embeddings.ndim != 2 or embeddings.size(-1) != expected_dim:
        raise ValueError(
            f"Loaded embeddings have shape {tuple(embeddings.shape)}, "
            f"expected (?, {expected_dim}). Regenerate ingredient_embeddings.pt."
        )

    # Align length if mismatch
    min_len = min(embeddings.size(0), len(product_names))
    embeddings = embeddings[:min_len]
    product_names = product_names[:min_len]
    ingredient_lists = ingredient_lists[:min_len]

    return product_names, ingredient_lists, embeddings


def _load_user_memory_raw() -> List[Dict]:
    if not os.path.exists(USER_MEMORY_PATH):
        return []
    entries: List[Dict] = []
    with open(USER_MEMORY_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def load_user_memory() -> Tuple[List[str], List[str], torch.Tensor, List[Dict]]:
    """
    Load user-generated product memory. Returns names, ingredients,
    embeddings tensor, and the raw entries (for metadata like timestamp).
    """
    entries = _load_user_memory_raw()
    if not entries:
        return [], [], torch.empty((0, 384), dtype=torch.float32), []

    names = []
    ingredients = []
    embedding_rows = []
    for entry in entries:
        names.append(entry.get("product_name", "Untitled"))
        ingredients.append(entry.get("ingredients", ""))
        embedding_rows.append(entry.get("embedding", []))

    tensor = torch.as_tensor(embedding_rows, dtype=torch.float32)
    return names, ingredients, tensor, entries


def append_user_memory(entry: Dict) -> None:
    """
    Append a single entry to the user memory JSONL file.
    """
    Path(USER_MEMORY_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(USER_MEMORY_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry))
        fh.write("\n")


def build_flat_ingredient_embeddings(ingredient_lists: List[str], model: SentenceTransformer, device=None) -> Tuple[List[str], torch.Tensor]:
    """
    Flatten ingredient lists into unique ingredient strings and build embeddings.
    """
    flat: List[str] = []
    seen = set()
    for ing_list in ingredient_lists:
        parts = [p.strip() for p in str(ing_list).split(",") if p.strip()]
        for part in parts:
            if part not in seen:
                seen.add(part)
                flat.append(part)

    if not flat:
        return flat, torch.empty((0, model.get_sentence_embedding_dimension()), dtype=torch.float32)

    tensor = model.encode(flat, convert_to_tensor=True).float()
    if device:
        tensor = tensor.to(device)
    return flat, tensor


def most_similar_ingredients(query_text: str, flat_ingredients: List[str], flat_embeddings: torch.Tensor, model: SentenceTransformer, top_k: int = 1) -> Dict[str, Dict[str, float]]:
    """
    For each ingredient in query_text, find the closest known ingredient.
    """
    user_ingredients = [p.strip() for p in str(query_text).split(",") if p.strip()]

    if not user_ingredients or flat_embeddings.numel() == 0:
        return {}

    results: Dict[str, Dict[str, float]] = {}
    for ing in user_ingredients:
        ing_embedding = embed_text(ing, model, device=flat_embeddings.device)
        sims = util.cos_sim(ing_embedding.unsqueeze(0), flat_embeddings)[0]
        top_k_eff = min(top_k, sims.numel())
        if top_k_eff == 0:
            continue
        top_scores, top_idx = torch.topk(sims, k=top_k_eff)
        results[ing] = {
            "closest_ingredient": flat_ingredients[int(top_idx[0])],
            "score": float(top_scores[0]),
        }
    return results


def find_similar_products(query_embedding: torch.Tensor, names: List[str], ingredient_lists: List[str], embeddings: torch.Tensor, top_k: int = 5) -> List[Dict[str, object]]:
    """
    Find nearest products for a given embedding.
    """
    if embeddings.numel() == 0:
        return []

    query_tensor = query_embedding
    if query_tensor.ndim == 1:
        query_tensor = query_tensor.unsqueeze(0)
    sims = util.cos_sim(query_tensor, embeddings)[0]
    top_k_eff = min(top_k, sims.numel())
    if top_k_eff == 0:
        return []
    top_scores, top_idx = torch.topk(sims, k=top_k_eff)

    return [
        {
            "product_name": names[int(i)],
            "ingredients": ingredient_lists[int(i)],
            "score": float(top_scores[j]),
        }
        for j, i in enumerate(top_idx)
    ]
