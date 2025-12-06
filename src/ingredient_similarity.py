"""
Compatibility wrapper around the new embeddings utilities.
"""
from typing import Dict, List

import torch

from src import embeddings_utils

# Initialise shared models and memory on import for backward compatibility
sentence_model = embeddings_utils.load_sentence_model()
product_names, ingredient_lists, base_embeddings = embeddings_utils.load_base_product_memory(sentence_model)
flat_ingredients, flat_embeddings = embeddings_utils.build_flat_ingredient_embeddings(
    ingredient_lists, sentence_model, device=base_embeddings.device if base_embeddings.numel() > 0 else None
)


def embed_text(text) -> torch.Tensor:
    return embeddings_utils.embed_text(text, sentence_model, device=base_embeddings.device if base_embeddings.numel() > 0 else None)


def most_similar_ingredients(query_text: str, top_k: int = 1) -> Dict[str, Dict[str, float]]:
    return embeddings_utils.most_similar_ingredients(
        query_text,
        flat_ingredients,
        flat_embeddings,
        sentence_model,
        top_k=top_k,
    )


def find_similar_products(query_embedding, top_k: int = 5) -> List[Dict[str, object]]:
    return embeddings_utils.find_similar_products(
        query_embedding,
        product_names,
        ingredient_lists,
        base_embeddings,
        top_k=top_k,
    )
