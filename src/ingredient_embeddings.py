import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

PRODUCT_MEMORY_PATH = "data/product_memory.csv"
EMBEDDING_OUTPUT_PATH = "models/ingredient_embeddings.pt"

# Load product memory CSV and drop any blank rows
df = (
    pd.read_csv(PRODUCT_MEMORY_PATH)
    .dropna(subset=["ingredients"])
)

# Embedder
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get ingredient texts
texts = df["ingredients"].astype(str).tolist()

# Compute embeddings as a tensor for safe loading with weights_only=True
embeddings = model.encode(texts, convert_to_tensor=True).float()

# Save the tensor (keeps dtype and shape; safe to load with weights_only=True)
torch.save(embeddings.cpu(), EMBEDDING_OUTPUT_PATH)

print(f"Saved clean embeddings to {EMBEDDING_OUTPUT_PATH}")
