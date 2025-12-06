"""
Helpers for storing and retrieving user favourites locally.
Data is persisted as JSONL under data/user_favourites.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

FAVOURITES_PATH = Path("data/user_favourites.jsonl")


def load_favourites() -> List[Dict]:
    if not FAVOURITES_PATH.exists():
        return []
    favourites: List[Dict] = []
    with FAVOURITES_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                favourites.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return favourites


def add_favourite(entry: Dict) -> None:
    """
    Append an entry if it is not already present (by product name + ingredients).
    """
    existing = load_favourites()
    signature = (entry.get("product_name", "").lower(), entry.get("ingredients_raw", "").lower())
    if any(
        (fav.get("product_name", "").lower(), fav.get("ingredients_raw", "").lower()) == signature
        for fav in existing
    ):
        return
    FAVOURITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FAVOURITES_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry))
        fh.write("\n")


def clear_favourites() -> None:
    if FAVOURITES_PATH.exists():
        FAVOURITES_PATH.unlink()
