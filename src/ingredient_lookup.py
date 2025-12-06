"""
Lightweight ingredient lookup by product name using DuckDuckGo HTML results.
This avoids API keys and keeps the dependency surface small while providing
best-effort extraction of ingredient snippets.
"""
from __future__ import annotations

import re
from typing import Dict, Optional
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}
SEARCH_URL = "https://duckduckgo.com/html/?q={query}"


def _extract_ingredients_from_text(text: str) -> Optional[str]:
    """
    Try to find an ingredient list inside arbitrary text.
    """
    if not text:
        return None
    cleaned = re.sub(r"\s+", " ", text)
    match = re.search(r"ingredients?:\s*([A-Za-z0-9 ,.;:/\\-()]+)", cleaned, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()

    # Fallback: if the text already looks like a comma-separated list
    if "," in cleaned and len(cleaned.split(",")) > 3:
        return cleaned.strip()
    return None


def _fetch_search_page(query: str) -> BeautifulSoup:
    url = SEARCH_URL.format(query=quote_plus(query))
    response = requests.get(url, headers=HEADERS, timeout=8)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def search_ingredients_by_product_name(product_name: str) -> Dict[str, str]:
    """
    Attempt to retrieve an ingredient list for a given product name.
    Returns a dictionary with keys: status, message, ingredients, source.
    """
    if not product_name or not product_name.strip():
        return {
            "status": "missing",
            "message": "Please provide a product name to search.",
            "ingredients": "",
            "source": "",
        }

    query = f"{product_name} ingredients"
    base_response = {
        "status": "not_found",
        "message": "No ingredient list detected online.",
        "ingredients": "",
        "source": "",
    }

    try:
        soup = _fetch_search_page(query)
    except requests.RequestException as exc:
        base_response["status"] = "error"
        base_response["message"] = f"Lookup failed: {exc}"
        return base_response

    results = soup.select(".result")
    for block in results:
        title_el = block.select_one("a.result__a")
        snippet_el = block.select_one(".result__snippet")
        title_text = title_el.get_text(" ", strip=True) if title_el else ""
        snippet_text = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        combined_text = " ".join([title_text, snippet_text])

        ingredients = _extract_ingredients_from_text(combined_text)
        if not ingredients:
            ingredients = _extract_ingredients_from_text(snippet_text)

        if ingredients:
            return {
                "status": "success",
                "message": "Fetched ingredients from web search.",
                "ingredients": ingredients,
                "source": title_text or "DuckDuckGo",
            }

    return base_response
