"""
Store availability checker for UK retailers (Boots and Superdrug).
Uses HTML scraping with BeautifulSoup to avoid heavy dependencies or APIs.
"""
from __future__ import annotations

import functools
import re
from typing import Dict, List
from urllib.parse import quote_plus, urljoin

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}


def _fetch_html(url: str) -> BeautifulSoup:
    response = requests.get(url, headers=HEADERS, timeout=8)
    response.raise_for_status()
    return BeautifulSoup(response.text, "lxml")


def _extract_price(text: str) -> str:
    match = re.search(r"£\s?\d[\d.,]*", text)
    return match.group(0) if match else text.strip()


def check_boots_stock(product_name: str) -> Dict[str, object]:
    """
    Boots search is heavily templated; use multiple selectors to detect tiles.
    If a direct Boots URL is passed, treat it as available.
    """
    result = {"store": "Boots", "available": False, "price": None, "link": None, "error": None}
    if not product_name:
        result["error"] = "No product name provided."
        return result

    # If the user pasted a direct Boots URL, mark as available.
    if product_name.startswith("http") and "boots.com" in product_name:
        result.update({"available": True, "link": product_name})
        return result

    try:
        url = f"https://www.boots.com/search?searchTerm={quote_plus(product_name)}"
        soup = _fetch_html(url)
    except requests.RequestException as exc:
        result["error"] = str(exc)
        return result

    # Try multiple selectors for product cards
    candidates = soup.select("a[data-productid]") or soup.select("a[data-test='product-link']") or soup.select(
        "a[href*='/p/']"
    )
    if candidates:
        href = candidates[0].get("href")
        if href:
            result["link"] = href if href.startswith("http") else urljoin("https://www.boots.com", href)
        result["available"] = True

    price_el = (
        soup.select_one("[data-e2e='product-card-price']")
        or soup.select_one(".product_price")
        or soup.select_one(".price")
    )
    if price_el:
        result["price"] = _extract_price(price_el.get_text(" ", strip=True))
    return result


def check_superdrug_stock(product_name: str) -> Dict[str, object]:
    result = {"store": "Superdrug", "available": False, "price": None, "link": None, "error": None}
    if not product_name:
        result["error"] = "No product name provided."
        return result
    try:
        url = f"https://www.superdrug.com/search?text={quote_plus(product_name)}"
        soup = _fetch_html(url)
    except requests.RequestException as exc:
        result["error"] = str(exc)
        return result

    link_el = soup.select_one("a[data-test='product-tile']") or soup.select_one("a[href*='/p/']")
    if link_el and link_el.get("href"):
        href = link_el["href"]
        result["link"] = href if href.startswith("http") else urljoin("https://www.superdrug.com", href)
        result["available"] = True

    price_el = soup.select_one("[data-test='product-price']") or soup.select_one(".price") or soup.select_one(".ProductPrice")
    if price_el:
        result["price"] = _extract_price(price_el.get_text(" ", strip=True))
    return result


@functools.lru_cache(maxsize=64)
def check_store_availability(product_name: str) -> List[Dict[str, object]]:
    """
    Return a list of availability dicts for configured stores.
    Cached to avoid repeated scraping during a single session.
    """
    return [
        check_boots_stock(product_name),
        check_superdrug_stock(product_name),
    ]
