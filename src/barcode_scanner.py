"""
Barcode decoding utilities for DermaLens.
Uses pyzbar + Pillow to decode and a lightweight web search to guess product names.
"""
from __future__ import annotations

import io
from typing import Dict, List
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup
from PIL import Image

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
}

try:
    from pyzbar.pyzbar import ZBarSymbol, decode  # type: ignore
except ImportError:  # pragma: no cover
    decode = None
    ZBarSymbol = None


def decode_barcodes(file_obj) -> List[str]:
    """
    Decode barcodes from a Streamlit UploadedFile or bytes object.
    Returns a list of unique barcode strings.
    """
    if decode is None:
        return []

    if file_obj is None:
        return []

    data = file_obj if isinstance(file_obj, (bytes, bytearray)) else file_obj.read()
    if hasattr(file_obj, "seek"):
        try:
            file_obj.seek(0)
        except Exception:
            pass

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return []

    symbols = [ZBarSymbol.EAN13, ZBarSymbol.EAN8, ZBarSymbol.UPCA, ZBarSymbol.CODE128] if ZBarSymbol else None
    results = decode(image, symbols=symbols) if symbols else decode(image)
    return list({res.data.decode("utf-8") for res in results if res.data})


def _search_first_result(query: str) -> Dict[str, str]:
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    response = requests.get(url, headers=HEADERS, timeout=8)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "lxml")
    first = soup.select_one(".result")
    if not first:
        return {}
    title_el = first.select_one("a.result__a")
    snippet_el = first.select_one(".result__snippet")
    return {
        "title": title_el.get_text(" ", strip=True) if title_el else "",
        "snippet": snippet_el.get_text(" ", strip=True) if snippet_el else "",
        "link": title_el["href"] if title_el and title_el.get("href") else "",
    }


def lookup_product_from_barcode(barcode: str) -> Dict[str, str]:
    """
    Try to resolve a product name using the decoded barcode via web search.
    Returns best-effort metadata.
    """
    if not barcode:
        return {"status": "missing", "message": "No barcode detected.", "product_name": "", "link": ""}

    try:
        result = _search_first_result(f"{barcode} skincare product")
    except requests.RequestException as exc:
        return {"status": "error", "message": f"Barcode lookup failed: {exc}", "product_name": "", "link": ""}

    if not result:
        return {"status": "not_found", "message": "No product match found for this barcode.", "product_name": "", "link": ""}

    title = result.get("title", "")
    product_name = title.split(" - ")[0] if title else ""
    return {
        "status": "success",
        "message": "Found a possible product match from barcode search.",
        "product_name": product_name,
        "link": result.get("link", ""),
        "snippet": result.get("snippet", ""),
    }
