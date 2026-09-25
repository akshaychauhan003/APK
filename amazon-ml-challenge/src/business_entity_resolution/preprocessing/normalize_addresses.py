"""
Address normalization utilities for business entity resolution.
"""

import re
import unicodedata

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bhwy\b": "highway",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
    r"\bbldg\b": "building",
    r"\bdist\b": "district",
    r"\bctr\b": "center",
    r"\bintl\b": "international",
    r"\bno\b": "number",
}


def extract_postal_code(address: str) -> str:
    """Extract potential 5-digit or 6-digit postal/zip code from address string."""
    if not address:
        return ""
    # Find 6-digit (e.g. India PIN) or 5-digit (e.g. US zip) numbers
    match = re.search(r"\b\d{5,6}\b", address)
    return match.group(0) if match else ""


def normalize_address(address: str) -> str:
    """Normalize an address string by expanding abbreviations and cleaning whitespace."""
    if not address or not isinstance(address, str):
        return ""

    address = unicodedata.normalize("NFKD", address).encode("ASCII", "ignore").decode("utf-8")
    address = address.lower()

    # Apply abbreviation mappings
    for pattern, replacement in ADDRESS_ABBREVIATIONS.items():
        address = re.sub(pattern, replacement, address)

    # Clean non-alphanumeric except spaces
    address = re.sub(r"[^\w\s]", " ", address)

    tokens = address.strip().split()
    return " ".join(tokens)
