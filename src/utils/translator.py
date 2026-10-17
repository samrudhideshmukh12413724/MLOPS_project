"""
Zero-dependency Cross-Lingual Translation Bridge for Enterprise RAG.
Supports seamless translation between English, Hindi, Spanish, French, and German
while preserving inline citation markers (e.g., [DOC-HR-001], [chunk_id]).
"""

import os
import re
import json
import logging
import urllib.parse
import urllib.request
from typing import Dict, List, Tuple

logger = logging.getLogger("ragops_translator")

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
}


def _protect_citations(text: str) -> Tuple[str, List[str]]:
    """Replaces citation brackets [DOC-...] or [chunk_id] with placeholders to prevent translation tampering."""
    citations: List[str] = []

    def replacer(match):
        citations.append(match.group(0))
        return f" __CITE_{len(citations) - 1}__ "

    protected_text = re.sub(r"\[[A-Za-z0-9_-]+\]", replacer, text)
    return protected_text, citations


def _restore_citations(text: str, citations: List[str]) -> str:
    """Restores protected citation brackets back to their original form."""
    restored = text
    for i, cite in enumerate(citations):
        pattern = re.compile(rf"\s*__\s*CITE_{i}\s*__\s*", re.IGNORECASE)
        restored = pattern.sub(f" {cite} ", restored)
    return re.sub(r"\s{2,}", " ", restored).strip()


def translate_text(text: str, source_lang: str = "auto", target_lang: str = "en") -> str:
    """
    Translates text between source_lang and target_lang using Google's public translation endpoint.
    Automatically handles language detection when translating to English.
    """
    if not text or not text.strip():
        return text

    target = (target_lang or "en").lower().strip()
    source = (source_lang or "auto").lower().strip()

    if source == target:
        return text
    if target == "en" and text.isascii():
        return text
    if target == "en":
        source = "auto"

    protected_text, citations = _protect_citations(text)

    try:
        url = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={urllib.parse.quote(source)}&tl={urllib.parse.quote(target)}&dt=t&q="
            + urllib.parse.quote(protected_text)
        )
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            res_json = json.loads(response.read().decode("utf-8"))
            if res_json and isinstance(res_json, list) and len(res_json) > 0 and isinstance(res_json[0], list):
                translated_segments = [
                    segment[0] for segment in res_json[0] if segment and len(segment) > 0 and segment[0]
                ]
                translated_text = "".join(translated_segments)
                if translated_text:
                    return _restore_citations(translated_text, citations)
    except Exception as e:
        logger.warning(f"Translation error ({source} -> {target}): {e}. Using original text.")

    return text
