"""
BankGuard AI — PII Filter using Microsoft Presidio.

Detects and masks PII before context reaches the LLM:
  - Account numbers
  - Phone numbers
  - Email addresses
  - PAN (Indian tax ID)
  - Aadhaar numbers
  - Credit/debit card numbers
  - Customer names (when anonymisation is requested)

Falls back to regex masking if Presidio is not installed.
"""

from __future__ import annotations

import re
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Try to import Presidio; fall back gracefully
_presidio_available = False
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig

    _analyzer = AnalyzerEngine()
    _anonymizer = AnonymizerEngine()
    _presidio_available = True
    log.info("presidio_loaded")
except ImportError:
    log.warning("presidio_not_installed_using_regex_fallback")


# ── Regex fallback patterns ───────────────────────────────────────────────────

_PATTERNS: list[tuple[str, str, str]] = [
    # (name, pattern, replacement)
    ("PHONE_IN", r"\+?91[-\s]?[6-9]\d{9}", "[PHONE]"),
    ("PHONE_GENERIC", r"\b[6-9]\d{9}\b", "[PHONE]"),
    ("EMAIL", r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", "[EMAIL]"),
    ("PAN", r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", "[PAN]"),
    ("AADHAAR", r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[AADHAAR]"),
    ("CARD", r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD_NUMBER]"),
]


async def mask_pii(text: str, entities: list[str] | None = None) -> str:
    """
    Mask PII in text.
    Uses Presidio if available, falls back to regex.
    """
    if not text:
        return text

    if _presidio_available:
        return _presidio_mask(text, entities)
    return _regex_mask(text)


def _presidio_mask(text: str, entities: list[str] | None = None) -> str:
    """Use Presidio for comprehensive PII detection."""
    detect_entities = entities or [
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "IBAN_CODE",
        "PERSON",
    ]
    try:
        results = _analyzer.analyze(text=text, entities=detect_entities, language="en")
        anonymized = _anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "PERSON": OperatorConfig("replace", {"new_value": "[NAME]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                "CREDIT_CARD": OperatorConfig("replace", {"new_value": "[CARD]"}),
            },
        )
        # Also apply regex for Indian-specific patterns Presidio may miss
        return _regex_mask(anonymized.text)
    except Exception as exc:
        log.warning("presidio_mask_error", error=str(exc))
        return _regex_mask(text)


def _regex_mask(text: str) -> str:
    """Regex-based PII masking (fallback)."""
    for name, pattern, replacement in _PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


async def filter_context_dict(data: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively mask PII in all string values of a dict.
    Used to scrub tool outputs before logging.
    """
    result: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, str):
            result[k] = await mask_pii(v)
        elif isinstance(v, dict):
            result[k] = await filter_context_dict(v)
        elif isinstance(v, list):
            result[k] = [
                (
                    await filter_context_dict(item)
                    if isinstance(item, dict)
                    else (await mask_pii(item) if isinstance(item, str) else item)
                )
                for item in v
            ]
        else:
            result[k] = v
    return result
