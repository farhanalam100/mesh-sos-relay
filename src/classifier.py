"""Robust model loader and a safe `classify()` API.

This module attempts to load the serialized TF-IDF vectorizer and two
classification heads on first use. Failures are handled gracefully by
falling back to a simple rule-based classifier so the demo remains usable
even when models are missing or incompatible.
"""

import joblib
import os
import logging
from typing import Dict

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_vectorizer = None
_cat_clf = None
_sev_clf = None
_use_fallback = False

logger = logging.getLogger(__name__)


def _fallback_classify(text: str) -> Dict:
    """Very small rule-based fallback when trained models are unavailable.

    The goal is to provide reasonable demo behavior and low confidence so
    delivered messages get flagged for human review.
    """
    t = text.lower()
    if any(k in t for k in ("fire", "smoke")):
        cat = "fire"
        sev = 5
    elif any(k in t for k in ("flood", "water", "inundat")):
        cat = "flood"
        sev = 4
    elif any(k in t for k in ("injur", "bleed", "hurt", "wound")):
        cat = "medical"
        sev = 5
    elif any(k in t for k in ("collapse", "trapped", "stuck")):
        cat = "rescue"
        sev = 5
    else:
        cat = "unknown"
        sev = 2

    conf = 0.30
    return {"category": cat, "severity": sev, "confidence": conf, "needs_review": True}


def _load():
    """Load models once. If any load fails we switch to fallback mode.

    This function is deliberately defensive: errors during unpickling are
    logged and do not raise, so the rest of the application keeps running.
    """
    global _vectorizer, _cat_clf, _sev_clf, _use_fallback
    if _vectorizer is not None or _use_fallback:
        return

    try:
        vec_path = os.path.join(_MODELS_DIR, "tfidf_vectorizer.joblib")
        cat_path = os.path.join(_MODELS_DIR, "category_classifier.joblib")
        sev_path = os.path.join(_MODELS_DIR, "severity_classifier.joblib")

        _vectorizer = joblib.load(vec_path)
        _cat_clf = joblib.load(cat_path)
        _sev_clf = joblib.load(sev_path)
        logger.info("Loaded TF-IDF vectorizer and classifiers from %s", _MODELS_DIR)
    except Exception as e:  # keep broad so a bad/unpicklable model doesn't crash the app
        logger.exception("Failed to load trained models; switching to fallback classifier: %s", e)
        _use_fallback = True


def classify(text: str) -> Dict:
    """Classify `text` and return a dict with category, severity, confidence.

    If the trained models cannot be used this returns a low-confidence
    fallback result and sets `needs_review` to True.
    """
    _load()
    if _use_fallback:
        return _fallback_classify(text)

    try:
        X = _vectorizer.transform([text])
        category = _cat_clf.predict(X)[0]
        severity = int(_sev_clf.predict(X)[0])
        # predict_proba may not exist for some classifiers; fall back safely
        try:
            confidence = float(_cat_clf.predict_proba(X).max())
        except Exception:
            confidence = 0.5
        return {"category": category, "severity": severity, "confidence": confidence,
                "needs_review": confidence < 0.4}
    except Exception:
        logger.exception("Runtime error while classifying text; using fallback")
        return _fallback_classify(text)
