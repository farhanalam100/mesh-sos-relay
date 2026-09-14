from src import classifier


def test_classify_returns_keys():
    res = classifier.classify("There is a fire and smoke in the building")
    assert isinstance(res, dict)
    assert set(["category", "severity", "confidence", "needs_review"]).issubset(res.keys())
    assert isinstance(res["severity"], int)
    assert isinstance(res["confidence"], float)


def test_fallback_low_confidence():
    # If models are incompatible/missing, fallback should return needs_review True
    res = classifier.classify("unknown unobvious message content")
    assert "needs_review" in res
    assert isinstance(res["needs_review"], bool)
