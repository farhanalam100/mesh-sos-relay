"""
Loads the trained classifier and tests it on brand-new messages that were
never in the training templates -- this is the real test of whether it
learned anything general, versus just memorizing template phrasing.
"""

import joblib

vectorizer = joblib.load("models/tfidf_vectorizer.joblib")
cat_clf = joblib.load("models/category_classifier.joblib")
sev_clf = joblib.load("models/severity_classifier.joblib")

NOVEL_MESSAGES = [
    "Please help, the ceiling just came down on top of us and my son can't breathe properly",
    "There's a huge crack running across our apartment wall and it's getting bigger every hour",
    "Grandpa collapsed and isn't breathing right, we don't know what to do",
    "Water level outside our window is almost at the roof, we need a boat right now",
    "We haven't eaten in three days, kids are getting very weak",
    "Hey has anyone seen my dog, he ran off during the storm",
    "Wanted to know if the main road towards the city is passable now",
    "A live wire fell near the entrance, nobody can get in or out safely",
]

def classify(text: str):
    X = vectorizer.transform([text])
    category = cat_clf.predict(X)[0]
    severity = sev_clf.predict(X)[0]
    cat_conf = cat_clf.predict_proba(X).max()
    return category, severity, cat_conf

print("Testing on messages NEVER seen during training:\n")
for msg in NOVEL_MESSAGES:
    category, severity, conf = classify(msg)
    print(f"  [{category:18s} sev={severity} conf={conf:.2f}]  {msg}")
