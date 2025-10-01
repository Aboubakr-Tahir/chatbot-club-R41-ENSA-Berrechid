import os, csv
from rapidfuzz import fuzz, process
import json

def load_questions():
    # repo root -> data/faq.csv
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, "..", ".."))
    path = os.path.join(root, "data", "faq.csv")
    items = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            items.append((r["question"].strip(), r["answer"].strip()))
    return items

FAQ_PAIRS = load_questions()
FAQ_QUESTIONS = [q for q,_ in FAQ_PAIRS]

def try_fastpath(user_q: str, threshold: int = 85):
    # fuzzy match against canonical FAQ questions
    match = process.extractOne(user_q, FAQ_QUESTIONS, scorer=fuzz.token_set_ratio)
    if match and match[1] >= threshold:
        # return the matched answer
        idx = FAQ_QUESTIONS.index(match[0])
        return FAQ_PAIRS[idx][1]
    return None
