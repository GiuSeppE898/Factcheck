"""
fever_eval.py
-------------
Evaluates the fact-checking pipeline on the FEVER subset.
- Works entirely in English (no translation step)
- Uses a pre-built static FAISS index (kb/fever_static_index)
- Hybrid retrieval: FAISS + BM25 + RRF
"""

import json
import sys
sys.path.insert(0, "src")

from retriever import carica_kb_statica, recupera_da_kb
from scorer import calcola_score

import os
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET_PATH = os.path.join(BASE_DIR, "data", "fever_subset_it.json")
KB_PATH     = os.path.join(BASE_DIR, "kb", "fever_static_index")
OUT_PATH    = os.path.join(BASE_DIR, "data", "fever_results.json")

# Carica l'indice statico una volta sola
print("Loading static KB...")
try:
    db = carica_kb_statica(KB_PATH)
    print("KB loaded.\n")
except Exception as e:
    print(f"ERROR: could not load KB from {KB_PATH}: {e}")
    print("Run data/kb_build_static.py first.")
    sys.exit(1)

# Carica subset
with open(SUBSET_PATH, "r") as f:
    subset = json.load(f)

print(f"Evaluating {len(subset)} claims...\n")

risultati = []
corretti = 0

for i, item in enumerate(subset):
    claim_en   = item["claim"]          # English claim, no translation
    label_reale = item["label"]

    if not item.get("evidence_pages"):
        # NOT ENOUGH INFO: no evidence pages in the dataset
        verdict     = "NOT ENOUGH INFO"
        score       = 0.5
        explanation = "No evidence pages available"
    else:
        chunks = recupera_da_kb(db, claim_en, k=3)
        if not chunks:
            verdict     = "NOT ENOUGH INFO"
            score       = 0.5
            explanation = "No chunks retrieved"
        else:
            out         = calcola_score(claim_en, chunks)
            verdict     = out["verdict"]
            score       = out["score"]
            explanation = out.get("explanation", "")

    corretto = verdict == label_reale
    if corretto:
        corretti += 1

    risultati.append({
        "id":          item["id"],
        "claim":       claim_en,
        "label_reale": label_reale,
        "verdict":     verdict,
        "score":       score,
        "explanation": explanation,
        "corretto":    corretto,
    })

    simbolo = "✅" if corretto else "❌"
    print(f"[{i+1:02d}] {simbolo} {label_reale:20s} → {verdict}")

# Metriche finali
accuracy = corretti / len(subset)
print(f"\n{'='*50}")
print(f"ACCURACY: {corretti}/{len(subset)} = {accuracy:.2%}")

for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
    items_label   = [r for r in risultati if r["label_reale"] == label]
    corretti_label = sum(1 for r in items_label if r["corretto"])
    print(f"  {label:20s}: {corretti_label}/{len(items_label)}")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(risultati, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to {OUT_PATH}")
