"""

Testa l'impatto del claim_extractor sulla pipeline FEVER.

Invece di usare il gold claim direttamente, passa ogni claim FEVER
attraverso il claim_extractor (riscrittura LLM), poi usa il claim
riscritto per retrieval e scoring.

Confronta con il baseline gold per misurare la propagazione dell'errore.

Configurazioni:
  GOLD    = gold claim → retriever → scorer         (baseline)
  FULL    = gold claim → claim_extractor → retriever → scorer
"""

import json
import sys
import os

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBSET_PATH = os.path.join(BASE_DIR, "data", "fever_subset.json")
KB_PATH     = os.path.join(BASE_DIR, "kb", "fever_static_index")
OUT_PATH    = os.path.join(BASE_DIR, "data", "fever_results_fullpipeline.json")

sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from retriever import carica_kb_statica, recupera_da_kb
from scorer import calcola_score_twostep
from claim_extractor import chiedi_llm, parse_risposta

print("Loading static KB...")
try:
    db = carica_kb_statica(KB_PATH)
    print("KB loaded.\n")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

with open(SUBSET_PATH) as f:
    subset = json.load(f)

# Solo claim con evidence_pages (SUPPORTS e REFUTES)
# NOT ENOUGH INFO non ha evidence → lo gestiamo separatamente
print(f"Evaluating {len(subset)} claims...\n")

risultati = []
corretti_gold = 0
corretti_full = 0
estratti_ok = 0
estratti_fallback = 0

for i, item in enumerate(subset):
    claim_gold  = item["claim"]
    label_reale = item["label"]

    # --- Configurazione GOLD (baseline) ---
    if not item.get("evidence_pages"):
        verdict_gold = "NOT ENOUGH INFO"
        score_gold   = 0.5
        expl_gold    = "No evidence pages"
    else:
        chunks = recupera_da_kb(db, claim_gold, k=3)
        out    = calcola_score_twostep(claim_gold, chunks)
        verdict_gold = out["verdict"]
        score_gold   = out["score"]
        expl_gold    = out.get("explanation", "")

    # --- Configurazione FULL (claim_extractor in the loop) ---
    raw = chiedi_llm(claim_gold)
    parsed = parse_risposta(raw)

    if parsed and parsed.get("verifiable") and parsed.get("claim"):
        claim_estratto = parsed["claim"]
        estratti_ok += 1
    else:
        # Fallback: usa il gold claim se il claim_extractor non lo riscrive
        claim_estratto = claim_gold
        estratti_fallback += 1

    if not item.get("evidence_pages"):
        verdict_full = "NOT ENOUGH INFO"
        score_full   = 0.5
        expl_full    = "No evidence pages"
    else:
        chunks = recupera_da_kb(db, claim_estratto, k=3)
        out    = calcola_score_twostep(claim_estratto, chunks)
        verdict_full = out["verdict"]
        score_full   = out["score"]
        expl_full    = out.get("explanation", "")

    corretto_gold = verdict_gold == label_reale
    corretto_full = verdict_full == label_reale
    if corretto_gold: corretti_gold += 1
    if corretto_full: corretti_full += 1

    risultati.append({
        "id":             item["id"],
        "claim_gold":     claim_gold,
        "claim_estratto": claim_estratto,
        "label_reale":    label_reale,
        "verdict_gold":   verdict_gold,
        "verdict_full":   verdict_full,
        "corretto_gold":  corretto_gold,
        "corretto_full":  corretto_full,
        "claim_modificato": claim_gold != claim_estratto,
    })

    sym_g = "✅" if corretto_gold else "❌"
    sym_f = "✅" if corretto_full else "❌"
    print(f"[{i+1:02d}] GOLD {sym_g} | FULL {sym_f} | {label_reale:20s} | estratto: {claim_estratto[:60]}")

# --- Metriche finali ---
n = len(subset)
print(f"\n{'='*60}")
print(f"GOLD accuracy:  {corretti_gold}/{n} = {corretti_gold/n:.2%}")
print(f"FULL accuracy:  {corretti_full}/{n} = {corretti_full/n:.2%}")
print(f"Δ:              {(corretti_full - corretti_gold)/n:+.2%}")
print(f"\nClaim riscritti dal claim_extractor: {estratti_ok}/{n}")
print(f"Fallback (gold usato):               {estratti_fallback}/{n}")

print(f"\n--- Per label ---")
for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
    items = [r for r in risultati if r["label_reale"] == label]
    cg = sum(1 for r in items if r["corretto_gold"])
    cf = sum(1 for r in items if r["corretto_full"])
    print(f"  {label:20s}: GOLD {cg}/{len(items)} | FULL {cf}/{len(items)}")

# Casi interessanti: gold corretto ma full sbagliato (errore introdotto)
peggiorati = [r for r in risultati if r["corretto_gold"] and not r["corretto_full"]]
migliorati = [r for r in risultati if not r["corretto_gold"] and r["corretto_full"]]
print(f"\nCasi peggiorati (gold ✅ → full ❌): {len(peggiorati)}")
print(f"Casi migliorati (gold ❌ → full ✅): {len(migliorati)}")

if peggiorati:
    print("\n--- Esempi peggiorati ---")
    for r in peggiorati[:3]:
        print(f"  Gold:     {r['claim_gold']}")
        print(f"  Estratto: {r['claim_estratto']}")
        print(f"  Label: {r['label_reale']} | Gold: {r['verdict_gold']} | Full: {r['verdict_full']}\n")

with open(OUT_PATH, "w") as f:
    json.dump(risultati, f, indent=2, ensure_ascii=False)
print(f"Risultati salvati in {OUT_PATH}")
