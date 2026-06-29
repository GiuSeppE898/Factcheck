"""
ner_recall_eval.py
------------------
Valuta quante delle evidence_pages di FEVER sarebbero raggiungibili
usando solo le entità estratte da spaCy sul testo del claim.

Metrica: NER Recall@evidence
= pagine FEVER trovabili tramite NER / totale pagine FEVER

Output: report su stdout + ner_recall_results.json
"""

import json
import spacy
import re
from pathlib import Path

SUBSET_PATH = Path(__file__).parent / "fever_subset.json"
OUT_PATH    = Path(__file__).parent / "ner_recall_results.json"

nlp = spacy.load("en_core_web_lg")


def normalizza(testo: str) -> str:
    """Normalizza un titolo per il confronto: lowercase, decodifica FEVER, rimuove punteggiatura."""
    testo = testo.replace("-LRB-", "(").replace("-RRB-", ")") \
                 .replace("-LSB-", "[").replace("-RSB-", "]") \
                 .replace("_", " ")
    testo = testo.lower()
    testo = re.sub(r"[^a-z0-9\s]", "", testo)
    return " ".join(testo.split())


def estrai_entita(claim: str) -> list[str]:
    """Estrae le entità nominate dal claim con spaCy."""
    doc = nlp(claim)
    return [
        ent.text for ent in doc.ents
        if ent.label_ in {"PERSON", "GPE", "LOC", "ORG", "FAC", "WORK_OF_ART", "EVENT", "NORP"}
    ]


def match(entita: list[str], evidence_pages: list[str]) -> tuple[list[str], list[str]]:
    """
    Controlla quali evidence_pages sono raggiungibili dalle entità NER.
    Una pagina è raggiungibile se il suo titolo normalizzato contiene
    (o è contenuto in) almeno una entità normalizzata.
    """
    trovate, mancanti = [], []
    for page in evidence_pages:
        page_norm = normalizza(page)
        raggiunta = any(
            normalizza(ent) in page_norm or page_norm in normalizza(ent)
            for ent in entita
        )
        if raggiunta:
            trovate.append(page)
        else:
            mancanti.append(page)
    return trovate, mancanti


# --- Main ---
with open(SUBSET_PATH) as f:
    subset = json.load(f)

# Solo claim con evidence_pages (SUPPORTS e REFUTES)
con_evidence = [item for item in subset if item.get("evidence_pages")]
print(f"Claim con evidence_pages: {len(con_evidence)}/{len(subset)}\n")

risultati = []
totale_pages = 0
totale_trovate = 0

for item in con_evidence:
    claim     = item["claim"]
    evidence  = item["evidence_pages"]
    entita    = estrai_entita(claim)
    trovate, mancanti = match(entita, evidence)

    recall = len(trovate) / len(evidence) if evidence else 0
    totale_pages   += len(evidence)
    totale_trovate += len(trovate)

    risultati.append({
        "id":            item["id"],
        "claim":         claim,
        "label":         item["label"],
        "evidence_pages": evidence,
        "entita_ner":    entita,
        "trovate":       trovate,
        "mancanti":      mancanti,
        "recall":        round(recall, 3),
    })

# --- Metriche ---
recall_globale = totale_trovate / totale_pages if totale_pages else 0
recall_perfetto = sum(1 for r in risultati if r["recall"] == 1.0)
recall_zero     = sum(1 for r in risultati if r["recall"] == 0.0)

print(f"{'='*50}")
print(f"NER Recall@evidence globale: {totale_trovate}/{totale_pages} = {recall_globale:.2%}")
print(f"Claim con recall 100%:        {recall_perfetto}/{len(risultati)}")
print(f"Claim con recall 0%:          {recall_zero}/{len(risultati)}")
print(f"{'='*50}")

# Per label
for label in ["SUPPORTS", "REFUTES"]:
    items = [r for r in risultati if r["label"] == label]
    if not items:
        continue
    r_medio = sum(r["recall"] for r in items) / len(items)
    print(f"  {label:10s}: recall medio = {r_medio:.2%}")

# Esempi di miss interessanti
print(f"\n--- Esempi pagine non trovate dal NER ---")
miss = [r for r in risultati if r["mancanti"]][:5]
for r in miss:
    print(f"\nClaim:    {r['claim']}")
    print(f"Entità:   {r['entita_ner']}")
    print(f"Mancanti: {r['mancanti']}")

# Salva
with open(OUT_PATH, "w") as f:
    json.dump(risultati, f, indent=2, ensure_ascii=False)
print(f"\nRisultati salvati in {OUT_PATH}")
