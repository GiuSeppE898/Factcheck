from preprocessor import preprocess
from claim_extractor import estrai_claim
from retriever import recupera_contesto
from scorer import calcola_score

TEST_ARTICLE = """
Cristiano Ronaldo, the Portuguese footballer, has been making headlines again.
He recently scored a hat-trick in a thrilling match against Barcelona.
The game took place at the Santiago Bernabéu Stadium in Madrid, Spain.
Fans from all over the world watched as Ronaldo showcased his incredible skills.
The event was also significant as it marked his return to the national team after a brief hiatus.
"""

def fact_check(articolo: str, k: int = 3) -> dict:
    frasi = preprocess(articolo)
    claims = estrai_claim(frasi)

    if not claims:
        return {"claims": [], "summary": {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}}

    risultati = []
    for c in claims:
        chunks = recupera_contesto(c["claim"], k=k)
        verdict = calcola_score(c["claim"], chunks)
        risultati.append({
            "claim": c["claim"],
            "original_text": c["testo_originale"],
            "entities": c["entita"],
            "verdict": verdict["verdict"],
            "score": verdict["score"],
            "explanation": verdict.get("explanation", ""),
        })

    conteggio = {"SUPPORTS": 0, "REFUTES": 0, "NOT ENOUGH INFO": 0}
    for r in risultati:
        conteggio[r["verdict"]] = conteggio.get(r["verdict"], 0) + 1

    return {"claims": risultati, "summary": conteggio}


if __name__ == "__main__":
    import json
    result = fact_check(TEST_ARTICLE)
    print(json.dumps(result, indent=2, ensure_ascii=False))