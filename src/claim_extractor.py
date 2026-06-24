import ollama
import json
import re

PROMPT_TEMPLATE = """You are a fact-checking assistant.

Analyze the following sentence and reply ONLY with a valid JSON object.
Do not use markdown code blocks. Do not add any text before or after the JSON.

Sentence: {frase}

Rules:
- "verifiable" is true if the sentence contains an objective, checkable fact (arrests, locations, dates, formal charges, official statements, named people or organizations).
- "verifiable" is false if the sentence is an opinion, vague description, or contextual narration without a formal fact.
- If verifiable, "claim" contains the sentence rewritten as a short, direct, atomic statement.
- If not verifiable, "claim" is null.

Reply EXACTLY in this format, nothing else:
{{"verifiable": true, "claim": "the claim text", "reason": "max 8 words"}}"""


def chiedi_llm(frase: str) -> str:
    risposta = ollama.generate(
        # model="phi3.5",  # locale 
        model="llama3.1:8b",  # Colab
        prompt=PROMPT_TEMPLATE.format(frase=frase),
        options={"temperature": 0.0, "num_predict": 512}
    )
    return risposta["response"]


def parse_risposta(raw: str) -> dict | None:
    # 1. Prova il parse diretto (caso normale)
    inizio = raw.find("{")
    fine = raw.rfind("}") + 1
    if inizio >= 0 and fine > inizio:
        try:
            return json.loads(raw[inizio:fine])
        except json.JSONDecodeError:
            pass  # JSON malformato — proviamo estrazione manuale

    # 2. Estrazione manuale dal testo grezzo
    if '"verifiable": false' in raw:
        return {"verifiable": False, "claim": None, "reason": ""}
    if '"verifiable": true' in raw:
        match = re.search(r'"claim":\s*"([^"]+)', raw)
        claim_text = match.group(1) if match else None
        return {"verifiable": True, "claim": claim_text, "reason": "[partial]"}

    return None

def estrai_claim(frasi: list[dict]) -> list[dict]:
    risultati = []
    for frase in frasi:
        raw = chiedi_llm(frase["testo"])
        risposta = parse_risposta(raw)

        if risposta is None:
            print(f"  [WARN] Frase {frase['id']}: risposta non parsabile →\n{raw}\n---")
            continue

        if risposta.get("verifiable") and risposta.get("claim"):
            risultati.append({
                "id": frase["id"],
                "testo_originale": frase["testo"],
                "claim": risposta["claim"],
                "motivo": risposta.get("reason", ""),
                "entita": frase["entita"]
            })

    return risultati


if __name__ == "__main__":
    from preprocessor import preprocess

    articolo = """
    Cristiano Ronaldo, the Portuguese footballer, has been making headlines again.
    He recently scored a hat-trick in a thrilling match against Barcelona.
    The game took place at the Santiago Bernabéu Stadium in Madrid, Spain.
    Fans from all over the world watched as Ronaldo showcased his incredible skills.
    The event was also significant as it marked his return to the national team after a brief hiatus.
    """

    frasi = preprocess(articolo)
    print(f"Frasi in input: {len(frasi)}\n")

    claims = estrai_claim(frasi)

    print(f"\nClaim verificabili trovati: {len(claims)}\n")
    for c in claims:
        print(f"[{c['id']}] {c['testo_originale']}")
        print(f"     → {c['claim']}")
        print(f"     Motivo: {c['motivo']}")
        print(f"     Entità: {c['entita']}\n")