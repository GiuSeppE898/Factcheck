import ollama
import json
import re

PROMPT_ZERO_SHOT = """You are a fact-checking system.

You have received a claim to verify and some excerpts from reference sources.

CLAIM TO VERIFY:
{claim}

SOURCE EXCERPTS:
{contesto}

Based ONLY on the provided excerpts, reply with a JSON object. Do not invent information.
Do not use markdown. Nothing other than the JSON.

Verdict rules:
- "SUPPORTS": the excerpts directly confirm the claim
- "REFUTES": the excerpts contradict the claim
- "NOT ENOUGH INFO": the excerpts do not contain enough information to decide

{{"verdict": "SUPPORTS/REFUTES/NOT ENOUGH INFO", "score": 0.0-1.0, "explanation": "max 15 words"}}

Where "score" is your confidence (1.0 = certain, 0.5 = uncertain).
"""
PROMPT_FEW_SHOT = """You are a fact-checking system.

Here are examples of how to evaluate claims:

EXAMPLE 1:
Claim: "The Eiffel Tower is located in Berlin."
Context: [Source 1] The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
Verdict: {{"verdict": "REFUTES", "score": 0.95, "explanation": "Context says Paris, claim says Berlin."}}

EXAMPLE 2:
Claim: "Barack Obama served as the 44th President of the United States."
Context: [Source 1] Barack Obama is an American politician who served as the 44th president of the United States from 2009 to 2017.
Verdict: {{"verdict": "SUPPORTS", "score": 0.99, "explanation": "Context directly confirms the claim."}}

EXAMPLE 3:
Claim: "Marie Curie won three Nobel Prizes."
Context: [Source 1] Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.
Verdict: {{"verdict": "NOT ENOUGH INFO", "score": 0.5, "explanation": "Context does not mention Nobel Prizes."}}

Now evaluate this:

CLAIM TO VERIFY:
{claim}

SOURCE EXCERPTS:
{contesto}

Reply ONLY with a JSON object, no markdown, nothing else:
{{"verdict": "SUPPORTS/REFUTES/NOT ENOUGH INFO", "score": 0.0-1.0, "explanation": "max 15 words"}}"""

PROMPT_CHAIN_OF_THOUGHT = """You are a fact-checking system. Think step by step before giving a verdict.

CLAIM TO VERIFY:
{claim}

SOURCE EXCERPTS:
{contesto}

Instructions:
1. Identify what the claim asserts.
2. Check if any excerpt confirms, contradicts, or is silent about the claim.
3. If an excerpt says something DIFFERENT from the claim, the verdict is REFUTES.
4. Reply ONLY with a JSON object, no markdown.

{{"verdict": "SUPPORTS/REFUTES/NOT ENOUGH INFO", "score": 0.0-1.0, "explanation": "max 15 words"}}"""

PROMPT_TEMPLATE = PROMPT_CHAIN_OF_THOUGHT  
def formatta_contesto(chunks: list[dict]) -> str:
    """Prepara i chunk recuperati come testo leggibile per l'LLM."""
    if not chunks:
        return "No excerpts available."
    righe = []
    for i, c in enumerate(chunks):
        righe.append(f"[Source {i+1} - {c['entita']}]\n{c['testo'][:400]}")
    return "\n\n".join(righe)


def calcola_score(claim: str, chunks: list[dict]) -> dict:
    """
    Dato un claim e i chunk recuperati, chiede all'LLM il verdict.
    Restituisce: {verdict, score, spiegazione, num_fonti}
    """
    contesto = formatta_contesto(chunks)
    prompt = PROMPT_TEMPLATE.format(claim=claim, contesto=contesto)

    risposta_raw = ollama.generate(
        # model="phi3.5",  # locale MacBook
        model="llama3.1:8b",  # Colab
        prompt=prompt,
        options={"temperature": 0.0, "num_predict": 256}
    )["response"]

    # Parse JSON con fallback
    inizio = risposta_raw.find("{")
    fine = risposta_raw.rfind("}") + 1
    try:
        if fine > inizio:
            risultato = json.loads(risposta_raw[inizio:fine])
        else:
            raise ValueError("JSON troncato")
    except (json.JSONDecodeError, ValueError):
        # Estrazione manuale del verdict
        verdict = "NOT ENOUGH INFO"
        for v in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
            if v in risposta_raw:
                verdict = v
                break
        risultato = {"verdict": verdict, "score": 0.5, "explanation": "[partial]"}

    risultato["num_fonti"] = len(chunks)
    risultato.setdefault("score", 0.5)
    risultato.setdefault("explanation", "")
    risultato.setdefault("verdict", "NOT ENOUGH INFO")
    return risultato
    


if __name__ == "__main__":
    from preprocessor import preprocess
    from claim_extractor import estrai_claim
    from retriever import recupera_contesto

    articolo = """
    Un uomo di 34 anni è stato arrestato ieri sera a Napoli dai carabinieri
    della stazione Vomero con l'accusa di rapina aggravata. La vittima,
    una donna di 60 anni, stava rientrando a casa in via Scarlatti quando
    è stata avvicinata dal sospettato. Sul posto è intervenuto anche il
    pubblico ministero della Procura di Napoli.
    """

    frasi = preprocess(articolo)
    claims = estrai_claim(frasi)

    print(f"\n{'='*60}")
    print("RISULTATI FACT-CHECKING")
    print(f"{'='*60}\n")

    for c in claims:
        chunks = recupera_contesto(c["claim"], k=3)
        risultato = calcola_score(c["claim"], chunks)

        print(f"Claim: {c['claim']}")
        print(f"Verdict:     {risultato['verdict']}")
        print(f"Score:       {risultato['score']}")
        print(f"Spiegazione: {risultato['spiegazione']}")
        print(f"Fonti usate: {risultato['num_fonti']}")
        print()