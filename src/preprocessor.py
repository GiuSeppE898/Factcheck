import spacy

# nlp = spacy.load("it_core_news_lg") PER FATTI DI CRONACA ITALIANA
nlp = spacy.load("en_core_web_lg")

def preprocess(testo: str) -> list[dict]:
    doc = nlp(testo)

    frasi = []
    for i, frase in enumerate(doc.sents):
        testo_pulito = " ".join(frase.text.split())  # rimuove newline e spazi multipli
        if not testo_pulito:  # salta frasi vuote
            continue

        """
        Se usiamo il modello italiano, possiamo filtrare le entità in questo modo:
        entita = [
            {"testo": ent.text, "tipo": ent.label_}
            for ent in frase.ents
            if ent.label_ in {"PER", "LOC", "ORG", "DATE", "EVENT"}
        ]
        """
        entita = [
            {"testo": ent.text, "tipo": ent.label_}
            for ent in frase.ents
                if ent.label_ in {"PERSON", "GPE", "LOC", "ORG", "DATE", "EVENT", "FAC"}
]
        frasi.append({
            "id": len(frasi),  # id ricalcolato dopo il filtro
            "testo": testo_pulito,
            "entita": entita
        })

    return frasi


if __name__ == "__main__":
    articolo = """
    Cristiano Ronaldo, the Portuguese footballer, has been making headlines again. He recently scored a hat-trick in a thrilling match against Barcelona. The game took place at the Santiago Bernabéu Stadium in Madrid, Spain. Fans from all over the world watched as Ronaldo showcased his incredible skills. The event was also significant as it marked his return to the national team after a brief hiatus.
    """

    risultati = preprocess(articolo)

    for frase in risultati:
        print(f"\n[Frase {frase['id']}] {frase['testo']}")
        print(f"  Entità: {frase['entita'] if frase['entita'] else 'nessuna'}")