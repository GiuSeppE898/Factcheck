import json
import ollama

def traduci(testo: str) -> str:
    risposta = ollama.generate(
        model="phi3.5",
        prompt=f"""Traduci in italiano questa frase. 
REGOLE IMPORTANTI:
- Mantieni INVARIATI: nomi di persone, titoli di film/serie TV/libri/canzoni, nomi di aziende, luoghi geografici
- Traduci solo le parole comuni (verbi, aggettivi, articoli)
- Rispondi SOLO con la traduzione, nient'altro, nessuna nota o spiegazione

Esempi corretti:
- "The Hunger Games is a film." → "The Hunger Games è un film."
- "Jimmy Carter was president." → "Jimmy Carter era presidente."
- "Stranger Things is set in Indiana." → "Stranger Things è ambientato in Indiana."

Frase da tradurre: {testo}""",
        options={"temperature": 0.0, "num_predict": 150}
    )["response"].strip()
    
    # Rimuovi eventuali note del modello dopo il punto finale
    linee = risposta.split('\n')
    return linee[0].strip()
# Carica subset
with open("data/fever_subset.json", "r") as f:
    subset = json.load(f)

print(f"Traduco {len(subset)} claim...\n")

tradotti = []
for i, item in enumerate(subset):
    traduzione = traduci(item["claim"])
    tradotti.append({**item, "claim_it": traduzione})
    print(f"[{i+1}/{len(subset)}] {item['claim']}")
    print(f"          → {traduzione}\n")

with open("data/fever_subset_it.json", "w", encoding="utf-8") as f:
    json.dump(tradotti, f, ensure_ascii=False, indent=2)

print(" Salvato in data/fever_subset_it.json")