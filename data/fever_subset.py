import json

# Carica il file scaricato manualmente
print("Carico FEVER...")
esempi = []
with open("data/train.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            esempi.append(json.loads(line))

print(f"Esempi totali: {len(esempi)}")

# Prendiamo 30 per label, solo quelli verificabili con evidence
subset = {"SUPPORTS": [], "REFUTES": [], "NOT ENOUGH INFO": []}
target = 30

for e in esempi:
    label = e.get("label")
    if label not in subset or len(subset[label]) >= target:
        continue
    # Per SUPPORTS e REFUTES vogliamo solo quelli con pagina Wikipedia
    evidence_pages = []
    if label != "NOT ENOUGH INFO":
        for ev_group in e.get("evidence", []):
            for ev in ev_group:
                if ev[2]:  # ev[2] è il titolo della pagina Wikipedia
                    evidence_pages.append(ev[2].replace("_", " "))
        if not evidence_pages:
            continue

    subset[label].append({
        "id": e["id"],
        "claim": e["claim"],
        "label": label,
        "evidence_pages": list(set(evidence_pages))
    })

tutti = subset["SUPPORTS"] + subset["REFUTES"] + subset["NOT ENOUGH INFO"]
print(f"\nTotale: {len(tutti)} claim")
for label, items in subset.items():
    print(f"  {label}: {len(items)}")

with open("data/fever_subset.json", "w", encoding="utf-8") as f:
    json.dump(tutti, f, ensure_ascii=False, indent=2)

print("\nEsempi:")
for label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]:
    ex = subset[label][0]
    print(f"\n[{label}] {ex['claim']}")
    print(f"  Pagine: {ex.get('evidence_pages', [])}")