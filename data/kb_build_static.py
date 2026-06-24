"""
kb_build_static.py
------------------
One-time script: builds a static FAISS index from the Wikipedia evidence pages
referenced in the FEVER subset. Run this once before fever_eval.py.

Output: kb/fever_static_index/  (FAISS index saved to disk)
"""

import json
import os
import wikipediaapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

SUBSET_PATH  = "data/fever_subset_it.json"
KB_OUT_PATH  = "kb/fever_static_index"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
MAX_CHARS     = 20000   # primi 20k caratteri per pagina (riduce rumore)

wiki     = wikipediaapi.Wikipedia(language="en", user_agent="factcheck-tesi/1.0")
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embedder = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def decode_fever_title(titolo: str) -> str:
    """Decodifica i titoli FEVER nel formato Wikipedia leggibile.
    FEVER usa Stanford CoreNLP che codifica le parentesi:
    -LRB- → (   -RRB- → )   -LSB- → [   -RSB- → ]
    """
    return (titolo
            .replace("-LRB-", "(")
            .replace("-RRB-", ")")
            .replace("-LSB-", "[")
            .replace("-RSB-", "]")
            .replace("_", " "))


def load_evidence_pages(path: str) -> list[str]:
    """Estrae tutti i titoli unici delle evidence pages dal subset FEVER."""
    with open(path, "r") as f:
        subset = json.load(f)
    titoli = set()
    for item in subset:
        for p in item.get("evidence_pages", []):
            if p:
                titoli.add(decode_fever_title(p))
    print(f"Pagine uniche da scaricare: {len(titoli)}")
    return sorted(titoli)


def download_and_chunk(titoli: list[str]) -> tuple[list[str], list[dict]]:
    """Scarica le pagine Wikipedia EN e le divide in chunk."""
    testi, metadati = [], []
    for i, titolo in enumerate(titoli):
        pagina = wiki.page(titolo)
        if not pagina.exists():
            print(f"  [{i+1}/{len(titoli)}] SKIP (non trovata): {titolo}")
            continue
        print(f"  [{i+1}/{len(titoli)}] OK: {pagina.title} ({len(pagina.text)} char)")
        for chunk in splitter.split_text(pagina.text[:MAX_CHARS]):
            testi.append(chunk)
            metadati.append({"fonte": pagina.title})
    return testi, metadati


def build_and_save(testi: list[str], metadati: list[dict], out_path: str):
    """Crea l'indice FAISS e lo salva su disco."""
    print(f"\nCreazione indice FAISS su {len(testi)} chunk...")
    db = FAISS.from_texts(testi, embedder, metadatas=metadati)
    os.makedirs(out_path, exist_ok=True)
    db.save_local(out_path)
    print(f"Indice salvato in: {out_path}")
    return db


if __name__ == "__main__":
    print("=== KB Static Builder ===\n")
    titoli = load_evidence_pages(SUBSET_PATH)
    testi, metadati = download_and_chunk(titoli)
    print(f"\nTotale chunk: {len(testi)}")
    build_and_save(testi, metadati, KB_OUT_PATH)
    print("\nDone.")
