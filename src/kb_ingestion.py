import wikipediaapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Modello di embedding multilingue leggero (~120MB, gira in CPU)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KB_PATH = "kb/faiss_index"

wiki = wikipediaapi.Wikipedia(
    language="it",
    user_agent="factcheck-tesi/1.0"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)


def scarica_wikipedia(entita: str) -> str | None:
    """Scarica il testo di una pagina Wikipedia italiana."""
    pagina = wiki.page(entita)
    if not pagina.exists():
        print(f"  [WARN] Wikipedia: nessuna pagina trovata per '{entita}'")
        return None
    print(f"  [OK] Scaricata pagina: '{pagina.title}' ({len(pagina.text)} caratteri)")
    return pagina.text


def costruisci_kb(entita: list[str]):
    """
    Dato un elenco di entità, scarica le pagine Wikipedia,
    le divide in chunk e costruisce l'indice FAISS.
    """
    print(f"\nCostruisco KB per {len(entita)} entità...\n")

    testi = []
    metadati = []

    for e in entita:
        testo = scarica_wikipedia(e)
        if testo is None:
            continue
        chunks = splitter.split_text(testo)
        for chunk in chunks:
            testi.append(chunk)
            metadati.append({"fonte": "wikipedia_it", "entita": e})

    if not testi:
        print("[ERRORE] Nessun testo scaricato.")
        return

    print(f"\nChunk totali da indicizzare: {len(testi)}")
    print("Carico il modello di embedding ...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_texts(testi, embeddings, metadatas=metadati)

    os.makedirs("kb", exist_ok=True)
    db.save_local(KB_PATH)
    print(f"\n✅ KB salvata in '{KB_PATH}' con {len(testi)} chunk.")


if __name__ == "__main__":
    # Entità estratte dai claim del Modulo 2
    entita = ["Napoli", "Vomero", "Procura della Repubblica"]

    costruisci_kb(entita)