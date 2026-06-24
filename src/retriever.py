import re
import wikipediaapi
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
KB_PATH = "kb/faiss_index"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

_db_statico = None

wiki = wikipediaapi.Wikipedia(language="it", user_agent="factcheck-tesi/1.0")
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# Mappa keyword → pagine Wikipedia aggiuntive da scaricare
_TOPIC_MAP = {
    r"rapin": ["Rapina", "Codice penale italiano"],
    r"omicid": ["Omicidio", "Codice penale italiano"],
    r"arresta|fermato|fermo": ["Arresto (diritto)", "Misure cautelari"],
    r"procura|magistrat|pm\b|pubblico ministero": ["Procura della Repubblica"],
    r"carabinier": ["Arma dei Carabinieri"],
    r"polizia": ["Polizia di Stato"],
    r"furto": ["Furto", "Codice penale italiano"],
    r"spaccio|droga|stupefacen": ["Traffico di stupefacenti"],
    r"aggravata|aggravato": ["Circostanze aggravanti"],
}


def _espandi_topic(claim: str) -> list[str]:
    """Restituisce pagine Wikipedia extra basandosi su keyword nel claim."""
    extra = []
    claim_lower = claim.lower()
    for pattern, pagine in _TOPIC_MAP.items():
        if re.search(pattern, claim_lower):
            extra.extend(pagine)
    return list(dict.fromkeys(extra))  # deduplica mantenendo ordine


def _scarica_pagina(titolo: str) -> tuple[str, str] | None:
    """Scarica una pagina Wikipedia. Ritorna (testo, titolo_reale) o None."""
    pagina = wiki.page(titolo)
    if not pagina.exists():
        print(f"  [WARN] Wikipedia: nessuna pagina per '{titolo}'")
        return None
    print(f"  [OK] Wikipedia: '{pagina.title}' ({len(pagina.text)} char)")
    return pagina.text, pagina.title


def _bm25_rank(query: str, chunks: list[str]) -> list[int]:
    """Restituisce gli indici dei chunk in ordine BM25 decrescente."""
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    return sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)


def recupera_contesto_dinamico(
    claim: str,
    entita: list[dict],
    k: int = 5,
) -> list[dict]:
    """
    KB dinamica: scarica Wikipedia per le entità del claim + topic semantici,
    costruisce un mini-FAISS in memoria, poi fonde il ranking con BM25.

    entita: lista di {"testo": "...", "tipo": "..."} da spacy
    """
    # Raccogli tutti i titoli da scaricare
    titoli = [e["testo"] for e in entita if e.get("testo")]
    titoli += _espandi_topic(claim)
    titoli = list(dict.fromkeys(titoli))  # deduplica

    testi_raw, metadati_raw = [], []
    titoli_scaricati: set[str] = set()
    for titolo in titoli:
        risultato = _scarica_pagina(titolo)
        if risultato is None:
            continue
        testo, titolo_reale = risultato
        if titolo_reale in titoli_scaricati:
            print(f"  [SKIP] '{titolo_reale}' già scaricata")
            continue
        titoli_scaricati.add(titolo_reale)
        for chunk in splitter.split_text(testo):
            testi_raw.append(chunk)
            metadati_raw.append({"fonte": "wikipedia_it", "entita": titolo_reale})

    if not testi_raw:
        print("  [WARN] Nessun testo scaricato — uso fallback statico")
        return recupera_contesto(claim, k=k)

    print(f"  [INFO] Chunk dinamici: {len(testi_raw)}")

    # --- Ranking ibrido: FAISS semantico + BM25 lessicale ---

    # 1. FAISS: top-k*4 candidati (pool ampio per BM25)
    n_candidati = min(k * 4, len(testi_raw))
    db_tmp = FAISS.from_texts(testi_raw, embeddings, metadatas=metadati_raw)
    faiss_results = db_tmp.similarity_search_with_score(claim, k=n_candidati)

    candidati_testi = [doc.page_content for doc, _ in faiss_results]
    candidati_meta  = [doc.metadata for doc, _ in faiss_results]
    candidati_score = [score for _, score in faiss_results]

    # 2. BM25 sugli stessi candidati FAISS
    bm25_rank_di = {idx: rank for rank, idx in enumerate(_bm25_rank(claim, candidati_testi))}

    # 3. Fusione Reciprocal Rank (RRF) — robusto, nessun iperparametro critico
    #    score_rrf = 1/(K + rank_faiss) + 1/(K + rank_bm25)  con K=60
    K = 60
    n = len(candidati_testi)
    punteggi_rrf = []
    for i in range(n):
        rrf = 1.0 / (K + i) + 1.0 / (K + bm25_rank_di[i])
        punteggi_rrf.append((rrf, i))

    punteggi_rrf.sort(reverse=True)
    top_idx = [idx for _, idx in punteggi_rrf[:k]]

    return [
        {
            "testo": candidati_testi[i],
            "fonte": candidati_meta[i].get("fonte", ""),
            "entita": candidati_meta[i].get("entita", ""),
            "score": round(float(candidati_score[i]), 4),
        }
        for i in top_idx
    ]


def recupera_contesto(claim: str, k: int = 5) -> list[dict]:
    """Retrieval sull'indice FAISS statico con fusione BM25+RRF."""
    global _db_statico
    if _db_statico is None:
        try:
            _db_statico = FAISS.load_local(
                KB_PATH, embeddings, allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"  [WARN] Indice statico non disponibile: {e}")
            return []

    # Pool ampio per BM25
    n_candidati = min(k * 4, 20)
    faiss_results = _db_statico.similarity_search_with_score(claim, k=n_candidati)

    candidati_testi = [doc.page_content for doc, _ in faiss_results]
    candidati_meta  = [doc.metadata for doc, _ in faiss_results]
    candidati_score = [score for _, score in faiss_results]

    # BM25 sugli stessi candidati
    bm25_rank_di = {idx: rank for rank, idx in enumerate(_bm25_rank(claim, candidati_testi))}

    # RRF fusion
    K = 60
    punteggi_rrf = []
    for i in range(len(candidati_testi)):
        rrf = 1.0 / (K + i) + 1.0 / (K + bm25_rank_di[i])
        punteggi_rrf.append((rrf, i))
    punteggi_rrf.sort(reverse=True)
    top_idx = [idx for _, idx in punteggi_rrf[:k]]

    return [
        {
            "testo": candidati_testi[i],
            "fonte": candidati_meta[i].get("fonte", ""),
            "entita": candidati_meta[i].get("fonte", ""),  # usa titolo pagina
            "score": round(float(candidati_score[i]), 4),
        }
        for i in top_idx
    ]


def carica_kb_statica(path: str) -> object:
    """Carica un indice FAISS da disco (per fever_eval.py)."""
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def recupera_da_kb(db, claim: str, k: int = 3) -> list[dict]:
    """
    Retrieval ibrido FAISS+BM25+RRF su un indice FAISS già caricato.
    Usato da fever_eval.py passando l'indice statico condiviso.
    """
    n_candidati = min(k * 4, 20)
    faiss_results = db.similarity_search_with_score(claim, k=n_candidati)

    candidati_testi = [doc.page_content for doc, _ in faiss_results]
    candidati_meta  = [doc.metadata for doc, _ in faiss_results]
    candidati_score = [score for _, score in faiss_results]

    bm25_rank_di = {idx: rank for rank, idx in enumerate(_bm25_rank(claim, candidati_testi))}

    K = 60
    punteggi_rrf = []
    for i in range(len(candidati_testi)):
        rrf = 1.0 / (K + i) + 1.0 / (K + bm25_rank_di[i])
        punteggi_rrf.append((rrf, i))
    punteggi_rrf.sort(reverse=True)
    top_idx = [idx for _, idx in punteggi_rrf[:k]]

    return [
        {
            "testo": candidati_testi[i],
            "fonte": candidati_meta[i].get("fonte", ""),
            "entita": candidati_meta[i].get("fonte", ""),
            "score": round(float(candidati_score[i]), 4),
        }
        for i in top_idx
    ]


if __name__ == "__main__":
    claims_test = [
        {
            "claim": "Un uomo di 34 anni è stato arrestato a Napoli dai carabinieri per rapina aggravata",
            "entita": [
                {"testo": "Napoli", "tipo": "LOC"},
                {"testo": "carabinieri", "tipo": "ORG"},
            ],
        },
        {
            "claim": "Il pubblico ministero della Procura di Napoli è intervenuto sul posto",
            "entita": [{"testo": "Napoli", "tipo": "LOC"}],
        },
    ]

    for item in claims_test:
        print(f"\nClaim: {item['claim']}")
        print("-" * 60)
        chunks = recupera_contesto_dinamico(item["claim"], item["entita"], k=3)
        for i, c in enumerate(chunks):
            print(f"  [{i+1}] Score FAISS: {c['score']} | Entità: {c['entita']}")
            print(f"       {c['testo'][:200]}...")
