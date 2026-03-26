from functools import lru_cache
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from config import load_settings


@lru_cache(maxsize=1)
def get_retriever():
    settings = load_settings()
    base_dir = Path(__file__).parent
    pdf_file = Path(base_dir.parent, settings.pdf_path).resolve()

    if not pdf_file.exists():
        raise FileNotFoundError(f"KB PDF not found at: {pdf_file}")

    persist_dir = Path(base_dir, settings.chroma_dir).resolve()
    embedding = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    # Build once (persistent Chroma). If already exists, load it.
    if persist_dir.exists() and any(persist_dir.iterdir()):
        vector_db = Chroma(
            persist_directory=str(persist_dir),
            embedding_function=embedding,
        )
    else:
        # First run: read PDF, split into chunks, then store vectors.
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
        splits = splitter.split_documents(documents)

        vector_db = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=str(persist_dir),
        )

    # We will use similarity_search_with_score to judge relevance.
    return vector_db


def retrieve_evidence(query: str, k: int = 5):
    # Returns (found, evidence_chunks).
    settings = load_settings()
    vector_db = get_retriever()

    # Chroma's "score" is a distance for some configurations (lower is more similar).
    # We'll treat "not enough evidence" as either:
    # - low total character count, or
    # - very poor top match.
    docs_with_scores = vector_db.similarity_search_with_score(query, k=k)
    if not docs_with_scores:
        return False, []

    top_score = docs_with_scores[0][1]
    contexts = []
    total_chars = 0
    for doc, _score in docs_with_scores:
        text = doc.page_content.strip()
        if len(text) < 40:
            continue
        meta = doc.metadata or {}
        page = meta.get("page", meta.get("page_number", "unknown"))
        sourcestr = f"[page {page}] {text}"
        contexts.append(sourcestr)
        total_chars += len(text)

    found_by_length = total_chars >= settings.rag_min_chars
    # Conservative: if top_score is extremely bad, assume no grounding.
    # We don't know exact metric direction, so we use a broad threshold.
    found_by_score = True
    try:
        found_by_score = top_score is not None and float(top_score) <= 1.25
    except Exception:
        found_by_score = True

    found = found_by_length and found_by_score
    return found, contexts

