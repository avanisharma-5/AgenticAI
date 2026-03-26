import os
from pathlib import Path

from dotenv import load_dotenv


# Load values from .env in this folder.
load_dotenv(dotenv_path=str(Path(__file__).parent / ".env"))


class Settings:
    def __init__(
        self,
        groq_api_key,
        serpapi_api_key,
        groq_model,
        embedding_model,
        pdf_path,
        chroma_dir,
        rag_min_chars,
    ):
        self.groq_api_key = groq_api_key
        self.serpapi_api_key = serpapi_api_key
        self.groq_model = groq_model
        self.embedding_model = embedding_model
        self.pdf_path = pdf_path
        self.chroma_dir = chroma_dir
        self.rag_min_chars = rag_min_chars


def load_settings():
    groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
    serpapi_api_key = os.environ.get("SERPAPI_API_KEY", "").strip()

    if not groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment (backend/.env).")

    # SerpAPI is optional. If missing, web fallback returns no results.
    if not serpapi_api_key:
        serpapi_api_key = ""

    return Settings(
        groq_api_key=groq_api_key,
        serpapi_api_key=serpapi_api_key,
        groq_model=os.environ.get("GROQ_MODEL", "groq/llama-3.1-8b-instant"),
        embedding_model=os.environ.get(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        pdf_path=os.environ.get("KB_PDF_PATH", "Health_Companion.pdf"),
        chroma_dir=os.environ.get("CHROMA_DIR", "chroma_store"),
        rag_min_chars=int(os.environ.get("RAG_MIN_CHARS", "250")),
    )

