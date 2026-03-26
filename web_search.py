from typing import List

import serpapi

from config import load_settings


def search_web(query: str, num_results: int = 5) -> List[str]:
    settings = load_settings()
    if not settings.serpapi_api_key:
        return []

    params = {
        "engine": "google",
        "q": query,
        "api_key": settings.serpapi_api_key,
        "num": num_results,
    }

    client = serpapi.Client(api_key=settings.serpapi_api_key)
    result = client.search(params)
    if isinstance(result, dict):
        organic = result.get("organic_results", [])
    else:
        organic = result.as_dict().get("organic_results", [])

    snippets: List[str] = []
    for item in organic[:num_results]:
        title = item.get("title", "").strip()
        link = item.get("link", "").strip()
        snippet = item.get("snippet", "").strip() or item.get("snippet_highlighted_words", "")
        if not title and not snippet:
            continue
        src = f"- {title} ({link})"
        if snippet:
            src = f"{src}\n  {snippet}"
        snippets.append(src)

    return snippets

