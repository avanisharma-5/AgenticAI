import json

import requests

from config import load_settings
from rag import retrieve_evidence
from web_search import search_web


def groq_chat(messages, temperature=0.1):
    settings = load_settings()
    model_id = settings.groq_model
    if "/" in model_id:
        model_id = model_id.split("/", 1)[1]

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {"model": model_id, "messages": messages, "temperature": temperature}

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def classify_insurance_intent(question):
    prompt = (
        "You are an intent classifier for an insurance assistant.\n"
        "Determine whether the user question is about insurance topics like: "
        "policy coverage, premiums, deductibles, life/health/auto/property insurance.\n"
        "If it is NOT insurance-related, return is_insurance_related=false.\n\n"
        "Return STRICT JSON only in the form:\n"
        '{ \"is_insurance_related\": true|false, \"reason\": \"...\" }'
    )

    resp = groq_chat(
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )

    try:
        data = json.loads(resp)
        if data.get("is_insurance_related") is True:
            return True, ""
        return False, data.get("reason") or "This does not look like an insurance question."
    except Exception:
        return True, ""


def run_insurance_agents(question):
    is_related, reason = classify_insurance_intent(question)
    if not is_related:
        return (
            "I can help with insurance questions (coverage, premiums, deductibles, "
            "life/health/auto/property policies).\n\n"
            f"Your query does not seem insurance-related. Reason: {reason}\n\n"
            "Please rephrase with insurance details, and I’ll help."
        )

    # ---- Researcher step (RAG -> optional SerpAPI fallback) ----
    found, evidence = retrieve_evidence(question, k=5)
    used_source = "PDF_RAG"

    if found:
        evidence_block = "\n".join(evidence[:5])
        availability = "true"
    else:
        used_source = "WEB_SERPAPI"
        web_snippets = search_web(question, num_results=5)
        if web_snippets:
            evidence_block = "\n".join(web_snippets[:5])
            availability = "maybe"
        else:
            evidence_block = ""
            availability = "false"

    if availability == "false":
        return (
            "I couldn’t find grounded insurance information in the provided PDF KB, "
            "and web search didn’t return useful sources for your question.\n\n"
            "Please share more details (policy type, country, coverage goal, and any terms "
            "you want explained), and I’ll help."
        )

    # Writer step: use only evidence and produce a structured answer.
    system = (
        "You are an insurance writer. You must NOT guess.\n"
        "Only use the provided EVIDENCE. If the evidence does not support an answer, "
        "say you couldn't find it and ask for clarification.\n"
        "Write a structured response for a user.\n"
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"INSURANCE_INFO_AVAILABLE: {availability}\n"
        f"SOURCE_USED: {used_source}\n\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        "Return:\n"
        "1) Coverage Summary (2-4 bullets)\n"
        "2) Key Terms/What to Check (2-4 bullets)\n"
        "3) Next Steps for the user (1-3 bullets)\n"
        "4) Limitations (1-2 bullets: what the evidence does NOT cover)\n"
        "5) Sources/Citations: include up to 3 citation lines copied verbatim from the evidence (page numbers or URLs/snippets).\n"
        "If the evidence is weak, include a short note asking for more details."
    )

    return groq_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )

