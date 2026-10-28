"""Experiment 7 API adapter reusing the customer-support FAQ from Experiment 6."""
from __future__ import annotations
import os
import re
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INSUFFICIENT = (
    "I couldn't find a reliable answer in the available FAQ. Please contact the support team "
    "through the Help section. Do not share your password, one-time code, or full payment-card details."
)

# Reuse the fictional customer-support policies defined in Experiment 6.
SUPPORT_FAQ = """
Order Tracking
Customers can view shipment updates in the Orders section of their account. A tracking link is shown after an order is dispatched. If tracking has not updated for 48 hours, contact support with the order reference. The assistant cannot access live order status.

Returns and Refunds
Unused items in their original packaging may be requested for return within 14 days of delivery. Some items may be excluded. To start a return, open Orders, choose the item, and select Return Request. After an approved return is received, a refund is usually processed to the original payment method within 5 to 7 business days. The assistant cannot approve a refund.

Account Access
If you cannot sign in, use the Forgot Password option on the sign-in page. Check your email for the password reset link. If the email does not arrive, check the spam folder and confirm that you entered the registered email address. Never share your password or one-time verification code.

Billing and Payments
Payment status is shown in the order details. If a payment appears to have been charged twice, do not submit repeated payments. Contact the billing support team with the transaction date and order reference. Do not share full card numbers or security codes.

Technical Support
If the website is not loading, refresh the page, check your internet connection, and try an updated browser. If the problem continues, clear the browser cache or try a private window. When contacting support, describe the error message and the steps that led to it. Do not share passwords or verification codes.

Contact Support
For an issue that is not covered by these FAQs, contact the support team through the Help section of the website. Include a brief description and relevant order reference, but do not send passwords, one-time codes, or full payment-card details.
""".strip()

def _split_faq(text: str, max_chars: int = 450, overlap: int = 60) -> list[str]:
    """Split the Experiment 6 FAQ into overlapping chunks, preserving section boundaries where possible."""
    chunks = []
    for section in re.split(r"\n(?=[A-Z][^\n]{2,40}\n)", text):
        section = section.strip()
        if not section:
            continue
        start = 0
        while start < len(section):
            end = min(len(section), start + max_chars)
            chunks.append(section[start:end].strip())
            if end == len(section):
                break
            start = max(start + 1, end - overlap)
    return chunks

# Initialize the retrieval index once when this module loads, not for every API request.
CHUNKS = _split_faq(SUPPORT_FAQ)
VECTORIZER = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
MATRIX = VECTORIZER.fit_transform(CHUNKS)

def classify_intent(question: str) -> str:
    """Keep the rule-based intent categories used in Experiment 6."""
    q = question.lower()
    if any(word in q for word in ["refund", "return", "exchange"]):
        return "returns_refunds"
    if any(word in q for word in ["track", "tracking", "shipment", "delivery", "order status", "live status", "where is my order"]):
        return "order_tracking"
    if any(word in q for word in ["password", "sign in", "login", "account", "verification"]):
        return "account_access"
    if any(word in q for word in ["charged", "payment", "billing", "invoice", "card"]):
        return "billing_payments"
    if any(word in q for word in ["website", "error", "loading", "browser", "technical", "internet"]):
        return "technical_support"
    return "general_support"

def retrieve(question: str, top_k: int = 3, min_score: float = 0.05) -> list[dict]:
    """Return top FAQ chunks with similarity scores."""
    q_vec = VECTORIZER.transform([question])
    scores = cosine_similarity(q_vec, MATRIX).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    results = []
    for idx in ranked:
        score = float(scores[idx])
        if score >= min_score:
            results.append({"chunk_id": int(idx), "text": CHUNKS[int(idx)], "score": round(score, 4)})
    return results

def _mock_generate(question: str, context: str) -> str:
    """Deterministic offline baseline: use only FAQ context and enforce Experiment 6 safety boundaries."""
    q = question.lower()
    # Preserve the explicit sensitive-data and live-status safeguards from Experiment 6.
    if any(term in q for term in ["tell me the password", "what is my password", "give me my password", "password for my account", "one-time code", "verification code"]):
        return ("I can't provide or retrieve passwords or verification codes. Use the Forgot Password "
                "option on the sign-in page, and never share your password or one-time code.")
    if any(term in q for term in ["live status", "current status", "status of order", "where is my order"]):
        return ("I can't access live order status. You can view shipment updates in the Orders section "
                "of your account; contact support if tracking has not updated for 48 hours.")
    if not context:
        return INSUFFICIENT

    # A lexical-overlap gate helps avoid answering unrelated questions from vaguely similar chunks.
    q_terms = {w for w in re.findall(r"[a-z0-9]+", q) if len(w) > 2}
    ctx_terms = set(re.findall(r"[a-z0-9]+", context.lower()))
    overlap = len(q_terms & ctx_terms) / max(1, len(q_terms))
    if overlap < 0.12:
        return INSUFFICIENT

    # Use direct matching sentences rather than inventing policy details.
    sentences = re.split(r"(?<=[.!?])\s+", context.replace("\n", " "))
    ranked_sentences = []
    for sentence in sentences:
        terms = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        score = len(q_terms & terms) / max(1, len(q_terms))
        if sentence.strip():
            ranked_sentences.append((score, sentence.strip()))
    ranked_sentences.sort(key=lambda item: item[0], reverse=True)
    selected = [s for score, s in ranked_sentences[:2] if score > 0]
    return " ".join(dict.fromkeys(selected)) if selected else INSUFFICIENT

def _ollama_generate(question: str, context: str) -> str:
    """Call a local Ollama model; no API key is required."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "qwen2.5:0.5b")
    prompt = (
        "You are a customer-support assistant. Answer only using the FAQ context. "
        "Do not invent order status, refund approvals, or company policies. "
        "Never reveal passwords or verification codes. If unsupported, reply exactly: "
        + INSUFFICIENT + f"\n\nFAQ context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return answer or INSUFFICIENT
    except requests.RequestException as exc:
        raise RuntimeError("Ollama backend request failed.") from exc

def _hf_generate(question: str, context: str) -> str:
    """Call Hugging Face Inference Providers chat completions using environment-only credentials."""
    token = os.getenv("HF_TOKEN", "").strip()
    model = os.getenv("HF_MODEL", "").strip()
    endpoint = os.getenv("HF_CHAT_COMPLETIONS_URL", "https://router.huggingface.co/v1/chat/completions").strip()
    if not token or not model:
        raise RuntimeError("Hugging Face backend is not configured.")
    system_prompt = (
        "You are a customer-support assistant. Answer only from the supplied FAQ context. "
        "Never invent policies or reveal secrets. If unsupported, say: " + INSUFFICIENT
    )
    try:
        response = requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"model": model, "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"FAQ context:\n{context}\n\nQuestion: {question}"}
            ], "temperature": 0},
            timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip() or INSUFFICIENT
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("Cloud inference request failed.") from exc

def answer_question(question: str) -> dict:
    """RAG adapter called by FastAPI; returns answer, source chunk metadata, intent, and backend."""
    docs = retrieve(question)
    context = "\n\n".join(d["text"] for d in docs)
    backend = os.getenv("RAG_BACKEND", "mock").strip().lower()
    if backend == "mock":
        answer = _mock_generate(question, context)
    elif backend == "ollama":
        answer = _ollama_generate(question, context)
    elif backend == "hf":
        answer = _hf_generate(question, context)
    elif backend == "mock_fail":
        raise RuntimeError("Simulated backend failure for the lab test.")
    else:
        raise RuntimeError("Unknown LLM backend configuration.")
    return {
        "answer": answer,
        "sources": [{"chunk_id": d["chunk_id"], "score": d["score"]} for d in docs],
        "intent": classify_intent(question),
        "backend": backend,
    }
