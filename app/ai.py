import os
import json
import functools

import anthropic

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

_client = None


def client():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


@functools.lru_cache(maxsize=1)
def _embedder():
    # Loaded lazily so the app still boots without the model downloaded yet.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    model = _embedder()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def structure_answer(category_name: str, question: str, answer: str) -> dict:
    """Turn a raw spoken/typed answer into a structured knowledge entry."""
    system = (
        "You help catalogue a domain expert's personal knowledge base. "
        f"The category is '{category_name}'. Given a question and the expert's "
        "raw answer, produce a JSON object with keys: "
        "'title' (short, under 8 words), "
        "'summary' (a clear, well-written rewrite of the knowledge in 2-5 sentences, "
        "preserving every technical detail, number, and specific fact given — do not "
        "invent details that weren't stated), and "
        "'tags' (3-6 short lowercase keyword tags). "
        "Respond with ONLY the JSON object, no other text."
    )
    msg = client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=system,
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nAnswer: {answer}",
        }],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    try:
        data = _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        data = {"title": question[:60], "summary": answer, "tags": []}
    data.setdefault("title", question[:60])
    data.setdefault("summary", answer)
    data.setdefault("tags", [])
    return data


def generate_followup(category_name: str, recent_entries: list[dict]) -> str | None:
    """
    Ask Claude for the next best question to deepen coverage of this category,
    based on what's already been captured. Returns None if Claude thinks the
    session should pause (it can always be resumed later).
    """
    system = (
        f"You are interviewing a domain expert to extract their tacit knowledge "
        f"about '{category_name}' for a personal knowledge base. Below is a list "
        "of question/summary pairs already captured. Ask ONE new, specific "
        "follow-up question that would surface knowledge not yet covered — "
        "favor practical judgment calls, troubleshooting steps, rules of thumb, "
        "and things a textbook wouldn't say, over generic questions. "
        "Respond with ONLY the question text, nothing else."
    )
    history = "\n".join(
        f"- Q: {e['question_text']}\n  Captured: {e['structured_summary']}"
        for e in recent_entries
    ) or "(nothing captured yet)"

    msg = client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=200,
        system=system,
        messages=[{"role": "user", "content": history}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text").strip()
    return text or None
