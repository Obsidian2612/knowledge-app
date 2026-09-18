import os
import json
import functools

import httpx

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")


@functools.lru_cache(maxsize=1)
def _embedder():
    # Loaded lazily so the app still boots without the model downloaded yet.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str):
    model = _embedder()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def _extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def _chat(system: str, user_content: str, max_tokens: int = 600, json_mode: bool = False) -> str:
    """Call Ollama's /api/chat endpoint. json_mode forces valid JSON output,
    which matters a lot for small models."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens},
    }
    if json_mode:
        payload["format"] = "json"

    resp = httpx.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


def structure_answer(category_name: str, question: str, answer: str, images: list[dict] | None = None) -> dict:
    """
    Turn a raw spoken/typed answer into a structured knowledge entry.
    `images` is accepted for interface compatibility but ignored here —
    no vision model is configured, so photos are stored/shown but not
    analyzed. Swap in a vision-capable Ollama model later to re-enable that.
    """
    system = (
        "You help catalogue a domain expert's personal knowledge base. "
        f"The category is '{category_name}'. Given a question and the expert's "
        "raw answer, respond with ONLY a JSON object with exactly these keys: "
        '"title" (short, under 8 words), '
        '"summary" (a clear rewrite of the knowledge in 2-5 sentences, keeping every '
        "technical detail, number, and specific fact given — do not invent anything "
        'not stated), "tags" (3-6 short lowercase keyword tags as a JSON array of strings). '
        "No other text, no markdown fences, just the JSON object."
    )
    user_content = f"Question: {question}\n\nAnswer: {answer}"

    try:
        text = _chat(system, user_content, max_tokens=700, json_mode=True)
        data = _extract_json(text)
    except (httpx.HTTPError, json.JSONDecodeError, ValueError):
        data = {}

    if not isinstance(data, dict):
        data = {}
    data.setdefault("title", question[:60])
    data.setdefault("summary", answer)
    data.setdefault("tags", [])
    if not isinstance(data.get("tags"), list):
        data["tags"] = []
    return data


def generate_starter_questions(name: str, description: str) -> list[str]:
    """Generate opening interview questions for a brand-new category."""
    system = (
        "You help set up a personal knowledge-capture interview. Given a subject "
        "area name and description, respond with ONLY a JSON object of the form "
        '{"questions": ["...", "...", "...", "..."]} containing exactly 4 opening '
        "questions that would draw out a practitioner's real hands-on knowledge — "
        "favor common mistakes, diagnostic approaches, rules of thumb, and judgment "
        "calls over generic textbook questions. No other text."
    )
    user_content = f"Subject area: {name}\nDescription: {description or '(none given)'}"

    try:
        text = _chat(system, user_content, max_tokens=400, json_mode=True)
        data = _extract_json(text)
        questions = data.get("questions") if isinstance(data, dict) else data
        if isinstance(questions, list) and all(isinstance(q, str) for q in questions) and questions:
            return questions[:6]
    except (httpx.HTTPError, json.JSONDecodeError, ValueError):
        pass

    return [
        f"What's the most common mistake you see in {name}?",
        f"What's a rule of thumb you use in {name} that isn't written down anywhere?",
        f"Walk me through how you approach a typical {name} problem.",
        f"What's something about {name} that took you years to learn?",
    ]


def generate_scenario_followup(category_name: str, turns: list[dict]) -> str | None:
    """
    Given the running Q&A for one specific real-world case, ask ONE more
    targeted follow-up to dig deeper into that case — what was tried, what
    didn't work, what finally worked, and why. Returns None once the model
    judges the case has enough detail to write up (the user can also just
    click "Finish" at any point regardless).
    """
    system = (
        f"You are helping a {category_name} expert document one specific real "
        "case they handled (e.g. a hard-to-diagnose repair), so it can be "
        "referenced later. Below is the case so far, as question/answer pairs. "
        "Ask ONE new, specific follow-up question that digs deeper into this "
        "particular case — prioritize: what was tried that DIDN'T work and why "
        "it seemed reasonable at the time, what changed things and led to the "
        "real cause, exact symptoms/measurements/part names, and the reasoning "
        "that connected the fix to the root cause. "
        "If the case already seems fully documented (root cause and fix are "
        "both clear), respond with exactly: DONE. "
        "Otherwise respond with ONLY the question text, nothing else."
    )
    history = "\n".join(
        f"- Q: {t['question']}\n  A: {t['answer']}" for t in turns
    ) or "(nothing yet)"

    try:
        text = _chat(system, history, max_tokens=200, json_mode=False)
    except httpx.HTTPError:
        return None

    text = text.strip().strip('"').strip()
    if not text or text.upper() == "DONE":
        return None
    return text


def summarize_scenario(category_name: str, turns: list[dict]) -> dict:
    """Turn a full scenario Q&A thread into a structured case write-up."""
    system = (
        f"You help catalogue a {category_name} expert's real-world case "
        "histories. Below is a full interview about one specific case, as "
        "question/answer pairs. Respond with ONLY a JSON object with exactly "
        "these keys: "
        '"title" (short, under 10 words, naming the problem), '
        '"summary" (a clear narrative in 4-10 sentences covering: the initial '
        "symptom, what was tried that didn't work and why it seemed reasonable, "
        "what finally worked, keeping every specific detail, part name, and "
        'number mentioned — do not invent anything not stated), '
        '"root_cause" (1-2 sentences: what was actually wrong, in plain terms), '
        '"tags" (3-6 short lowercase keyword tags as a JSON array of strings). '
        "No other text, no markdown fences, just the JSON object."
    )
    history = "\n".join(
        f"- Q: {t['question']}\n  A: {t['answer']}" for t in turns
    ) or "(nothing captured)"

    try:
        text = _chat(system, history, max_tokens=900, json_mode=True)
        data = _extract_json(text)
    except (httpx.HTTPError, json.JSONDecodeError, ValueError):
        data = {}

    if not isinstance(data, dict):
        data = {}
    data.setdefault("title", "Untitled case")
    data.setdefault("summary", history)
    data.setdefault("root_cause", "")
    data.setdefault("tags", [])
    if not isinstance(data.get("tags"), list):
        data["tags"] = []
    return data


def generate_followup(category_name: str, recent_entries: list[dict]) -> str | None:
    """
    Ask the model for the next best question to deepen coverage of this
    category, based on what's already been captured. Returns None if
    nothing useful comes back (session can always be resumed later).
    """
    system = (
        f"You are interviewing a domain expert to extract their tacit knowledge "
        f"about '{category_name}' for a personal knowledge base. You'll be given a "
        "list of question/summary pairs already captured. Ask ONE new, specific "
        "follow-up question that would surface knowledge not yet covered — favor "
        "practical judgment calls, troubleshooting steps, and rules of thumb over "
        "generic questions. Respond with ONLY the question text, nothing else, no "
        "quotes, no preamble."
    )
    history = "\n".join(
        f"- Q: {e['question_text']}\n  Captured: {e['structured_summary']}"
        for e in recent_entries
    ) or "(nothing captured yet)"

    try:
        text = _chat(system, history, max_tokens=200, json_mode=False)
    except httpx.HTTPError:
        return None

    # Small models sometimes wrap the answer in quotes anyway; strip those.
    text = text.strip().strip('"').strip()
    return text or None
