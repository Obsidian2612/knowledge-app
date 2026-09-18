from dotenv import load_dotenv
load_dotenv()

import os
import re
import base64
import uuid

from fastapi import FastAPI, Request, Depends, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func, text

from app.database import get_db, engine, Base
from app.models import (
    Category, StarterQuestion, KnowledgeEntry, EntryImage,
    Scenario, ScenarioTurn, ScenarioImage,
)
from app import ai

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Knowledge Catalogue")
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# pgvector ships in the image but isn't enabled in a fresh database by default
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    conn.commit()

Base.metadata.create_all(bind=engine)


@app.on_event("startup")
def seed_on_startup():
    from app.seed_data import run
    run()


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or uuid.uuid4().hex[:8]


@app.get("/", response_class=HTMLResponse)
def home(request: Request, db: Session = Depends(get_db)):
    categories = db.query(Category).order_by(Category.name).all()
    counts = dict(
        db.query(KnowledgeEntry.category_id, func.count(KnowledgeEntry.id))
        .group_by(KnowledgeEntry.category_id)
        .all()
    )
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "categories": categories, "counts": counts},
    )


@app.get("/categories/new", response_class=HTMLResponse)
def new_category_form(request: Request):
    return templates.TemplateResponse("new_category.html", {"request": request})


@app.post("/categories", response_class=HTMLResponse)
def create_category(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    custom_questions: str = Form(""),
    db: Session = Depends(get_db),
):
    slug = _slugify(name)
    base_slug = slug
    n = 1
    while db.query(Category).filter_by(slug=slug).first():
        n += 1
        slug = f"{base_slug}-{n}"

    lines = [q.strip() for q in custom_questions.splitlines() if q.strip()]
    questions = lines if lines else ai.generate_starter_questions(name, description)

    category = Category(slug=slug, name=name, description=description)
    db.add(category)
    db.flush()
    for i, q in enumerate(questions):
        db.add(StarterQuestion(category_id=category.id, order=i, text=q))
    category.current_question = questions[0] if questions else ""
    db.commit()

    return RedirectResponse(f"/category/{slug}", status_code=303)


@app.get("/category/{slug}", response_class=HTMLResponse)
def category_page(slug: str, request: Request, db: Session = Depends(get_db)):
    category = db.query(Category).filter_by(slug=slug).first()
    if not category:
        return RedirectResponse("/")
    recent = (
        db.query(KnowledgeEntry)
        .filter_by(category_id=category.id)
        .order_by(KnowledgeEntry.created_at.desc())
        .limit(20)
        .all()
    )
    scenarios = (
        db.query(Scenario)
        .filter_by(category_id=category.id)
        .order_by(Scenario.created_at.desc())
        .limit(20)
        .all()
    )
    return templates.TemplateResponse(
        "category.html",
        {"request": request, "category": category, "entries": recent, "scenarios": scenarios},
    )


def _save_uploaded_images(images: list[UploadFile]) -> list[str]:
    saved = []
    for img in images:
        if not img.filename:
            continue
        raw = img.file.read()
        if not raw:
            continue
        ext = os.path.splitext(img.filename)[1] or ".jpg"
        stored_name = f"{uuid.uuid4().hex}{ext}"
        with open(os.path.join(UPLOAD_DIR, stored_name), "wb") as f:
            f.write(raw)
        saved.append(stored_name)
    return saved


@app.get("/category/{slug}/cases/new", response_class=HTMLResponse)
def new_scenario_form(slug: str, request: Request, db: Session = Depends(get_db)):
    category = db.query(Category).filter_by(slug=slug).first()
    return templates.TemplateResponse(
        "new_scenario.html", {"request": request, "category": category}
    )


@app.post("/category/{slug}/cases", response_class=HTMLResponse)
def create_scenario(
    slug: str,
    description: str = Form(...),
    images: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
):
    category = db.query(Category).filter_by(slug=slug).first()

    scenario = Scenario(category_id=category.id, status="open")
    db.add(scenario)
    db.flush()

    turn = ScenarioTurn(
        scenario_id=scenario.id, order=0,
        question="Describe what happened.", answer=description,
    )
    db.add(turn)

    for fname in _save_uploaded_images(images):
        db.add(ScenarioImage(scenario_id=scenario.id, filename=fname))

    next_q = ai.generate_scenario_followup(category.name, [{"question": turn.question, "answer": turn.answer}])
    scenario.current_question = next_q or ""
    db.commit()

    return RedirectResponse(f"/case/{scenario.id}", status_code=303)


@app.get("/case/{scenario_id}", response_class=HTMLResponse)
def scenario_page(scenario_id: int, request: Request, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).get(scenario_id)
    if not scenario:
        return RedirectResponse("/")
    return templates.TemplateResponse(
        "scenario.html", {"request": request, "scenario": scenario}
    )


@app.post("/case/{scenario_id}/turn", response_class=HTMLResponse)
def submit_scenario_turn(
    scenario_id: int,
    request: Request,
    answer: str = Form(...),
    images: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
):
    scenario = db.query(Scenario).get(scenario_id)
    question = scenario.current_question

    next_order = (
        db.query(func.coalesce(func.max(ScenarioTurn.order), -1))
        .filter_by(scenario_id=scenario.id).scalar() + 1
    )
    turn = ScenarioTurn(scenario_id=scenario.id, order=next_order, question=question, answer=answer)
    db.add(turn)

    for fname in _save_uploaded_images(images):
        db.add(ScenarioImage(scenario_id=scenario.id, filename=fname))

    all_turns = [{"question": t.question, "answer": t.answer} for t in scenario.turns] + [
        {"question": question, "answer": answer}
    ]
    next_q = ai.generate_scenario_followup(scenario.category.name, all_turns)
    scenario.current_question = next_q or ""
    db.commit()
    db.refresh(scenario)
    db.refresh(turn)

    return templates.TemplateResponse(
        "scenario_turn_partial.html", {"request": request, "scenario": scenario, "turn": turn}
    )


@app.post("/case/{scenario_id}/finish", response_class=HTMLResponse)
def finish_scenario(scenario_id: int, request: Request, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).get(scenario_id)
    all_turns = [{"question": t.question, "answer": t.answer} for t in scenario.turns]

    result = ai.summarize_scenario(scenario.category.name, all_turns)
    scenario.title = result["title"]
    scenario.summary = result["summary"]
    scenario.root_cause = result["root_cause"]
    scenario.tags = result["tags"]
    scenario.status = "resolved"
    scenario.current_question = ""
    try:
        scenario.embedding = ai.embed_text(f"{result['title']}. {result['summary']}")
    except Exception:
        pass
    db.commit()

    return RedirectResponse(f"/case/{scenario.id}", status_code=303)


@app.get("/case/{scenario_id}/delete")
def delete_scenario(scenario_id: int, db: Session = Depends(get_db)):
    scenario = db.query(Scenario).get(scenario_id)
    if scenario:
        slug = scenario.category.slug
        db.delete(scenario)
        db.commit()
        return RedirectResponse(f"/category/{slug}", status_code=303)
    return RedirectResponse("/", status_code=303)


def _next_question(db: Session, category: Category) -> str | None:
    """Advance interview state and return the next question, or None."""
    starters = (
        db.query(StarterQuestion)
        .filter_by(category_id=category.id)
        .order_by(StarterQuestion.order)
        .all()
    )
    if category.phase == "starter":
        next_index = category.starter_index + 1
        if next_index < len(starters):
            category.starter_index = next_index
            return starters[next_index].text
        else:
            category.phase = "adaptive"
            # fall through to adaptive generation

    recent = (
        db.query(KnowledgeEntry)
        .filter_by(category_id=category.id)
        .order_by(KnowledgeEntry.created_at.desc())
        .limit(8)
        .all()
    )
    recent_dicts = [
        {"question_text": e.question_text, "structured_summary": e.structured_summary}
        for e in reversed(recent)
    ]
    return ai.generate_followup(category.name, recent_dicts)


@app.post("/category/{slug}/answer", response_class=HTMLResponse)
def submit_answer(
    slug: str,
    request: Request,
    answer: str = Form(...),
    images: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db),
):
    category = db.query(Category).filter_by(slug=slug).first()
    question = category.current_question

    # Save uploaded images to disk (stored/shown regardless of whether the
    # configured model can actually look at them)
    saved_filenames = []
    image_payloads = []
    for img in images:
        if not img.filename:
            continue
        raw = img.file.read()
        if not raw:
            continue
        ext = os.path.splitext(img.filename)[1] or ".jpg"
        stored_name = f"{uuid.uuid4().hex}{ext}"
        with open(os.path.join(UPLOAD_DIR, stored_name), "wb") as f:
            f.write(raw)
        saved_filenames.append(stored_name)
        image_payloads.append({
            "media_type": img.content_type or "image/jpeg",
            "data": base64.b64encode(raw).decode("utf-8"),
        })

    structured = ai.structure_answer(category.name, question, answer, images=image_payloads)
    try:
        embedding = ai.embed_text(f"{structured['title']}. {structured['summary']}")
    except Exception:
        embedding = None

    entry = KnowledgeEntry(
        category_id=category.id,
        question_text=question,
        raw_answer=answer,
        title=structured["title"],
        structured_summary=structured["summary"],
        tags=structured["tags"],
        embedding=embedding,
    )
    db.add(entry)
    db.flush()  # get entry.id before attaching images

    for fname in saved_filenames:
        db.add(EntryImage(entry_id=entry.id, filename=fname))

    next_q = _next_question(db, category)
    category.current_question = next_q or ""
    db.commit()
    db.refresh(entry)

    return templates.TemplateResponse(
        "entry_partial.html",
        {"request": request, "entry": entry, "category": category, "next_question": next_q},
    )


@app.get("/entry/{entry_id}/delete")
def delete_entry(entry_id: int, db: Session = Depends(get_db)):
    entry = db.query(KnowledgeEntry).get(entry_id)
    if entry:
        slug = entry.category.slug
        db.delete(entry)
        db.commit()
        return RedirectResponse(f"/category/{slug}", status_code=303)
    return RedirectResponse("/", status_code=303)


@app.get("/search", response_class=HTMLResponse)
def search(request: Request, q: str = "", db: Session = Depends(get_db)):
    entry_results = []
    scenario_results = []
    if q.strip():
        try:
            vec = ai.embed_text(q)
            entry_results = (
                db.query(KnowledgeEntry)
                .filter(KnowledgeEntry.embedding.isnot(None))
                .order_by(KnowledgeEntry.embedding.cosine_distance(vec))
                .limit(10)
                .all()
            )
            scenario_results = (
                db.query(Scenario)
                .filter(Scenario.embedding.isnot(None))
                .order_by(Scenario.embedding.cosine_distance(vec))
                .limit(10)
                .all()
            )
        except Exception:
            # fall back to simple text search if embeddings aren't ready
            like = f"%{q}%"
            entry_results = (
                db.query(KnowledgeEntry)
                .filter(
                    KnowledgeEntry.structured_summary.ilike(like)
                    | KnowledgeEntry.title.ilike(like)
                )
                .limit(10)
                .all()
            )
            scenario_results = (
                db.query(Scenario)
                .filter(Scenario.summary.ilike(like) | Scenario.title.ilike(like))
                .limit(10)
                .all()
            )
    return templates.TemplateResponse(
        "search.html",
        {"request": request, "q": q, "results": entry_results, "scenario_results": scenario_results},
    )
