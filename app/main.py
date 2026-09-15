from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db, engine, Base
from app.models import Category, StarterQuestion, KnowledgeEntry
from app import ai

app = FastAPI(title="Knowledge Catalogue")
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

Base.metadata.create_all(bind=engine)


@app.on_event("startup")
def seed_on_startup():
    from app.seed_data import run
    run()


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
    return templates.TemplateResponse(
        "category.html",
        {"request": request, "category": category, "entries": recent},
    )


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
    db: Session = Depends(get_db),
):
    category = db.query(Category).filter_by(slug=slug).first()
    question = category.current_question

    structured = ai.structure_answer(category.name, question, answer)
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
    results = []
    if q.strip():
        try:
            vec = ai.embed_text(q)
            results = (
                db.query(KnowledgeEntry)
                .filter(KnowledgeEntry.embedding.isnot(None))
                .order_by(KnowledgeEntry.embedding.cosine_distance(vec))
                .limit(15)
                .all()
            )
        except Exception:
            # fall back to simple text search if embeddings aren't ready
            like = f"%{q}%"
            results = (
                db.query(KnowledgeEntry)
                .filter(
                    KnowledgeEntry.structured_summary.ilike(like)
                    | KnowledgeEntry.title.ilike(like)
                )
                .limit(15)
                .all()
            )
    return templates.TemplateResponse(
        "search.html", {"request": request, "q": q, "results": results}
    )
