# Knowledge Catalogue

A self-hosted app that interviews you, category by category, and turns your
answers into a structured, searchable personal knowledge base — using Claude
to ask smart follow-up questions and clean up your raw answers into clear
summaries + tags.

## How it works

- Each category (Farming, Auto Mechanics, Electrical, SQL Server, etc.) has
  4 fixed starter questions (edit these in `app/seed_data.py`).
- Once you've answered all the starters for a category, Claude starts
  generating adaptive follow-up questions based on what you've already told
  it — trying to fill in gaps rather than ask generic questions.
- Every answer you give is sent to Claude, which rewrites it into a clean
  title + summary + tags, and is stored in Postgres.
- A local embedding model (`sentence-transformers`, runs inside the
  container, no extra API key) embeds each entry so search finds
  conceptually similar answers, not just keyword matches.

## Setup

1. Copy the env file and add your Anthropic API key:

   ```bash
   cp .env.example .env
   # edit .env and set ANTHROPIC_API_KEY
   ```

2. Build and run:

   ```bash
   docker compose up --build
   ```

   First build will take a while — it downloads the embedding model
   dependencies (torch etc). Subsequent builds are cached.

3. Open http://localhost:8000

## Customizing categories

Edit `CATEGORIES` in `app/seed_data.py` — add categories, change starter
questions, adjust descriptions. Restart the app; new categories/questions
are added automatically, existing ones are left alone (so it's safe to
re-run without losing your data).

## Project layout

```
docker-compose.yml       # app + postgres(pgvector) containers
Dockerfile
requirements.txt
.env.example
app/
  main.py                # FastAPI routes
  models.py               # SQLAlchemy models (categories, entries)
  database.py             # DB engine/session
  ai.py                    # Claude calls + local embeddings
  seed_data.py             # your categories + starter questions
  templates/               # Jinja2 + HTMX templates
  static/style.css
```

## Extending ideas

- Export a category (or everything) to Markdown/PDF as a personal wiki.
- Add authentication if you ever expose this beyond your local network.
- Swap Postgres full-text search in alongside vector search for exact-term
  lookups (pgvector handles semantic search already).
- Add a "revisit/edit" flow to refine older entries as your answers evolve.
- Point the interview at a voice-to-text input so you can talk through
  answers instead of typing.
