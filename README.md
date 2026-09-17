# Knowledge Catalogue

A self-hosted app that interviews you, category by category, and turns your
answers into a structured, searchable personal knowledge base — using a
local Ollama model to ask smart follow-up questions and clean up your raw
answers into clear summaries + tags.

## How it works

- Each category (Farming, Auto Mechanics, Electrical, SQL Server, etc.) has
  4 fixed starter questions (edit these in `app/seed_data.py`).
- Once you've answered all the starters for a category, the model starts
  generating adaptive follow-up questions based on what you've already told
  it — trying to fill in gaps rather than ask generic questions.
- Every answer you give is sent to your Ollama server, which rewrites it
  into a clean title + summary + tags, stored in Postgres.
- A local embedding model (`sentence-transformers`, runs inside the
  container) embeds each entry so search finds conceptually similar
  answers, not just keyword matches.


## Setup

1. Find the Docker network your Ollama container is on:

   ```bash
   docker inspect ollama --format '{% raw %}{{json .NetworkSettings.Networks}}{% endraw %}'
   ```

   Take that network name and put it in `docker-compose.yml`, replacing
   `CHANGE_ME_TO_YOUR_OLLAMA_NETWORK`. (If Ollama's container is actually
   named something other than `ollama`, also update `OLLAMA_HOST` in the
   same file.)

2. In Portainer, set the stack environment variable `OLLAMA_MODEL` to a
   model you've already pulled on that Ollama server, e.g.:

   ```bash
   docker exec -it ollama ollama pull llama3.2:1b
   ```

   Small/CPU-friendly options: `llama3.2:1b`, `qwen2.5:1.5b`, `llama3.2:3b`
   — bigger generally means better-structured summaries and follow-up
   questions, at the cost of slower responses on CPU.

3. Deploy the stack. The app talks to Ollama over `http://ollama:11434`
   using its OpenAI/chat-style API — no API key needed.

4. Open `http://<host>:8009`.

Note: no vision model is wired up, so photos you attach are stored and
shown in the entry, but not analyzed by the AI. If you later pull a vision
model (`llava`, `qwen2.5vl`, `llama3.2-vision`), the `structure_answer`
function in `app/ai.py` is the place to re-add image content to the
Ollama request.

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
