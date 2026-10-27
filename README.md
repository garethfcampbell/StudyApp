# QUB Finance: Revision App

An interactive self-study web application for Queen's University Belfast Finance students. Upload your lecture notes (PDF, PowerPoint or Word) and get AI-powered tutoring, quizzes, and practice calculations — all tailored to your own material.

---

## Features

- **Document Upload** — Drag and drop PDF, PowerPoint or Word files; text is extracted automatically
- **AI Chat Tutor** — Ask questions about your notes and get clear, context-aware explanations
- **Key Concepts** — Generate a structured summary of the main ideas in your document
- **Executive Summary** — Get a concise revision sheet of the lecture content
- **Retrieval Quiz** — Auto-generated multiple-choice questions to test your recall
- **Essay Questions** — Practice longer-form answers with model responses
- **Calculation Practice** — Step-by-step worked examples with interactive answer checking
- **Session Persistence** — Your document, conversation and practice state are kept for 24 hours after your last activity, including if you close the browser; reloading the page restores the conversation

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python / Flask / Gunicorn |
| AI | OpenAI (`gpt-6-luna` in Fast mode (`service_tier="fast"`) for every text feature, medium reasoning effort (low for the summary and equation extraction); `gpt-6.1-terra` as fallback and for calculation verification and answer checking; `gpt-image-2.5-sunburst` for infographics) |
| Database | PostgreSQL (Replit managed) |
| Frontend | Bootstrap 5 / Vanilla JS / MathJax 3 |
| File parsing | PyPDF2, python-pptx |

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL database
- OpenAI API key

### Environment Variables

Create the following secrets/environment variables before running:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `SESSION_SECRET` | Yes | Random secret string for Flask sessions |
| `DATABASE_URL` | Yes | PostgreSQL connection string |
| `ADMIN_PASSWORD` | No | Password for the activity log page at `/admin/activity` (every upload and feature use - executive summary, essay questions, calculation questions, quiz, infographic, chat - with date/time, document and status; `?event=upload` to filter, `?format=csv` to download). The browser prompts for it; username is `ADMIN_USER` (default `admin`). If neither this nor `ADMIN_TOKEN` is set, the page returns 404. |
| `ADMIN_TOKEN` | No | Optional alternative for scripts: send it as an `X-Admin-Token` header. |
| `PREWARM_FEATURES` | No | Which features are pre-generated in the background right after an upload so the first click is instant: default `essay,calculation,quiz,infographic_brief` (the infographic's text brief only; add `infographic` to pre-generate the image too, at an image-generation cost per upload). Empty disables pre-generation. |
| `RESEND_API_KEY` | No | Enables the "Email infographic as PDF" option, sent via Resend over HTTPS (works on Replit). If neither this nor `SMTP_HOST` is set, the option is hidden. |
| `RESEND_FROM` | No | From address, e.g. `QUB Finance: Revision App <tutor@yourdomain.ac.uk>` (domain must be verified in Resend). Defaults to `onboarding@resend.dev`, which only delivers to the Resend account owner - for testing. |
| `SMTP_HOST` | No | Fallback SMTP transport, used only when `RESEND_API_KEY` is unset (e.g. `smtp.office365.com`; outbound SMTP may be blocked on Replit). |
| `SMTP_PORT` | No | `587` (STARTTLS, default) or `465` (SSL) |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | No | SMTP login / password or API key |
| `SMTP_FROM` | No | From address (defaults to `SMTP_USERNAME`) |
| `SMTP_FROM_NAME` | No | Display name (default `QUB Finance: Revision App`) |

### Installation

Dependencies are declared in `pyproject.toml` (with a `uv.lock` lockfile):

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install .
```

### Running Locally

```bash
# Development
python main.py

# Production (Gunicorn)
gunicorn --config gunicorn.conf.py main:app
```

The app will be available at `http://localhost:5000`.

---

## Project Structure

```
├── app.py                      # Main Flask application and routes
├── tutor_ai.py                 # AI tutoring logic (OpenAI gpt-6-luna, gpt-6.1-terra fallback)
├── pdf_processor.py            # PDF and PowerPoint text extraction
├── models.py                   # SQLAlchemy database models
├── database.py                 # Database initialisation
├── database_storage_manager.py # PostgreSQL session storage
├── performance_optimizations.py# In-memory caching and rate limiting
├── deployment_config.py        # Health check, HTTPS redirect and security headers
├── gunicorn.conf.py            # Gunicorn worker configuration
├── main.py                     # Application entry point
├── templates/                  # Jinja2 HTML templates
│   └── base.html
└── static/                     # CSS, JS, and image assets
```

---

## Architecture

The application uses a **session-based architecture** — no user accounts required. Each visitor gets a unique session ID, and their uploaded document, chat history, quiz progress, and scores are stored against that session.

### AI Model Priority

All AI features use OpenAI models with automatic fallback:

| Feature | Primary | Fallback | Streaming | Max Tokens |
| --- | --- | --- | --- | --- |
| Chat / tutoring | `gpt-6-luna` | `gpt-6.1-terra` | Yes | 15,000 |
| Executive summary | `gpt-6-luna` | `gpt-6.1-terra` | Yes | 15,000 |
| Key concepts | `gpt-6-luna` | `gpt-6.1-terra` | Yes | 15,000 |
| Essay questions | `gpt-6-luna` | `gpt-6.1-terra` | Yes | 15,000 |
| Quiz generation | `gpt-6-luna` | `gpt-6.1-terra` | No | 15,000 |
| Equation extraction | `gpt-6-luna` | `gpt-6.1-terra` | No | 2,000 |
| Calculation questions | `gpt-6-luna` | `gpt-6.1-terra` | No | 5,000 |
| Answer evaluation | `gpt-6-luna` | `gpt-6.1-terra` | No | 5,000 |

Temperature is not sent to the gpt-6 models (they reject non-default values); every call runs at the model default, and output length is governed by the reasoning effort and the token budget. A call cut off by its token budget is retried once with a doubled budget; a streamed answer that is cut off ends with a visible "[Response cut short ...]" note instead of reading as complete.

Chat, summary, key concepts, and essay questions use **Server-Sent Events (SSE)** for real-time streaming. Quiz generation, equation extraction, calculation questions, and answer evaluation use **background polling**.

### Document text and context

- PDF pages and slides with no extractable text are kept in sequence with a note saying the page is an image, so the model knows material is missing there (the server log reports how many pages had no text). PowerPoint text inside grouped shapes is extracted. Word files get `--- Page N ---` markers from the page breaks Word recorded, so citations and page selection work for them too.
- Essay marking reads the whole document up to the chat budget (120k characters) or the pages most relevant to the question and the answer, not a head-and-tail slice.
- Document-type classification samples the head, middle and tail of the document.
- Each upload is fenced: background jobs (early summary, pre-generations, classification, equation list) of a previous upload discard their results once a newer upload has started, and the document text served to a request is checked against a fingerprint in the session cookie so a worker's cached copy of a previous document is never used.
- Every document prompt is sent document-first: the material (as `Lecture Notes:` / `Exam Paper:` ...) opens the message, and the feature's persona and task follow it. All luna features (summary, key concepts, essay rounds and marking, quiz, equation and exam-question extraction, calculation drafts and worked examples, infographic brief) therefore share one prompt-cache prefix per document, and the calculation verification and answer check share one on gpt-6.1-terra, so after the first call the document tokens are billed at the cached rate (a tenth on luna, a twentieth on terra) for about 30 minutes.
- Failure notices and interrupted streams are never cached, pre-generated, stored as the current question or fed back into the chat history.

### Storage

Session data is stored in PostgreSQL with an in-memory cache (10-minute TTL) sitting in front to reduce database load. The system automatically retries on SSL connection drops with exponential backoff.

### Performance

- Gzip compression on all responses
- Deferred loading of non-critical JavaScript
- MathJax loaded only when calculation questions are displayed
- Background task polling for all AI-powered operations
- Per-worker in-memory caching of deterministic AI results (summary, key concepts, quiz, essay questions) keyed by document hash, 1-hour TTL
- Per-session rate limiting: 30 requests/minute on chat endpoints, 10/minute on generation endpoints, 5/minute on uploads

---

## Deployment

The application is designed for deployment on **Replit Autoscale** with Gunicorn. Run `gunicorn --config gunicorn.conf.py main:app` to start the server; all worker, timeout, and connection settings are defined in `gunicorn.conf.py`.

Key configuration:

```python
# gunicorn.conf.py
worker_class = "gthread"   # Thread-based workers required for SSE streaming
threads = 12               # Threads per worker
workers = min(cpu, 6)      # Auto-scaled to available CPUs (max 6)
timeout = 300              # Extended for long AI generation tasks
max_requests = 3000        # Recycle workers to prevent memory leaks
preload_app = True         # App loaded in master; DB pool disposed per worker in post_fork
```

Health check available at `/health` (registered in all environments).

---

## Security

The app is session-based with no user accounts; the JSON API endpoints are exempt from CSRF tokens and are protected by per-session rate limiting and strict input validation instead.

- All API keys and secrets stored as environment variables — never hardcoded
- Per-session rate limiting on every AI endpoint (30/min chat, 10/min generation, 5/min upload)
- Content Security Policy and security headers (HSTS, X-Frame-Options, nosniff) in production
- Input validation with message length limits (5,000 characters) and type checks
- Filenames sanitised with `secure_filename` and HTML-escaped before being returned to the client
- Generic error messages to clients; full exception detail only in server logs
- Secure (HTTPS-only) session cookies and HTTPS redirection in production
- `/metrics` and `/security_metrics` restricted to localhost, with a single trusted proxy hop (`ProxyFix` applied exactly once)

---

## License

This project was developed for educational use at Queen's University Belfast.

## Models and reasoning effort by feature

Set in `tutor_ai.py` (constants near the top). Every text call tries its first-choice model up to 3 times, then the other model once. Every gpt-6-luna call, including the vision review of the infographic, is sent with `service_tier="fast"` (OpenAI Fast mode); gpt-6.1-terra calls use the default tier.

| Stage / button | Model | Reasoning effort |
|---|---|---|
| Document type classification (after upload) | gpt-6-luna (terra fallback) | medium |
| Executive summary | gpt-6-luna | low |
| Chat | gpt-6-luna | medium |
| Essay questions - round generation | gpt-6-luna | medium |
| Essay questions - marking the student's answer | gpt-6-luna | medium |
| Multiple choice quiz | gpt-6-luna | medium |
| Calculation questions - equation extraction (two stages: first equation, then full list) | gpt-6-luna | low |
| Calculation questions - exam question extraction | gpt-6-luna | medium |
| Calculation questions - first question (pre-generated at upload) | drafted by gpt-6-luna, then verified by gpt-6.1-terra when shown | medium |
| Calculation questions - every later question or worked example (generated on the click) | drafted by gpt-6.1-terra, then verified by gpt-6.1-terra | medium |
| Calculation questions - checking the answer | gpt-6.1-terra (luna fallback) | medium |
| Revision infographic - text brief | gpt-6-luna | medium |
| Revision infographic - image | gpt-image-2.5-sunburst, quality high | n/a |
| Revision infographic - review and spell check | gpt-6-luna (terra fallback), vision | medium |
| Revision infographic - masked repair | gpt-image-2.5-sunburst edits | n/a |

The calculation model is the expensive one and never shares a prompt cache with the other features, so it is not sent the whole document. Equation extraction (on luna) returns each equation with its name, variable definitions and the pages/slides where it is introduced, explained or applied; a calculation question is then generated from that equation plus those pages, their neighbouring pages and any other page that mentions the equation (up to `CALC_CONTEXT_CHARS`, 40k characters; documents that fit that budget are sent whole). A cited page is used only if it mentions the equation, and exam-paper pages that are mostly numbers or headed as tables, formula sheets or appendices are always kept so present-value tables reach the model. Answer checking and exam worked examples likewise receive only the pages relevant to the question.

After an upload the equation extraction runs in **two stages at low reasoning effort**: a first-equation-only call (about 5 s) lets the calculation pre-generation start at once, while the full list is extracted in a background thread and reconciled so that index 0 is the equation already pre-generated. A "next question" click that arrives before the full list is stored waits for it (up to 30 s) rather than reporting the notes as exhausted.

The first calculation question is **drafted on luna** (pre-generated after upload, so it costs little when nobody clicks) and every later one, including exam worked examples, is **drafted on gpt-6.1-terra** on the click; every draft is then **verified on gpt-6.1-terra**: the draft is shown immediately; the strong model then checks it against the same pages and replies `OK` or a corrected question. Minor rounding differences, style and equivalent forms are explicitly not grounds for a rewrite. The student is not told about the check: the answer box is available at once, and if a correction comes back (typically within 3-5 s) the question is silently re-rendered, the view returns to the top of it, and the stored question used for answer marking is replaced. If the verification call fails the draft stands. The verifier also computes the challenge's answer; it is stored as a reference for marking, and an answer submitted while the check is still running waits for the verdict (up to 15 s) so it is marked against the question the student sees. A verdict that arrives after the student has moved on is discarded. Marking accepts answers within 1% of the correct figure or differing only by rounding, and uses the same equation and conventions as the question's worked example. A chat message typed during practice that reads as a question to the tutor (no number, no formula) is answered in chat rather than graded. Measured on two sample decks, 9 of 10 drafts passed; this cuts a clicked question to roughly 60% of a sol-generated one and an unclicked pre-warm to about 5%.
