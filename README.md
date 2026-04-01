# QUB Finance AI Tutor

An interactive self-study web application for Queen's University Belfast Finance students. Upload your lecture notes (PDF or PowerPoint) and get AI-powered tutoring, quizzes, and practice calculations — all tailored to your own material.

---

## Features

- **Document Upload** — Drag and drop PDF or PowerPoint files; text is extracted automatically
- **AI Chat Tutor** — Ask questions about your notes and get clear, context-aware explanations
- **Key Concepts** — Generate a structured summary of the main ideas in your document
- **Executive Summary** — Get a concise overview of the lecture content
- **Retrieval Quiz** — Auto-generated multiple-choice questions to test your recall
- **Essay Questions** — Practice longer-form answers with model responses
- **Calculation Practice** — Step-by-step worked examples with interactive answer checking
- **Session Persistence** — Your progress is saved across browser sessions

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python / Flask / Gunicorn |
| AI (primary) | OpenAI (`gpt-5.4-mini`) |
| AI (fallback) | OpenAI (`gpt-5.4-nano`) |
| Database | PostgreSQL (Replit managed) |
| Frontend | Bootstrap 5 / Vanilla JS / MathJax |
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

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
# or with uv:
uv sync
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
├── tutor_ai.py                 # AI tutoring logic (Gemini + OpenAI)
├── pdf_processor.py            # PDF and PowerPoint text extraction
├── models.py                   # SQLAlchemy database models
├── database.py                 # Database initialisation
├── database_storage_manager.py # PostgreSQL session storage
├── performance_optimizations.py# In-memory caching and rate limiting
├── speed_optimizations.py      # Connection pooling and async cleanup
├── deployment_config.py        # HTTPS redirect and security headers
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

| Feature | Primary | Fallback |
|---|---|---|
| Chat / tutoring | `gpt-5.4-mini` | `gpt-5.4-nano` |
| Calculation questions | `gpt-5.4-mini` | `gpt-5.4-nano` |
| Quiz generation | `gpt-5.4-mini` | `gpt-5.4-nano` |
| Executive summary | `gpt-5.4-mini` | `gpt-5.4-nano` |
| Essay questions | `gpt-5.4-mini` | `gpt-5.4-nano` |
| Key concepts | `gpt-5.4-mini` | `gpt-5.4-nano` |

### Storage

Session data is stored in PostgreSQL with an in-memory cache (10-minute TTL) sitting in front to reduce database load. The system automatically retries on SSL connection drops with exponential backoff.

### Performance

- Gzip compression on all responses
- Deferred loading of non-critical JavaScript
- MathJax loaded only when calculation questions are displayed
- Background task polling for all AI-powered operations
- 30 requests/minute rate limiting per session on chat endpoints

---

## Deployment

The application is designed for deployment on **Replit Autoscale** with Gunicorn. Run `gunicorn --config gunicorn.conf.py main:app` to start the server; all worker, timeout, and connection settings are defined in `gunicorn.conf.py`.

Key configuration:

```python
# gunicorn.conf.py
workers = 4          # Adjust based on available CPU
timeout = 120
max_requests = 1000  # Recycle workers to prevent memory leaks
preload_app = True
```

Health check available at `/health`.

---

## Security

- All API keys and secrets stored as environment variables — never hardcoded
- CSRF protection via Flask-WTF on all form submissions
- Content Security Policy headers to prevent XSS
- Input validation with message length limits (5,000 characters)
- Secure filename handling for all uploads
- HTTPS enforced in production
- Rate limiting on all AI endpoints

---

## License

This project was developed for educational use at Queen's University Belfast.
