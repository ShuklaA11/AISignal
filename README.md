# AI Newsletter

A personalized AI news aggregator that fetches articles from multiple sources, processes them with LLMs, and delivers personalized daily digests via email. Features ML-based personalization that learns from user behavior over time.

## Features

- **Multi-source aggregation** -- RSS feeds, arXiv papers, HuggingFace, Reddit, Twitter/X, GitHub trending repos, and Anthropic blog
- **LLM-powered processing** -- Automatic summarization, categorization, topic tagging, and importance scoring via LiteLLM (supports Claude, OpenAI, Ollama)
- **Two-tower recommendation** -- Per-user MLP (128 -> 256 -> 1024) trained via contrastive learning against article embeddings (mxbai-embed-large, 1024-dim) for semantic preference matching
- **Hybrid scoring** -- Rule-based scoring blended with ML-learned weights via alpha decay (starts 100% rules, shifts to 70% learned as interaction data grows)
- **Role-based summaries** -- Three summary variants per article (student, industry, enthusiast) with on-demand difficulty adaptation (beginner, intermediate, advanced)
- **Thompson sampling exploration** -- Prevents filter bubbles by surfacing articles outside the user's typical preferences
- **Email digests** -- Daily personalized digests with click tracking
- **Analytics dashboard** -- CTR, nDCG@10, personalization lift, and per-user ML profile metrics
- **Admin review interface** -- Approve/reject articles and edit summaries before sending

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | Python 3.12, FastAPI |
| Frontend | Jinja2 templates, HTMX, TailwindCSS (CDN) |
| Database | SQLite via SQLModel (SQLAlchemy) |
| Migrations | Alembic |
| LLM | LiteLLM (multi-provider: Ollama, Anthropic, OpenAI) |
| ML | PyTorch (user embedding tower), NumPy |
| Email | Resend API, Gmail SMTP, generic SMTP, console (dev) |
| Auth | bcrypt, session-based, CSRF tokens, signed email links |
| Scheduler | Custom asyncio-based SimpleScheduler |

## Prerequisites

- **Python 3.11+**
- **[Ollama](https://ollama.ai)** (default LLM and embedding provider)
- (Optional) Anthropic or OpenAI API key for cloud LLM providers
- (Optional) API keys for Twitter, Reddit fetchers

## Installation

### macOS

```bash
# Install Python 3.12 via Homebrew (skip if already installed)
brew install python@3.12

# Install Ollama
brew install ollama

# Clone and set up
git clone <repo-url>
cd ai-newsletter

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Pull default LLM and embedding models
ollama pull qwen3.5:4b
ollama pull mxbai-embed-large
```

### Linux (Ubuntu/Debian)

```bash
# Install Python 3.12
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Clone and set up
git clone <repo-url>
cd ai-newsletter

python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Pull default models
ollama pull qwen3.5:4b
ollama pull mxbai-embed-large
```

### Windows

```powershell
# Install Python 3.12 from https://www.python.org/downloads/
# Make sure to check "Add Python to PATH" during installation

# Install Ollama from https://ollama.ai/download/windows

# Clone and set up
git clone <repo-url>
cd ai-newsletter

python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"

# Pull default models
ollama pull qwen3.5:4b
ollama pull mxbai-embed-large
```

## Configuration

### 1. Environment Variables

Copy the example env file and fill in your values:

```bash
cp config/.env.example config/.env
```

**Required:**

| Variable | Description |
|----------|-------------|
| `NEWSLETTER_SECRET_KEY` | Session/token signing key. Generate: `python -c "import secrets; print(secrets.token_urlsafe(32))"` |

**Optional (by feature):**

| Variable | When Needed |
|----------|-------------|
| `NEWSLETTER_BASE_URL` | Production deployment (used in email links) |
| `NEWSLETTER_ANTHROPIC_API_KEY` | Using Claude as LLM provider |
| `NEWSLETTER_OPENAI_API_KEY` | Using OpenAI as LLM provider |
| `NEWSLETTER_TWITTER_BEARER_TOKEN` | Twitter/X fetcher |
| `NEWSLETTER_REDDIT_CLIENT_ID` | Reddit fetcher |
| `NEWSLETTER_REDDIT_CLIENT_SECRET` | Reddit fetcher |
| `NEWSLETTER_RESEND_API_KEY` | Resend email provider |
| `NEWSLETTER_EMAIL__SMTP_PASSWORD` | Gmail/SMTP email provider |

Nested config values use double underscores: `NEWSLETTER_{SECTION}__{FIELD}` (e.g., `NEWSLETTER_LLM__PROVIDER=anthropic`).

### 2. Application Config

Copy and customize the YAML config:

```bash
cp config/config.yaml.example config/config.yaml
```

This file controls LLM provider/model, email delivery settings, fetch schedule (UTC), RSS feed URLs, arXiv categories, Reddit subreddits, Twitter search query, topic taxonomy, and role-based defaults.

## Running the Application

### Development

```bash
# Make sure Ollama is running
ollama serve

# Start the dev server (in a separate terminal)
source .venv/bin/activate
uvicorn src.web.app:app --reload --port 8000
```

Visit `http://localhost:8000` to access the web UI.

### Production

```bash
uvicorn src.web.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Use `--workers 1` because SQLite does not support concurrent writes from multiple processes. For multi-worker setups, migrate to PostgreSQL.

### Creating an Admin User

Sign up through the web UI, then promote to admin:

```bash
sqlite3 data/newsletter.db "UPDATE users SET is_admin = 1 WHERE email = 'your@email.com';"
```

## Architecture

```
Fetch              Process                 Personalize            Deliver
  |                   |                        |                     |
  v                   v                        v                     v
7 Fetchers  -->  Dedup + Ingest  -->  LLM Summarize (3 roles)  -->  Score per user
(RSS, arXiv,    (URL + title           + Categorize                 (rule + ML blend)
 HF, Reddit,    fingerprint)           + Tag topics                 --> Build digest
 Twitter,                              + Rate importance            --> Send email
 GitHub,                               + Embed articles
 Anthropic)
```

### Data Sources

| Source | Module | Description |
|--------|--------|-------------|
| RSS Feeds | `src/fetchers/rss.py` | Configurable list of AI/ML blog feeds |
| arXiv | `src/fetchers/arxiv_fetcher.py` | Papers from cs.AI, cs.CL, cs.CV, cs.LG |
| Reddit | `src/fetchers/reddit.py` | Top posts from ML subreddits |
| Twitter/X | `src/fetchers/twitter.py` | AI-related tweets via API v2 |
| HuggingFace | `src/fetchers/huggingface.py` | Trending models and datasets |
| GitHub | `src/fetchers/github_trending.py` | Trending ML repositories |
| Anthropic Blog | `src/fetchers/anthropic_blog.py` | Anthropic's official blog |

### Two-Tower Recommendation Architecture

The personalization system uses a **two-tower architecture** where user preferences and article content are independently projected into a shared 1024-dimensional embedding space, then scored via cosine similarity.

```
Article Tower                          User Tower
(pre-computed)                         (per-user MLP)

title + summary                        engagement history
      |                                       |
      v                                       v
mxbai-embed-large                      128-dim feature vector
      |                                (pooled embeddings +
      v                                 engagement stats)
1024-dim embedding                            |
      |                                       v
      |                              Linear(128 -> 256) + ReLU
      |                                       |
      |                                       v
      |                              Linear(256 -> 1024)
      |                                       |
      v                                       v
  L2 normalize                           L2 normalize
      |                                       |
      +----------> cosine similarity <--------+
                          |
                          v
                  embedding factor
                    [0.3, 2.0]
```

**Article tower**: Each article is embedded using `mxbai-embed-large` (1024-dim) via Ollama. Input is `"{title} | {summary}"` — using the enthusiast-level LLM summary when available, falling back to the first 500 chars of raw content. Embeddings are pre-computed in batch and stored as float32 blobs.

**User tower**: A lightweight 3-layer MLP trained per-user via contrastive learning (CosineEmbeddingLoss, margin=0.2). The 128-dim input feature vector is constructed from:

| Feature Slice | Dimensions | Source |
|---------------|------------|--------|
| Saved article pool | 0-31 | Mean-pooled embeddings of 50 most recent saved articles, projected from 1024 to 32-dim |
| Clicked article pool | 32-63 | Same projection for clicked (but not saved) articles |
| Skipped article pool | 64-95 | Same projection for shown-but-not-engaged articles |
| Engagement statistics | 96-127 | CTR, save rate, skip rate, interaction maturity, embedding coverage, diversity metrics |

**Training**: Contrastive learning with saved/clicked articles as positives and skipped articles as negatives (3:1 negative sampling ratio). Requires at least 10 positive examples per user. Trained nightly with Adam (lr=0.001, 50 epochs).

### Scoring & Blending

Final article scores blend two independent scoring systems:

**Rule-based score** (always computed):
```
score = base_importance x role_factor x topic_match x level_filter x source_weight
```

**ML-learned score** (when sufficient interaction data exists):
```
score = base x source_factor x category_factor x topic_factor
        x difficulty_factor x entity_factor x embedding_factor
```

Where `embedding_factor` maps cosine similarity to a multiplicative boost: `1.0 + similarity`, clamped to [0.3, 2.0].

**Blending**: `final = alpha x rule_score + (1 - alpha) x learned_score`

Alpha decays exponentially with user interactions:
- 0 interactions: alpha = 1.0 (pure rules)
- 40 interactions: alpha ~ 0.65
- 100+ interactions: alpha -> 0.3 (70% learned, 30% rule-based floor)

**Exploration**: Thompson sampling surfaces articles outside the user's typical preferences to prevent filter bubbles.

**Nightly adaptation loop**: Weight decay prevents stale preferences (decay rate scales with signal confidence), and metrics-driven adaptation adjusts alpha and learning rates if personalization lift or nDCG trends decline.

### Scheduled Jobs (UTC)

| Time | Job |
|------|-----|
| 06:00 | Fetch articles from all sources |
| 06:30 | LLM processing (summarize, categorize, tag) |
| 15:00 | Build and send personalized digests |
| 02:00 | Skip stale article processing |
| 02:30 | Weight decay on learned preferences |
| 03:00 | Daily metrics calculation |
| 03:15 | Metrics-driven adaptation |
| 03:30 | User embedding model training |
| 04:00 | Expired token cleanup |

All times are configurable in `config/config.yaml`.

## Scripts

Manual pipeline control and utilities:

```bash
# Trigger article fetching from all sources
python scripts/run_fetch.py

# Run LLM processing on unprocessed articles
python scripts/run_process.py

# Build and send digests for all active users
python scripts/run_send.py

# Backfill article embeddings
python scripts/run_embeddings.py

# Backfill daily scoring metrics (last 30 days)
python scripts/run_metrics.py

# Offline evaluation: rule-based vs ML vs blended scoring
python scripts/offline_eval.py

# t-SNE visualization of article embeddings
python scripts/visualize_embeddings.py
```

## Testing

```bash
# Run full test suite
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_scoring.py
```

Tests use `unittest.mock` for mocking external services. No live network access or API keys required.

## Project Structure

```
src/
  config.py                  # Settings (pydantic-settings), YAML + env vars
  logging_config.py          # Rotating file + stderr logging
  utils.py                   # Shared utilities
  fetchers/                  # 7 source fetchers (RSS, arXiv, Twitter, Reddit, HF, GitHub, Anthropic)
  pipeline/
    orchestrator.py          # Fetch orchestration (asyncio.gather)
    processor.py             # LLM batch processing (summaries, metadata)
    scheduler.py             # SimpleScheduler + scheduled job functions
  storage/
    models.py                # SQLModel table definitions
    database.py              # Engine, session_scope context manager
    queries.py               # Reusable query functions
  personalization/
    scorer.py                # Multiplicative scoring
    learner.py               # EMA weight learning, metrics adaptation
    digest_builder.py        # MMR + Thompson sampling digest composition
    exploration.py           # Thompson sampling for exploration/exploitation
  embeddings/                # Article embeddings + user tower MLP
  metrics/calculator.py      # Daily CTR, nDCG, save rate, personalization lift
  email_delivery/
    sender.py                # Multi-provider email with retry
    templates/               # Jinja2 email templates
  llm/provider.py            # LiteLLM wrapper with fallback chain
  web/
    app.py                   # FastAPI app, middleware, lifespan
    csrf.py                  # Session-bound CSRF middleware
    rate_limit.py            # Rate limiter (slowapi + custom)
    auth_utils.py            # Password hashing, auth dependencies
    template_engine.py       # Jinja2 template setup
    routes/                  # auth, feed, profile, review, api, onboarding, analytics
    static/                  # CSS, JS assets
    templates/               # Jinja2 page templates
config/
  .env.example               # Environment variable template
  config.yaml.example        # Application config template
alembic/                     # Database migration scripts
scripts/                     # Manual pipeline scripts
tests/                       # Test suite
```

## Database Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Create a new migration after model changes
alembic revision --autogenerate -m "description of change"
```

## Switching LLM Providers

The default config uses Ollama (free, local). To switch to a cloud provider:

1. Set the provider in `config/config.yaml`:
   ```yaml
   llm:
     provider: "anthropic"  # or "openai"
     model: "claude-sonnet-4-20250514"  # or "gpt-4o"
   ```
2. Set the corresponding API key in `config/.env`:
   ```
   NEWSLETTER_ANTHROPIC_API_KEY=sk-ant-...
   ```
3. (Optional) Configure fallback providers:
   ```yaml
   llm:
     fallbacks: ["openai", "ollama"]
   ```

## License

MIT License. See [LICENSE](LICENSE) for details.
