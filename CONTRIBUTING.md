# Contributing

## Running Tests

Install dev dependencies and run the test suite:

```bash
pip install -e ".[dev]"
pytest tests/
```

Tests use `pytest` with `unittest.mock` for mocking external services (APIs, feeds, LLM calls). No live network access is required to run the test suite.

## Adding a New Fetcher

All fetchers live in `src/fetchers/` and follow the same pattern:

1. **Create a new file** in `src/fetchers/` (e.g., `my_source.py`).

2. **Subclass `BaseFetcher`** from `src.fetchers.base` and implement two required members:
   - `source_name` property -- Returns a short string identifier (e.g., `"my_source"`).
   - `fetch()` async method -- Returns a `list[RawArticle]` with normalized article data.

   Example skeleton:

   ```python
   from src.fetchers.base import BaseFetcher, RawArticle

   class MySourceFetcher(BaseFetcher):
       @property
       def source_name(self) -> str:
           return "my_source"

       async def fetch(self) -> list[RawArticle]:
           # Fetch and return articles as RawArticle instances
           ...
   ```

3. **Use `RawArticle`** to return normalized data. Required fields are `url` and `title`. Optional fields include `content`, `author`, `published_at`, `source_name`, `source_type`, and `extra_metadata`.

4. **Register the fetcher** by adding it to the fetcher list in `src/fetchers/__init__.py` or in the pipeline orchestrator where fetchers are instantiated.

5. **Write tests** in `tests/test_fetchers/`. Mock any HTTP calls or external APIs so tests run offline. See `tests/test_fetchers/test_fetchers.py` for examples of mocking patterns.

The pipeline calls `safe_fetch()` (inherited from `BaseFetcher`), which wraps `fetch()` with automatic retries and error handling so a single failing fetcher never crashes the pipeline.

## Code Style

- Follow PEP 8 conventions.
- Use Google-style docstrings for public functions and classes.
- Use type hints for function signatures.
- Keep imports sorted: standard library, third-party, local.

## Pull Request Process

1. Create a feature branch from `main`.
2. Make your changes and add tests for new functionality.
3. Run `pytest tests/` and confirm all tests pass.
4. Open a pull request with a clear description of the change.
