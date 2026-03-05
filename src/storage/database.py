import threading
from contextlib import contextmanager
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from src.config import DATA_DIR, load_settings

_engine = None
_engine_lock = threading.Lock()


def get_engine(database_url: str | None = None):
    """Return the SQLAlchemy engine, creating it on first call.

    Uses a module-level singleton. Pass a URL to create a non-default engine.
    """
    global _engine

    if _engine is not None and database_url is None:
        return _engine

    with _engine_lock:
        # Double-check after acquiring lock
        if _engine is not None and database_url is None:
            return _engine

        if database_url is None:
            settings = load_settings()
            database_url = settings.database_url

        # Ensure data directory exists for SQLite
        if "sqlite" in database_url:
            db_path = database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        engine = create_engine(database_url, echo=False)

        # Cache the default engine
        if _engine is None:
            _engine = engine

        return engine


def init_db(database_url: str | None = None) -> None:
    """Create/migrate all tables using Alembic if available.

    Falls back to SQLModel.metadata.create_all() if the Alembic
    migrations directory doesn't exist (e.g. in tests or fresh checkouts).
    """
    engine = get_engine(database_url)

    alembic_dir = Path(__file__).parent.parent.parent / "alembic"
    if alembic_dir.exists():
        try:
            from alembic import command
            from alembic.config import Config
            alembic_cfg = Config(str(alembic_dir.parent / "alembic.ini"))
            alembic_cfg.set_main_option("sqlalchemy.url", str(engine.url))
            command.upgrade(alembic_cfg, "head")
            return
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).error(f"Alembic migration failed: {e}")
            raise

    SQLModel.metadata.create_all(engine)


def get_session(database_url: str | None = None) -> Session:
    """Open a new database session. Caller is responsible for closing it."""
    engine = get_engine(database_url)
    return Session(engine)


@contextmanager
def session_scope(database_url: str | None = None):
    """Context manager that yields a Session and ensures it is closed."""
    session = get_session(database_url)
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
