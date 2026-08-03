"""Alembic must not tear down application logging when driven programmatically.

alembic/env.py calls logging.config.fileConfig(). By default that disables
every logger created before it runs, and applies alembic.ini's root level of
WARNING. The pipeline scripts call init_db() at startup, so that silenced all
src.* logging for the rest of the process — which is how a batch-parsing
failure went unnoticed long enough to leave ~1000 articles unprocessed.

These tests drive env.py via `stamp` rather than `upgrade`: it runs the same
env.py (and therefore the same fileConfig call) without requiring the
migration chain to apply.
"""

import logging
from unittest.mock import patch

import pytest
from alembic.config import Config

from alembic import command
from src.storage.database import init_db

ALEMBIC_INI = "alembic.ini"
PIPELINE_LOGGERS = ("src.pipeline.processor", "src.llm.summarizer")


@pytest.fixture
def preserved_logging():
    """Save and restore global logging state around the test."""
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_disabled = {
        name: logging.getLogger(name).disabled for name in PIPELINE_LOGGERS
    }
    yield
    root.setLevel(saved_level)
    root.handlers = saved_handlers
    for name, was_disabled in saved_disabled.items():
        logging.getLogger(name).disabled = was_disabled


def _config(tmp_path, configure_logger=None):
    cfg = Config(ALEMBIC_INI)
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{tmp_path / 'stamp.db'}")
    if configure_logger is not None:
        cfg.attributes["configure_logger"] = configure_logger
    return cfg


def test_programmatic_run_leaves_pipeline_loggers_enabled(tmp_path, preserved_logging):
    for name in PIPELINE_LOGGERS:
        logging.getLogger(name)  # exists before Alembic runs, as in real scripts

    command.stamp(_config(tmp_path, configure_logger=False), "head")

    for name in PIPELINE_LOGGERS:
        assert not logging.getLogger(name).disabled, f"{name} was disabled by Alembic"


def test_programmatic_run_does_not_raise_root_log_level(tmp_path, preserved_logging):
    """alembic.ini sets root to WARNING; that must not leak into the app."""
    logging.getLogger().setLevel(logging.INFO)

    command.stamp(_config(tmp_path, configure_logger=False), "head")

    assert logging.getLogger().level <= logging.INFO


def test_pipeline_logger_still_emits_after_programmatic_run(
    tmp_path, preserved_logging, caplog
):
    """The property that actually matters: log records reach handlers."""
    logger = logging.getLogger("src.pipeline.processor")
    logging.getLogger().setLevel(logging.INFO)

    command.stamp(_config(tmp_path, configure_logger=False), "head")

    with caplog.at_level(logging.INFO):
        logger.info("Processing batch 1")
        logger.warning("2 article(s) will not be processed")

    assert "Processing batch 1" in caplog.text
    assert "will not be processed" in caplog.text


def test_cli_run_still_configures_logging_without_disabling_loggers(
    tmp_path, preserved_logging
):
    """Without the flag (i.e. the `alembic` CLI), fileConfig still runs — but
    disable_existing_loggers=False keeps application loggers alive."""
    for name in PIPELINE_LOGGERS:
        logging.getLogger(name)

    command.stamp(_config(tmp_path), "head")

    for name in PIPELINE_LOGGERS:
        assert not logging.getLogger(name).disabled


def test_init_db_opts_out_on_fresh_database(tmp_path):
    """A fresh database is stamped, not upgraded — the flag must be set there too."""
    captured = {}

    def fake_stamp(cfg, revision):
        captured["configure_logger"] = cfg.attributes.get("configure_logger")

    with patch("alembic.command.stamp", side_effect=fake_stamp):
        init_db(f"sqlite:///{tmp_path / 'fresh.db'}")

    assert captured["configure_logger"] is False


def test_init_db_opts_out_on_existing_database(tmp_path):
    """init_db must set the flag, or the fix never reaches the pipeline scripts."""
    url = f"sqlite:///{tmp_path / 'existing.db'}"
    init_db(url)  # build it once so the second call takes the upgrade path

    captured = {}

    def fake_upgrade(cfg, revision):
        captured["configure_logger"] = cfg.attributes.get("configure_logger")

    with patch("alembic.command.upgrade", side_effect=fake_upgrade):
        init_db(url)

    assert captured["configure_logger"] is False
