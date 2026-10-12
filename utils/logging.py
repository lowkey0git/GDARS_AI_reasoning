"""
Production Logging
==================
Two modes controlled by the ENVIRONMENT setting:

  development — human-readable Rich console output, no file
  production  — structured JSON to stdout (picked up by CloudWatch/Datadog)

Engineering note on why we don't use structlog's PrintLogger:
  PrintLogger has no .name attribute. Python's standard logging internals
  call .name on loggers in several places (particularly in handlers and
  the logging.Manager). This crashes silently or loudly depending on which
  version of Python you're on. We use stdlib logging as the backend and
  structlog only as a processor chain sitting on top of it.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

import structlog
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: Optional[str] = None,
                  environment: str = "development") -> None:
    """
    Configure structlog + stdlib logging.

    Args:
        level:       One of DEBUG / INFO / WARNING / ERROR
        log_file:    If set, also write to a rotating file
        environment: "development" → Rich console; "production" → JSON stdout
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # ── Handlers ──────────────────────────────────────────────
    handlers: list[logging.Handler] = []

    if environment == "development":
        handlers.append(RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            markup=True,
            show_path=False,
            level=numeric_level,
        ))
    else:
        # Production: plain stdout so Docker/ECS log drivers capture it
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        ))
        handlers.append(stream_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        handlers.append(file_handler)

    # ── Root logger ───────────────────────────────────────────
    # force=True clears any handlers set before this call
    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)

    # ── Structlog config ──────────────────────────────────────
    # We bind structlog to stdlib so it uses the handlers above.
    # This is the correct way to combine them — structlog processes
    # the record, then hands it to stdlib, which dispatches to handlers.
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
    ]

    if environment == "production":
        shared_processors.append(structlog.processors.JSONRenderer())
    else:
        shared_processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=shared_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),   # ← uses stdlib, not PrintLogger
        wrapper_class=structlog.stdlib.BoundLogger,        # ← has .name, .handlers, etc.
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Returns a structlog logger bound to a stdlib logger.
    This logger has a .name attribute so it's safe everywhere.
    """
    return structlog.get_logger(name)
