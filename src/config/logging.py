"""
Structured logging configuration using structlog.
Provides consistent, structured logging across the application.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

import structlog
from structlog.stdlib import ProcessorFormatter

from .settings import settings


def get_log_filename(base_name: str) -> str:
    """
    Generate a log filename with timestamp format: log_yyyymmddhhmmss.log
    
    Args:
        base_name: Base name for the log file (ignored, kept for compatibility)
        
    Returns:
        Log filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"log_{timestamp}.log"


def configure_logging(
    log_level: Optional[str] = None,
    json_logs: Optional[bool] = None,
    log_dir: Optional[str] = None,
) -> None:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                   Defaults to settings.log_level.
        json_logs: If True, output JSON format. If False, use console renderer.
                   Defaults to settings.json_logs.
        log_dir: Directory to store log files.
                   Defaults to settings.log_dir.
    """
    level = log_level or settings.log_level
    log_level_int = getattr(logging, level.upper(), logging.INFO)
    
    use_json = json_logs if json_logs is not None else settings.json_logs
    log_directory = log_dir or settings.log_dir

    # Shared processors for both structlog and stdlib logging
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Choose renderer based on environment
    if use_json:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.rich_traceback
            if _rich_available()
            else structlog.dev.plain_traceback,
        )
    
    # Configure structlog
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure ProcessorFormatter for stdlib logging integration
    formatter = ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level_int)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_directory:
        # Create log directory if it doesn't exist
        os.makedirs(log_directory, exist_ok=True)
        
        # Generate timestamped log filename in the log directory
        timestamped_log_file = os.path.join(log_directory, get_log_filename(None))
        
        # Use RotatingFileHandler for size-based log rotation
        # Logs rotate when they reach max_bytes, keeping backup_count backups
        # Old logs beyond retention_days are cleaned up separately
        file_handler = logging.handlers.RotatingFileHandler(
            timestamped_log_file,
            maxBytes=settings.log_max_bytes,
            backupCount=settings.log_backup_count,
            encoding="utf-8"
        )
        # Use JSON for file logs
        file_formatter = ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level_int)
        root_logger.addHandler(file_handler)
    
    root_logger.setLevel(log_level_int)
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def _rich_available() -> bool:
    """Check if rich library is available for better traceback formatting."""
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger instance.
    
    Args:
        name: Logger name. If None, uses the caller's module name.
        
    Returns:
        A bound structlog logger.
    """
    return structlog.get_logger(name)


def bind_request_context(
    request_id: Optional[str] = None,
    method: Optional[str] = None,
    path: Optional[str] = None,
    client_ip: Optional[str] = None,
    **kwargs,
) -> None:
    """
    Bind request context to the current context variables.
    All subsequent log entries will include these fields.
    
    Args:
        request_id: Unique request identifier
        method: HTTP method
        path: Request path
        client_ip: Client IP address
        **kwargs: Additional context fields
    """
    context = {}
    if request_id:
        context["request_id"] = request_id
    if method:
        context["method"] = method
    if path:
        context["path"] = path
    if client_ip:
        context["client_ip"] = client_ip
    context.update(kwargs)
    
    if context:
        structlog.contextvars.bind_contextvars(**context)


def clear_request_context() -> None:
    """Clear all bound context variables."""
    structlog.contextvars.clear_contextvars()
