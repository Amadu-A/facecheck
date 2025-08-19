# mlface_verify/logging_config.py
import logging
import logging.config
import json
import inspect
from datetime import datetime

from .filters import AsciiOnlyFilter
from .logging_handlers import SizeAndTimeRotatingFileHandler

class JsonAdapter(logging.LoggerAdapter):
    """Адаптер: превращает сообщения в компактный JSON."""
    def process(self, msg, kwargs):
        frame = inspect.currentframe()
        outer_frames = inspect.getouterframes(frame)
        func_name = outer_frames[3].function if len(outer_frames) > 3 else "unknown"
        levelname = logging.getLevelName(self.logger.level)
        log_entry = {
            "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
            "level": levelname,
            "func": func_name,
            "message": msg,
        }
        return json.dumps(log_entry, ensure_ascii=False), kwargs

class ErrorLevelFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR

class InfoAndAboveFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.INFO

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "filters": {
        "ascii_only": {"()": AsciiOnlyFilter},
        "only_errors": {"()": ErrorLevelFilter},
        "info_and_above": {"()": InfoAndAboveFilter},
    },
    "formatters": {
        "standard": {
            "format": "%(levelname)s | %(name)s | %(asctime)s | line %(lineno)d | %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "app_stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "filters": ["ascii_only"],
        },
        "file_out": {
            "()": "mlface_verify.logging_handlers.SizeAndTimeRotatingFileHandler",
            "level": "INFO",
            "filename": "logout.log",
            "max_bytes": 10 * 1024 * 1024,
            "days": 3,
            "filters": ["ascii_only", "info_and_above"],
        },
        "file_err": {
            "()": "mlface_verify.logging_handlers.SizeAndTimeRotatingFileHandler",
            "level": "ERROR",
            "filename": "log_err.log",
            "max_bytes": 10 * 1024 * 1024,
            "days": 3,
            "filters": ["ascii_only", "only_errors"],
        },
    },
    "loggers": {
        "app": {
            "handlers": ["app_stdout", "file_out", "file_err"],
            "level": "INFO",
            "propagate": False,
        },
        "utils": {
            "handlers": ["file_out", "file_err"],
            "level": "DEBUG",
            "propagate": False,
        },
        # Глушим шум
        "urllib3": {"level": "WARNING", "handlers": [], "propagate": False},
        "http.client": {"level": "WARNING", "handlers": [], "propagate": False},
        "ultralytics": {"level": "WARNING", "handlers": [], "propagate": True},
    },
    "root": {"level": "CRITICAL", "handlers": []},
}
