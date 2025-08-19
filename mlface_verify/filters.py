# mlface_verify/filters.py
import logging

class AsciiOnlyFilter(logging.Filter):
    """Пример фильтра. Здесь пропускаем всё."""
    def filter(self, record: logging.LogRecord) -> bool:
        return True
