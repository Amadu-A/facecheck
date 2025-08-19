# mlface_verify/decorators.py
import functools
import logging
import time

logger = logging.getLogger("app")

def log_call(name: str = None):
    """Декоратор: логируем вход/выход функции/метода с таймингом."""
    def deco(fn):
        display = name or fn.__qualname__
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"CALL {display}")
            try:
                res = fn(*args, **kwargs)
                took = (time.time() - start) * 1000
                logger.info(f"OK {display} ({took:.1f} ms)")
                return res
            except Exception as e:
                took = (time.time() - start) * 1000
                logger.error(f"ERR {display}: {e} ({took:.1f} ms)")
                raise
        return wrapper
    return deco
