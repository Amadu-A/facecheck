# mlface_verify/logging_handlers.py
import os
import time
from logging.handlers import BaseRotatingHandler

class SizeAndTimeRotatingFileHandler(BaseRotatingHandler):
    """
    Ротация по размеру (max_bytes) И по времени (backup_count дней).
    - Если прошло более backup_count суток с момента создания файла — ротируем.
    - Если размер превысил max_bytes — ротируем.
    """
    def __init__(self, filename, mode='a', max_bytes=10 * 1024 * 1024, days=3, encoding=None, delay=False):
        self.max_bytes = int(max_bytes)
        self.days = int(days)
        self.base_ctime = None  # время создания базового файла
        super().__init__(filename, mode, encoding, delay)
        self._ensure_ctime()

    def _ensure_ctime(self):
        try:
            self.base_ctime = os.path.getctime(self.baseFilename)
        except OSError:
            self.base_ctime = time.time()

    def shouldRollover(self, record):
        if self.stream is None:
            self.stream = self._open()

        # Размер
        if self.max_bytes > 0:
            self.stream.flush()
            if self.stream.tell() >= self.max_bytes:
                return True

        # Время
        now = time.time()
        if (now - self.base_ctime) >= self.days * 86400:
            return True

        return False

    def doRollover(self):
        if self.stream:
            self.stream.close()

        # Сдвигаем backup файлы
        for i in range(self.days - 1, 0, -1):
            sfn = f"{self.baseFilename}.{i}"
            dfn = f"{self.baseFilename}.{i + 1}"
            if os.path.exists(sfn):
                if os.path.exists(dfn):
                    os.remove(dfn)
                os.rename(sfn, dfn)

        # Текущий файл -> .1
        dfn = f"{self.baseFilename}.1"
        if os.path.exists(dfn):
            os.remove(dfn)
        if os.path.exists(self.baseFilename):
            os.rename(self.baseFilename, dfn)

        self.stream = self._open()
        self._ensure_ctime()
