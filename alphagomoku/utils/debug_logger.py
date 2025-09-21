"""Simple debug logging utility that writes to debug.log"""

import os
import threading
from datetime import datetime
from typing import Optional


class DebugLogger:
    """Thread-safe debug logger that writes to debug.log"""

    def __init__(self, enabled: bool = True, log_file: str = "debug.log"):
        self._enabled = enabled
        self._file_lock = threading.Lock()
        self._log_file = log_file

    def debug(self, message: str, flush: bool = True) -> None:
        """Write a debug message to debug.log with timestamp and PID"""
        if not self._enabled:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        pid = os.getpid()
        formatted_message = f"[{timestamp}] PID {pid}: {message}\n"

        with self._file_lock:
            with open(self._log_file, 'a', encoding='utf-8') as f:
                f.write(formatted_message)
                if flush:
                    f.flush()


class NoOpLogger:
    """No-operation logger that does nothing"""

    def debug(self, message: str, flush: bool = True) -> None:
        """No-op debug method"""
        pass