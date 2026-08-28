"""Synchronization for model recognition and lifecycle operations."""

from collections.abc import Generator
from contextlib import contextmanager
from threading import Condition

from .errors import ServerBusyError


class ModelLifecycleGate:
    """Coordinate parallel recognition with exclusive model lifecycle changes.

    Recognition leases may coexist. Once a lifecycle operation starts, new
    recognition work is rejected so an unload cannot be starved by traffic;
    the lifecycle operation then waits for existing leases to finish.
    """

    def __init__(self) -> None:
        self._condition = Condition()
        self._active_recognitions = 0
        self._lifecycle_in_progress = False

    @contextmanager
    def recognition(self) -> Generator[None, None, None]:
        """Acquire a shared lease for one recognition request."""
        with self._condition:
            if self._lifecycle_in_progress:
                raise ServerBusyError(
                    "model lifecycle operation is in progress; try again later"
                )
            self._active_recognitions += 1
        try:
            yield
        finally:
            with self._condition:
                self._active_recognitions -= 1
                if self._active_recognitions == 0:
                    self._condition.notify_all()

    @contextmanager
    def lifecycle(self) -> Generator[None, None, None]:
        """Acquire exclusive access, after current recognition work drains."""
        with self._condition:
            if self._lifecycle_in_progress:
                raise ServerBusyError(
                    "model lifecycle operation is already in progress; try again later"
                )
            self._lifecycle_in_progress = True
            while self._active_recognitions:
                self._condition.wait()
        try:
            yield
        finally:
            with self._condition:
                self._lifecycle_in_progress = False
                self._condition.notify_all()
