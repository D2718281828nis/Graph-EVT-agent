"""Optional progress reporting for callers that embed the library.

``verbose=False`` (the default) is silent. ``verbose=True`` draws a progress
bar on stderr: ``tqdm`` is used when installed, otherwise a dependency-free
text bar. A ``callback(event)`` receives every update as a
:class:`ProgressEvent`, so GUIs, web services and loggers can render progress
themselves without parsing terminal output.
"""

from dataclasses import dataclass
from importlib import import_module
import sys
import time
import warnings
from typing import Any, Callable


@dataclass(frozen=True)
class ProgressEvent:
    """One progress update: ``step`` of ``total`` for the task ``desc``."""

    desc: str
    step: int
    total: int
    message: str = ""
    elapsed: float = 0.0

    @property
    def fraction(self) -> float:
        return self.step / self.total if self.total else 1.0


ProgressCallback = Callable[[ProgressEvent], None]


class _TextBar:
    """Minimal single-line progress bar used when tqdm is unavailable."""

    def __init__(self, total: int, desc: str, stream: Any, width: int = 24):
        self.total, self.desc, self.stream, self.width = total, desc, stream, width
        self.step, self.message = 0, ""
        self._render()

    def _render(self) -> None:
        filled = int(self.width * self.step / self.total) if self.total else self.width
        bar = "#" * filled + "-" * (self.width - filled)
        suffix = f" {self.message}" if self.message else ""
        self.stream.write(f"\r{self.desc}: [{bar}] {self.step}/{self.total}{suffix}\033[K")
        self.stream.flush()

    def update(self, step: int, message: str) -> None:
        self.step, self.message = step, message
        self._render()

    def close(self) -> None:
        self.stream.write("\n")
        self.stream.flush()


class _TqdmBar:
    def __init__(self, total: int, desc: str, stream: Any, tqdm: Any):
        self._bar = tqdm(total=total, desc=desc, file=stream, leave=True)
        self._step = 0

    def update(self, step: int, message: str) -> None:
        if message:
            self._bar.set_postfix_str(message, refresh=False)
        self._bar.update(step - self._step)  # redraws at most every mininterval
        self._step = step

    def close(self) -> None:
        self._bar.close()


def _make_bar(total: int, desc: str, stream: Any) -> Any:
    try:
        with warnings.catch_warnings():  # tqdm.auto warns when ipywidgets is missing
            warnings.simplefilter("ignore")
            tqdm = import_module("tqdm.auto").tqdm
    except ImportError:
        return _TextBar(total, desc, stream)
    return _TqdmBar(total, desc, stream, tqdm)


class Progress:
    """Context manager that reports ``advance`` calls to a bar and/or callback.

    ``verbose`` may be ``False``/``0`` (silent), ``True``/``1`` (one bar per
    top-level task) or ``2`` (also bars for nested tasks such as training
    epochs inside an evaluation). ``depth`` is the nesting level of the task.
    """

    def __init__(self, total: int, desc: str, verbose: bool | int = False,
                 callback: ProgressCallback | None = None, depth: int = 0,
                 stream: Any = None):
        self.total, self.desc, self.callback = max(int(total), 0), desc, callback
        self.step = 0
        self._start = time.perf_counter()
        show = int(verbose) > depth
        self._bar = _make_bar(self.total, desc, stream or sys.stderr) if show else None

    def advance(self, message: str = "", steps: int = 1) -> None:
        self.step = min(self.total, self.step + steps)
        if self._bar is not None:
            self._bar.update(self.step, message)
        if self.callback is not None:
            self.callback(ProgressEvent(self.desc, self.step, self.total, message,
                                        time.perf_counter() - self._start))

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def __enter__(self) -> "Progress":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
