"""Process-wide runtime objects (locks, tokenizer, answer function) shared by the views."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable

from trid.onprem.config import Config


@dataclass
class Runtime:
    config: Config
    tokenizer: object
    answer: Callable[[object, str], dict]
    registry: 'SessionRegistry'  # defined in a later task; None until then
    gen_lock: threading.Lock
    parse_lock: threading.Lock
    insecure_dev: bool
    port: int


_runtime: Runtime | None = None


def install(rt: Runtime) -> None:
    global _runtime
    _runtime = rt


def get() -> Runtime:
    if _runtime is None:
        raise RuntimeError('runtime is not installed')
    return _runtime
