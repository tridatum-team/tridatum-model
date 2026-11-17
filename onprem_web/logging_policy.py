"""Content-free logging: log lines carry the logger, level and an exception type or status, never content."""

import logging


class TypeOnlyFormatter(logging.Formatter):
    """'<logger> <levelname> <ExceptionType>' with exc_info, else '<logger> <levelname> <status_code or ->'.

    Message arguments, the request, the path and tracebacks are never written (any of them can hold
    file names, questions or document text).
    """

    def format(self, record: logging.LogRecord) -> str:
        if record.exc_info and record.exc_info[0] is not None:
            detail = record.exc_info[0].__name__
        else:
            status = getattr(record, 'status_code', None)
            detail = str(status) if isinstance(status, int) else '-'
        return f'{record.name} {record.levelname} {detail}'


def _print_type(exc_type) -> None:
    import sys
    print(getattr(exc_type, '__name__', 'Exception'), file=sys.stderr, flush=True)


def install_excepthooks() -> None:
    """Uncaught exceptions (main thread and worker threads) print only their type name, never a traceback."""
    import sys
    import threading
    sys.excepthook = lambda exc_type, exc, tb: _print_type(exc_type)
    threading.excepthook = lambda args: _print_type(args.exc_type)
