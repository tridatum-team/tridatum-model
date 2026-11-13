"""Upload / ask / clear on a session's in-memory registry entry.

Direct ports of trid.onprem.server load() and ask() (and its /api/clear branch): same lock waits,
status codes, Korean messages and audit details. Nothing here writes file names, questions,
answers or PDF text anywhere: audit details carry only hashes, counts, seconds and status words,
and stderr gets exception class names only.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sys
import time
from dataclasses import dataclass
from typing import Callable

from onprem_web import runtime
from onprem_web.accounts.registry import Entry
from onprem_web.audit.services import log
from trid.document_case import CONTEXT_OVERFLOW
from trid.onprem.server import _STATUS_RE, BUSY_ASK, BUSY_PARSE, GENERIC_ERROR, MODEL_DOWN
from trid.pdf_documents import DocumentError
from trid.pdf_isolated import parse_pdf  # module-level so tests can patch it here

LOGIN_REQUIRED = '로그인이 필요합니다.'
NO_DOCUMENT = '이 브라우저에서 문서를 먼저 넣어 주세요.'
REUPLOAD = '문서를 다시 올려 주세요.'


@dataclass
class Result:
    status: int
    data: dict


def load_pdf(entry: Entry, body: bytes, username: str, ip,
             attach: Callable[[object, str], bool] | None = None) -> Result:
    """Parse `body` and keep the document on `entry`.

    `attach(doc, document_id)` stores the document only if the session is still registered (the
    view passes SessionRegistry.attach); without it the entry is updated directly.
    """
    rt = runtime.get()
    config = rt.config
    digest = hashlib.sha256(body).hexdigest()
    if not rt.parse_lock.acquire(timeout=config.pdf.wall_seconds + config.lock_wait):
        return Result(503, {'error': BUSY_PARSE})
    try:
        # The file name never leaves the browser: the parser always sees 'document.pdf'.
        try:
            doc = parse_pdf(body, 'document.pdf', rt.tokenizer, limits=config.pdf)
        except DocumentError as e:
            log('pdf_upload', username, ip, 'error', sha256=digest, status='error')
            return Result(400, {'error': str(e)})
        except Exception as e:
            print(type(e).__name__, file=sys.stderr)
            log('pdf_upload', username, ip, 'error', sha256=digest, status='error')
            return Result(500, {'error': GENERIC_ERROR})
        document_id = secrets.token_urlsafe(24)
        if attach is None:
            entry.document = doc
            entry.document_id = document_id
        elif not attach(doc, document_id):
            # The login ended while parsing (logout, admin action, idle expiry): keep nothing.
            log('pdf_upload', username, ip, 'error', sha256=digest, status='error', error='session_revoked')
            return Result(401, {'error': LOGIN_REQUIRED})
        log('pdf_upload', username, ip, 'ok', sha256=digest, pages=len(doc.pages), status='ok')
        return Result(200, {'document_id': document_id, 'name': doc.name,
                            'pages': doc.pages, 'warnings': doc.warnings})
    finally:
        rt.parse_lock.release()


def ask(entry: Entry, question, document_id, username: str, ip,
        still_valid: Callable[[], bool], had_document: bool) -> Result:
    rt = runtime.get()
    config = rt.config
    if (not isinstance(question, str) or len(question) > config.max_question
            or not isinstance(document_id, str)):
        return Result(400, {'error': f'질문은 {config.max_question:,}자 이하로 입력해 주세요.'})
    document = entry.document
    current = entry.document_id
    if document is None or current is None:
        return Result(409, {'error': REUPLOAD if had_document else NO_DOCUMENT})
    if not hmac.compare_digest(document_id.encode(), current.encode()):
        return Result(409, {'error': NO_DOCUMENT})
    started = time.monotonic()

    def audit(result, **detail):
        log('ask', username, ip, result, seconds=round(time.monotonic() - started, 2), **detail)

    if not rt.gen_lock.acquire(timeout=config.lock_wait):
        audit('error', status='error', error='busy')
        return Result(503, {'error': BUSY_ASK})
    try:
        result = rt.answer(document, question)
    except DocumentError as e:
        overflow = str(e) == CONTEXT_OVERFLOW
        audit('error', status='error', error='context_overflow' if overflow else 'bad_request')
        return Result(422 if overflow else 400, {'error': str(e)})
    except OSError:
        audit('error', status='error', error='model_unavailable')
        return Result(502, {'error': MODEL_DOWN})
    except Exception as e:
        print(type(e).__name__, file=sys.stderr)
        audit('error', status='error')
        return Result(500, {'error': GENERIC_ERROR})
    finally:
        rt.gen_lock.release()
    if not still_valid():
        audit('error', status='error', error='session_revoked')
        return Result(401, {'error': LOGIN_REQUIRED})
    status = result.get('status')
    audit('ok', status=status if isinstance(status, str) and _STATUS_RE.fullmatch(status) else 'unknown')
    return Result(200, {k: v for k, v in result.items() if k not in ('checked', 'raw_output')})


def clear(entry: Entry, username: str, ip) -> Result:
    entry.document = None
    entry.document_id = None
    log('clear', username, ip, 'ok')
    return Result(200, {'cleared': True})
