"""On-prem commands of the Django build: serve, create-admin, purge, import-legacy.

Option names, messages and exit codes are those of trid/onprem/__main__.py. Each command points
TRID_WEB_CONFIG / TRID_WEB_DEV at its config before django.setup(). Nothing printed here contains
questions, file names or document text.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from trid.onprem.__main__ import TOKENIZER_SHA256, load_tokenizer  # noqa: E402
from trid.onprem.config import load_config  # noqa: E402

PURGE_INTERVAL = 24 * 3600
PRUNE_INTERVAL = 60
UPLOAD_SLACK = 64 * 1024  # headers and framing on top of the largest allowed PDF


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _setup(config_path: Path, dev: bool = False):
    """Load the config, point Django at it, set Django up and apply migrations."""
    config = load_config(config_path)
    os.environ['TRID_WEB_CONFIG'] = str(config_path)
    if dev:
        os.environ['TRID_WEB_DEV'] = '1'
    else:
        os.environ.pop('TRID_WEB_DEV', None)
    os.environ.pop('TRID_TEST_ROUTES', None)
    os.environ['DJANGO_SETTINGS_MODULE'] = 'onprem_web.settings'
    config.data_dir.mkdir(parents=True, exist_ok=True)
    import django
    django.setup()
    from django.core.management import call_command
    call_command('migrate', interactive=False, verbosity=0)
    return config


def run_purge(config, now: float | None = None) -> int:
    from onprem_web.audit.services import log, purge_before
    cutoff = (time.time() if now is None else now) - config.retention_days * 86400
    count = purge_before(cutoff)
    log('retention_purge', None, None, 'ok', count=count)
    return count


def cmd_create_admin(args) -> int:
    load_config(args.config)  # a bad config fails before the password is asked for
    if args.password_stdin:
        password = sys.stdin.readline().rstrip('\r\n')
    else:
        password = getpass.getpass('비밀번호: ')
        if getpass.getpass('비밀번호 확인: ') != password:
            return _fail('두 비밀번호가 일치하지 않습니다.')
    _setup(args.config)
    from onprem_web.accounts.services import create_user
    from onprem_web.audit.services import log
    try:
        user = create_user(args.username, password, 'admin', must_change=False)
    except ValueError as e:
        return _fail(str(e))
    log('admin_create_user', None, None, 'ok', target=user.username, action='create')
    print(f'관리자 계정을 만들었습니다: {user.username}')
    return 0


def cmd_purge(args) -> int:
    config = _setup(args.config)
    count = run_purge(config)
    print(f'{count}건을 정리했습니다.')
    return 0


def cmd_import_legacy(args) -> int:
    from django.db import DatabaseError
    from onprem_web.legacy_import import LegacyImportError, import_legacy, legacy_snapshot

    config = load_config(args.config)
    # Reject a source/target alias and unsupported source before Django can open
    # its writable database or apply migrations.
    with legacy_snapshot(args.source, config.data_dir / 'web.db') as source:
        try:
            _setup(args.config)
            accounts, events = import_legacy(source)
        except DatabaseError:
            raise LegacyImportError('Django target could not be prepared; no records imported.') from None
    print(f'Imported accounts={accounts} audit_events={events}')
    return 0


def sweep(registry) -> None:
    """Idle logins out of the registry (and their DB rows), then registry keys whose DB session is gone."""
    from django.contrib.sessions.models import Session
    from onprem_web.accounts.middleware import prune
    prune(registry)
    keys = registry.keys()
    if keys:
        live = set(Session.objects.filter(session_key__in=keys).values_list('session_key', flat=True))
        for key in keys - live:
            registry.drop(key)


def _background(config, registry, stop: threading.Event) -> None:
    from django.db import close_old_connections
    last_purge = time.monotonic()
    while not stop.wait(PRUNE_INTERVAL):
        try:
            sweep(registry)
            if time.monotonic() - last_purge >= PURGE_INTERVAL:
                last_purge = time.monotonic()
                run_purge(config)
        except Exception as e:  # keep the maintenance thread alive; class name only
            print(type(e).__name__, file=sys.stderr)
        finally:
            close_old_connections()


def make_waitress(application, config, host: str):
    """The WSGI server: one thread per slot, request bodies capped at the PDF limit plus framing.

    inbuf_overflow = the same cap, so waitress keeps every accepted body in memory: above its
    default (512 KB) it would buffer the upload in a TemporaryFile on disk.
    """
    from waitress import create_server
    limit = int(config.pdf.max_bytes) + UPLOAD_SLACK
    return create_server(application, host=host, port=config.port, threads=config.slots,
                         max_request_body_size=limit, inbuf_overflow=limit, ident=None,
                         clear_untrusted_proxy_headers=True)


def cmd_serve(args) -> int:
    if args.insecure_dev and args.bind not in (None, '127.0.0.1'):
        return _fail('--insecure-dev에서는 --bind 127.0.0.1만 사용할 수 있습니다.')
    host = args.bind if args.bind is not None else ('127.0.0.1' if args.insecure_dev else '0.0.0.0')
    config = load_config(args.config)
    try:
        digest = hashlib.sha256(config.tokenizer_path.read_bytes()).hexdigest()
    except OSError:
        return _fail('토크나이저 파일을 읽을 수 없습니다.')
    if digest != TOKENIZER_SHA256:
        message = f'토크나이저 sha256이 기대값과 다릅니다. 기대값: {TOKENIZER_SHA256}'
        if not args.insecure_dev:
            return _fail(message)
        print('경고(개발용): ' + message, file=sys.stderr)
    try:
        tok = load_tokenizer(config.tokenizer_path)
    except Exception as e:
        return _fail(f'토크나이저를 불러올 수 없습니다. ({type(e).__name__})')

    from onprem_web.logging_policy import install_excepthooks
    install_excepthooks()
    logging.captureWarnings(True)
    config = _setup(args.config, dev=args.insecure_dev)

    from django.core.wsgi import get_wsgi_application

    from onprem_web import runtime
    from onprem_web.accounts.registry import SessionRegistry
    from trid.onprem.server import _default_answer

    run_purge(config)
    application = get_wsgi_application()
    server = make_waitress(application, config, host)
    port = server.effective_port
    registry = SessionRegistry(config.max_sessions, config.idle_seconds, config.per_minute)
    runtime.install(runtime.Runtime(
        config=config, tokenizer=tok, answer=_default_answer(config), registry=registry,
        gen_lock=threading.Lock(), parse_lock=threading.Lock(), insecure_dev=args.insecure_dev, port=port))
    stop = threading.Event()
    threading.Thread(target=_background, args=(config, registry, stop), daemon=True).start()

    def on_term(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, on_term)
    except (ValueError, OSError):
        pass
    print(f'서버를 시작합니다: {host}:{port}', flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m onprem_web')
    sub = parser.add_subparsers(dest='command', required=True)
    serve = sub.add_parser('serve', help='서버 시작')
    serve.add_argument('--config', required=True, type=Path)
    serve.add_argument('--bind', help='수신 주소 (기본: 0.0.0.0, 개발용: 127.0.0.1)')
    serve.add_argument('--insecure-dev', action='store_true', help='개발용: 127.0.0.1, HTTP, 쿠키 trid-dev')
    admin = sub.add_parser('create-admin', help='관리자 계정 만들기')
    admin.add_argument('--config', required=True, type=Path)
    admin.add_argument('--username', required=True)
    admin.add_argument('--password-stdin', action='store_true', help='표준 입력 한 줄을 비밀번호로 사용')
    purge = sub.add_parser('purge', help='보관 기간이 지난 사용 기록 삭제')
    purge.add_argument('--config', required=True, type=Path)
    legacy = sub.add_parser('import-legacy', help='빈 Django DB로 기존 계정과 이용 기록 가져오기')
    legacy.add_argument('--config', required=True, type=Path)
    legacy.add_argument('--source', required=True, type=Path)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handler = {'serve': cmd_serve, 'create-admin': cmd_create_admin, 'purge': cmd_purge,
               'import-legacy': cmd_import_legacy}[args.command]
    try:
        return handler(args)
    except (ValueError, OSError) as e:
        # Config/ValueError messages are ours; OSError text can carry a path but never document data.
        return _fail(f'오류: {e}' if isinstance(e, ValueError) else f'오류: {type(e).__name__}')


if __name__ == '__main__':
    sys.exit(main())
