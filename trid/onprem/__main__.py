"""On-prem commands: serve, create-admin, purge.

Only the standard library and `tokenizers` are imported (no torch, no numpy).
Nothing printed here contains questions, file names or document text.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import os
import signal
import sys
import threading
import time
from pathlib import Path

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from trid.onprem.auth import Auth  # noqa: E402
from trid.onprem.config import load_config  # noqa: E402
from trid.onprem.server import make_server  # noqa: E402
from trid.onprem.store import Store  # noqa: E402

TOKENIZER_SHA256 = '73551e65acfa80834c235602900f31dc353ce230bfb69939ffdf44ca26ebc58d'
PURGE_INTERVAL = 24 * 3600
PRUNE_INTERVAL = 60


def load_tokenizer(path: Path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(path))


def open_store(config) -> Store:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    return Store(config.data_dir / 'onprem.db')


def run_purge(store: Store, config, now: float | None = None) -> int:
    cutoff = (time.time() if now is None else now) - config.retention_days * 86400
    count = store.purge_before(cutoff)
    store.log('retention_purge', None, None, 'ok', count=count)
    return count


def _fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def cmd_create_admin(args) -> int:
    config = load_config(args.config)
    if args.password_stdin:
        password = sys.stdin.readline().rstrip('\r\n')
    else:
        password = getpass.getpass('비밀번호: ')
        if getpass.getpass('비밀번호 확인: ') != password:
            return _fail('두 비밀번호가 일치하지 않습니다.')
    store = open_store(config)
    try:
        try:
            user = store.create_user(args.username, password, 'admin', must_change=False)
        except ValueError as e:
            return _fail(str(e))
        store.log('admin_create_user', None, None, 'ok', target=user.username, action='create')
    finally:
        store.close()
    print(f'관리자 계정을 만들었습니다: {user.username}')
    return 0


def cmd_purge(args) -> int:
    config = load_config(args.config)
    store = open_store(config)
    try:
        count = run_purge(store, config)
    finally:
        store.close()
    print(f'{count}건을 정리했습니다.')
    return 0


def _background(store, auth, config, stop: threading.Event) -> None:
    last_purge = time.monotonic()
    while not stop.wait(PRUNE_INTERVAL):
        try:
            auth.prune()
            if time.monotonic() - last_purge >= PURGE_INTERVAL:
                last_purge = time.monotonic()
                run_purge(store, config)
        except Exception as e:  # keep the maintenance thread alive; class name only
            print(type(e).__name__, file=sys.stderr)


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
    store = open_store(config)
    auth = Auth(store, max_sessions=config.max_sessions, idle_seconds=config.idle_seconds)
    run_purge(store, config)
    server = make_server(config, store, auth, tok, insecure_dev=args.insecure_dev, host=host)
    stop = threading.Event()
    threading.Thread(target=_background, args=(store, auth, config, stop), daemon=True).start()

    def on_term(signum, frame):
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, on_term)
    except (ValueError, OSError):
        pass
    print(f'서버를 시작합니다: {host}:{server.server_address[1]}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='python -m trid.onprem')
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
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    handler = {'serve': cmd_serve, 'create-admin': cmd_create_admin, 'purge': cmd_purge}[args.command]
    try:
        return handler(args)
    except (ValueError, OSError) as e:
        # Config/ValueError messages are ours; OSError text can carry a path but never document data.
        return _fail(f'오류: {e}' if isinstance(e, ValueError) else f'오류: {type(e).__name__}')


if __name__ == '__main__':
    sys.exit(main())
