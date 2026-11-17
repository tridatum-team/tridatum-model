"""python -m onprem_web serve / create-admin / purge (ports test_onprem_cli.py).

Django settings are process-global and the test process already has its own, so every command runs
in a subprocess, the way the container runs it (python -I -m onprem_web ...).
"""

import contextlib
import hashlib
import io
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

import webapp_support as S  # noqa: F401  (boots Django for the in-process hook test)

from onprem_web import logging_policy
from trid.onprem.__main__ import TOKENIZER_SHA256

PW = 'correct horse 1'
ROOT = Path(__file__).resolve().parent.parent


def make_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        self.tok = self.root / 'tokenizer.json'
        self.tok.write_text('{"secret-tokenizer-content": 1}', encoding='utf-8')
        self.cfg = self.root / 'onprem.json'
        self.write_config()

    def write_config(self, **extra):
        data = {'hostname': 'x.example', 'data_dir': str(self.data), 'tokenizer_path': str(self.tok), 'port': 0}
        data.update(extra)
        self.cfg.write_text(json.dumps(data), encoding='utf-8')

    def env(self):
        env = {k: v for k, v in os.environ.items()
               if k not in ('TRID_WEB_CONFIG', 'TRID_WEB_DEV', 'TRID_TEST_ROUTES', 'DJANGO_SETTINGS_MODULE')}
        env['PYTHONIOENCODING'] = 'utf-8'
        return env

    def run_cli(self, *argv, stdin='', timeout=120):
        p = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-m', 'onprem_web', *argv], input=stdin, capture_output=True,
                           text=True, encoding='utf-8', timeout=timeout, env=self.env(), cwd=self.root)
        return p.returncode, p.stdout, p.stderr

    def db(self):
        con = sqlite3.connect(self.data / 'web.db')
        self.addCleanup(con.close)
        return con

    # -- create-admin ------------------------------------------------------------

    def test_create_admin_password_stdin(self):
        code, out, err = self.run_cli('create-admin', '--config', str(self.cfg), '--username', 'Root',
                                      '--password-stdin', stdin=PW + '\n')
        self.assertEqual(code, 0, err)
        self.assertIn('관리자 계정을 만들었습니다: root', out)
        self.assertNotIn(PW, out + err)
        row = self.db().execute('SELECT role, must_change, is_active FROM accounts_user WHERE username=?',
                                ('root',)).fetchone()
        self.assertEqual(row, ('admin', 0, 1))
        events = self.db().execute("SELECT username, detail FROM audit_event WHERE event='admin_create_user'").fetchall()
        self.assertEqual(len(events), 1)
        self.assertIsNone(events[0][0])
        self.assertEqual(json.loads(events[0][1]), {'target': 'root', 'action': 'create'})
        code, _, err = self.run_cli('create-admin', '--config', str(self.cfg), '--username', 'root',
                                    '--password-stdin', stdin=PW + '\n')
        self.assertNotEqual(code, 0)
        self.assertIn('이미', err)

    def test_create_admin_rejects_short_password(self):
        code, out, err = self.run_cli('create-admin', '--config', str(self.cfg), '--username', 'root',
                                      '--password-stdin', stdin='short\n')
        self.assertEqual(code, 1)
        self.assertTrue(err.strip())
        self.assertNotIn('Traceback', err)
        self.assertNotIn('short\n', out)
        count = self.db().execute("SELECT COUNT(*) FROM accounts_user").fetchone()[0]
        self.assertEqual(count, 0)

    def test_create_admin_getpass_must_match(self):
        # getpass reads the console, not stdin, so this one runs main() with getpass patched in a child.
        code = (
            'import sys\nfrom unittest import mock\nimport onprem_web.__main__ as m\n'
            f'with mock.patch("getpass.getpass", side_effect=[{PW!r}, {PW + "x"!r}]):\n'
            f'    sys.exit(m.main(["create-admin", "--config", {str(self.cfg)!r}, "--username", "root"]))\n')
        p = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-c', code], capture_output=True, text=True, encoding='utf-8',
                           env=self.env(), timeout=120)
        self.assertEqual(p.returncode, 1, p.stderr)
        self.assertIn('두 비밀번호가 일치하지 않습니다.', p.stderr)

    # -- purge -------------------------------------------------------------------

    def test_purge_removes_old_and_logs_count(self):
        self.write_config(retention_days=10)
        self.assertEqual(self.run_cli('create-admin', '--config', str(self.cfg), '--username', 'root',
                                      '--password-stdin', stdin=PW + '\n')[0], 0)
        con = self.db()
        insert = "INSERT INTO audit_event (ts, username, ip, event, result, detail) VALUES (?, NULL, NULL, ?, 'ok', '{}')"
        con.execute(insert, (time.time() - 11 * 86400, 'clear'))
        con.execute(insert, (time.time() - 9 * 86400, 'logout'))
        con.commit()
        code, out, err = self.run_cli('purge', '--config', str(self.cfg))
        self.assertEqual(code, 0, err)
        self.assertIn('1건을 정리했습니다.', out)
        rows = [(e, json.loads(d)) for e, d in con.execute('SELECT event, detail FROM audit_event ORDER BY id')]
        self.assertEqual(rows, [('admin_create_user', {'target': 'root', 'action': 'create'}), ('logout', {}),
                                ('retention_purge', {'count': 1})])

    # -- serve -------------------------------------------------------------------

    def test_serve_refuses_wrong_tokenizer_in_prod(self):
        code, out, err = self.run_cli('serve', '--config', str(self.cfg), timeout=60)
        self.assertEqual(code, 1)
        self.assertIn(TOKENIZER_SHA256, err)
        self.assertNotIn('secret-tokenizer-content', out + err)
        self.assertNotIn('서버를 시작합니다', out)
        self.assertFalse((self.data / 'web.db').exists())

    def start_dev(self, port):
        make_tokenizer(self.tok)
        self.write_config(port=port)
        p = subprocess.Popen([sys.executable, '-I', '-X', 'utf8', '-m', 'onprem_web', 'serve', '--config', str(self.cfg),
                              '--insecure-dev'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding='utf-8', env=self.env(), cwd=self.root)

        def stop():
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(10)
            p.stdout.close()
            p.stderr.close()
        self.addCleanup(stop)
        first = p.stdout.readline()  # '서버를 시작합니다: host:port' once listening (or '' if it exited)
        return p, first

    def test_serve_dev_warns_on_tokenizer_mismatch(self):
        p, first = self.start_dev(free_port())
        self.assertIn('서버를 시작합니다: 127.0.0.1:', first)
        p.terminate()
        _, err = p.communicate(timeout=20)
        self.assertIn('경고(개발용): 토크나이저 sha256이 기대값과 다릅니다.', err)
        self.assertNotIn('Traceback', err)

    def test_serve_applies_migrations_on_empty_dir(self):
        port = free_port()
        p, first = self.start_dev(port)
        self.assertIn(f'127.0.0.1:{port}', first)
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=10) as r:
            self.assertEqual(r.status, 200)
        tables = {t for (t,) in self.db().execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({'accounts_user', 'audit_event', 'django_session'} <= tables, tables)
        purges = self.db().execute("SELECT COUNT(*) FROM audit_event WHERE event='retention_purge'").fetchone()[0]
        self.assertEqual(purges, 1)  # the purge also runs once at start

    def test_dev_binds_loopback_only(self):
        port = free_port()
        p, first = self.start_dev(port)
        self.assertIn(f'127.0.0.1:{port}', first)
        addresses = {info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)}
        addresses.discard('127.0.0.1')
        for address in addresses:
            with self.assertRaises(OSError, msg=address):
                socket.create_connection((address, port), timeout=1).close()
        socket.create_connection(('127.0.0.1', port), timeout=5).close()

    def test_offline_env_is_forced(self):
        code = ('import os, onprem_web.__main__\n'
                'print(os.environ["HF_HUB_OFFLINE"], os.environ["TOKENIZERS_PARALLELISM"])\n')
        p = subprocess.run([sys.executable, '-I', '-X', 'utf8', '-c', code], capture_output=True, text=True, env=self.env(),
                           timeout=60)
        self.assertEqual(p.stdout.split(), ['1', 'false'], p.stderr)


class HookTests(unittest.TestCase):
    def test_excepthooks_print_type_only(self):
        saved = sys.excepthook, threading.excepthook
        self.addCleanup(setattr, sys, 'excepthook', saved[0])
        self.addCleanup(setattr, threading, 'excepthook', saved[1])
        logging_policy.install_excepthooks()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            try:
                raise ValueError('secret-file-name.pdf')
            except ValueError as e:
                sys.excepthook(type(e), e, e.__traceback__)
                threading.excepthook(threading.ExceptHookArgs((type(e), e, e.__traceback__, None)))
        self.assertEqual(err.getvalue().splitlines(), ['ValueError', 'ValueError'])

    def test_root_logger_is_type_only(self):
        from django.conf import settings
        self.assertEqual(settings.LOGGING['root']['handlers'], ['stderr'])


class SweepTests(unittest.TestCase):
    def test_sweep_drops_registry_keys_without_db_session(self):
        from django.contrib.sessions.backends.db import SessionStore
        from onprem_web.__main__ import sweep
        rt = S.make_runtime()
        live = SessionStore()
        live.create()
        self.addCleanup(live.delete)
        self.assertTrue(rt.registry.admit(live.session_key, 'alice'))
        self.assertTrue(rt.registry.admit('gone-key', 'bob'))
        sweep(rt.registry)
        self.assertEqual(rt.registry.keys(), {live.session_key})


class WaitressTests(unittest.TestCase):
    def test_upload_body_never_spills_to_a_temp_file(self):
        """waitress buffers bodies over inbuf_overflow (default 512 KB) in a TemporaryFile; uploads must stay in memory."""
        import dataclasses
        import http.client
        from unittest import mock
        from onprem_web.__main__ import make_waitress
        from trid.onprem.config import load_config
        config = dataclasses.replace(load_config(S.CONFIG_PATH), port=0)
        seen = []

        def app(environ, start_response):
            seen.append(len(environ['wsgi.input'].read()))
            start_response('200 OK', [('Content-Type', 'text/plain')])
            return [b'ok']

        server = make_waitress(app, config, '127.0.0.1')
        threading.Thread(target=server.run, daemon=True).start()
        self.addCleanup(server.close)
        body = b'%PDF' + b'x' * (int(config.pdf.max_bytes) - 4)
        real = tempfile.TemporaryFile
        with mock.patch('tempfile.TemporaryFile', side_effect=real) as spill:
            conn = http.client.HTTPConnection('127.0.0.1', server.effective_port, timeout=30)
            conn.request('POST', '/api/load', body=body, headers={'Content-Type': 'application/pdf'})
            r = conn.getresponse()
            r.read()
            conn.close()
        self.assertEqual((r.status, seen), (200, [len(body)]))
        spill.assert_not_called()
