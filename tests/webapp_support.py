"""Shared fixtures for the Django on-prem tests.

Django settings are process-global, so one temp data_dir/config is created per process, before settings load.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path

HOST = 'test.example'
ORIGIN = 'https://' + HOST
PW = 'correct horse 1'

_TMP = Path(tempfile.mkdtemp(prefix='trid-web-test-'))
atexit.register(shutil.rmtree, _TMP, True)
DATA_DIR = _TMP / 'data'
DATA_DIR.mkdir()
CONFIG_PATH = _TMP / 'onprem.json'
CONFIG_PATH.write_text(json.dumps({
    'hostname': HOST, 'port': 0, 'data_dir': str(DATA_DIR),
    'tokenizer_path': str(_TMP / 'tokenizer.json'),
}), encoding='utf-8')
os.environ['TRID_WEB_CONFIG'] = str(CONFIG_PATH)
os.environ.pop('TRID_WEB_DEV', None)
os.environ['TRID_TEST_ROUTES'] = '1'
os.environ['DJANGO_SETTINGS_MODULE'] = 'onprem_web.settings'

import django  # noqa: E402

django.setup()

from django.core.management import call_command  # noqa: E402
from django.test import Client  # noqa: E402

call_command('migrate', verbosity=0)

from trid.onprem.config import load_config  # noqa: E402
from onprem_web import runtime  # noqa: E402
from onprem_web.accounts.registry import SessionRegistry  # noqa: E402


def make_runtime(answer=None, tok=None, registry=None, clock=None, **config_overrides) -> runtime.Runtime:
    """Install a runtime with a fresh SessionRegistry from the config (or `registry`); `clock` drives its idle timer."""
    config = dataclasses.replace(load_config(CONFIG_PATH), **config_overrides)
    if registry is None:
        registry = SessionRegistry(config.max_sessions, config.idle_seconds, config.per_minute,
                                   clock=clock or time.monotonic)
    rt = runtime.Runtime(
        config=config, tokenizer=tok,
        answer=answer or (lambda document, question: {'status': 'ok'}),
        registry=registry, gen_lock=threading.Lock(), parse_lock=threading.Lock(),
        insecure_dev=False, port=0)
    runtime.install(rt)
    return rt


def client() -> Client:
    return Client(enforce_csrf_checks=True, HTTP_HOST=HOST, HTTP_ORIGIN=ORIGIN, HTTP_X_REAL_IP='10.0.0.5')
