# 사내망 PDF 근거 확인 서비스 — Django 판 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 기존 사내망 서비스(`trid/onprem`)와 같은 기능·규칙·JSON API를 가진 Django 프로젝트 `onprem_web/`를 같은 저장소에 만든다. 기존 판은 건드리지 않는다.

**Architecture:** Django 5.2 LTS 단일 프로세스, waitress 스레드 서버. 업무 로직은 앱별 `services.py`에 두고 dict를 돌려준다. JSON 뷰(`/api/*`)는 그 dict를 그대로 돌려주고, htmx 뷰(`/ui/*`)는 템플릿 조각을 그린다. 업로드 문서는 프로세스 메모리의 `SessionRegistry`에만 두고, DB(SQLite)에는 계정·세션 식별·이용 기록만 저장한다. 답변 엔진·PDF 추출·설정·문구·모델 정보는 기존 `trid.*` 모듈을 import해서 쓴다.

**Tech Stack:** Python 3.12, Django 5.2.x, waitress 3.x, htmx 2.0.x(파일로 저장소에 포함), SQLite(WAL), tokenizers 0.23.3, pypdf 6.21.2, unittest.

**Spec:** `docs/superpowers/specs/2026-11-13-onprem-django-design.md` (executors read it; the plan argues from it). 기존 판 설계서 `docs/superpowers/specs/2026-11-13-onprem-pdf-web-design.md`도 참조한다.

## Global Constraints

- 기존 판 파일(`trid/onprem/**`, `trid/onprem_ui/**`, `deploy/onprem/**`, `tests/test_onprem_*.py`)은 수정하지 않는다. 기존 판 테스트 127개가 계속 통과해야 한다: `PYTHONUTF8=1 .venv/Scripts/python -m unittest discover -s tests -p 'test_onprem_*.py'` → `OK`.
- 새 판 테스트 파일 이름은 `tests/test_webapp_*.py`다. 실행: `PYTHONUTF8=1 .venv/Scripts/python -m unittest discover -s tests -p 'test_webapp_*.py'`. (리눅스 컨테이너에서는 `.venv/bin/python`)
- 다음은 기존 모듈을 import해서 쓰고 복사하지 않는다.
  - 설정: `trid.onprem.config.Config`, `load_config`
  - 계정 규칙: `trid.onprem.store.normalize_username`, `EVENTS`, `DETAIL_KEYS`, `ROLES`, `trid.onprem.passwords.check_new_password`
  - 문구: `trid.onprem.server`의 `BUSY_ASK`, `BUSY_PARSE`, `BAD_LOGIN`, `MODEL_DOWN`, `GENERIC_ERROR`, `MUST_CHANGE`, `ADMIN_ONLY`, `CSRF_FAIL`, `CSP`
  - 모델 정보: `trid.onprem.server.model_card_html`, `trid.onprem.model_card.MODEL_CARD`
  - 시작 처리: `trid.onprem.__main__.load_tokenizer`
  - 답변·PDF 엔진: `trid.document_case`, `trid.pdf_isolated.parse_pdf`, `trid.pdf_documents.DocumentError`, `trid.llama_client`
- 파일명, 질문, 질문 해시, 답변, PDF 본문, 원문 구간은 DB(세션 포함), 로그, stdout, stderr, 임시 파일 어디에도 남기지 않는다. 서버는 항상 `'document.pdf'`라는 이름으로 파싱한다.
- 외부 URL(`http://`, `https://`, `//fonts`, `@import url(`)을 템플릿·정적 파일에 넣지 않는다. 인라인 `<script>`와 `on*=` 속성도 쓰지 않는다. JS에서 `innerHTML =`, `insertAdjacentHTML`, `document.write`를 쓰지 않는다.
- 쿠키: 운영 `__Host-trid`(Secure, HttpOnly, SameSite=Strict, Path=/, Domain 없음), 개발 `trid-dev`(Secure 없음, 127.0.0.1에만 바인딩). 로그인 응답의 `Set-Cookie`는 세션 쿠키 하나뿐이다(`CSRF_USE_SESSIONS = True`).
- CSRF 헤더는 `X-Trid-Token`이다(`CSRF_HEADER_NAME = 'HTTP_X_TRID_TOKEN'`). 토큰은 `<meta name="csrf-token" content="…">`로 내려 준다. 비밀번호 변경이 필요한 계정이면 `<meta name="must-change" content="1">`도 넣는다.
- 모든 POST는 Origin이 운영 `https://<hostname>`, 개발 `http://127.0.0.1:<port>`와 정확히 같아야 한다. 다르면 403 `{'error': '허용되지 않은 요청입니다.'}`이다.
- 모든 응답에 헤더 `Content-Security-Policy: <trid.onprem.server.CSP>`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`를 붙인다. `Cache-Control`은 `/fonts/*.woff2`만 `private, max-age=86400`이고 나머지는 `no-store`다.
- 의존성은 `deploy/onprem_web/requirements-app.txt`에 정확한 버전과 `--hash`로 고정하고 `--no-deps --require-hashes`로 설치한다. torch·numpy는 넣지 않는다. 추가 패키지(django-axes, django-csp 등)도 쓰지 않는다.
- JSON 오류 응답 형태는 `{'error': '<한국어 문구>'}`이고, 상태 코드와 문구는 기존 판 `trid/onprem/server.py`와 같다.

## Review Focus

1. **PDF 본문이나 답에 `<script>`·`&`·`<b>` 같은 HTML이 들어 있는 경우:** 문서 보기와 답 조각에 글자 그대로 보여야 하고 실행되면 안 된다. 담당 Task 7 `test_document_and_answer_escape_markup`.
2. **업로드가 한도를 넘거나, 비어 있거나, PDF가 아닌 경우:** 기존 판과 같은 413·400 한국어 문구가 나와야 하고, 임시 폴더나 `data_dir`에 새 파일이 생기면 안 된다. 담당 Task 6 `test_upload_rejections_leave_no_files`.
3. **앱 재시작 뒤 DB 세션은 살아 있는데 메모리 문서가 없는 경우:** 화면은 열리고, 질문은 409 `'문서를 다시 올려 주세요.'`를 받아야 한다. 담당 Task 6 `test_after_restart_question_asks_for_reupload`.
4. **두 직원이 동시에 질문하는 경우:** 두 번째 직원은 `lock_wait` 뒤 503 `BUSY_ASK`를 받는다. 다른 직원의 질문이 생성 중이어도 업로드는 성공해야 한다. 담당 Task 6 `test_second_question_gets_busy_and_upload_still_works`.
5. **답 생성 도중 관리자가 그 직원을 비활성화하는 경우:** 401이 나야 하고, 답은 전달되지 않으며, 메모리 문서는 지워져야 한다. 담당 Task 6 `test_revoked_during_generation_returns_401_and_drops_document`.

---

## File Structure

```
onprem_web/
  __init__.py
  __main__.py            # serve / create-admin / purge (Task 9)
  settings.py            # reads TRID_WEB_CONFIG, TRID_WEB_DEV (Task 1)
  urls.py                # all routes (grows per task)
  runtime.py             # Runtime dataclass: config, tokenizer, answer, locks, registry (Task 1)
  web.py                 # security headers middleware, json helpers, static/font views, healthz, error handlers (Task 1–2)
  logging_policy.py      # no-content log formatter (Task 2)
  audit/                 # Event model + triggers, log/query/csv/purge (Task 3)
  accounts/              # User model, login service, registry, session middleware, auth API (Task 4–5)
  workspace/             # load/ask/clear services, JSON + htmx views, present_answer (Task 6–7)
  staffadmin/            # 한국어 관리자 화면 (Task 8)
  templates/             # base.html, login.html, workspace.html, admin.html, partials/_*.html
  static/                # common.css(복사본), app.js, admin.js, login.js, vendor/htmx.min.js
deploy/onprem_web/       # Dockerfile.app, compose.yaml, requirements-app.txt, requirements-dev.txt, build_bundle.sh, LICENSES.md (Task 11)
docs/ONPREM_WEB.md       # 한국어 설치·운영 안내 (Task 11)
tests/webapp_support.py  # Django 부팅·런타임 설치·클라이언트 도우미 (Task 1)
tests/test_webapp_*.py
```

글꼴은 복사하지 않고 `trid/onprem_ui/fonts/PretendardVariable.woff2`를 `/fonts/PretendardVariable.woff2` 주소로 제공한다. `common.css`는 `onprem_web/static/common.css`로 복사한다. 새 판 화면이 앞으로 달라질 수 있기 때문이다.

---

### Task 1: 프로젝트 뼈대, 설정, 런타임, 보안 헤더, 정적 파일

**Files:**
- Create: `onprem_web/__init__.py`, `onprem_web/settings.py`, `onprem_web/urls.py`, `onprem_web/runtime.py`, `onprem_web/web.py`, `onprem_web/static/common.css` (copy of `trid/onprem_ui/common.css`), `onprem_web/static/vendor/htmx.min.js`, `deploy/onprem_web/requirements-dev.txt`, `tests/webapp_support.py`, `tests/test_webapp_boot.py`
- Modify: `pyproject.toml`. Include the `onprem_web` package and its templates/static in the package data, the same way `trid` is included.

**Interfaces:**
- Produces:
  - `onprem_web.settings` reads env `TRID_WEB_CONFIG` (path to onprem.json, required) and `TRID_WEB_DEV` (`'1'` = insecure dev), and calls `trid.onprem.config.load_config`. It exposes `TRID_CONFIG: Config` and `TRID_DEV: bool` as settings.
  - `DATABASES['default']` is SQLite at `config.data_dir / 'web.db'`, with `OPTIONS={'timeout': 5, 'init_command': 'PRAGMA journal_mode=WAL;'}`.
  - `onprem_web.runtime`:
    - `@dataclass Runtime(config: Config, tokenizer: object, answer: Callable[[object, str], dict], registry: 'SessionRegistry', gen_lock: threading.Lock, parse_lock: threading.Lock, insecure_dev: bool, port: int)`
    - `install(rt: Runtime) -> None`
    - `get() -> Runtime`, which raises `RuntimeError` if nothing is installed.
    - Task 5 defines `registry`; until then tests pass `None`.
  - `onprem_web.web`:
    - `json_response(status: int, data) -> HttpResponse`: UTF-8 with `ensure_ascii=False`.
    - `SecurityHeadersMiddleware`.
    - `healthz(request)`: returns `{'ok': True}`.
    - `static_file(request, name)` and `font_file(request, name)`.
  - `tests.webapp_support`: on import it
    1. creates a temp `data_dir`;
    2. writes onprem.json with `hostname='test.example'` and `port=0`;
    3. sets the env vars, then calls `django.setup()` and `call_command('migrate', verbosity=0)`.

    It exports `HOST`, `ORIGIN`, `PW` with the same values as `tests/test_onprem_server.py`, and these helpers:
    - `make_runtime(answer=None, tok=None, **config_overrides) -> Runtime`, which installs the runtime. From Task 5 on, it also builds a fresh `SessionRegistry` from the config.
    - `client() -> django.test.Client(enforce_csrf_checks=True, HTTP_ORIGIN=ORIGIN, HTTP_X_REAL_IP='10.0.0.5')`, with `secure=True` on requests.

- [ ] **Step 1: Set up the env and pin dependencies.**
  - `requirements-dev.txt` holds `Django==5.2.*` (the latest 5.2 patch on the build day, exact version), `waitress==3.*` (exact version), `tokenizers==0.23.3`, `pypdf==6.21.2` and `playwright==1.49.1`.
  - Create the env with `uv venv .venv --python 3.12; uv pip install -r deploy/onprem_web/requirements-dev.txt; uv pip install --no-deps -e .`.
  - Copy htmx 2.0.x `dist/htmx.min.js` from the npm package `htmx.org` to `onprem_web/static/vendor/htmx.min.js`, and write its version and sha256 in a comment on the file's first line.
- [ ] **Step 2: Write the failing test `tests/test_webapp_boot.py`.**

```python
class BootTests(django.test.TestCase):
    def test_healthz_and_security_headers(self):
        r = S.client().get('/healthz', secure=True)
        self.assertEqual(r.status_code, 200); self.assertEqual(r.json(), {'ok': True})
        self.assertEqual(r['Content-Security-Policy'], CSP)
        for k, v in {'X-Content-Type-Options': 'nosniff', 'X-Frame-Options': 'DENY',
                     'Referrer-Policy': 'no-referrer', 'Cache-Control': 'no-store'}.items():
            self.assertEqual(r[k], v)
    def test_font_cached_others_no_store(self):
        r = S.client().get('/fonts/PretendardVariable.woff2', secure=True)
        self.assertEqual(r['Content-Type'], 'font/woff2'); self.assertEqual(r['Cache-Control'], 'private, max-age=86400')
        self.assertEqual(b''.join(r.streaming_content if r.streaming else [r.content])[:4], b'wOF2')
        self.assertEqual(S.client().get('/static/vendor/htmx.min.js', secure=True)['Cache-Control'], 'no-store')
        self.assertEqual(S.client().get('/static/../settings.py', secure=True).status_code, 404)
    def test_django_defaults_are_overridden(self):
        from django.conf import settings as s
        self.assertFalse(s.DEBUG); self.assertEqual(s.ALLOWED_HOSTS, ['test.example'])
        self.assertEqual(s.CSRF_TRUSTED_ORIGINS, ['https://test.example'])
        self.assertEqual(s.FILE_UPLOAD_HANDLERS, ['django.core.files.uploadhandler.MemoryFileUploadHandler'])
        self.assertTrue(s.CSRF_USE_SESSIONS); self.assertEqual(s.CSRF_HEADER_NAME, 'HTTP_X_TRID_TOKEN')
        self.assertEqual(s.SESSION_COOKIE_NAME, '__Host-trid'); self.assertEqual(s.SESSION_COOKIE_SAMESITE, 'Strict')
        self.assertNotIn('django.contrib.admin', s.INSTALLED_APPS); self.assertNotIn('django.contrib.messages', s.INSTALLED_APPS)
        self.assertEqual(s.PASSWORD_HASHERS[0], 'django.contrib.auth.hashers.ScryptPasswordHasher')
        self.assertEqual(s.ADMINS, []); self.assertEqual(s.DATA_UPLOAD_MAX_MEMORY_SIZE, 2 * 1024 * 1024)
    def test_unknown_path_is_korean_json_404(self):
        r = S.client().get('/nope', secure=True)
        self.assertEqual((r.status_code, r.json()), (404, {'error': '없는 페이지입니다.'}))
```

- [ ] **Step 3: Run it and confirm it fails.**
  - Run the `test_webapp_boot` module with the new-version test command.
  - Expected: `ModuleNotFoundError: onprem_web`.
- [ ] **Step 4: Implement settings, runtime and web.**
  - `DEBUG = TRID_DEV and False`: DEBUG stays False even in dev, so errors never render.
  - `SECURE_*` redirects stay off, because Caddy terminates TLS.
  - `DATA_UPLOAD_MAX_MEMORY_SIZE = config.pdf.max_bytes`.
  - `SESSION_EXPIRE_AT_BROWSER_CLOSE = True`.
  - `SESSION_ENGINE = 'django.contrib.sessions.backends.db'`.
  - `INSTALLED_APPS` holds only `contenttypes`, `auth`, `sessions`, then `onprem_web.audit`, `.accounts`, `.workspace`, `.staffadmin`. These app modules are empty until their tasks.
  - `handler404` and `handler500` return Korean JSON: the 500 handler returns `GENERIC_ERROR`.
  - Static views serve only allow-listed names (a fixed dict, as `STATIC` in `trid/onprem/server.py` does), read from `onprem_web/static`, plus the font from `trid/onprem_ui/fonts`.
- [ ] **Step 5: Run the boot tests and the old suite.**
  - Expected: boot tests `OK`; old suite `Ran 127 tests … OK`.
- [ ] **Step 6: Commit.** `git commit -m "feat(onprem-web): Django skeleton, settings, security headers, static"`

### Task 2: 로그에 내용이 남지 않게 하기

**Files:**
- Create: `onprem_web/logging_policy.py`, `tests/test_webapp_logging.py`
- Modify: `onprem_web/settings.py` (`LOGGING`), `onprem_web/urls.py` (test-only route is not allowed; use a view registered only when `settings.TRID_TEST_ROUTES` is True, set by `webapp_support`)

**Interfaces:**
- Produces:
  - `logging_policy.TypeOnlyFormatter(logging.Formatter)`: writes `"<logger> <levelname> <ExceptionType>"` when `exc_info` is set; otherwise `"<logger> <levelname> <status_code or ->"`. It never writes the message arguments, the request or the path's query string.
  - `LOGGING` applies it to the `django`, `django.request`, `django.server`, `django.security` and `waitress` loggers, with one stderr handler, `disable_existing_loggers=False`, and `mail_admins` removed.

- [ ] **Step 1: Write the failing test.**
  - The test-only view `/__boom` raises `ValueError(SECRET_QUESTION + SECRET_NAME)`.
  - Capture stderr and the root logging output, then request `/__boom?q=SECRET`.
  - Assert the response is 500 `{'error': GENERIC_ERROR}`.
  - Assert the captured output contains `ValueError` and does not contain `SECRET_QUESTION`, `SECRET_NAME`, `'q=SECRET'` or `'Traceback'`.
  - Also log `logging.getLogger('waitress').exception('x %s', SECRET_NAME)` inside an except block, and assert the secret is absent from the output.
- [ ] **Step 2: Run it and confirm it fails.** The secret appears in the output.
- [ ] **Step 3: Implement `TypeOnlyFormatter` and `LOGGING`.**
- [ ] **Step 4: Run the logging tests and the boot tests.** Expected: `OK`.
- [ ] **Step 5: Commit.** `git commit -m "feat(onprem-web): content-free logging policy"`

### Task 3: 이용 기록(audit) — 추가만 가능, 조회, CSV, 보관 정리

**Files:**
- Create: `onprem_web/audit/{__init__,apps,models,services}.py`, `onprem_web/audit/migrations/0001_initial.py`, `0002_append_only_triggers.py`, `tests/test_webapp_audit.py`

**Interfaces:**
- Produces:
  - `audit.models.Event`: fields `ts: float` (db_index), `username: str|None`, `ip: str|None`, `event: str`, `result: str` and `detail: str` (a JSON text with sorted keys). `save()` refuses an update, raising `PermissionError` when `pk` is already set; `delete()` always raises `PermissionError`.
  - `audit.models.PurgeGate`: a single-row table with `open: int` (default 0), created and seeded by migration 0002.
  - Migration 0002 runs `RunSQL` with two triggers:
    - `BEFORE UPDATE ON audit_event BEGIN SELECT RAISE(ABORT,'append-only'); END`
    - `BEFORE DELETE ON audit_event WHEN (SELECT open FROM audit_purgegate WHERE id=1)=0 BEGIN SELECT RAISE(ABORT,'append-only'); END`
  - `audit.services.log(event: str, username: str|None, ip: str|None, result: str, **detail) -> None`. Its validation matches `trid.onprem.store.Store.log` exactly: the event is in `EVENTS`, the result matches `[a-z_]{1,32}`, keys are in `DETAIL_KEYS`, and string values match `[A-Za-z0-9._:-]{1,128}`. Ints, floats and bools are allowed. It raises `ValueError` otherwise.
  - `audit.services.query_events(since=None, until=None, username=None, event=None, limit=50, offset=0) -> dict`: the same return shape and ordering as `trid.onprem.store.Store.query_events`, so read that function and match it.
  - `audit.services.export_csv(since=None, until=None, username=None, event=None) -> bytes`: the same columns and header row as `Store.export_csv`, with a UTF-8 BOM. Cells that start with `= + - @ \t \r` get a leading `'`.
  - `audit.services.purge_before(ts: float) -> int`: in one `transaction.atomic()` it sets the gate open, deletes, then closes the gate. It returns the count.

- [ ] **Step 1: Write the failing tests.**
  - `test_log_rejects_free_text`: covers the detail key `question`, a value with a space, an unknown event, and the result `'OK!'`; each raises `ValueError`.
  - `test_update_and_delete_blocked_at_db_level`: for each of `Event.objects.filter(pk=e.pk).update(result='x')`, `Event.objects.all().delete()` and `Event.objects.bulk_update([...], ['result'])`, assert `django.db.utils.IntegrityError` or `DatabaseError` is raised and the row is unchanged.
  - `test_purge_deletes_only_old_rows_and_gate_closes`: purge two old rows and keep one new row; the return value is 2, and `PurgeGate.objects.get().open == 0` afterwards.
  - `test_csv_matches_old_store`: write the same three events, including username `'=evil'`, to `trid.onprem.store.Store` (temp db) and to `audit.services`. Assert both CSV outputs are byte-equal after replacing the timestamp column with a fixed `ts`; inject a fixed clock into both. The CSV starts with `b'\xef\xbb\xbf'` and contains `"'=evil"`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement the models, migrations and services.**
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat(onprem-web): append-only audit log with DB triggers, CSV, purge"`

### Task 4: 계정 모델과 로그인 규칙

**Files:**
- Create: `onprem_web/accounts/{__init__,apps,models,services}.py`, `onprem_web/accounts/migrations/0001_initial.py`, `tests/test_webapp_accounts.py`
- Modify: `onprem_web/settings.py` (`AUTH_USER_MODEL = 'accounts.User'`, `AUTH_PASSWORD_VALIDATORS = []`, because validation goes through `check_new_password`)

**Interfaces:**
- Consumes: `audit.services.log`, `trid.onprem.store.normalize_username`, `ROLES`, `trid.onprem.passwords.check_new_password`, the `BAD_LOGIN` text, and `_LAST_ADMIN` from `trid.onprem.auth`.
- Produces:
  - `accounts.models.User(AbstractBaseUser)`:
    - Fields: `username` (unique; already normalized), `role` (`'user'` or `'admin'`), `must_change: bool=True`, `failed: int=0`, `locked_until: float=0`, `is_active: bool=True` (the old version's `disabled` is its inverse), `password_version: int=0`.
    - `USERNAME_FIELD='username'`.
    - `get_session_auth_hash()` also mixes in `password_version`, so bumping the version ends all sessions.
  - `accounts.services` (one module-level `threading.RLock` serializes account mutations, as `Auth._acct_lock` does):
    - `create_user(username, password, role, must_change=True) -> User`. It raises `ValueError` with the old version's message.
    - `attempt_login(username: str, password: str, ip: str|None, now: float|None=None) -> User|None`. It ports `trid/onprem/auth.py` `Auth.login` failure handling: 5 failures lock the account for 900 s; an unknown account is checked against a dummy hash; and it logs `login_ok`, `login_failed` or `login_locked` with the same detail values as the old version. Unknown, locked and disabled accounts all return `None`.
    - `change_password(user, current: str, new: str, ip) -> None`. It raises `ValueError(msg)`, bumps `password_version`, sets `must_change=False` and logs `password_changed`.
    - `admin_create(actor, username, password, role, ip) -> User`.
    - `admin_update(actor, target: str, *, role=None, disabled=None, reset_password=None, unlock=None, ip) -> None`. It refuses to disable or demote the last active admin with `ValueError(_LAST_ADMIN)`. A reset sets `must_change=True` and bumps `password_version`. It logs the same events, `action` values and `target` as `trid/onprem/auth.py` `admin_update`/`admin_unlock` and `trid/onprem/admin.py` `do_create`.

- [ ] **Step 1: Write the failing tests.** Port the account-rule scenarios from `tests/test_onprem_auth.py` to the service functions, keeping the same values:
  - `test_five_failures_lock_for_15_minutes`
  - `test_unknown_locked_disabled_all_return_none_and_log`
  - `test_unknown_user_runs_dummy_hash`: patch the hasher's verify and assert it is called once.
  - `test_password_rules_10_to_1024`
  - `test_username_normalized` (`' Alice '` → `'alice'`)
  - `test_last_admin_cannot_be_disabled_or_demoted`
  - `test_reset_sets_must_change_and_bumps_version`
  - `test_password_hash_is_scrypt`: assert `user.password.startswith('scrypt$')`.
  - `test_no_free_text_in_audit_for_login_failures`: no row contains the attempted password or an unknown username that does not match `[A-Za-z0-9._:-]`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement the model, migration and services.**
- [ ] **Step 4: Run the tests and confirm they pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat(onprem-web): user model and login rules"`

### Task 5: 세션 관리와 인증 API (`/api/login`, `/api/logout`, `/api/password`)

**Files:**
- Create: `onprem_web/accounts/registry.py`, `onprem_web/accounts/middleware.py`, `onprem_web/accounts/views.py`, `tests/test_webapp_sessions.py`
- Modify: `onprem_web/settings.py` (`MIDDLEWARE` order), `onprem_web/urls.py`, `onprem_web/runtime.py` (type of `registry`)

**Interfaces:**
- Consumes: `accounts.services.*` and `audit.services.log`.
- Produces:
  - `registry.Entry`: a dataclass with `username: str`, `touched: float`, `document: object|None = None`, `document_id: str|None = None` and `calls: list[float]`.
  - `registry.SessionRegistry(max_sessions: int, idle_seconds: float, per_minute: int, clock=time.monotonic)`, guarded by an internal lock. Methods:
    - `admit(key: str, username: str) -> bool`: prunes idle entries first and returns False when full.
    - `get(key) -> Entry|None`.
    - `touch(key) -> bool`: returns False when the entry is idle, and drops it.
    - `allow_call(key) -> bool`: allows at most `per_minute` calls in a 60 s window.
    - `drop(key) -> None`, `drop_user(username: str, keep: str|None=None) -> None` and `keys() -> set[str]`.
  - Any dropped `Entry` has `document` and `document_id` set to `None` before it is released.
  - `middleware.TridSessionMiddleware` goes after `AuthenticationMiddleware` and before `CsrfViewMiddleware`. For each request it:
    1. Checks the Origin of every POST.
    2. Handles a session with no registry entry, which means the app restarted:
       - If the user is authenticated, it admits a new entry and keeps `request.session['had_document']`. If the cap is full, it flushes the session and returns 401.
       - If the user is anonymous but the session key is in the registry, it drops the entry. This covers a password version change or a disabled account.
    3. Expires sessions: if idle for more than `idle_seconds`, it logs `session_expired`, flushes and drops, then returns 401 for an API request or the login page for a page request.
    4. Applies `allow_call` to `/api/*` and `/ui/*` POSTs, returning 429 `'요청이 많습니다. 1분 뒤 다시 시도해 주세요.'` when over the limit.
    5. Gates must-change: unless the path is `/api/password`, `/ui/password`, `/api/logout` or a GET of `/`, it returns 403 `{'error': MUST_CHANGE, 'must_change': True}`.
    6. Sets `request.trid_entry` and `request.trid_ip`. The IP comes from `X-Real-IP` when it parses as an IP address; otherwise it is `None`.
  - `views.api_login`:
    - It is `@csrf_exempt`; the middleware already checked Origin.
    - Body: JSON `{username, password}`.
    - On success it calls `django.contrib.auth.login()`, sets `session['pv']` and admits the session to the registry. If the registry is full it returns 503 `'현재 접속 인원이 많습니다. 잠시 후 다시 시도해 주세요.'`.
    - It returns 200 `{'status': 'must_change'|'ok'}`, or 401 `BAD_LOGIN`.
    - A missing username or password returns 400 `'아이디와 비밀번호를 입력해 주세요.'`.
  - `api_logout`: logs `logout`, then `auth.logout()` and `registry.drop()`, and returns 200 `{'ok': True}`.
  - `api_password`: body `{current, new}`. On success it calls `update_session_auth_hash`, runs `registry.drop_user(username, keep=current_key)` and deletes the other DB sessions for that user. Error statuses and messages match `server.py` `password()`.
  - Unauthenticated API calls return 401 `{'error': '로그인이 필요합니다.'}`; CSRF failure returns 403 `CSRF_FAIL` through `CSRF_FAILURE_VIEW`.

- [ ] **Step 1: Write the failing tests.**
  - `test_login_sets_single_host_cookie`: exactly one `Set-Cookie`, named `__Host-trid`, with `Path=/`, no `Domain`, `Secure`, `HttpOnly` and `SameSite=Strict`.
  - `test_bad_origin_rejected_even_on_login`: Origin `https://evil.example` returns 403.
  - `test_post_without_token_is_csrf_fail`: the result is 403 `CSRF_FAIL`. With `X-Trid-Token` read from `/` meta `csrf-token`, the same request succeeds.
  - `test_idle_30_minutes_expires_and_logs`: inject the registry clock.
  - `test_session_cap_32_returns_503`.
  - `test_per_minute_20_returns_429`.
  - `test_must_change_gate_blocks_load`: returns 403 with `must_change`, and the `/` page contains `<meta name="must-change" content="1">`.
  - `test_password_change_ends_other_sessions_and_their_documents`: the other client's registry entry is gone and its next call returns 401.
  - `test_admin_disable_ends_session`: after `admin_update(disabled=True)`, the target's next call returns 401 and the registry has no entry.
  - `test_session_store_holds_no_content`: after login, decode every `Session` row and assert its keys are a subset of `{'_auth_user_id', '_auth_user_backend', '_auth_user_hash', '_csrftoken', 'pv', 'had_document'}`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement the registry, middleware and views.**
- [ ] **Step 4: Run the new-version suite and the old suite.**
- [ ] **Step 5: Commit.** `git commit -m "feat(onprem-web): session registry, middleware, auth API"`

### Task 6: 업로드·질문·지우기 서비스와 JSON API (`/api/load`, `/api/ask`, `/api/clear`)

**Files:**
- Create: `onprem_web/workspace/{__init__,apps,services,api}.py`, `tests/test_webapp_workspace.py`
- Modify: `onprem_web/urls.py`

**Interfaces:**
- Consumes:
  - `runtime.get()`: provides `config`, `tokenizer`, `answer`, `gen_lock`, `parse_lock` and `registry`.
  - `request.trid_entry` and `request.trid_ip`.
  - `audit.services.log`.
- Produces:
  - `services.Result`: a dataclass with `status: int` and `data: dict`.
  - `services.load_pdf(entry: Entry, body: bytes, username: str, ip) -> Result`: a direct port of `server.py` `load()`.
    - `parse_lock` waits `config.pdf.wall_seconds + config.lock_wait`, and returns 503 `BUSY_PARSE` on timeout.
    - Parsing is `parse_pdf(body, 'document.pdf', tok, limits=config.pdf)`.
    - Logging matches the old version: `pdf_upload` with `sha256`, `pages` and `status`.
    - It sets `entry.document` and `entry.document_id = secrets.token_urlsafe(24)`.
    - Success returns 200 `{'document_id', 'name', 'pages', 'warnings'}`.
  - `services.ask(entry, question, document_id, username, ip, still_valid: Callable[[], bool], had_document: bool) -> Result`: a direct port of `server.py` `ask()`, with the same status codes, messages and `log` details.
    - Only one change: when there is no document and `had_document` is True, the message is `'문서를 다시 올려 주세요.'` (409). Otherwise it is the old `'이 브라우저에서 문서를 먼저 넣어 주세요.'`.
    - `still_valid` is checked after generation. When it fails, it returns 401 and the view also drops the registry entry.
    - The response strips `checked` and `raw_output`.
  - `services.clear(entry, username, ip) -> Result`: logs `clear` and returns 200 `{'cleared': True}`.
  - `api.load`: reads the raw request body (`request.body`; `application/pdf`; no multipart).
    - `RequestDataTooBig` returns 413 `'입력이 너무 큽니다.'`.
    - A missing `Content-Length` returns 411 `'요청 길이가 필요합니다.'`.
    - A success sets `session['had_document']=True`.
  - `api.ask` and `api.clear` take JSON bodies of at most `SMALL_BODY` (16 KiB), the same limit as the old version.
  - `still_valid` reloads the user and compares `session['_auth_user_hash']` with `user.get_session_auth_hash()` and `user.is_active`.

- [ ] **Step 1: Write the failing tests.** Port the scenarios from `tests/test_onprem_server.py`, reusing its `fake_doc`, `ok_answer`, `SECRET_NAME` and `SECRET_QUESTION`, and from `test_onprem_upload_limit.py`. As the old tests do, patch `onprem_web.workspace.services.parse_pdf`: return `fake_doc('document.pdf')` for a body starting `%PDF`, and raise `DocumentError('PDF를 읽지 못했습니다.')` otherwise. The repo has no tokenizer or real PDF fixture.
  - `test_load_ignores_file_name_header`: send `X-File-Name: SECRET_NAME`; the response `name` is `'document.pdf'` and no audit row contains the secret.
  - `test_ask_logs_only_status_and_seconds`.
  - `test_model_down_is_502_korean`.
  - `test_context_overflow_is_422`.
  - `test_document_id_mismatch_is_409`.
  - `test_question_over_2000_is_400`, with the message `'질문은 2,000자 이하로 입력해 주세요.'`.
  - Plus the Review Focus tests:
    - `test_upload_rejections_leave_no_files`: snapshot `tempfile.gettempdir()` and `data_dir` listings before and after uploads of `max_bytes+1` bytes, an empty body and `b'not a pdf'`. The statuses are 413, 400 and 400, and the listings are unchanged.
    - `test_after_restart_question_asks_for_reupload`: upload, then replace the runtime registry with a new empty one. `GET /` still returns 200, and ask returns 409 `'문서를 다시 올려 주세요.'`.
    - `test_second_question_gets_busy_and_upload_still_works`: the answer fake blocks on an `Event`. Thread A asks; B asks with `lock_wait=0.1` and gets 503 `BUSY_ASK`; B's upload returns 200 while A is still blocked.
    - `test_revoked_during_generation_returns_401_and_drops_document`: the answer fake calls `admin_update(disabled=True)` on the asker before returning. The response is 401, no answer field is present, and `registry.get(key)` is `None`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement the services and API views.**
- [ ] **Step 4: Run the new-version suite and the old suite.**
- [ ] **Step 5: Commit.** `git commit -m "feat(onprem-web): load/ask/clear services and JSON API"`

### Task 7: 직원 화면과 로그인 화면 (템플릿 + htmx)

**Files:**
- Create:
  - Templates: `onprem_web/templates/base.html`, `login.html`, `workspace.html`, `partials/_document.html`, `partials/_answer.html`, `partials/_password.html`.
  - Code: `onprem_web/workspace/present.py`, `onprem_web/workspace/ui.py`.
  - Static: `onprem_web/static/app.js`, `onprem_web/static/login.js`.
  - Tests: `tests/test_webapp_ui.py`.
- Modify: `onprem_web/urls.py`

**Interfaces:**
- Consumes: `services.load_pdf`, `ask`, `clear` (Task 6) and `model_card_html()`.
- Produces:
  - `present.STATUS: dict[str, str]`: copied exactly from `trid/onprem_ui/app.js` `STATUS`.
  - `present.present_answer(result: dict) -> dict`: a port of `app.js` `renderAnswer`, `contextLimitLine`, `highlighted` and `placeOf`. It returns `{'answer', 'tag', 'tag_bad', 'limits': [str], 'warn_notes': [str], 'quotes': [{'place', 'before', 'mark', 'after', 'context'}], 'commentary', 'table_warning', 'info': [str]}`.
    - The `before`, `mark` and `after` slices use code-point offsets, the same as `Array.from`.
    - The section labels are the same Korean strings in the same order.
  - Pages and fragments:
    - `GET /`: renders `login.html` when the user is logged out, otherwise `workspace.html`.
    - `workspace.html` keeps the element IDs and layout of `trid/onprem_ui/index.html`, with `MODEL_CARD_PLACEHOLDER` replaced by `model_card_html()` and the same `#model-status` text.
    - `POST /ui/ask`: form fields `question` and `document_id`; returns `_answer.html`.
    - `GET /ui/document`: returns `_document.html` for the session's in-memory document, or an empty fragment.
    - `POST /ui/clear`.
    - `POST /ui/password`: returns `_password.html` with an error, or `HX-Redirect: /`.
  - `base.html`:
    - `<meta name="csrf-token">`, `<meta name="username">` and `<meta name="must-change">`.
    - `<meta name="htmx-config" content='{"allowEval":false,"allowScriptTags":false,"includeIndicatorStyles":false,"selfRequestsOnly":true}'>`.
    - `<body hx-headers='{"X-Trid-Token": "{{ csrf_token }}"}'>`.
    - It loads `/static/vendor/htmx.min.js` and the page JS.
  - `app.js` (under 600 lines; no `innerHTML`):
    - Uploading: shows the local file name; checks the size before sending (against `<meta name="trid-max-bytes">`); POSTs the raw bytes with `fetch` to `/api/load`; on 200 sets `#document_id`; then calls `htmx.ajax('GET', '/ui/document', '#doc-view')`.
    - The character counter.
    - `doc-view.open` on width ≥ 1024px, as in the old version.

- [ ] **Step 1: Write the failing tests.**
  - `test_templates_have_no_external_urls_inline_scripts_or_innerhtml`: the same assertions as `tests/test_onprem_ui.py` `UiFileTests`, applied to every file under `onprem_web/templates` and `onprem_web/static` except `vendor/`.
  - `test_staff_page_matches_old_contract`: the page contains the notice `'질문마다 완전한 문장으로 써 주세요. 이전 질문은 이어서 해석하지 않습니다.'`, `maxlength="2000"` and `<meta name="username" content="alice">`. `STATUS_TEXT` appears before the first `<details`. The model card contains `<dt>` for each of the seven terms and `<li>이미지·스캔 PDF는 읽지 못합니다.</li>`. Every `src` and `href` asset returns 200.
  - `test_present_answer_statuses`: for each `STATUS` key, `present_answer({'status': k, 'answer': 'a'})['tag'] == STATUS[k]`, and `tag_bad` is True only for `model_failure`.
  - `test_present_answer_quote_slices_by_code_point`: a text with emoji and span `[1,3]` slices by code point.
  - `test_document_and_answer_escape_markup`: a fake document page reads `'<script>alert(1)</script> & <b>x</b>'`, and the answer quote contains `'<img src=x>'`. The `/ui/document` and `/ui/ask` fragments contain `&lt;script&gt;`, `&amp;` and `&lt;img`, and do not contain `<script>alert` or `<img src=x>`.
  - `test_ui_ask_without_token_is_403`.
  - **Browser check** (playwright, skipped if it is not installed): run waitress in a thread in this test process on a free port, so `parse_pdf` can be patched to `fake_doc`, then log in. Upload a `%PDF-1.4 fake` file and ask with the fake answer. Assert `.answer .tag` is visible and that no console CSP errors occur. Take screenshots at 1440 and 390 widths into the scratchpad for a manual check against the old version.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement the templates, `present.py`, the UI views and the JS.**
- [ ] **Step 4: Run the new-version suite, then compare the screenshots with the old version's screens.** Layout differences count as failures.
- [ ] **Step 5: Commit.** `git commit -m "feat(onprem-web): staff and login screens with htmx"`

### Task 8: 한국어 관리자 화면

**Files:**
- Create: `onprem_web/staffadmin/{__init__,apps,views}.py`, `onprem_web/templates/admin.html`, `partials/_users.html`, `partials/_events.html`, `onprem_web/static/admin.js`, `tests/test_webapp_admin.py`
- Modify: `onprem_web/urls.py`

**Interfaces:**
- Consumes:
  - `accounts.services.admin_create` and `admin_update`.
  - `audit.services.query_events`, `export_csv` and `log`.
- Produces:
  - `GET /admin`: admin only. It returns 403 `ADMIN_ONLY` for staff and shows the login page when logged out. The layout follows `trid/onprem_ui/admin.html`.
  - `POST /ui/admin/users`: create; re-renders `_users.html` with the message.
  - `POST /ui/admin/users/<username>`: role, disabled, reset_password and unlock; re-renders `_users.html`.
  - `GET /ui/admin/events?since&until&username&event&offset`: re-renders `_events.html`, with `PAGE_SIZE` from `trid.onprem.admin`.
  - `POST /api/admin/events.csv`: form fields are the filters and the token goes in `csrfmiddlewaretoken`.
    - It returns `text/csv; charset=utf-8` with `Content-Disposition: attachment; filename="events.csv"` and logs `admin_export`.
    - It is POST rather than the old GET-with-`?token=`, so the token never appears in a URL.
  - The sha256 cell shows the first 12 characters plus `…`, with the full value in `title`, as in the old `admin.js`.

- [ ] **Step 1: Write the failing tests.** Port the scenarios from `tests/test_onprem_admin.py`:
  - `test_staff_cannot_open_admin_or_csv` (403).
  - `test_create_logs_target_and_action`.
  - `test_last_admin_message_shown`.
  - `test_events_filter_and_paging`.
  - `test_csv_requires_post_with_token`: a GET returns 405 and a POST without the token returns 403.
  - `test_admin_cannot_see_staff_documents`: no admin route returns document text after a staff upload; scan the `/admin` and `/ui/admin/events` HTML for the fake page text.
  - `test_no_edit_or_delete_route_for_events`: no URL pattern contains `events` together with `delete` or `update`.
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement the views, templates and admin.js.**
- [ ] **Step 4: Run the new-version suite.**
- [ ] **Step 5: Commit.** `git commit -m "feat(onprem-web): Korean admin screens"`

### Task 9: 실행 명령 (`serve`, `create-admin`, `purge`)

**Files:**
- Create: `onprem_web/__main__.py`, `tests/test_webapp_cli.py`

**Interfaces:**
- Consumes: everything above, plus `trid.onprem.__main__.load_tokenizer`.
- Produces:
  - `python -I -m onprem_web serve --config PATH [--insecure-dev]`, `create-admin --config PATH --username NAME [--password-stdin]` and `purge --config PATH`.
    - The option names, messages and exit codes match `trid/onprem/__main__.py`, so read `build_parser`, `cmd_*` and `_fail`.
    - Each command sets `TRID_WEB_CONFIG` and `TRID_WEB_DEV` before calling `django.setup()`.
  - `serve`:
    1. Checks the tokenizer sha256 against `73551e65acfa80834c235602900f31dc353ce230bfb69939ffdf44ca26ebc58d`. On a mismatch it fails in production and warns only in dev.
    2. Runs `call_command('migrate', interactive=False, verbosity=0)`.
    3. Installs `Runtime`.
    4. Starts a background thread every 24 h: `purge_before(now - retention_days*86400)` and a registry sweep, which drops registry keys whose DB session no longer exists. The purge also runs once at start.
    5. Runs `waitress.serve(application, host='0.0.0.0' (prod) | '127.0.0.1' (dev), port=config.port, threads=config.slots, max_request_body_size=config.pdf.max_bytes + 64*1024, ident=None, clear_untrusted_proxy_headers=True)`.
    6. Handles SIGTERM for a clean stop.
  - `purge` logs `retention_purge` with `count`, as the old `run_purge` does.

- [ ] **Step 1: Write the failing tests.** Port from `tests/test_onprem_cli.py`:
  - `test_create_admin_password_stdin`
  - `test_create_admin_rejects_short_password`
  - `test_purge_removes_old_and_logs_count`
  - `test_serve_refuses_wrong_tokenizer_in_prod`
  - `test_serve_dev_warns_on_tokenizer_mismatch`
  - `test_serve_applies_migrations_on_empty_dir`: start serve in a subprocess on port 0 in dev with a temp `data_dir`, poll `/healthz`, assert `web.db` exists with the `accounts_user` table, then terminate.
  - `test_dev_binds_loopback_only`
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Implement `__main__.py`.**
- [ ] **Step 4: Run the new-version suite.**
- [ ] **Step 5: Commit.** `git commit -m "feat(onprem-web): serve/create-admin/purge commands"`

### Task 10: 두 판의 동작 비교와 배포 전 점검 공유

**Files:**
- Create: `tests/test_webapp_parity.py`

**Interfaces:**
- Consumes:
  - The old `trid.onprem.server.make_server(config, store, auth, tok, answer=fake)`, run in a thread.
  - The new `onprem_web serve`, run as in Task 9, or waitress in a thread with `make_runtime(answer=fake)`.
  - `deploy/onprem/release_check.py`, loaded the way `tests/test_onprem_release_check.py` `load_module()` does.

- [ ] **Step 1: Write the tests.**
  - `test_same_fake_answer_same_json`: run the same login → load → ask sequence through real HTTP on both servers, with one fake `answer` that returns a quoted result containing `candidates`, `commentary`, `context_limits` and the `checked`/`raw_output` keys. Assert the `/api/ask` JSON bodies are equal, and that both lack `checked` and `raw_output`.
  - `test_same_errors_same_json`: compare the status and body pairs from both servers for:
    - a 2001-character question;
    - a wrong `document_id`;
    - a model-down `OSError`;
    - a `DocumentError(CONTEXT_OVERFLOW)`;
    - no token;
    - a bad Origin;
    - a bad login.
  - `test_release_check_runs_unmodified_against_new_version`: run `release_check.run()` against the new server, as `test_onprem_release_check.py` does against its fake. The exit code is 0, and stdout contains neither `SECRET_ANSWER` nor the PDF text.
- [ ] **Step 2: Run them.** Expected: pass. A failure here means an earlier task diverged from the old version; fix the new version, never the old one.
- [ ] **Step 3: Commit.** `git commit -m "test(onprem-web): parity with stdlib version and shared release check"`

### Task 11: 배포 묶음과 안내서

**Files:**
- Create:
  - `deploy/onprem_web/Dockerfile.app`, `compose.yaml`, `requirements-app.txt`, `build_bundle.sh`, `onprem.example.json` (the same keys as `deploy/onprem/onprem.example.json`) and `LICENSES.md`.
  - `docs/ONPREM_WEB.md`.
  - `tests/test_webapp_deploy.py`.

**Interfaces:**
- Consumes: `deploy/onprem/Dockerfile.llm`, `Caddyfile` and `make_self_signed.sh`, by reference. `compose.yaml` builds or uses the llm image `trid-onprem-llm:latest` and mounts `../onprem/Caddyfile`.
- Produces:
  - `Dockerfile.app`:
    - It follows `deploy/onprem/Dockerfile.app` line for line, with three changes:
      - Its wheels and requirements come from `deploy/onprem_web/`.
      - It also copies `onprem_web/` and `trid/onprem/` (shared constants and config) plus `trid/onprem_ui/fonts/` into site-packages.
      - It ends with `CMD ["python", "-I", "-m", "onprem_web", "serve", "--config", "/data/onprem.json"]`.
    - It keeps uid 10001, `WORKDIR /`, the tokenizer sha256 check and the no torch/numpy build check.
  - `compose.yaml`:
    - `name: trid-onprem-web`, with app image `trid-onprem-web-app:latest`.
    - The services keep the old version's settings: `read_only`, `cap_drop: [ALL]`, an internal network, only `tls` publishing 443, log rotation (`max-size`, `max-file`) and `TRID_MAX_BODY`.
  - `requirements-app.txt`: exact versions plus `--hash` for Django, asgiref, sqlparse, waitress, tokenizers and pypdf.
  - `LICENSES.md`:
    - The old rows, plus Django (BSD-3-Clause), asgiref (BSD-3-Clause), sqlparse (BSD-3-Clause), waitress (ZPL-2.1) and htmx (0BSD).
    - The htmx row records the version and sha256.
  - `docs/ONPREM_WEB.md` (Korean) covers:
    - install;
    - first admin (`create-admin`);
    - certificate replacement;
    - backup of `data/web.db*`;
    - upgrade (migrations apply automatically at start);
    - specs;
    - "한 서버에는 한 판만 설치" (install only one version per server);
    - how to run the new-version tests.

- [ ] **Step 1: Write the failing tests.** Port the static checks from `tests/test_onprem_deploy.py` to the new files:
  - `test_requirements_all_hashed_and_no_torch_numpy`
  - `test_compose_only_tls_publishes_and_app_read_only_cap_drop`
  - `test_compose_project_and_image_names_differ_from_old` (`trid-onprem-web`, `trid-onprem-web-app`)
  - `test_dockerfile_runs_isolated_as_10001_and_checks_tokenizer`
  - `test_dockerignore_allowlist_includes_onprem_web`: add the new paths to the repo's `.dockerignore` allow-list without removing any old entries.
  - `test_licenses_list_every_requirement`: each package name in `requirements-app.txt` appears in `LICENSES.md`, plus htmx.
  - `test_guide_mentions_single_version_per_server_and_create_admin`
- [ ] **Step 2: Run the tests and confirm they fail.**
- [ ] **Step 3: Write the deploy files and the guide.** Download the wheels with `build_bundle.sh`, which downloads to `deploy/onprem_web/wheels` (git-ignored, like the old version).
- [ ] **Step 4: Run both full suites.** Expected: the new-version suite `OK`; the old suite `Ran 127 tests … OK`. If Docker is available on the build machine, run `docker compose -f deploy/onprem_web/compose.yaml build app` and the old `release_check.py` against it. Otherwise record in the report that this step is left for the build machine.
- [ ] **Step 5: Commit.** `git commit -m "feat(onprem-web): Docker packaging, licenses, Korean install guide"`
