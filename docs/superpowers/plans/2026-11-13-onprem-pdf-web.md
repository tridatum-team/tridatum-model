# 고객 내부망 설치형 PDF 근거 확인 웹 서비스 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 테스트 사이트의 "PDF 업로드 → 근거 확인" 경로를 자체 계정·이용 기록을 갖춘 웹 서비스로 만들고, 인터넷 없는 x86_64 리눅스 서버에 Docker로 설치할 수 있게 묶는다.

**Architecture:** `scripts/shared_demo.py`의 첨부 문서 답변 경로(PDF 격리 파싱, `answer_document_case`, llama-server 호출)를 `trid/` 모듈로 옮기고 테스트 사이트는 그 모듈을 쓴다. 설치판은 `trid/onprem/` 패키지로, 표준 라이브러리 HTTP 서버 + 내장 SQLite(계정·이용 기록) + scrypt로 만든다. docker compose는 `app` / `llm`(llama.cpp CPU) / `tls`(Caddy) 세 서비스로 구성한다.

**Tech Stack:** Python 3.12(표준 라이브러리 `http.server`, `sqlite3`, `hashlib.scrypt`), 기존 의존성 `tokenizers`·`pypdf`, llama.cpp `8ca4060` CPU 빌드, Caddy, docker compose, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-13-onprem-pdf-web-design.md`

## Global Constraints

- Python `>=3.12,<3.13`. 설치판 런타임에 **새 PyPI 의존성 금지**. `torch`·`numpy`를 설치판 코드 경로에서 import하지 않는다(설치판 이미지는 `tokenizers`, `pypdf`만 설치).
- 설치·실행 중에 외부 네트워크 접근이 없어야 한다(다운로드, 텔레메트리, 웹폰트, CDN 모두 포함). 화면 CSP는 `default-src 'self'`.
- PDF 본문, 파일명, 질문, 답변이 디스크(SQLite, 로그, 임시 파일)나 표준 출력·오류에 남으면 안 된다.
- 모델 GGUF `runs/open-weight-base/Qwen3-1.7B-Q8_0.gguf` sha256 `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a`. 업로드 토크나이저 `runs/reader-factorial-01-cpt-v1-v2-v4/tokenizer.json` sha256 `73551e65acfa80834c235602900f31dc353ce230bfb69939ffdf44ca26ebc58d`. llama.cpp 커밋 `8ca4060`.
- 답변 설정: family `qwen3`, context `4096`, `max_tokens` `512`, greedy, 사고 모드 끔(`deploy/macmini-site/versions.json`의 `trid-assistant-qwen3-1.7b`).
- 업로드 한도 기본값: 요청 2MiB, 20쪽, 5만 자, 파싱 15초. 설치판에서만 설정으로 바꿀 수 있다. 테스트 사이트 값은 바뀌지 않는다.
- 계정: 역할 `user`·`admin`. 5회 연속 실패 시 15분 잠금, 30분 무활동 시 세션 종료, 최초·초기화 후 비밀번호 변경 강제, 비밀번호 최소 10자.
- 이용 기록 보관 기본 365일. 웹에서는 수정·삭제 불가. 다운로드는 CSV.
- 세션 상한 32, 요청 슬롯 8, 세션당 분당 20회, 생성은 한 번에 하나.
- 대상 서버: x86_64 리눅스, AVX2 이상, `GGML_NATIVE=OFF`.
- 화면 문구는 한국어. 모델 상태 문구 "실험 서비스 · 원본 모델(미특화) · 품질 기준 미달"은 그대로 표시한다.
- 테스트 실행: `python -m unittest discover -s tests -p '<pattern>'`(저장소 루트에서, Python 3.12). 개발 PC에 3.12가 없으면 3.12 환경을 먼저 만든다.

## Review Focus

1. **질문 처리 중이나 업로드 중에 세션이 만료되는 경우:** 401과 "다시 로그인해 주세요"를 돌려주고, 문서를 메모리에서 지우고, 화면은 로그인으로 돌아가야 한다. → Task 6 `test_expired_session_mid_ask_returns_401_and_drops_document`
2. **사용자 이름의 대소문자·앞뒤 공백 차이(`Kim `와 `kim`):** 같은 계정으로 취급해야 한다. 그래야 중복 계정을 만들거나, 대소문자를 바꿔 잠금을 우회할 수 없다. → Task 4 `test_username_normalized_on_create_and_lookup`
3. **관리자가 자기 자신이나 마지막 관리자를 잠그거나 일반 사용자로 바꾸는 경우:** 거부해야 한다. 허용하면 설치본 관리가 불가능해진다. → Task 5 `test_cannot_lock_or_demote_last_active_admin`
4. **llm 컨테이너가 꺼져 있거나 응답이 180초를 넘는 경우:** "모델 서버에 연결할 수 없습니다"를 보여 주고, 생성 잠금을 풀고, 기록에는 `error=model_unavailable`을 남긴다. 표준 오류에는 질문이 든 traceback을 출력하지 않는다. → Task 6 `test_model_down_releases_lock_and_logs_kind_only`
5. **없는 계정으로 로그인을 시도하는 경우:** 있는 계정과 같은 응답을 돌려줘 계정 존재 여부가 드러나지 않게 한다. 기록에는 입력한 이름 대신 `(unknown)`을 남긴다. 사용자가 이름 칸에 비밀번호를 잘못 입력했을 수 있기 때문이다. → Task 5 `test_unknown_user_same_response_logged_as_unknown`

---

## File Structure

| 파일 | 책임 |
|---|---|
| `trid/llama_client.py` (신규) | llama-server HTTP 호출: `request`, `complete`, `QWEN3_NO_THINK` (`scripts/evaluate_external_chat.py`에서 이동) |
| `trid/document_case.py` (신규) | 첨부 문서 `case` 답변: `answer_document_case`, `fit_own_passages`, 인용·숫자 검사 보조 함수 (`shared_demo.py`에서 이동) |
| `trid/pdf_isolated.py`, `trid/pdf_extract_worker.py` (신규) | 별도 프로세스에서 제한을 걸고 PDF를 파싱 |
| `scripts/shared_demo.py`, `scripts/evaluate_external_chat.py`, `scripts/shared_pdf_extract.py` (수정) | 새 모듈을 쓰는 얇은 래퍼. 동작은 그대로 |
| `trid/onprem/config.py` | 설정 dataclass와 로딩 |
| `trid/onprem/passwords.py` | scrypt 해시·검증·규칙 |
| `trid/onprem/store.py` | SQLite 계정·이용 기록, 보관 기간 정리, CSV |
| `trid/onprem/auth.py` | 로그인·잠금·세션 |
| `trid/onprem/model_card.py` | 화면에 보일 모델 정보 문구 |
| `trid/onprem/server.py` | HTTP 라우트(직원·관리자) |
| `trid/onprem/__main__.py` | `python -m trid.onprem serve | create-admin | purge` |
| `trid/onprem_ui/` | `login.html`, `index.html`, `admin.html`, `common.css`, `login.js`, `app.js`, `admin.js` |
| `deploy/onprem/` | `Dockerfile.app`, `Dockerfile.llm`, `Caddyfile`, `compose.yaml`, `onprem.example.json`, `build_bundle.sh`, `release_check.py`, `release-questions.json` |
| `docs/ONPREM.md` | 한국어 설치·운영 안내서 |
| `tests/test_llama_client.py`, `tests/test_document_case.py`, `tests/test_pdf_isolated.py`, `tests/test_onprem_*.py` | 테스트 |

---

### Task 1: llama-server 클라이언트를 `trid/`로 옮기기

**Files:**
- Create: `trid/llama_client.py`
- Modify: `scripts/evaluate_external_chat.py:39`, `:159-196`. `QWEN3_NO_THINK`, `request`, `complete`를 지우고 `from trid.llama_client import QWEN3_NO_THINK, request, complete`로 다시 내보낸다.
- Test: `tests/test_llama_client.py`

**Interfaces:**
- Produces:
  - `request(base: str, endpoint: str, payload: dict | None = None, timeout: float = 180) -> dict`
  - `complete(base, messages, family, max_tokens, context, grammar=None) -> dict`. 반환 키는 기존과 같다(`generated`, `truncated_output`, `skipped_context`, `prompt_tokens`, …).
  - `count_prompt_tokens(base, messages) -> int | None`. `shared_demo.prompt_tokens`와 같은 동작이다.
  - `QWEN3_NO_THINK`

- [ ] **Step 1: 기준선을 기록한다.** `python -m unittest discover -s tests -p 'test_shared_demo.py'`, `-p 'test_external_chat_eval.py'`를 실행해 통과 개수를 커밋 메시지에 적는다. Expected: 모두 OK.
- [ ] **Step 2: 실패하는 테스트를 쓴다.** `tests/test_llama_client.py`에서 `patch('trid.llama_client.request')`를 쓴다.
  - `test_complete_rejects_missing_qwen3_no_think_prefix`: `/apply-template`이 `'...'`를 돌려주면 `ValueError`가 난다.
  - `test_complete_skips_when_prompt_plus_max_tokens_exceeds_context`: 토큰 4000개, `max_tokens=512`, `context=4096`이면 `result['skipped_context'] is True`이고 `/completion`은 호출되지 않는다.
  - `test_complete_payload_is_greedy`: `/completion` payload가 `temperature=0, seed=42, samplers=['temperature'], repeat_penalty=1, cache_prompt=False`이다.
  - `test_count_prompt_tokens_returns_none_on_oserror`
- [ ] **Step 3: 실행해서 실패를 확인한다.** Expected: `ModuleNotFoundError: trid.llama_client`.
- [ ] **Step 4: 구현한다.** 코드를 그대로 옮기고 `timeout` 인자만 추가한다(기본 180).
- [ ] **Step 5: 새 테스트와 Step 1의 두 묶음을 실행한다.** Expected: 모두 OK, 개수는 기준선과 같다.
- [ ] **Step 6: 커밋한다.** `refactor: move llama-server client into trid.llama_client`

### Task 2: 첨부 문서 `case` 답변을 `trid/document_case.py`로 옮기기

**Files:**
- Create: `trid/document_case.py`
- Modify: `scripts/shared_demo.py:187-297`, `:622-626`. `LEADING_TAGS`, `NUMBER`, `MIN_QUOTE_CHARS`, `_numbers`, `NORMALIZE`, `_norm`, `evidence_start`, `CONTEXT_OVERFLOW`를 새 모듈에서 import해 같은 이름으로 둔다. `answer_document_case`는 래퍼로 바꾼다.
- Test: `tests/test_document_case.py`

**Interfaces:**
- Consumes: `trid.pdf_documents.retrieve`, `trid.case_prompts.CASE_SYSTEM`, `trid.prompts.rag_prompt`
- Produces:
  - `answer_document_case(document, question, base, family, context, max_tokens=256, selected=None, topical=None, provenance=None, *, complete: Callable, fit_context: Callable) -> dict`. 본문과 반환 dict는 현재 `shared_demo.answer_document_case`와 같다. `complete(base, messages, family, max_tokens, context)`와 `fit_context(selected, build, base, context, max_tokens) -> (selected, notes, limits)`는 주입받는다.
  - `fit_own_passages(selected, build, count_tokens: Callable[[list], int | None], context: int, max_tokens: int) -> tuple[list, list[str], dict]`. 첨부 문서 전용이다. 첫(가장 관련 높은) 구간은 남기고 뒤에서부터 지운다. 안내 문구는 `"모델 문맥 한도로 적용 문서의 관련도 낮은 구간 {n}개를 빼었습니다."`이고, `limits`는 `{'linked_passages': 0, 'table_rows': 0, 'tables': 0, 'own_passages': n}`이다. `doc_id`·`derived` 속성을 요구하지 않는다(`trid.pdf_documents.Chunk`에는 없음).
  - `shared_demo.answer_document_case(*args, **kwargs)`는 `complete=lambda *a, **k: complete(*a, **k)`, `fit_context=lambda *a: fit_context(*a)`로 **호출할 때 모듈 전역을 찾게** 넘긴다. 그래야 기존 테스트의 `patch('scripts.shared_demo.complete')`, `patch('scripts.shared_demo.prompt_tokens')`가 계속 동작한다.

- [ ] **Step 1: 실패하는 테스트를 쓴다.** `tests/test_document_case.py`(`test_pdf_documents.CharTokenizer`와 `make_chunks` 사용)
  - `test_fit_own_passages_keeps_best_and_drops_weakest_first`: 구간 3개, `count_tokens`가 [5000, 4000, 3000]을 차례로 돌려주고 `context=4096`, `max_tokens=512`이면 결과는 첫 구간 하나, `limits['own_passages'] == 2`이다.
  - `test_fit_own_passages_works_on_plain_pdf_chunks`: `trid.pdf_documents.Chunk`로 예산을 넘겨도 `AttributeError`가 나지 않는다.
  - `test_case_quote_requires_verbatim_match`: 주입한 `complete`가 원문에 있는 문장 하나와 없는 문장 하나를 돌려주면 `status == 'quoted'`, `len(candidates) == 1`이고, 없는 문장은 `commentary`에만 있다.
  - `test_case_withholds_sentence_with_unsupported_number`
  - `test_case_overflow_raises_context_overflow`: `complete`가 `{'skipped_context': True}`를 돌려주면 `DocumentError(CONTEXT_OVERFLOW)`가 난다.
- [ ] **Step 2: 실행해서 실패를 확인한다.** Expected: `ModuleNotFoundError`.
- [ ] **Step 3: 구현한다.** 본문은 그대로 옮기고 `complete`·`fit_context`를 인자로 받는다. `fit_own_passages`는 새로 쓴다.
- [ ] **Step 4: 새 테스트와 `test_shared_demo.py`를 실행한다.** Expected: 새 테스트 OK, `test_shared_demo.py`는 Task 1 기준선과 개수가 같고 테스트 파일은 수정하지 않았다.
- [ ] **Step 5: 커밋한다.** `refactor: share document case answering via trid.document_case`

### Task 3: 격리 PDF 파싱을 `trid/`로 옮기고 한도를 인자로 받기

**Files:**
- Create: `trid/pdf_isolated.py`, `trid/pdf_extract_worker.py`
- Modify: `scripts/shared_demo.py:1814-1850`. `parse_pdf(body, name, tok)`를 `pdf_isolated.parse_pdf(body, name, tok, limits=DEMO_LIMITS)` 호출로 바꾼다. `scripts/shared_pdf_extract.py`는 worker의 `main()`을 테스트 사이트 한도로 호출하는 2줄짜리로 바꾼다.
- Test: `tests/test_pdf_isolated.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class PdfLimits: max_bytes: int = 2*1024*1024; max_pages: int = 20; max_characters: int = 50_000; wall_seconds: float = 15.0; cpu_seconds: int = 10; address_space_bytes: int = 1024**3`
  - `DEMO_LIMITS = PdfLimits()`
  - `parse_pdf(body: bytes, name: str, tok, limits: PdfLimits = DEMO_LIMITS) -> trid.pdf_documents.Document`. 실패하면 `DocumentError`가 난다.
  - worker 실행: `[sys.executable, '-I', '-m', 'trid.pdf_extract_worker', str(max_bytes), str(max_pages), str(max_characters), str(cpu_seconds), str(address_space_bytes)]`. stdin은 PDF 바이트, stdout은 `{'pages','warnings'}` 또는 `{'error'}` JSON이다. `resource`는 POSIX에서만 import한다. 오류 문구는 `f'텍스트가 있는 PDF {max_bytes//(1024*1024)}MB·{max_pages}페이지·{max_characters//10000}만 자 이하로 다시 시도해 주세요.'`이다(기본값이면 지금 문구와 같다).
  - 리눅스 메모리 제한은 worker의 `RLIMIT_AS`가 맡는다. macOS의 `ps` RSS 감시는 `sys.platform == 'darwin'`에서만 그대로 둔다.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**
  - `test_parse_pdf_passes_limits_to_worker`: `subprocess.Popen`을 patch해서 argv에 한도 5개가 순서대로 들어가는지 본다.
  - `test_parse_pdf_page_limit_message_uses_limits`: 3쪽 PDF(`tests/fixtures`에 있으면 재사용, 없으면 `pypdf.PdfWriter`로 빈 페이지 생성)와 `max_pages=2`로 실행하면 `'2페이지'`가 든 `DocumentError`가 난다. `@unittest.skipUnless(os.name == 'posix')`
  - `test_parse_pdf_kills_worker_after_wall_timeout`: `wall_seconds=0.2`이고 worker가 멈춰 있으면(Popen을 patch해 `poll()`이 계속 None) `'시간이 초과'`가 든 `DocumentError`가 나고 `kill()`이 호출된다.
  - `test_demo_limits_unchanged`: `DEMO_LIMITS == PdfLimits(2*1024*1024, 20, 50_000, 15.0, 10, 1024**3)`
- [ ] **Step 2: 실행해서 실패를 확인한다.**
- [ ] **Step 3: 구현한다.** 기존 `parse_pdf` 본문을 옮기고 한도만 인자로 바꾼다.
- [ ] **Step 4: 새 테스트와 `test_shared_demo.py`, `test_pdf_documents.py`를 실행한다.** Expected: 모두 OK, 기준선 개수와 같다.
- [ ] **Step 5: 커밋한다.** `refactor: isolated PDF parsing with explicit limits in trid.pdf_isolated`

### Task 4: 비밀번호와 SQLite 저장소

**Files:**
- Create: `trid/onprem/__init__.py`, `trid/onprem/passwords.py`, `trid/onprem/store.py`
- Test: `tests/test_onprem_store.py`

**Interfaces:**
- Produces:
  - `passwords.hash_password(pw: str) -> str`. 형식은 `scrypt$16384$8$1$<salt b64>$<hash b64>`(salt 16바이트, dklen 32).
  - `passwords.verify_password(pw: str, stored: str) -> bool`. 비교는 `hmac.compare_digest`로 한다.
  - `passwords.check_new_password(pw: str) -> None`. 10자 미만이면 `ValueError('비밀번호는 10자 이상이어야 합니다.')`
  - `store.normalize_username(name: str) -> str`. NFC, 앞뒤 공백 제거, 소문자. 허용 문자는 `[a-z0-9._-]{2,32}`이고 아니면 `ValueError`.
  - `@dataclass class User: username: str; role: str; pw_hash: str; must_change: bool; failed: int; locked_until: float; disabled: bool; password_version: int`
  - `class Store(path: Path, clock=time.time)`
    - `create_user(username, password, role, must_change=True) -> User`
    - `get_user(username) -> User | None`
    - `list_users() -> list[User]`
    - `update_user(username, **fields) -> None`. 허용 필드는 `role, pw_hash, must_change, failed, locked_until, disabled, password_version`이다.
    - `count_active_admins() -> int`
    - `log(event: str, username: str | None, ip: str | None, result: str, **detail) -> None`
    - `query_events(since=None, until=None, username=None, event=None, limit=200, offset=0) -> list[dict]`
    - `export_csv(...) -> str`. `query_events`와 같은 필터를 받고, 열은 `ts,username,ip,event,result,detail`이다.
    - `purge_before(ts: float) -> int`
  - 이벤트 이름: `login_ok, login_failed, login_locked, logout, session_expired, password_changed, pdf_upload, ask, clear, admin_create_user, admin_update_user, admin_reset_password, admin_export`.
  - `detail`에 허용하는 키는 `sha256, pages, seconds, status, error, target, action, count`뿐이다. 그 밖의 키를 넘기면 `ValueError`가 난다(파일명·질문이 기록에 섞이지 않도록 하는 코드상의 장치).

- [ ] **Step 1: 실패하는 테스트를 쓴다.** 임시 디렉터리 DB를 쓴다.
  - `test_hash_roundtrip_and_salted`: 같은 비밀번호도 해시가 둘 다르고, 둘 다 검증을 통과한다.
  - `test_short_password_rejected`: 9자면 거부, 10자면 통과.
  - `test_username_normalized_on_create_and_lookup`: `create_user(' Kim ', …)` 뒤 `get_user('KIM').username == 'kim'`이고, `create_user('kim', …)`를 다시 하면 `ValueError`가 난다.
  - `test_log_rejects_unknown_detail_keys`: `log('pdf_upload', 'kim', '10.0.0.1', 'ok', filename='a.pdf')`면 `ValueError`가 난다.
  - `test_events_never_store_question_or_filename`: 허용 키로 `log`한 뒤 DB 파일 바이트에 테스트용 질문·파일명 문자열이 없다.
  - `test_purge_before_removes_only_older_events`
  - `test_export_csv_header_and_rows`
- [ ] **Step 2: 실행해서 실패를 확인한다.**
- [ ] **Step 3: 구현한다.** `sqlite3`, WAL 모드, `PRAGMA user_version=1`로 스키마를 만든다. 스레드마다 연결하거나 `threading.Lock`으로 한 연결을 감싼다(빌더 선택).
- [ ] **Step 4: 실행해서 통과를 확인한다.**
- [ ] **Step 5: 커밋한다.** `feat(onprem): password hashing and SQLite account/event store`

### Task 5: 로그인·잠금·세션

**Files:**
- Create: `trid/onprem/auth.py`
- Test: `tests/test_onprem_auth.py`

**Interfaces:**
- Consumes: Task 4 `Store`, `User`, `passwords.*`, `normalize_username`
- Produces:
  - `@dataclass class Session: sid: str; csrf: str; username: str; role: str; password_version: int; must_change: bool; touched: float; document: object = None; document_id: str | None = None; calls: list = field(default_factory=list)`
  - `class Auth(store, clock=time.monotonic, wall=time.time, max_sessions=32, idle_seconds=1800, max_failures=5, lock_seconds=900)`
    - `login(username: str, password: str, ip: str) -> tuple[str, Session | None]`. 상태는 `'ok' | 'must_change' | 'invalid' | 'locked' | 'busy'`이다. `'invalid'`와 `'locked'`는 화면에서 같은 문구를 쓴다. `'ok'`·`'must_change'`면 세션을 만든다.
    - `session(sid: str) -> Session | None`. 만료되면 문서를 지우고 `session_expired`를 기록한 뒤 `None`을 돌려준다. 사용자가 비활성화됐거나 `password_version`이 다르면 무효다.
    - `logout(sid: str, ip: str) -> None`
    - `change_password(session, current: str, new: str, ip: str) -> None`. `password_version += 1`, `must_change=False`가 되고 그 사용자의 다른 세션은 끝난다.
    - `admin_update(actor: Session, target: str, *, role=None, disabled=None, reset_password: str | None = None, ip: str) -> None`. 마지막 활성 관리자를 잠그거나 강등하면 `ValueError('마지막 관리자는 잠그거나 권한을 바꿀 수 없습니다.')`. 비밀번호 초기화는 `must_change=True`로 하고 대상의 세션을 끝낸다.
    - `prune() -> None`

- [ ] **Step 1: 실패하는 테스트를 쓴다.** 가짜 clock을 쓴다.
  - `test_lock_after_five_failures_for_fifteen_minutes`: 5번 틀린 뒤에는 맞는 비밀번호도 `'locked'`이고, clock을 +900초 하면 `'ok'`이다.
  - `test_success_resets_failure_count`
  - `test_unknown_user_same_response_logged_as_unknown`: 상태 `'invalid'`, 기록 `username is None`(CSV에는 `(unknown)`), `scrypt`는 호출된다(가짜 해시로 검증해 시간 차를 줄인다).
  - `test_idle_session_expires_after_1800s_and_drops_document`
  - `test_must_change_on_first_login`
  - `test_password_change_ends_other_sessions`
  - `test_cannot_lock_or_demote_last_active_admin`: 관리자가 한 명일 때 자기 자신의 `disabled=True`·`role='user'`는 `ValueError`이고, 관리자 두 명이면 허용된다.
  - `test_session_cap_returns_busy`: 세션 32개 뒤 33번째는 `'busy'`
- [ ] **Step 2: 실행해서 실패를 확인한다.**
- [ ] **Step 3: 구현한다.** 세션 id·csrf는 `secrets.token_urlsafe(32)`로 만든다.
- [ ] **Step 4: 실행해서 통과를 확인한다.**
- [ ] **Step 5: 커밋한다.** `feat(onprem): login, lockout and session lifecycle`

### Task 6: 설치판 HTTP 서버 — 직원 기능

**Files:**
- Create: `trid/onprem/config.py`, `trid/onprem/model_card.py`, `trid/onprem/server.py`
- Create(임시 최소본, Task 8에서 완성): `trid/onprem_ui/login.html`, `index.html`
- Test: `tests/test_onprem_server.py`

**Interfaces:**
- Consumes: Task 1 `complete`, `count_prompt_tokens`. Task 2 `answer_document_case`, `fit_own_passages`. Task 3 `parse_pdf`, `PdfLimits`. Task 5 `Auth`, `Session`.
- Produces:
  - `@dataclass(frozen=True) class Config`. 필드와 기본값: `hostname: str`(필수), `port=8080`, `data_dir=Path('/data')`, `llm_base='http://llm:8080'`, `tokenizer_path=Path('/opt/trid/tokenizer.json')`, `retention_days=365`, `pdf=PdfLimits()`, `max_question=2000`, `idle_seconds=1800`, `max_sessions=32`, `slots=8`, `per_minute=20`, `lock_wait=2.0`, `context=4096`, `max_tokens=512`, `family='qwen3'`. `load_config(path: Path) -> Config`는 JSON을 읽고 모르는 키는 거부한다.
  - `model_card.MODEL_CARD: dict`. 키는 `label, version, status, base_model, license, evaluation, limits`이다. `version`·`limits`는 spec 6절 문구를 그대로 쓰고, `status`·`base_model`·`license`·`evaluation`은 `deploy/macmini-site/versions.json`의 `trid-assistant-qwen3-1.7b` 값을 그대로 쓴다.
  - `make_server(config, store, auth, tok, answer=None) -> ThreadingHTTPServer`. `answer(document, question) -> dict`를 테스트에서 주입할 수 있다. 기본값은 `answer_document_case(document, question, config.llm_base, config.family, config.context, max_tokens=config.max_tokens, complete=complete, fit_context=lambda s, b, base, c, m: fit_own_passages(s, b, lambda msgs: count_prompt_tokens(base, msgs), c, m))`이다.
  - 라우트(직원):
    - `GET /`: 세션이 없으면 `login.html`, 있으면 `index.html`(csrf·모델 카드 치환).
    - `GET /app.js`, `/login.js`, `/common.css`, `GET /healthz` → `{'ok': True}`
    - `POST /api/login {username,password}` → 200 `{status}` + 쿠키, 401 `{error:'아이디 또는 비밀번호가 올바르지 않습니다.'}`(invalid·locked 공통), 503(busy)
    - `POST /api/logout`
    - `POST /api/password {current,new}`
    - `POST /api/load`(본문 PDF, 헤더 `X-File-Name`) → `{document_id, name, pages, warnings}`. `name`은 화면 표시용이고 기록하지 않는다.
    - `POST /api/ask {question, document_id}` → `answer` 결과 dict(`checked`, `raw_output` 제거)
    - `POST /api/clear`
  - 공통 규칙:
    - POST는 `Origin == 'https://' + hostname`과 `X-Trid-Token == session.csrf`를 확인한다(login 제외. login은 Origin만 확인).
    - 쿠키 `__Host-trid=<sid>; Path=/; Secure; HttpOnly; SameSite=Strict`
    - 보안 헤더와 CSP는 `shared_demo.make_server.respond`의 값을 그대로 쓴다.
    - `Content-Length` 필수, `Transfer-Encoding` 거부, 본문 상한은 `config.pdf.max_bytes`(ask·login은 16KiB).
    - 세션당 분당 20회를 넘으면 429. 슬롯 8개(`BoundedServer`와 같은 방식).
    - 생성 잠금은 `lock_wait`초 기다리고, 못 잡으면 503 `'다른 질문을 처리 중입니다. 잠시 후 다시 시도해 주세요.'`
    - `must_change` 세션은 `/api/password`, `/api/logout`만 허용하고 나머지는 403 `{error, must_change: true}`이다.
    - 클라이언트 IP는 `X-Real-IP`(Caddy가 설정)를 쓰고, 없으면 소켓 주소를 쓴다.
    - 예외는 잡아서 `{error}`만 응답한다. 표준 오류에는 `type(e).__name__`만 쓰고 `traceback.print_exc()`는 쓰지 않는다. `log_message`는 아무것도 하지 않는다.
  - 기록:
    - `pdf_upload`: `sha256=hashlib.sha256(body)`, `pages`, `status`(`ok`/`error`)
    - `ask`: `seconds`, `status`(결과 `status` 또는 `error`), 실패면 `error`(`model_unavailable` | `context_overflow` | `bad_request` | `busy`)
    - `clear`, `login_*`, `logout`, `password_changed`

- [ ] **Step 1: 실패하는 테스트를 쓴다.** 실제 서버를 `127.0.0.1:0`에 띄우고 `http.client`로 요청한다(`test_shared_demo.py` 방식). `answer`는 주입하고, PDF는 `patch('trid.onprem.server.parse_pdf')`로 대체한다.
  - `test_unauthenticated_get_serves_login_and_api_returns_401`
  - `test_login_sets_host_cookie_and_wrong_origin_rejected`
  - `test_csrf_required_on_post`
  - `test_must_change_blocks_load_until_password_changed`
  - `test_upload_then_ask_logs_hash_pages_seconds_status_only`: 업로드·질문 뒤 `store.query_events()`의 detail 키가 허용 집합 안에 있고, DB 파일 바이트에 파일명·질문 문자열이 없다.
  - `test_document_is_per_session`: A가 올린 `document_id`로 B가 질문하면 409.
  - `test_expired_session_mid_ask_returns_401_and_drops_document`: clock을 +1801초 하고 ask하면 401이고, 세션 문서는 None이다.
  - `test_model_down_releases_lock_and_logs_kind_only`: `answer`가 `ConnectionRefusedError`를 던지면 502 `{'error': '모델 서버에 연결할 수 없습니다. 관리자에게 문의해 주세요.'}`이고, 바로 다음 ask는 잠금을 얻으며, 기록은 `error='model_unavailable'`, stderr에는 질문 문자열이 없다(`contextlib.redirect_stderr`).
  - `test_generation_lock_busy_returns_503_after_lock_wait`
  - `test_rate_limit_21st_request_in_minute_is_429`
  - `test_oversize_body_413`
  - `test_index_page_shows_model_status_text`: 응답에 `'실험 서비스 · 원본 모델(미특화) · 품질 기준 미달'`이 있다.
- [ ] **Step 2: 실행해서 실패를 확인한다.**
- [ ] **Step 3: 구현한다.** `shared_demo.make_server`의 구조(respond, BoundedServer, 본문 읽기)를 따르고, 그래프·workspace·router·history 분기는 넣지 않는다.
- [ ] **Step 4: 실행해서 통과를 확인한다.**
- [ ] **Step 5: 커밋한다.** `feat(onprem): staff-facing HTTP server with auth and usage events`

### Task 7: 관리자 API

**Files:**
- Modify: `trid/onprem/server.py`
- Test: `tests/test_onprem_admin.py`

**Interfaces:**
- Consumes: Task 5 `Auth.admin_update`, Task 4 `Store.create_user/list_users/query_events/export_csv`
- Produces(모두 `role == 'admin'` 필요, 아니면 403):
  - `GET /admin` → `admin.html`
  - `POST /api/admin/users/list` → `[{username, role, disabled, locked, must_change}]`(해시 제외)
  - `POST /api/admin/users/create {username, role, password}` → 생성된 사용자는 `must_change=True`
  - `POST /api/admin/users/update {username, role?, disabled?, reset_password?}`. `unlock`은 `failed=0, locked_until=0`으로 처리한다.
  - `POST /api/admin/events {since?, until?, username?, event?, offset?}` → 최대 200행
  - `GET /api/admin/events.csv?since=&until=&username=&event=`(csrf는 쿼리 `token`) → `text/csv; charset=utf-8`, BOM 포함(엑셀 한글 표시), `Content-Disposition: attachment`. 다운로드할 때마다 `admin_export`를 기록한다.
  - 수정·삭제용 이벤트 API는 만들지 않는다.

- [ ] **Step 1: 실패하는 테스트를 쓴다.**
  - `test_non_admin_gets_403_on_every_admin_route`
  - `test_admin_creates_user_who_must_change_password`
  - `test_admin_reset_password_ends_target_sessions`
  - `test_admin_unlock_clears_lock`
  - `test_last_admin_demotion_rejected_over_http`: 400과 해당 문구를 돌려준다.
  - `test_users_list_never_contains_hash`
  - `test_events_csv_has_bom_and_logs_export`
  - `test_no_route_deletes_or_edits_events`: `/api/admin/events/delete` 등은 404.
- [ ] **Step 2: 실행해서 실패를 확인한다.**
- [ ] **Step 3: 구현한다.**
- [ ] **Step 4: 실행해서 통과를 확인한다.**
- [ ] **Step 5: 커밋한다.** `feat(onprem): admin account management and usage log API`

### Task 8: 화면 (로그인·직원·관리자)

**Files:**
- Create/Modify: `trid/onprem_ui/common.css`, `login.html`, `login.js`, `index.html`, `app.js`, `admin.html`, `admin.js`
- Test: `tests/test_onprem_ui.py`

**Interfaces:**
- Consumes: Task 6·7 라우트
- 화면 요구:
  - `trid/shared_ui/index.html`의 색 토큰과 글꼴 스택을 `common.css`로 옮겨 쓴다. 외부 폰트·CDN·인라인 스크립트는 쓰지 않는다(CSP `script-src 'self'`).
  - 서버 값은 모두 `textContent`로 넣는다(`shared_ui/app.js`와 같은 규칙).
  - 로그인 화면: 아이디, 비밀번호, 오류 문구. 비밀번호 변경이 필요하면 바로 변경 화면으로 간다.
  - 직원 화면: 상단에 사용자 이름과 로그아웃, 모델 정보 카드(`MODEL_CARD`), PDF 선택·문서 읽기·자료 지우기, 추출 문서 보기, 질문 입력(2,000자)과 안내 "질문마다 완전한 문장으로 써 주세요. 이전 질문은 이어서 해석하지 않습니다."
    - 답 표시는 `shared_ui/app.js`의 `case` 결과 렌더링(확인된 인용의 쪽·원문 강조, "검증되지 않은 해설" 표시, 안내 notes)을 가져오고, 그래프·대화·버전 선택 코드는 뺀다.
    - 401을 받으면 로그인 화면으로 이동하고 "세션이 만료됐습니다. 다시 로그인해 주세요."를 보여 준다.
  - 관리자 화면: 계정 목록·생성·잠금/해제·비밀번호 초기화·역할 변경, 이용 기록 필터(기간·계정·이벤트)·페이지 넘김·CSV 다운로드. 직원 화면으로 가는 링크를 둔다.
  - 휴대폰 너비(360px)에서 가로 스크롤이 없어야 한다.
- [ ] **Step 1: 실패하는 테스트를 쓴다.**
  - `test_ui_has_no_external_urls`: 모든 `trid/onprem_ui/*` 파일에 `http://`, `https://`, `//fonts`, `@import url(`이 없다.
  - `test_ui_has_no_inline_script_or_innerhtml_from_server`: html에 `<script>`(src 없는) 없음, js에 `innerHTML =` 없음.
  - `test_index_contains_independent_question_notice`
- [ ] **Step 2: 실행해서 실패를 확인한다.**
- [ ] **Step 3: 구현한다.**
- [ ] **Step 4: 실행해서 통과를 확인한다.** 그리고 로컬 확인: `python -m trid.onprem serve --config deploy/onprem/onprem.example.json --insecure-dev`(Task 9의 개발용 플래그, Secure 쿠키 대신 `trid-dev` 쿠키)로 띄워 브라우저에서 로그인 → 업로드 → 질문(llm 없이 오류 문구 확인) → 관리자 화면을 확인한다.
- [ ] **Step 5: 커밋한다.** `feat(onprem): login, staff and admin pages`

### Task 9: 실행 명령·보관 기간 정리

**Files:**
- Create: `trid/onprem/__main__.py`, `deploy/onprem/onprem.example.json`
- Test: `tests/test_onprem_cli.py`

**Interfaces:**
- Produces:
  - `python -m trid.onprem serve --config PATH [--insecure-dev]`: 설정·토크나이저 sha256(Global Constraints 값)을 확인하고 서버를 시작한다. 시작할 때와 이후 24시간마다 `store.purge_before(now - retention_days*86400)`를 실행하고 정리한 건수를 `count`로 기록한다. `--insecure-dev`는 `127.0.0.1`에만 bind하고 쿠키 이름을 `trid-dev`(Secure 없음), Origin을 `http://127.0.0.1:<port>`로 쓴다. 컨테이너에서는 쓰지 않는다.
  - `python -m trid.onprem create-admin --config PATH --username NAME`: 비밀번호는 `getpass`로 두 번 입력받거나 stdin에서 받는다(`--password-stdin`). 사용자가 이미 있으면 실패한다. 처음 로그인할 때 변경을 강제하지 않는다(설치 담당자가 직접 정한 비밀번호이므로).
  - `python -m trid.onprem purge --config PATH`
  - 환경변수 `HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`
- [ ] **Step 1: 실패하는 테스트를 쓴다.**
  - `test_create_admin_via_password_stdin`
  - `test_create_admin_refuses_existing_user`
  - `test_serve_refuses_tokenizer_with_wrong_sha256`
  - `test_purge_respects_retention_days`
  - `test_load_config_rejects_unknown_keys`
- [ ] **Step 2: 실행해서 실패를 확인한다.**
- [ ] **Step 3: 구현한다.**
- [ ] **Step 4: 실행해서 통과를 확인한다.**
- [ ] **Step 5: 커밋한다.** `feat(onprem): serve/create-admin/purge commands`

### Task 10: Docker 이미지·compose·오프라인 묶음

**Files:**
- Create: `deploy/onprem/Dockerfile.app`, `deploy/onprem/Dockerfile.llm`, `deploy/onprem/Caddyfile`, `deploy/onprem/compose.yaml`, `deploy/onprem/build_bundle.sh`, `deploy/onprem/LICENSES.md`
- Test: `tests/test_onprem_deploy.py`(파일 내용 검사). 실제 빌드는 Docker가 있는 x86_64 리눅스 빌드 머신에서 한다.

**Interfaces:**
- `Dockerfile.app`:
  - base `python:3.12-slim`(digest 고정)
  - `pip install --no-index --find-links wheels tokenizers==0.23.3 pypdf==6.21.2`(wheel은 빌드 머신에서 `pip download`)
  - `trid/` 중 설치판에 필요한 모듈만 복사: `trid/__init__.py`, `pdf_documents.py`, `pdf_isolated.py`, `pdf_extract_worker.py`, `document_case.py`, `llama_client.py`, `prompts.py`, `case_prompts.py`, `onprem/`, `onprem_ui/`, 그리고 이 모듈들이 import하는 것만. `trid/__init__.py`가 torch를 import하지 않는지 확인한다.
  - 토크나이저를 `/opt/trid/tokenizer.json`에 둔다. 비root 사용자, `VOLUME /data`, `CMD ["python","-m","trid.onprem","serve","--config","/data/onprem.json"]`
- `Dockerfile.llm`:
  - llama.cpp `8ca4060`을 `-DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_CUDA=OFF -DGGML_METAL=OFF -DLLAMA_CURL=OFF`로 빌드(멀티스테이지).
  - GGUF를 복사하고 빌드 중 sha256을 검증한다(불일치면 실패).
  - 실행: `llama-server -m /models/Qwen3-1.7B-Q8_0.gguf -c 4096 --host 0.0.0.0 --port 8080 -t ${LLM_THREADS:-4} --no-webui`
- `compose.yaml`:
  - 네트워크 `internal`(`internal: true`)과 `edge`
  - `llm`·`app`은 `internal`에만 붙인다. `tls`(Caddy, digest 고정)는 `internal`+`edge`에 붙이고 `443:443`만 연다.
  - `app` 볼륨 `./data:/data`, `tls` 볼륨 `./certs:/certs:ro`
  - 각 서비스에 `restart: unless-stopped`, `app`은 `mem_limit`을 둔다.
- `Caddyfile`:
  - `{$TRID_HOSTNAME}`, 인증서 `/certs/cert.pem`·`key.pem`
  - `reverse_proxy app:8080`, `header_up X-Real-IP {remote_host}`, `request_body max_size 3MB`
  - 자동 HTTPS·업데이트 확인·텔레메트리 끔
- `build_bundle.sh`:
  - 모델·토크나이저 sha256 검증 → 이미지 3개 빌드 → `docker save`로 `trid-onprem-<git short sha>.tar` 생성
  - compose·Caddyfile·`onprem.example.json`·`docs/ONPREM.md`·`LICENSES.md`·`SHA256SUMS`를 함께 `trid-onprem-<sha>/` 폴더에 넣는다.
  - 자체 서명 인증서 생성 스크립트 `make_self_signed.sh`(openssl, 호스트명 인자)를 포함한다.
- `LICENSES.md`: Qwen3 Apache-2.0, llama.cpp MIT, Caddy Apache-2.0, tokenizers Apache-2.0, pypdf BSD-3-Clause, Python PSF, 이미지 base 라이선스

- [ ] **Step 1: 실패하는 테스트를 쓴다.**
  - `test_compose_llm_and_app_only_on_internal_network`
  - `test_only_tls_publishes_port_443`
  - `test_dockerfile_llm_disables_native_and_pins_commit`
  - `test_dockerfile_llm_checks_gguf_sha256`(Global Constraints 값이 들어 있는지)
  - `test_app_image_copies_no_torch_modules`: `Dockerfile.app`의 COPY 목록 모듈을 `ast`로 파싱해 `torch`·`numpy` import가 없는지 확인한다.
- [ ] **Step 2: 실행해서 실패를 확인한다.**
- [ ] **Step 3: 파일을 작성한다.**
- [ ] **Step 4: 테스트를 실행한다.** 이후 빌드 머신에서 `bash deploy/onprem/build_bundle.sh`를 실행한다. Expected: tar 생성, `SHA256SUMS` 검증 OK.
- [ ] **Step 5: 빌드 머신에서 인터넷을 끄고(`docker network disconnect` 또는 네트워크 차단 VM) 설치 리허설을 한다.**
  - `docker load -i …tar` → 인증서 생성 → `docker compose up -d` → `docker compose exec app python -m trid.onprem create-admin …`
  - `curl -k https://<host>/healthz`가 `{"ok": true}`이고 `docker compose logs`에 외부 접속 시도가 없어야 한다.
- [ ] **Step 6: 커밋한다.** `feat(onprem): docker images, compose and offline bundle`

### Task 11: 설치 안내서와 배포 전 점검

**Files:**
- Create: `docs/ONPREM.md`, `deploy/onprem/release_check.py`, `deploy/onprem/release-questions.json`
- Test: `tests/test_onprem_release_check.py`

**Interfaces:**
- `release_check.py --base https://<host> --user <admin> --pdf <file> --questions release-questions.json --out report.json`
  - 실제 설치본에 로그인해 PDF를 올리고 고정 질문을 순서대로 묻는다. 각 질문의 `status`, 인용 수, 쪽 번호, 처리 시간, `commentary` 유무를 `report.json`에 쓴다(이 보고서는 배포 담당자 PC에만 남는다).
  - 기대값과 비교하지 않고 사람이 확인한다(spec 1-2절 ②).
- `release-questions.json`: 저장소의 시연용 가상 문서(`data/reading-demo-v3/travel-named.txt`를 PDF로 바꾼 파일을 `deploy/onprem/release.pdf`로 커밋, 실제 기관 자료 아님)에 대한 질문 6~8개. 답 있음, 답 없음, 숫자 질문을 섞는다.
- `docs/ONPREM.md`(한국어) 목차:
  1. 구성과 한계(모델 상태 문구, 해설 미검증, 스캔 PDF 불가, 질문별 독립 처리)
  2. 권장 사양: x86_64·AVX2, 동시 5~10명 기준 CPU 코어·메모리·디스크. **수치는 Task 10 리허설과 release_check 처리 시간 실측으로 채운다.**
  3. 설치(이미지 불러오기, 인증서, 설정 파일, 첫 관리자)
  4. 운영(계정 관리, 이용 기록 조회·CSV, 보관 기간, 기록 항목과 남기지 않는 항목 목록)
  5. 백업·복구(`./data`의 SQLite. 서비스 정지 후 복사하거나 `sqlite3 .backup`)
  6. 업그레이드·되돌리기(이전 tar 보관)
  7. 인증서 교체
  8. 문제 해결(모델 서버 연결 실패, 잠긴 계정, 세션 만료)
- [ ] **Step 1: 실패하는 테스트를 쓴다.**
  - `test_release_check_writes_one_row_per_question`(가짜 서버)
  - `test_release_pdf_is_synthetic_demo_material`: `release.pdf` 텍스트에 "가상" 표기가 있다.
- [ ] **Step 2: 실행해서 실패를 확인한다.**
- [ ] **Step 3: 구현하고 안내서를 쓴다.**
- [ ] **Step 4: 빌드 머신 설치본에 `release_check.py`를 실행한다.** 처리 시간 중앙값·최대값을 안내서 2절에 기입하고, 결과를 사람이 검토한다.
- [ ] **Step 5: 커밋한다.** `docs(onprem): install guide and release check`

---

## 범위 밖 / 확인 필요 메모

- 테스트 사이트(`test.tridatum.co`) 재배포는 하지 않는다. Task 1~3은 테스트 사이트 코드를 바꾸지만 동작은 바꾸지 않으며, 재배포 여부는 따로 정한다.
- 발견 사항: 테스트 사이트의 `shared_demo.fit_context`는 PDF `Chunk`에 없는 `doc_id`를 읽는다. 문맥 예산을 넘는 PDF 질문에서 `AttributeError`가 날 수 있지만, 상위 6구간이 보통 예산 안이라 지금은 드러나지 않는다. 설치판은 Task 2의 `fit_own_passages`를 쓰므로 영향이 없다. 테스트 사이트 수정은 범위 밖이다.
- 현재 개발 PC(Windows)에는 Python 3.12·uv·Docker가 없다. Task 1~9는 Python 3.12 환경만 있으면 되고, Task 10~11은 Docker가 있는 x86_64 리눅스 빌드 머신이 필요하다. Task 3의 일부 테스트는 POSIX 전용이다.
