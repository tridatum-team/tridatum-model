# Static Linux port of recovered qwen-serving-harness.py (original SHA256 prefix 9b07f62...).
"""Standalone no-retrieval Qwen serving measurement harness (qwen-recovery-003).

Runs ONLY as the child of supervisor006 (parent-owned process group, launched through the Linux guard): validates
every input (receipt absent, model file sha256, optional adapter GGUF sha256, server binary, LD_LIBRARY_PATH on the
CPU build dir, loopback port free and not a live demo port, suite shape, time budget) BEFORE authenticating guard
ownership (the Linux guard already consumed the G byte), then spawns runs/tools/llama.cpp-8ca4060/llama-server in the SAME process group (no new session),
bound to 127.0.0.1 only, CPU only. No Workspace, no bge-m3, no native reader, no RAG: the model sees only the chat
messages.

Backends: a merged GGUF (--model-file alone, arm `model`) or base GGUF + LoRA adapter GGUF (--lora, passed to
llama-server `--lora`; arms `adapter` = scale 1.0 and `base` = scale 0.0 switched inside the same process through
POST /lora-adapters and confirmed by GET, so both arms share one render; --arms adapter,base runs the suite twice).
Rendering follows scripts/base_dialogue_runner.py exactly: /apply-template with chat_template_kwargs
{"enable_thinking": false} (native Qwen3 no-thinking template, no system prompt), /tokenize, then /completion with
the token ids, greedy (temperature 0, seed 42), n_predict --max-new, cache_prompt off. Every rendered prompt must end
with the trainer's QWEN3_NO_THINK suffix; with --official-render the sample render must equal the official tokenizer
render (text and token ids) or the run aborts before any request.

Suite: either the serving suite (JSON {"sessions": [{id, turns: [{user, contains?, not_contains?, hangul?} | marker]}],
"oai_parity_session"?}, see qwen-serving-suite.json) or the evaluator input schema {"results": [{id, group, prompt |
turns: [user, ...]}]} unchanged (dev/final sets; --ids chunks a long set across several supervised children). Each
session keeps its own history made of the model's own answers; "__CLEAR__" / "<NEW_CHAT>" inside a session empties it;
a new session starts empty (reset / new-session isolation). One extra request repeats the first turn of
`oai_parity_session` through /v1/chat/completions to measure whether the OpenAI-compatible surface yields the same
text as the native path.

Each request records HTTP status, wall seconds, prompt/output token ids, stop type ('limit' = truncated), the server's
timings and generation settings, fit against the context window, the group's RSS and pressure. HTTP 200 is never read
as correctness: `surface` holds contains / not-contains / hangul measurements for the parent to interpret, not verdicts.
SIGTERM/SIGINT from the supervisor save the stop reason and the in-flight request at once, then shutdown proceeds
(SIGTERM to llama-server, grace, SIGKILL). Exit 0 only when every request returned 200, parity held and no group
member survived shutdown. Never modifies the repository.
"""
import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
from pathlib import Path

REPO = Path('/Users/eomgyuhyeon/사업/trid')
sys.path.insert(0, str(REPO / 'scripts'))
from base_dialogue_runner import NEW_CONVERSATION, request, sha256  # noqa: E402  (stdlib-only module)

SERVER_DIR = REPO / 'runs/tools/llama.cpp-8ca4060'
SERVER = SERVER_DIR / 'llama-server'
QWEN3_NO_THINK = '<think>\n\n</think>\n\n'  # scripts/train_open_weight_lora.py:56
ASSISTANT_NO_THINK = '<|im_start|>assistant\n' + QWEN3_NO_THINK
TEMPLATE_KWARGS = {'enable_thinking': False}
GREEDY = {'temperature': 0, 'seed': 42}
LIVE_DEMO_PORTS = {51873, 51912, 51917, 51918, 51990}  # shared_demo + live llama ports + base_dialogue_runner default
T_START = time.monotonic()  # the supervisor's timer is already armed when this process starts: startup counts
SUPERVISOR_CLEANUP_SECONDS = 2  # supervisor006 soft-stops the group at duration - 2
MIN_REQUEST_SECONDS = 30  # a derived per-request timeout below this cannot hold 384 CPU tokens; refuse instead
MODEL_BYTES_MAX = 5 * 1024**3  # weights alone must leave room under the supervisor's 6 GiB group RSS ceiling
HANGUL = re.compile('[가-힣]')
PS = '/bin/ps'
NATIVE_JOB = None


def now():
    return time.monotonic()


def members(pgid):
    """Group members as seen by ps, excluding the ps process itself (it inherits this pgid while it runs)."""
    proc = subprocess.Popen([PS, '-axo', 'pid=,pgid=,stat=,rss=,comm='], stdout=subprocess.PIPE, text=True)
    out, _ = proc.communicate(timeout=2)
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) == 5 and int(parts[1]) == pgid and int(parts[0]) != proc.pid:
            rows.append({'pid': int(parts[0]), 'stat': parts[2], 'rss_bytes': int(parts[3]) * 1024, 'comm': parts[4]})
    return rows


def native_guard():
    from qwen_cpu_native import authenticate_guard
    return authenticate_guard(NATIVE_JOB)


def pressure():
    from qwen_cpu_native import guard_module
    return guard_module(NATIVE_JOB).host_metrics()['memory_psi']


def sessions_of(suite):
    """Serving suite {"sessions": [{id, turns: [str marker | {user, ...}]}]} or evaluator input
    {"results": [{id, group, prompt | turns: [user, ...]}]} (the frozen dev/final schema; only id/group/user text read)."""
    if 'sessions' in suite:
        return suite['sessions']
    out = []
    for r in suite['results']:
        if ('prompt' in r) == ('turns' in r):
            raise ValueError(f'{r.get("id")}: exactly one of prompt / turns')
        users = [r['prompt']] if 'prompt' in r else list(r['turns'])
        out.append({'id': r['id'], 'group': r.get('group'),
                    'turns': [u if u in NEW_CONVERSATION else {'user': u} for u in users]})
    return out


def expand(suite, ids=None):
    """Suite -> ordered request specs {session, group, turn, user, reset, surface}. Markers ("__CLEAR__",
    "<NEW_CHAT>") are not requests: they set reset=True on the next turn of the same session. `ids` keeps a subset of
    sessions (parent chunks long sets across supervised children). Raises on malformed input."""
    specs = []
    sessions = sessions_of(suite)
    if ids is not None:
        missing = set(ids) - {s.get('id') for s in sessions}
        if missing:
            raise ValueError(f'unknown ids: {sorted(missing)}')
        sessions = [s for s in sessions if s.get('id') in ids]
    for session in sessions:
        if not session.get('id') or not session.get('turns'):
            raise ValueError('every session needs an id and turns')
        reset = False
        for index, turn in enumerate(session['turns']):
            if isinstance(turn, str):
                if turn not in NEW_CONVERSATION:
                    raise ValueError(f'{session["id"]}[{index}]: bare strings must be reset markers')
                reset = True
                continue
            user = turn.get('user')
            if not isinstance(user, str) or not user.strip():
                raise ValueError(f'{session["id"]}[{index}]: user text required')
            specs.append({'session': session['id'], 'group': session.get('group'), 'turn': index, 'user': user,
                          'reset': reset,
                          'surface': {k: turn.get(k) for k in ('contains', 'not_contains', 'hangul') if k in turn}})
            reset = False
    if len({s['session'] for s in specs}) != len(sessions):
        raise ValueError('session ids must be unique and every session needs at least one request')
    parity = suite.get('oai_parity_session')
    if parity and ids is not None and parity not in ids:
        parity = None  # chunk without the parity session: no OAI request
    if parity and parity not in {s['session'] for s in specs}:
        raise ValueError('oai_parity_session must name a session')
    return specs, parity


def arms_of(text, lora):
    """Comma list -> arms. With --lora: adapter (scale 1.0) and/or base (scale 0.0, same process, same render);
    without: only model."""
    arms = [x for x in text.split(',') if x]
    allowed = {'adapter', 'base'} if lora else {'model'}
    if not arms or set(arms) - allowed or len(set(arms)) != len(arms):
        raise ValueError(f'arms must be a non-repeating subset of {sorted(allowed)}')
    return arms


def surface(spec, text):
    """Measurements only (never a verdict): which expected substrings appear, which forbidden ones appear, hangul."""
    want = spec['surface']
    out = {}
    if 'contains' in want:
        out['contains'] = {k: k in text for k in want['contains']}
    if 'not_contains' in want:
        out['not_contains'] = {k: k not in text for k in want['not_contains']}
    if 'hangul' in want:
        out['hangul'] = bool(HANGUL.search(text)) == bool(want['hangul'])
    out['all_true'] = all(all(v.values()) if isinstance(v, dict) else v for v in out.values())
    return out


def budget(ready, request_timeout, shutdown_grace, n_requests, startup):
    """Seconds the window needs: measured startup so far + ready + requests + shutdown grace. Must stay within
    duration - SUPERVISOR_CLEANUP_SECONDS."""
    return startup + ready + n_requests * request_timeout + shutdown_grace


def derive_request_timeout(duration, ready, shutdown_grace, n_requests, startup):
    """Largest equal per-request timeout that fits the window (whole seconds), or None if below the minimum."""
    per = int((duration - SUPERVISOR_CLEANUP_SECONDS - startup - ready - shutdown_grace) // n_requests)
    return per if per >= MIN_REQUEST_SECONDS else None


def server_argv(model_file, port, context, threads, lora):
    argv = [str(SERVER), '-m', str(model_file), '--host', '127.0.0.1', '--port', str(port), '-c', str(context),
            '-np', '1', '-t', str(threads), '-tb', str(threads), '-b', '256', '-ub', '128', '-ngl', '0',
            '--no-repack', '--no-context-shift', '--jinja', '--no-webui']
    # --reasoning-format is left at the server default (auto). At 8ca4060 (tag b11946) the chat parser is built from
    # the template's generation prompt: common/chat-auto-parser-generator.cpp:133 consumes only the part before the
    # reasoning start tag (prefix(generation_prompt, "<think>")), then :104/:140-148 extract an optional
    # <think>...</think> block only when reasoning_format != none. With enable_thinking=false the generation prompt
    # ends with the empty think block, so 'none' leaves it in message.content (observed in base-serving-smoke) while
    # 'auto' consumes it and message.content is the generated text. The rendered prompt is unaffected: reasoning_format
    # reaches only the parser (common/chat.cpp:1487,1563), never the template render.
    if lora:
        argv += ['--lora', str(lora)]
    return argv


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model-file', type=Path, required=True, help='merged GGUF, or base GGUF when --lora is given')
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--lora', type=Path, help='LoRA adapter GGUF applied by llama-server --lora (scale 1.0)')
    p.add_argument('--lora-sha256', help='required with --lora')
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    p.add_argument('--server-log', type=Path, required=True)
    p.add_argument('--port', type=int, required=True, help='loopback port, isolated from the live demo')
    p.add_argument('--context', type=int, default=2048)
    p.add_argument('--max-new', type=int, default=384)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--ids', nargs='*', help='subset of session ids (chunk long sets across supervised children)')
    p.add_argument('--arms', default=None, help='with --lora: adapter | base | adapter,base (default adapter); else model')
    p.add_argument('--official-render', type=Path,
                   help='JSON {"messages": [...], "prompt": str, "token_ids": [...]} from the official HF tokenizer with '
                        'enable_thinking=False; the served render of the same messages must match or the run aborts')
    p.add_argument('--duration', type=float, required=True, help='supervisor duration cap in seconds (set by the guard)')
    p.add_argument('--ready-timeout', type=float, default=60)
    p.add_argument('--request-timeout', type=float, default=None,
                   help='per request; default: derived so ready + requests + grace fit the duration (>= 30 s each)')
    p.add_argument('--shutdown-grace', type=float, default=10)
    a = p.parse_args()

    # --- validation before native guard ownership: nothing is spawned if any of this fails ---
    receipt, model_file, suite_path, server_log = (x.resolve() for x in (a.receipt, a.model_file, a.suite, a.server_log))
    lora = a.lora.resolve() if a.lora else None
    if receipt.exists():
        sys.exit('refusing to overwrite an existing receipt')
    for path, what in ((model_file, 'model file'), (suite_path, 'suite'), (SERVER, 'llama-server')):
        if not path.exists():
            sys.exit(f'{what} missing: {path}')
    if os.environ.get('LD_LIBRARY_PATH') != str(SERVER_DIR):
        sys.exit(f'LD_LIBRARY_PATH must be {SERVER_DIR}')
    if bool(lora) != bool(a.lora_sha256):
        sys.exit('--lora and --lora-sha256 go together')
    if lora and not lora.exists():
        sys.exit(f'adapter missing: {lora}')
    if a.official_render and not a.official_render.exists():
        sys.exit(f'official render missing: {a.official_render}')
    if a.port in LIVE_DEMO_PORTS:
        sys.exit(f'port {a.port} belongs to a live/known service')
    try:
        with socket.socket() as s:
            s.bind(('127.0.0.1', a.port))
    except OSError as error:
        sys.exit(f'port {a.port} not bindable on loopback: {error}')
    model_bytes = model_file.stat().st_size + (lora.stat().st_size if lora else 0)
    if model_bytes > MODEL_BYTES_MAX:
        sys.exit(f'weights {model_bytes} bytes exceed {MODEL_BYTES_MAX}; cannot stay under the 6 GiB group ceiling')
    suite = json.loads(suite_path.read_text())
    try:
        specs, parity_session = expand(suite, set(a.ids) if a.ids else None)
        arms = arms_of(a.arms or ('adapter' if lora else 'model'), lora)
    except (ValueError, KeyError, TypeError) as error:
        sys.exit(f'suite/arms invalid: {error}')
    n_requests = len(specs) * len(arms) + (1 if parity_session else 0)
    native_guard()
    model_sha = sha256(model_file)
    if model_sha != a.expected_sha256:
        sys.exit(f'model sha256 {model_sha} != expected {a.expected_sha256}')
    if lora and sha256(lora) != a.lora_sha256:
        sys.exit('adapter sha256 mismatch')
    startup = round(time.monotonic() - T_START, 3)  # includes the hashes; measured, not guessed
    request_timeout = a.request_timeout
    if request_timeout is None:
        request_timeout = derive_request_timeout(a.duration, a.ready_timeout, a.shutdown_grace, n_requests, startup)
        if request_timeout is None:
            sys.exit(f'{n_requests} requests cannot each get {MIN_REQUEST_SECONDS} s within {a.duration} s '
                     f'(startup {startup}, ready {a.ready_timeout}, grace {a.shutdown_grace}); chunk with --ids')
    need = budget(a.ready_timeout, request_timeout, a.shutdown_grace, n_requests, startup)
    if need > a.duration - SUPERVISOR_CLEANUP_SECONDS:
        sys.exit(f'window needs {need} s (startup {startup} + ready + {n_requests} x {request_timeout} + grace) '
                 f'> duration {a.duration} - {SUPERVISOR_CLEANUP_SECONDS} s; chunk with --ids')

    # --- receipt skeleton and signal handling are in place before recording native guard ownership ---
    pgid = os.getpgid(0)
    argv = server_argv(model_file, a.port, a.context, a.threads, lora)
    record = {'harness': 'qwen-serving-harness', 'harness_pid': os.getpid(), 'pgid': pgid, 'argv': argv,
              'backend': 'base_gguf+lora' if lora else 'merged_or_base_gguf', 'llama_cpp': 'runs/tools/llama.cpp-8ca4060 (CPU build)',
              'model_file': str(model_file), 'model_sha256': model_sha, 'lora_file': str(lora) if lora else None,
              'lora_sha256': a.lora_sha256, 'weights_bytes': model_bytes, 'arms': arms, 'ids': sorted(a.ids) if a.ids else None,
              'arm_switches': [],
              'binding': {'host': '127.0.0.1', 'port': a.port, 'threads': a.threads, 'context': a.context,
                          'max_new': a.max_new, 'batch': 256, 'ubatch': 128, 'no_repack': True, 'no_context_shift': True,
                          'n_gpu_layers': 0, 'decoding': GREEDY, 'template_kwargs': TEMPLATE_KWARGS, 'system_prompt': None,
                          'reasoning_format': 'auto (server default; official parser consumes the template prefill)',
                          'cache_prompt': False, 'history': 'own answers; markers and new sessions reset',
                          'context_note': '2048 matches the frozen evaluation generation contract (384 new tokens within '
                                          '2048); 4096 would only double KV memory, no suite prompt needs it'},
              'env': {k: os.environ.get(k) for k in ('LD_LIBRARY_PATH', 'LLAMA_ARG_DEVICE', 'LLAMA_ARG_N_GPU_LAYERS',
                                                     'OMP_NUM_THREADS', 'TRID_UI_SUPERVISOR_PID')},
              'suite_file': str(suite_path), 'suite_sha256': sha256(suite_path),
              'timeouts': {'ready': a.ready_timeout, 'request': request_timeout, 'request_derived': a.request_timeout is None,
                           'shutdown_grace': a.shutdown_grace, 'duration': a.duration, 'startup': startup, 'window_needed': need},
              'phases': [], 'requests': [], 'in_flight': None, 'stop_signal': None, 'memory': {}, 'status': 'validated'}
    lock = threading.RLock()  # on_stop may fire while the main thread is inside save()
    t0 = now()

    def save():
        with lock:
            tmp = receipt.with_suffix(receipt.suffix + '.tmp')
            tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1))
            tmp.chmod(0o600)
            tmp.replace(receipt)

    stop_requested = threading.Event()

    def on_stop(signum, _frame):
        # Persist immediately (the supervisor allows <= 2 s before SIGKILL); shutdown proceeds in main flow.
        record['stop_signal'] = {'signal': signal.Signals(signum).name, 'elapsed': round(now() - t0, 3),
                                 'in_flight': record['in_flight']}
        if record['status'] not in ('complete', 'failed'):
            record['status'] = 'stopped_by_signal'
        stop_requested.set()
        try:
            save()
        except Exception:
            pass
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_stop)
    signal.signal(signal.SIGINT, on_stop)
    save()
    record['linux_guard_ownership'] = native_guard()

    peak = {'group_rss_bytes': 0, 'server_rss_bytes': 0, 'samples': 0}

    def sampler():
        while not stop_requested.is_set():
            rows = members(pgid)
            with lock:
                peak['samples'] += 1
                peak['group_rss_bytes'] = max(peak['group_rss_bytes'], sum(r['rss_bytes'] for r in rows))
                peak['server_rss_bytes'] = max(peak['server_rss_bytes'], *(r['rss_bytes'] for r in rows if r['comm'].endswith('llama-server')), 0)
            stop_requested.wait(0.5)

    def phase(name):
        record['phases'].append({'phase': name, 'elapsed': round(now() - t0, 3), 'members': members(pgid),
                                 'pressure': pressure(), 'peak': dict(peak)})
        save()

    def deadline_left():
        return a.duration - SUPERVISOR_CLEANUP_SECONDS - a.shutdown_grace - startup - (now() - t0)

    base = f'http://127.0.0.1:{a.port}'

    def serve(messages, timeout):
        """Rendered prompt and its token ids, as base_dialogue_runner.run().serve does; both calls share the request bound."""
        prompt = request(base, '/apply-template', {'messages': messages, 'chat_template_kwargs': TEMPLATE_KWARGS},
                         timeout=timeout)['prompt']
        bos = record['props'].get('bos_token') or ''
        tokens = request(base, '/tokenize', {'content': prompt, 'add_special': not (bos and prompt.startswith(bos)),
                                             'parse_special': True}, timeout=timeout)['tokens']
        return prompt, tokens

    def timed(fn, spec):
        """Run one request; every outcome (success, HTTP error, timeout, stop signal) is appended before returning."""
        record['in_flight'] = {**spec, 'started': round(now() - t0, 3)}
        save()
        started = now()
        row = dict(spec)
        try:
            row.update(fn(min(request_timeout, max(1, deadline_left()))))
        except urllib.error.HTTPError as error:
            row.update(http_status=error.code, error=error.read()[:500].decode('utf-8', 'replace'))
        except OSError as error:  # socket.timeout and connection failures
            row.update(http_status=None, error=f'{type(error).__name__}: {error}', transport_failure=True)
        except RuntimeError as error:  # contract failure on a 200 response (render suffix, context fit)
            row.update(http_status=None, error=f'{type(error).__name__}: {error}', contract_failure=True)
        except BaseException as error:  # KeyboardInterrupt from on_stop lands here mid-request
            row.update(http_status=None, error=f'{type(error).__name__}: {error}', interrupted=True)
            row['wall_seconds'] = round(now() - started, 3)
            record['requests'].append(row)
            record['in_flight'] = None
            save()
            raise
        row['wall_seconds'] = round(now() - started, 3)
        row['seconds'] = row['wall_seconds']
        row['pressure'] = pressure()
        row['group_rss_bytes'] = sum(r['rss_bytes'] for r in members(pgid))
        record['requests'].append(row)
        record['in_flight'] = None
        save()
        return row

    def set_arm(arm):
        """Adapter on/off inside one process: POST /lora-adapters sets the scale; GET confirms it."""
        if not lora:
            return
        scale = 1.0 if arm == 'adapter' else 0.0
        request(base, '/lora-adapters', [{'id': 0, 'scale': scale}], timeout=10)
        state = request(base, '/lora-adapters', timeout=5)
        if not (len(state) == 1 and state[0].get('scale') == scale):
            raise RuntimeError(f'adapter scale not {scale}: {state}')
        record['arm_switches'].append({'arm': arm, 'scale': scale, 'elapsed': round(now() - t0, 3)})

    def native(messages, spec):
        def go(timeout):
            prompt, tokens = serve(messages, timeout)
            if not prompt.endswith(ASSISTANT_NO_THINK):
                raise RuntimeError('served prompt does not end with the no-thinking assistant header')
            fits = len(tokens) + a.max_new <= a.context
            if not fits:
                raise RuntimeError(f'prompt {len(tokens)} + {a.max_new} new tokens exceeds context {a.context}')
            c = request(base, '/completion', dict(prompt=tokens, n_predict=a.max_new, cache_prompt=False,
                                                  return_tokens=True, **GREEDY), timeout=timeout)
            text = c['content'].strip()
            return {'http_status': 200, 'prompt': prompt, 'rendered_prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                    'prompt_token_ids': tokens, 'prompt_tokens': len(tokens),
                    'prompt_token_ids_sha256': hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                    'fits_context': fits, 'output': text, 'output_token_ids': c.get('tokens'),
                    'generated_tokens': c.get('tokens_predicted'), 'stop_type': c.get('stop_type'),
                    'stopped_at_eos': c.get('stop_type') == 'eos',
                    'truncated': c.get('stop_type') == 'limit' or bool(c.get('truncated')),
                    'stopping_word': c.get('stopping_word'), 'timings': c.get('timings'),
                    'generation_settings': {k: v for k, v in (c.get('generation_settings') or {}).items()
                                            if k not in ('n_predict', 'seed')},
                    'surface': surface(spec, text)}
        return timed(go, spec)

    def oai(messages, spec, native_text):
        """The OAI surface returns message.content after the server's chat parser (see server_argv on reasoning_format).
        raw_content_equal is the deployment-surface measurement (content as a client receives it == native generated
        text); no_think_prefill_echoed and generated_text_equal remain as diagnostics so a regression to the echo
        (base-serving-smoke under 'none') is visible rather than masked."""
        def go(timeout):
            c = request(base, '/v1/chat/completions', {'messages': messages, 'max_tokens': a.max_new, 'cache_prompt': False,
                                                       'chat_template_kwargs': TEMPLATE_KWARGS, **GREEDY}, timeout=timeout)
            choice = c['choices'][0]
            raw = choice['message'].get('content') or ''
            echoed = raw.startswith(QWEN3_NO_THINK)
            generated = (raw[len(QWEN3_NO_THINK):] if echoed else raw).strip()
            text = raw.strip()
            return {'http_status': 200, 'output': text, 'finish_reason': choice.get('finish_reason'),
                    'truncated': choice.get('finish_reason') == 'length', 'usage': c.get('usage'),
                    'reasoning_content': choice['message'].get('reasoning_content'),
                    'timings': c.get('timings'), 'no_think_prefill_echoed': echoed, 'raw_content_equal': text == native_text,
                    'generated_text_equal': generated == native_text, 'surface': surface(spec, generated)}
        return timed(go, spec)

    server_log.parent.mkdir(parents=True, exist_ok=True)
    log = open(server_log, 'wb')
    child = subprocess.Popen(argv, stdout=log, stderr=log, cwd=str(REPO))  # same session and pgid
    record['server_pid'] = child.pid
    record['status'] = 'running'
    threading.Thread(target=sampler, daemon=True).start()
    exit_code = None
    try:
        phase('spawned')
        ready = False
        while now() - t0 < a.ready_timeout and not stop_requested.is_set():
            if child.poll() is not None:
                break
            try:
                props = request(base, '/props', timeout=2)
                if Path(props['model_path']).resolve() == model_file:
                    record['props'] = props
                    ready = True
                    break
            except OSError:
                pass
            time.sleep(0.5)
        record['ready_seconds'] = round(now() - t0, 3)
        phase('ready' if ready else 'not_ready')
        if not ready:
            raise RuntimeError('llama-server not ready within timeout or exited')
        props = record['props']
        template = props.get('chat_template') or ''
        record['server'] = {'build': props.get('build_info'), 'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
                            'chat_template': template, 'default_params': (props.get('default_generation_settings') or {}).get('params'),
                            'eos_token': props.get('eos_token'), 'bos_token': props.get('bos_token'),
                            'health': request(base, '/health', timeout=5)}
        if lora:
            adapters = request(base, '/lora-adapters', timeout=5)
            record['server']['lora_adapters'] = adapters
            if not any(Path(x.get('path', '')).resolve() == lora and x.get('scale') == 1.0 for x in adapters):
                raise RuntimeError(f'adapter not loaded at scale 1.0: {adapters}')
        sample = [{'role': 'user', 'content': '안녕하세요'}, {'role': 'assistant', 'content': '안녕하세요.'},
                  {'role': 'user', 'content': '고마워요'}]
        if a.official_render:
            official = json.loads(a.official_render.read_text())
            sample = official['messages']
        served, served_ids = serve(sample, min(request_timeout, max(1, deadline_left())))
        parity = {'sample_messages': sample, 'served_prompt': served, 'served_token_ids': served_ids,
                  'no_think_suffix': served.endswith(ASSISTANT_NO_THINK)}
        if a.official_render:
            parity.update(official_render=str(a.official_render), text_equal=served == official['prompt'],
                          token_ids_equal=served_ids == official['token_ids'])
        else:
            parity.update(official_render=None, text_equal=None, token_ids_equal=None,
                          note='unconfirmed: no official render supplied; suffix check only')
        record['render_parity'] = parity
        save()
        if not parity['no_think_suffix'] or parity['token_ids_equal'] is False or parity['text_equal'] is False:
            raise RuntimeError('render parity failed; no requests made')
        histories = {}
        outputs = {}
        first_ids = {}
        for arm in arms:
            set_arm(arm)
            for spec in specs:
                if stop_requested.is_set():
                    raise KeyboardInterrupt
                key = (arm, spec['session'])
                history = [] if spec['reset'] else histories.get(key, [])
                messages = history + [{'role': 'user', 'content': spec['user']}]
                row = native(messages, {**spec, 'arm': arm, 'kind': 'native', 'history_turns': len(history) // 2})
                if row['http_status'] != 200:
                    raise RuntimeError(f'{arm}/{spec["session"]}[{spec["turn"]}] failed: {row.get("http_status")} {row.get("error")}')
                histories[key] = messages + [{'role': 'assistant', 'content': row['output']}]
                outputs.setdefault(key, row['output'])
                if not history:
                    first_ids.setdefault(spec['session'], {})[arm] = row['prompt_token_ids_sha256']
                record['health_after'] = request(base, '/health', timeout=5)
        record['cross_arm_first_prompt_parity'] = {
            s: {'token_ids_sha256': v, 'evaluated': len(arms) >= 2, 'equal': len(set(v.values())) == 1 if len(arms) >= 2 else None}
            for s, v in first_ids.items()}  # one arm per child: parity is the parent's cross-receipt comparison
        if parity_session:
            first = next(s for s in specs if s['session'] == parity_session)
            row = oai([{'role': 'user', 'content': first['user']}],
                      {'session': parity_session, 'group': first['group'], 'turn': first['turn'], 'user': first['user'],
                       'arm': arms[-1], 'kind': 'oai', 'reset': True, 'surface': first['surface']},
                      outputs[(arms[-1], parity_session)])
            if row['http_status'] != 200:
                raise RuntimeError(f'oai request failed: {row.get("http_status")} {row.get("error")}')
        record['status'] = 'requests_complete'
    except KeyboardInterrupt:
        record['error'] = {'type': 'StopSignal', 'message': json.dumps(record['stop_signal'])}
        record['status'] = 'stopped_by_signal'
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        record['status'] = 'failed'
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # a second signal must not interrupt the shutdown bookkeeping
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        save()
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
            try:
                exit_code = child.wait(timeout=a.shutdown_grace)
            except subprocess.TimeoutExpired:
                child.kill()
                exit_code = child.wait()
                record['server_killed'] = True
        else:
            exit_code = child.returncode
        record['server_exit_code'] = exit_code
        stop_requested.set()
        time.sleep(0.5)
        live = [m for m in members(pgid) if m['pid'] != os.getpid() and not m['stat'].startswith('Z')]
        record['live_members_after_shutdown'] = live
        record['memory'] = {**peak, 'ceiling_bytes': 6 * 1024**3, 'under_ceiling': peak['group_rss_bytes'] <= 6 * 1024**3}
        phase('after_shutdown')
        reqs = record['requests']
        record['summary'] = {'requests': len(reqs), 'http_200': sum(r.get('http_status') == 200 for r in reqs),
                             'truncated': sum(bool(r.get('truncated')) for r in reqs),
                             'surface_all_true': sum(bool((r.get('surface') or {}).get('all_true')) for r in reqs),
                             'oai_generated_text_equal': next((r['generated_text_equal'] for r in reqs if r.get('kind') == 'oai'), None),
                             'oai_no_think_prefill_echoed': next((r['no_think_prefill_echoed'] for r in reqs if r.get('kind') == 'oai'), None),
                             'wall_seconds_max': max((r.get('wall_seconds', 0) for r in reqs), default=None),
                             'predicted_per_second_min': min((r['timings']['predicted_per_second'] for r in reqs
                                                              if r.get('timings')), default=None)}
        ok = (record['status'] == 'requests_complete' and not live and reqs
              and all(r.get('http_status') == 200 for r in reqs) and record['memory']['under_ceiling'])
        if ok:
            record['status'] = 'complete'
        elif record['status'] == 'requests_complete':
            record['status'] = 'incomplete'
        record['elapsed_seconds'] = round(now() - t0, 3)
        save()
        log.close()
    sys.exit(0 if record['status'] == 'complete' else 1)


if __name__ == '__main__':
    main()
