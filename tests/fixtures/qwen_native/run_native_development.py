"""Native development controller for recovery-003 (Q8 backend amendment, dev-selection-rule-006-q8).

One explicit arm, one explicit checkpoint, a finite list of frozen DEV36 scenario ids. Every scenario runs in its own
bounded child: qwen-serving-guard.py (supervisor006: 180 s absolute deadline, pressure 1 gate, sampled 6 GiB group
RSS stop, no retry) -> qwen-serving-harness.py (CPU llama.cpp 8ca4060, exact Q8 base GGUF, optional converted LoRA
GGUF via --lora, own history, greedy seed 42, 384 new tokens within 2048, official-render parity, no RAG).

Arms (harness naming):
  model    baseline: the Q8 base alone, no --lora at all. Generated ONCE per id and reused across every checkpoint.
  adapter  candidate: the same Q8 base + the official converter's adapter.f32.gguf of ONE checkpoint (scale 1.0).

Sequential Popen+wait, never two model children, never a retry, never an overwritten output (--out must not exist).
Each child keeps its supervisor receipt, harness receipt (every turn: rendered prompt, token ids + hashes, output,
output token ids, stop type, timings, generation settings, pressure, RSS), llama-server log and guard/harness log.
The first unclean or non-binding child stops the run; finished cases are retained, remaining ids are listed, nothing
is deleted. SIGTERM/SIGINT are forwarded to the running guard so its supervisor cleans the owned group.

Cross-receipt parity (rule 006-q8 / contract 004-q8): with --baseline <model-arm out dir> each adapter-arm child is
compared to the baseline child of the same id: identical model_sha256, chat_template_sha256, server build, binding,
render parity, generation settings, per-turn user strings, and turn-0 rendered_prompt_sha256 + prompt_token_ids_sha256.
The adapter identity necessarily differs (baseline has none) and is never part of parity. A parity mismatch is
INVALID_NO_RETRY. --parity-only recomputes the comparison for a finished adapter run without any model child.

This controller reads only scenario ids / groups / user strings (the frozen generation inputs); it never opens dev
rubrics, dev gold rows or any final24 file, and it judges no output.

  .venv/bin/python -B run_native_development.py --arm model --group domain --out <fresh dir>
  .venv/bin/python -B run_native_development.py --arm adapter --checkpoint <train-out>/checkpoints/step-000023 \\
      --adapter-gguf <conversion out-dir>/adapter.f32.gguf --baseline <model-arm out dir> --group domain --out <fresh dir>
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

ROOT = Path('/private/tmp/chat-root-c-005')
HERE = ROOT / 'qwen-recovery-003'
REPO = Path('/Users/eomgyuhyeon/사업/trid')
PYTHON = REPO / '.venv/bin/python'
GUARD = HERE / 'qwen-serving-guard.py'
HARNESS = HERE / 'qwen-serving-harness.py'
SUPERVISOR = ROOT / 'institution-native-ranking-V/development_supervisor006.py'
DIALOGUE_RUNNER = REPO / 'scripts/base_dialogue_runner.py'  # imported by the harness (request/sha256/markers)
LLAMA_SERVER_DIR = REPO / 'runs/tools/llama.cpp-8ca4060'
LLAMA_SERVER = LLAMA_SERVER_DIR / 'llama-server'
BASE_GGUF = REPO / 'runs/model-repair-002-qwen3-1p7b/qwen3-1.7b-base-resaved.q8_0.gguf'
BASE_GGUF_SHA256 = '29508ae00d82764d261287099ea3077dd08dcbf5cb86fe14427e9407b913c3c7'
BASE_REVISION = '70d244cc86ccca08cf5af4e1e306ecf908b1ad5e'
SCENARIOS = REPO / 'runs/qwen-specialization-003/development/scenarios.json'  # frozen DEV36 generation inputs
OWNER_SCENARIOS = {  # the data-owner files the rule pins; SCENARIOS must be exactly their union (id, group, user text)
    REPO / 'data/qwen-specialization-003/domain-dev-scenarios-003.json': '13fa5a06f264678ee95611467b289f18bab81a5a15c07972ddd85af32824d4c8',
    REPO / 'data/qwen-specialization-003/behavior-dev-scenarios.json': 'bc613942b59dd6ec6ace8bfe865012da2f591161bd52ae9bec1da1d3b61f47d7',
}
MANIFEST = REPO / 'data/qwen-specialization-003/train-manifest-003.json'
RULE = HERE / 'evaluation/dev-selection-q8-006.json'
RULE_ID = 'dev-selection-rule-006-q8'
GENERATION_CONTRACT = HERE / 'evaluation/final24-generation-q8-004.json'
OFFICIAL_RENDER = HERE / 'official-render.json'  # HF-tokenizer render of a generic sample; harness aborts on mismatch
GROUPS = ('domain', 'uncertainty', 'retention')
GROUP_SIZE = 12
ARMS = ('model', 'adapter')
CHILD_SECONDS = 180  # contract.json runtime_authority.evaluation_child_seconds_max
READY_TIMEOUT = 30  # harness default is 60; base readiness measured 0.566 s, the margin goes to the per-turn budget
MAX_NEW = 384
CONTEXT = 2048
THREADS = 4
LIVE_DEMO_PORTS = {51873, 51912, 51917, 51918, 51990}  # same exclusion as the harness
PARENT_TOOL_TIMEOUT = 2700  # a child is started only if CHILD_SECONDS + 30 s cleanup still fit

# Frozen inputs and the exact reviewed source this controller was written against (sha256).
PINS = {
    SCENARIOS: 'a501be86d09bf01e614d8454775c8568e30a1c3b83dfcf99a25866f99f68ea52',
    MANIFEST: '74270df1620057b4a95505173c7d2e0abc693501ec4b59f38aee712b34407979',
    RULE: 'f61e4661c5940eb9b0a47ab7a1b7f6212980abab2eb04ae5a5cab92eecca611c',
    GENERATION_CONTRACT: '6c2ab376f064a4b21f6192276b5ec9704d773d227143a540166ee738084019d5',
    OFFICIAL_RENDER: '7d3fa3cb7e04e10087cbfb94d1066213e01ac766f8def8dfa449761a945480e1',
    HARNESS: '9b07f62effa358ff059601849e01ab6be0dfb21541d878cc3b1a58ee5c1154a8',
    GUARD: 'e9598f0e0ac250924e629812fd408155dd9caa65ddb266f360a49620eec1d129',
    SUPERVISOR: '0f31777b5f66199f56a91a73857440d6c316c1cf71148aef5be432fdd5e28070',
    DIALOGUE_RUNNER: 'e1a35e447565290fe49368a9b4ab0341ee3c0d6afcb2b5d1d6081772b33b7fda',
    LLAMA_SERVER: 'f7b5362cd81fdf30cf1bc4e7e0976452b3a2abcf2436b7418119717534509bdc',
    **OWNER_SCENARIOS,
}
# Harness receipt binding the contract fixes (final24-generation-q8-004.json#final_quality_backend.preferred.binding).
EXPECTED_BINDING = {'host': '127.0.0.1', 'threads': THREADS, 'context': CONTEXT, 'max_new': MAX_NEW, 'batch': 256,
                    'ubatch': 128, 'no_repack': True, 'no_context_shift': True, 'n_gpu_layers': 0,
                    'decoding': {'temperature': 0, 'seed': 42}, 'template_kwargs': {'enable_thinking': False},
                    'system_prompt': None, 'cache_prompt': False}
# Receipt fields that must be identical between a baseline child and an adapter child of the same id.
SHARED_IDENTITY = ('model_file', 'model_sha256', 'llama_cpp', 'suite_sha256')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b''):
            digest.update(chunk)
    return digest.hexdigest()


def safe_name(index, sid):
    return f'{index:02d}-' + re.sub(r'[^A-Za-z0-9._-]+', '_', sid)


def check_pins():
    """Every frozen input and every reviewed executable this controller depends on must still be byte-identical."""
    pins = {}
    for path, expected in PINS.items():
        if not path.is_file():
            raise ValueError(f'Pinned file missing: {path}')
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f'Frozen input or reviewed source changed: {path} ({actual})')
        pins[str(path)] = actual
    rule = json.loads(RULE.read_text())
    if rule['id'] != RULE_ID:
        raise ValueError(f'Selection rule id {rule["id"]!r} is not {RULE_ID}')
    if not BASE_GGUF.is_file():
        raise ValueError(f'Serving Q8 base missing: {BASE_GGUF}')
    actual = sha256(BASE_GGUF)
    if actual != BASE_GGUF_SHA256:
        raise ValueError(f'Serving Q8 base changed: {actual}')
    pins[str(BASE_GGUF)] = actual
    if not PYTHON.is_file():
        raise ValueError(f'Interpreter missing: {PYTHON}')
    return pins


def load_scenarios():
    """Frozen DEV36 generation inputs: id, group and user strings only. The combined file must be exactly the union of
    the data-owner files the rule pins (same ids, groups and user text), so both pin sets describe one input."""
    rows = json.loads(SCENARIOS.read_text())['results']
    if len(rows) != len(GROUPS) * GROUP_SIZE:
        raise ValueError(f'Expected {len(GROUPS) * GROUP_SIZE} frozen dev scenarios, found {len(rows)}')
    counts = {g: sum(r.get('group') == g for r in rows) for g in GROUPS}
    if any(counts[g] != GROUP_SIZE for g in GROUPS) or len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Dev groups are not {GROUP_SIZE} unique ids each: {counts}')
    owner = {}
    for path in OWNER_SCENARIOS:
        for r in json.loads(path.read_text())['results']:
            owner[r['id']] = r
    def users(r):
        if ('prompt' in r) == ('turns' in r):
            raise ValueError(f'{r["id"]}: exactly one of prompt / turns')
        return [r['prompt']] if 'prompt' in r else list(r['turns'])
    scenarios = {}
    for r in rows:
        o = owner.get(r['id'])
        if o is None or o.get('group') != r['group'] or users(o) != users(r):
            raise ValueError(f'{r["id"]}: combined dev file differs from the data-owner file')
        scenarios[r['id']] = {'group': r['group'], 'users': users(r), 'order': len(scenarios)}
    if set(owner) != set(scenarios):
        raise ValueError('Data-owner dev files hold ids the combined file lacks')
    return scenarios


def select_ids(scenarios, group, ids):
    if group:
        chosen = [sid for sid, s in scenarios.items() if s['group'] == group]
    else:
        chosen = []
        for sid in ids:
            if sid not in scenarios:
                raise ValueError(f'Unknown dev scenario id: {sid}')
            if sid in chosen:
                raise ValueError(f'Duplicate id: {sid}')
            chosen.append(sid)
    return sorted(chosen, key=lambda sid: scenarios[sid]['order'])


def bind_checkpoint(checkpoint, adapter_gguf):
    """The adapter must be a saved checkpoint of the approved run and the GGUF its official conversion: the step dir's
    training.json records this exact safetensors hash, the original base revision, the admitted TRAIN manifest and the
    serving Q8 base it was trained on; the converter receipt next to the GGUF records the same safetensors +
    training.json hashes, the serving Q8 base sha and the GGUF hash observed on disk."""
    checkpoint = checkpoint.resolve()
    adapter = checkpoint / 'adapter'
    training_json = checkpoint / 'training.json'
    for path in (adapter / 'adapter_config.json', adapter / 'adapter_model.safetensors', training_json):
        if not path.is_file():
            raise ValueError(f'Checkpoint lacks {path}')
    trained = json.loads(training_json.read_text())
    weights = sha256(adapter / 'adapter_model.safetensors')
    record = trained.get('checkpoint') or {}
    problems = []
    if trained.get('status') != 'checkpoint':
        problems.append(f'status={trained.get("status")!r} (expected checkpoint)')
    if record.get('adapter_weights_sha256') != weights:
        problems.append('checkpoint.adapter_weights_sha256 differs from adapter_model.safetensors on disk')
    if record.get('step') is not None and record.get('step') != trained.get('steps'):
        problems.append('checkpoint.step differs from steps')
    if not isinstance(trained.get('steps'), int) or trained['steps'] <= 0:
        problems.append(f'steps={trained.get("steps")!r}')
    if record.get('path') and Path(record['path']).resolve() != adapter:
        problems.append(f'checkpoint.path {record["path"]!r} is not {adapter}')
    if trained.get('base_revision') != BASE_REVISION:
        problems.append(f'base_revision={trained.get("base_revision")!r}')
    if trained.get('manifest_sha256') != PINS[MANIFEST]:
        problems.append('manifest_sha256 is not the admitted TRAIN manifest')
    base_weights = trained.get('base_weights') or {}
    if base_weights.get('representation') != 'q8_0' or base_weights.get('gguf_sha256') != BASE_GGUF_SHA256:
        problems.append(f'training.json base_weights {base_weights.get("representation")!r}/{base_weights.get("gguf_sha256")!r} '
                        'is not the frozen serving Q8 base')
    if problems:
        raise ValueError('Checkpoint binding failed: ' + '; '.join(problems))

    adapter_gguf = adapter_gguf.resolve()
    receipt_path = adapter_gguf.parent / 'adapter-convert-receipt.json'
    if not adapter_gguf.is_file() or not receipt_path.is_file():
        raise ValueError(f'Converted adapter or its adapter-convert-receipt.json missing next to {adapter_gguf}')
    receipt = json.loads(receipt_path.read_text())
    gguf_sha = sha256(adapter_gguf)
    out = receipt.get('outputs', {}).get('adapter_gguf') or {}
    inputs = receipt.get('inputs') or {}
    files = inputs.get('files') or {}
    conv = []
    if receipt.get('status') != 'complete':
        conv.append(f'receipt status={receipt.get("status")!r}')
    if out.get('sha256') != gguf_sha:
        conv.append('receipt outputs.adapter_gguf.sha256 differs from the GGUF on disk')
    if Path(out.get('path', '')).resolve() != adapter_gguf:
        conv.append('receipt outputs.adapter_gguf.path is not the given GGUF')
    if Path(inputs.get('adapter_dir', '')).resolve() != adapter:
        conv.append('receipt inputs.adapter_dir is not the checkpoint adapter')
    if (files.get('adapter_model.safetensors') or {}).get('sha256') != weights:
        conv.append('receipt safetensors sha differs from the checkpoint adapter')
    if (inputs.get('training_json') or {}).get('sha256') != sha256(training_json):
        conv.append('receipt was not produced against this checkpoint training.json')
    if (inputs.get('base') or {}).get('serving_base_gguf_sha256_expected') != BASE_GGUF_SHA256:
        conv.append('receipt serving base GGUF sha is not the serving Q8 base')
    if (inputs.get('base') or {}).get('revision') != BASE_REVISION:
        conv.append('receipt base revision mismatch')
    if conv:
        raise ValueError('Converted adapter binding failed: ' + '; '.join(conv))
    return dict(checkpoint=str(checkpoint), adapter=str(adapter), step=trained['steps'], planned_steps=trained.get('planned_steps'),
                training_json=str(training_json), training_json_sha256=sha256(training_json),
                adapter_config_sha256=sha256(adapter / 'adapter_config.json'), adapter_weights_sha256=weights,
                base_model_id=trained.get('base_model_id'), base_revision=trained['base_revision'],
                base_weights=base_weights,
                manifest_id=trained.get('manifest_id'), manifest_sha256=trained['manifest_sha256'],
                lora=trained.get('lora'), adapter_gguf=str(adapter_gguf), adapter_gguf_sha256=gguf_sha,
                conversion_receipt=str(receipt_path), conversion_receipt_sha256=sha256(receipt_path),
                converter_entry=(receipt.get('converter') or {}).get('entry') or receipt.get('subprocess_argv'))


def free_port():
    """A currently free loopback port outside the live-demo set; the harness re-checks bindability before spawning."""
    while True:
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0))
            port = s.getsockname()[1]
        if port not in LIVE_DEMO_PORTS:
            return port


def verify_child(case, supervisor_receipt, harness_receipt, pins, binding, arm, sid, users):
    """Both receipts must say the bounded child finished cleanly on the pinned inputs with exactly this scenario in
    exactly this arm, every user turn answered. Facts are recorded; nothing about quality is judged."""
    sup = json.loads(supervisor_receipt.read_text())
    case['receipt_summary'] = {k: sup.get(k) for k in (
        'harness_exit_code', 'stop_reason', 'first_soft_stop_reason', 'reaped', 'cleanup_state', 'cleanup_errors',
        'hard_alarm_fired', 'elapsed_seconds')}
    clean = (sup.get('harness_exit_code') == 0 and sup.get('reaped') is True and sup.get('cleanup_state') == 'completed'
             and not sup.get('cleanup_errors') and sup.get('stop_reason') == 'harness_exit'
             and sup.get('first_soft_stop_reason') is None and sup.get('hard_alarm_fired') is False)
    if not clean:
        raise RuntimeError('Supervisor receipt is not clean')
    r = json.loads(harness_receipt.read_text())
    mismatch = []
    expect = {'status': 'complete', 'model_file': str(BASE_GGUF), 'model_sha256': BASE_GGUF_SHA256,
              'arms': [arm], 'ids': [sid], 'suite_file': str(SCENARIOS), 'suite_sha256': pins[str(SCENARIOS)],
              'lora_file': binding['adapter_gguf'] if binding else None,
              'lora_sha256': binding['adapter_gguf_sha256'] if binding else None,
              'backend': 'base_gguf+lora' if binding else 'merged_or_base_gguf'}
    for k, v in expect.items():
        if r.get(k) != v:
            mismatch.append(f'{k}={r.get(k)!r}')
    binding_seen = r.get('binding') or {}
    for k, v in EXPECTED_BINDING.items():
        if binding_seen.get(k) != v:
            mismatch.append(f'binding.{k}={binding_seen.get(k)!r}')
    if (r.get('env') or {}).get('DYLD_LIBRARY_PATH') != str(LLAMA_SERVER_DIR):
        mismatch.append('env.DYLD_LIBRARY_PATH')
    if (r.get('timeouts') or {}).get('duration') != CHILD_SECONDS:
        mismatch.append('timeouts.duration')
    parity = r.get('render_parity') or {}
    if not (parity.get('official_render') and Path(parity['official_render']).resolve() == OFFICIAL_RENDER
            and parity.get('text_equal') is True and parity.get('token_ids_equal') is True and parity.get('no_think_suffix') is True):
        mismatch.append('render_parity')
    server = r.get('server') or {}
    if not server.get('build') or not server.get('chat_template_sha256'):
        mismatch.append('server identity')
    if binding:
        adapters = server.get('lora_adapters') or []
        if not any(Path(x.get('path', '')).resolve() == Path(binding['adapter_gguf']) and x.get('scale') == 1.0 for x in adapters):
            mismatch.append('server.lora_adapters')
        if [s.get('arm') for s in r.get('arm_switches') or []] != ['adapter']:
            mismatch.append('arm_switches')
    elif server.get('lora_adapters'):
        mismatch.append('baseline child reports a loaded adapter')
    if r.get('live_members_after_shutdown') or (r.get('memory') or {}).get('under_ceiling') is not True:
        mismatch.append('shutdown/memory')
    if mismatch:
        raise RuntimeError('Harness receipt does not bind to the pinned run: ' + ', '.join(mismatch))
    rows = r.get('requests') or []
    if len(rows) != len(users):
        raise RuntimeError(f'{len(rows)} requests recorded for {len(users)} user turns')
    settings = None
    for index, (row, user) in enumerate(zip(rows, users)):
        if (row.get('kind') != 'native' or row.get('arm') != arm or row.get('session') != sid or row.get('turn') != index
                or row.get('user') != user or row.get('http_status') != 200 or row.get('fits_context') is not True
                or row.get('history_turns') != index or row.get('reset') is not False
                or not row.get('rendered_prompt_sha256') or not row.get('prompt_token_ids_sha256')
                or not isinstance(row.get('output'), str)):
            raise RuntimeError(f'request {index} is not the expected own-history native turn for {sid}/{arm}')
        settings = settings or row.get('generation_settings')
        if row.get('generation_settings') != settings:
            raise RuntimeError('generation settings changed between turns')
    first = rows[0]
    case.update(
        turns_total=len(users), turns_completed=len(rows), seconds=round(sum(x.get('wall_seconds', 0) for x in rows), 3),
        elapsed_seconds=r.get('elapsed_seconds'), startup_seconds=(r.get('timeouts') or {}).get('startup'),
        request_timeout=(r.get('timeouts') or {}).get('request'),
        server_build=server.get('build'), chat_template_sha256=server.get('chat_template_sha256'),
        generation_settings_sha256=hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest(),
        turn0_rendered_prompt_sha256=first['rendered_prompt_sha256'], turn0_prompt_token_ids_sha256=first['prompt_token_ids_sha256'],
        prompt_tokens=[x.get('prompt_tokens') for x in rows], generated_tokens=[x.get('generated_tokens') for x in rows],
        stop_type=[x.get('stop_type') for x in rows], stopped_at_eos=all(x.get('stopped_at_eos') for x in rows),
        truncated=any(bool(x.get('truncated')) for x in rows), fits_context=all(x.get('fits_context') for x in rows),
        empty_output=any(not x['output'].strip() for x in rows),
        peak_group_rss_bytes=(r.get('memory') or {}).get('group_rss_bytes'), pressure=[x.get('pressure') for x in rows],
        harness_receipt_sha256=sha256(harness_receipt), supervisor_receipt_sha256=sha256(supervisor_receipt))


def compare_with_baseline(candidate_receipt, baseline_receipt, users):
    """Cross-receipt parity for one id: same base/template/server/settings and turn-0 render + token hashes; later turns
    carry the same user strings (own histories legitimately differ). Adapter identity is excluded by design."""
    c = json.loads(candidate_receipt.read_text())
    b = json.loads(baseline_receipt.read_text())
    cr, br = c.get('requests') or [], b.get('requests') or []
    checks = {k: c.get(k) == b.get(k) for k in SHARED_IDENTITY}
    # Each child owns a fresh loopback port; compare generation settings, not that transport address.
    checks['binding_equal'] = all(
        (c.get('binding') or {}).get(k) == (b.get('binding') or {}).get(k) for k in EXPECTED_BINDING
    )
    checks['server_build_equal'] = (c.get('server') or {}).get('build') == (b.get('server') or {}).get('build')
    checks['chat_template_sha256_equal'] = (c.get('server') or {}).get('chat_template_sha256') == (b.get('server') or {}).get('chat_template_sha256')
    checks['render_parity_equal'] = all((c.get('render_parity') or {}).get(k) == (b.get('render_parity') or {}).get(k)
                                        for k in ('served_prompt', 'served_token_ids', 'text_equal', 'token_ids_equal'))
    checks['baseline_arm_model_without_lora'] = b.get('arms') == ['model'] and b.get('lora_file') is None
    checks['candidate_arm_adapter'] = c.get('arms') == ['adapter'] and bool(c.get('lora_sha256'))
    checks['turn_count_equal'] = len(cr) == len(br) == len(users)
    checks['user_strings_equal'] = [x.get('user') for x in cr] == [x.get('user') for x in br] == list(users)
    # Adapter activation is the intended arm difference; its file and scale are checked by verify_child.
    checks['generation_settings_equal'] = bool(cr and br) and all(
        {k: v for k, v in (x.get('generation_settings') or {}).items() if k != 'lora'}
        == {k: v for k, v in (y.get('generation_settings') or {}).items() if k != 'lora'}
        for x, y in zip(cr, br)
    )
    checks['turn0_rendered_prompt_sha256_equal'] = bool(cr and br) and cr[0].get('rendered_prompt_sha256') == br[0].get('rendered_prompt_sha256')
    checks['turn0_prompt_token_ids_sha256_equal'] = bool(cr and br) and cr[0].get('prompt_token_ids_sha256') == br[0].get('prompt_token_ids_sha256')
    return {'equal': all(checks.values()), 'checks': checks, 'baseline_receipt': str(baseline_receipt),
            'baseline_receipt_sha256': sha256(baseline_receipt), 'candidate_receipt_sha256': sha256(candidate_receipt),
            'baseline_model_sha256': b.get('model_sha256'), 'candidate_lora_sha256': c.get('lora_sha256')}


def load_baseline(baseline_dir, ids):
    """A finished model-arm execution of this controller; every requested id must be among its completed cases."""
    execution = json.loads((baseline_dir / 'execution.json').read_text())
    if execution.get('arm') != 'model':
        raise ValueError('--baseline is not a model-arm run')
    cases = {c['id']: c for c in execution.get('cases', []) if c.get('verified') is True}
    missing = [sid for sid in ids if sid not in cases]
    if missing:
        raise ValueError(f'--baseline has no verified case for: {missing}')
    for sid in ids:
        if not Path(cases[sid]['receipt']).is_file():
            raise ValueError(f'--baseline receipt missing: {cases[sid]["receipt"]}')
    return {'dir': str(baseline_dir), 'execution': str(baseline_dir / 'execution.json'),
            'execution_sha256': sha256(baseline_dir / 'execution.json'), 'state': execution.get('state'),
            'receipts': {sid: cases[sid]['receipt'] for sid in ids}}


def parity_only(args, scenarios):
    out = args.out.resolve()
    execution = json.loads((out / 'execution.json').read_text())
    if execution.get('arm') != 'adapter':
        raise ValueError('--parity-only needs an adapter-arm run in --out')
    cases = [c for c in execution.get('cases', []) if c.get('verified') is True]
    ids = [c['id'] for c in cases]
    baseline = load_baseline(args.baseline.resolve(), ids)
    report = out / f'parity-vs-{args.baseline.resolve().name}.json'
    result = {sid: compare_with_baseline(Path(c['receipt']), Path(baseline['receipts'][sid]), scenarios[sid]['users'])
              for sid, c in zip(ids, cases)}
    payload = {'candidate_execution': str(out / 'execution.json'), 'baseline': baseline, 'ids': ids,
               'all_equal': all(v['equal'] for v in result.values()), 'parity': result, 'written_unix': time.time()}
    with report.open('x') as f:  # never overwrites an earlier comparison
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({'parity': str(report), 'all_equal': payload['all_equal'], 'ids': len(ids)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arm', choices=ARMS, required=True, help='model = Q8 base alone (baseline); adapter = base + converted LoRA')
    parser.add_argument('--checkpoint', type=Path, help='<train-out>/checkpoints/step-NNNNNN (holds adapter/ and training.json); adapter arm only')
    parser.add_argument('--adapter-gguf', type=Path, help='<conversion out-dir>/adapter.f32.gguf with adapter-convert-receipt.json beside it; adapter arm only')
    parser.add_argument('--baseline', type=Path, help='finished --arm model --out dir; adapter arm: per-id cross-receipt parity')
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--group', choices=GROUPS, help='all 12 frozen ids of one dev group')
    selection.add_argument('--ids', nargs='+', help='explicit dev scenario ids (finite parent batch)')
    parser.add_argument('--out', type=Path, required=True, help='fresh directory; never reused')
    parser.add_argument('--parity-only', action='store_true', help='no model child: compare a finished adapter run in --out with --baseline')
    args = parser.parse_args()
    if args.arm == 'adapter' and not args.parity_only and not (args.checkpoint and args.adapter_gguf):
        parser.error('--arm adapter needs --checkpoint and --adapter-gguf')
    if args.arm == 'model' and (args.checkpoint or args.adapter_gguf or args.baseline):
        parser.error('--arm model is the adapter-free baseline: no --checkpoint, --adapter-gguf or --baseline')
    if args.parity_only and not (args.arm == 'adapter' and args.baseline and args.out.is_dir()):
        parser.error('--parity-only needs --arm adapter, --baseline and an existing adapter-run --out')
    if not args.parity_only and args.out.exists():
        parser.error('--out must be a fresh directory; outputs are never replaced')
    pins = check_pins()
    scenarios = load_scenarios()
    if args.parity_only:
        parity_only(args, scenarios)
        return
    binding = bind_checkpoint(args.checkpoint, args.adapter_gguf) if args.arm == 'adapter' else None
    ids = select_ids(scenarios, args.group, args.ids)
    for sid in ids:
        if not re.fullmatch(r'[^\s,]+', sid):
            raise ValueError(f'Scenario id cannot be passed on the command line: {sid!r}')
    baseline = load_baseline(args.baseline.resolve(), ids) if args.baseline else None
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    plan = dict(controller=str(Path(__file__).resolve()), controller_sha256=sha256(__file__), rule_id=RULE_ID, pins=pins,
                arm=args.arm, checkpoint=binding, baseline=baseline, group=args.group, ids=ids,
                turns={sid: len(scenarios[sid]['users']) for sid in ids},
                base_gguf=str(BASE_GGUF), base_gguf_sha256=BASE_GGUF_SHA256, base_revision=BASE_REVISION,
                llama_server=str(LLAMA_SERVER), child_seconds=CHILD_SECONDS, ready_timeout=READY_TIMEOUT, max_new=MAX_NEW,
                context=CONTEXT, threads=THREADS, decoding=EXPECTED_BINDING['decoding'], official_render=str(OFFICIAL_RENDER),
                bound_seconds=len(ids) * CHILD_SECONDS, started_unix=time.time(),
                baseline_note='model arm = Q8 base without any --lora, generated once per id and reused for every checkpoint; '
                              'adapter identity is never part of cross-receipt parity',
                guard_contract='qwen-serving-guard.py -> supervisor006: pressure 1 gate, sampled 6 GiB group RSS stop, '
                               f'absolute deadline {CHILD_SECONDS} s, SIGTERM/SIGKILL cleanup, no retry')
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    result = dict(state='RUNNING', arm=args.arm, group=args.group, checkpoint_step=binding['step'] if binding else None,
                  checkpoint=binding['checkpoint'] if binding else None,
                  adapter_weights_sha256=binding['adapter_weights_sha256'] if binding else None,
                  adapter_gguf_sha256=binding['adapter_gguf_sha256'] if binding else None,
                  base_gguf_sha256=BASE_GGUF_SHA256, baseline=baseline['dir'] if baseline else None,
                  requested=len(ids), completed=0, remaining_ids=list(ids), automatic_retries=0, cases=[],
                  plan=str(out / 'plan.json'))
    started = time.monotonic()
    current = {'child': None, 'signal': None}

    def forward(signum, frame):
        """Hand the interruption to the running guard (its supervisor cleans the owned group); start nothing else."""
        current['signal'] = signal.Signals(signum).name
        if current['child'] is not None and current['child'].poll() is None:
            current['child'].send_signal(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, forward)
    try:
        for index, sid in enumerate(ids, 1):
            if current['signal']:
                result.update(state='INTERRUPTED_NO_RETRY', interrupted_by=current['signal'])
                raise RuntimeError('Interrupted before scenario ' + sid)
            if time.monotonic() - started + CHILD_SECONDS + 30 > PARENT_TOOL_TIMEOUT:
                result.update(state='BOUND_STOP_NO_RETRY', stopped_before=sid)
                raise RuntimeError('Remaining parent bound cannot hold another child; remaining ids retained, no automatic retry')
            name = safe_name(index, sid)
            sup_receipt, receipt, server_log, log = (out / f'{name}-supervisor.json', out / f'{name}.json',
                                                     out / f'{name}-server.log', out / f'{name}.log')
            port = free_port()
            harness_args = ['--model-file', str(BASE_GGUF), '--expected-sha256', BASE_GGUF_SHA256]
            if binding:
                harness_args += ['--lora', binding['adapter_gguf'], '--lora-sha256', binding['adapter_gguf_sha256']]
            harness_args += ['--arms', args.arm, '--suite', str(SCENARIOS), '--ids', sid, '--receipt', str(receipt),
                             '--server-log', str(server_log), '--port', str(port), '--context', str(CONTEXT),
                             '--max-new', str(MAX_NEW), '--threads', str(THREADS), '--ready-timeout', str(READY_TIMEOUT),
                             '--official-render', str(OFFICIAL_RENDER)]
            cmd = [str(PYTHON), '-B', str(GUARD), '--receipt', str(sup_receipt), '--seconds', str(CHILD_SECONDS), '--', *harness_args]
            case = dict(index=index, id=sid, group=scenarios[sid]['group'], arm=args.arm, receipt=str(receipt),
                        supervisor_receipt=str(sup_receipt), server_log=str(server_log), log=str(log), port=port,
                        argv=cmd, verified=False)
            result['cases'].append(case)
            child_started = time.monotonic()
            with log.open('x') as handle:
                current['child'] = subprocess.Popen(cmd, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT)
                returncode = current['child'].wait()
            current['child'] = None
            case.update(returncode=returncode, wall_seconds=round(time.monotonic() - child_started, 3))
            if current['signal']:
                result.update(state='INTERRUPTED_NO_RETRY', interrupted_by=current['signal'], failed_scenario=sid)
                raise RuntimeError('Interrupted during scenario ' + sid)
            if returncode != 0:
                result.update(state='FAILED_NO_RETRY', failed_scenario=sid)
                raise RuntimeError(f'Bounded scenario failed (exit {returncode}): {sid}; see {log}')
            try:
                verify_child(case, sup_receipt, receipt, pins, binding, args.arm, sid, scenarios[sid]['users'])
                if baseline:
                    case['parity'] = compare_with_baseline(receipt, Path(baseline['receipts'][sid]), scenarios[sid]['users'])
                    if not case['parity']['equal']:
                        failed = [k for k, v in case['parity']['checks'].items() if not v]
                        raise RuntimeError(f'cross-receipt parity with the baseline failed: {failed}')
            except Exception as exc:
                result.update(state='INVALID_NO_RETRY', failed_scenario=sid)
                raise RuntimeError(f'Scenario {sid} did not verify: {exc}') from exc
            case['verified'] = True
            result['completed'] += 1
            result['remaining_ids'].remove(sid)
            print(json.dumps({'completed': result['completed'], 'of': len(ids), 'id': sid, 'seconds': case['seconds'],
                              'turns': f'{case["turns_completed"]}/{case["turns_total"]}', 'fits_context': case['fits_context'],
                              'truncated': case['truncated'], 'stopped_at_eos': case['stopped_at_eos'],
                              'empty_output': case['empty_output'],
                              'parity': case['parity']['equal'] if baseline else 'not_compared'}, ensure_ascii=False), flush=True)
        result['state'] = 'COMPLETE'
    except BaseException as exc:
        result['error'] = repr(exc)
        if result['state'] == 'RUNNING':
            result['state'] = 'FAILED_NO_RETRY'
        raise
    finally:
        result['elapsed_seconds'] = round(time.monotonic() - started, 3)
        with (out / 'execution.json').open('x') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(json.dumps({'state': result['state'], 'completed': result['completed'], 'of': len(ids),
                          'remaining_ids': result['remaining_ids'], 'execution': str(out / 'execution.json')}), flush=True)


if __name__ == '__main__':
    main()
