"""Relocatable, parent-only Qwen013 serving qualification (stdlib; no model imports).

Commands: prepare --repo R --sources N --plan P --plan-sha256 H --runtime J
--runtime-sha256 H --out D; admit/run --prepared D --prepared-sha256 H
--candidate J --candidate-sha256 H --out D [run: --port N]; qualify --measurement J
--measurement-sha256 H --review J --review-sha256 H --out J.
All output directories/files must be fresh. prepare/admit never start a model.
Run ONLY after independent DEV selection; it does not deploy or open final24.

Runtime JSON schema trid-serving-runtime-v1: interpreter/server are {path,sha256};
libraries is a nonempty list of those bindings (the complete reviewed non-system
native dependency closure); helper binds scripts/base_dialogue_runner.py;
commit='8ca4060', cpu_only=true. Paths are explicit absolute local paths. Invoke
this driver with the exact bound interpreter path (not another venv sharing its
executable). The runtime JSON hash is an externally reviewed trust anchor, not
self-attestation; the parent must verify the actual native dependency closure.

Candidate JSON schema trid-serving-candidate-v1: selected_label plus bindings
selection_input, selection, pipeline_outcome, full_training, training, weights,
config, conversion, base, adapter. The independent selector input's pins.serving_candidates must map
ALL three blinded labels to identity dictionaries returned by candidate_identity;
its pins.baseline_preserved must be true; pins.independent_selection must contain
independent=true and a nonempty reviewer identity. Bind these before selection.
The original pure selector is replayed, not regraded. Original remote receipt paths
remain in receipts; local relocations are explicit bindings cross-checked by hash.
Preparation binds the frozen DEV domain/behavior scenario files and behavior rubric
under --repo/data/qwen-specialization-003. Admission authenticates those bytes and
requires the exact IDs, groups and rubric-derived behavior families before replay.

A run emits MEASURED_AWAITING_CONTENT_REVIEW, never automatic qualification from
surface substrings. Independent review schema trid-serving-content-review-v1:
measurement_sha256, reviewer (nonempty), independent=true, and decisions: one per
actual request, {key, output_sha256, pass:true, assertion, evidence}. key is
'chunki:kind:session:turn' (i=1..3); assertion must equal CONTENT_ASSERTIONS[session],
evidence must quote a nonempty portion of that output. A reviewer must read the
whole raw response and decide the assertion, not merely search for its substrings.
qualify checks all decisions and revalidates the measured source/receipt bindings.

Algorithm provenance: recovered supervisor006 is copied byte-for-byte and called
through supervise(); guard environment/bounds are explicit here. A visible harness
derivative changes repository/helper/runtime paths plus raw response capture and
per-request server PID evidence. The pinned request helper additionally rejects
non-200 HTTP success codes rather than misreporting them as 200. Prompting,
sampling, case expansion, deadlines and cleanup algorithms are preserved.
The supervisor's recorded child argv is also checked against the bound interpreter,
prepared derivative, artifact paths and exact chunk arguments before qualification.
preparation.json records each replacement and original/derivative source hashes.
No module globals are patched. No GPU source or bundle pins are changed.
No baseline generation or regrading is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

PINS = {
    'qwen-serving-harness.py': '9b07f62effa358ff059601849e01ab6be0dfb21541d878cc3b1a58ee5c1154a8',
    'qwen-serving-guard.py': 'e9598f0e0ac250924e629812fd408155dd9caa65ddb266f360a49620eec1d129',
    'development_supervisor006.py': '0f31777b5f66199f56a91a73857440d6c316c1cf71148aef5be432fdd5e28070',
    'qwen-serving-suite.json': 'dd7d7b292a9c0ab5d7346abf28c82e134ffc31208706354910026303e982ce09',
    'official-render.json': '7d3fa3cb7e04e10087cbfb94d1066213e01ac766f8def8dfa449761a945480e1',
    'select_native_development.py': '1f7fd0400b8a6c2f74e0ad7c821acde1baf92559ffa82fcb27a5f1d3bf58dad1',
}
DEV_PINS = {
    'domain-dev-scenarios-003.json': '13fa5a06f264678ee95611467b289f18bab81a5a15c07972ddd85af32824d4c8',
    'behavior-dev-scenarios.json': 'bc613942b59dd6ec6ace8bfe865012da2f591161bd52ae9bec1da1d3b61f47d7',
    'behavior-dev-rubric.json': '6ef21a0631a4f82d1cfca10b15fe9dd336e99f3f60ed2f575a6391d7bf653e2a',
}
BASE_SHA = '29508ae00d82764d261287099ea3077dd08dcbf5cb86fe14427e9407b913c3c7'
HELPER_SHA = 'e1a35e447565290fe49368a9b4ab0341ee3c0d6afcb2b5d1d6081772b33b7fda'
CONVERTER_SHA = 'ddb143ac34022b0646e51d9c35e9cfa3ec3d0a36091975ebc88f553dd18ca350'
CHUNKS = [['single', 'seq-1', 'seq-2'], ['history-correction', 'new-session-isolation'],
          ['clear-marker-isolation', 'seq-3']]
BOUNDS = dict(child_seconds=180, rss_stop_bytes=6 * 1024**3, pressure_required=1,
              ready_timeout=30, max_new=384, context=2048, threads=4)
PROTECTED_PORTS = {51873, 51912, 51917, 51918, 51990}
SUFFIX = '<|im_start|>assistant\n<think>\n\n</think>\n\n'
CONTENT_ASSERTIONS = {
    'single': 'Relevant Korean domestic-business-trip reimbursement documents; no fabricated mandatory rule.',
    'seq-1': 'Five relevant meeting-minutes fields, not merely five unrelated items.',
    'seq-2': 'One polite closing sentence suitable for an official civil-petition reply.',
    'history-correction': 'Each turn follows the actual conversation; correction is Friday/Sejong and final recall is Park Seoyeon/Friday/Sejong, without contradicting these facts.',
    'new-session-isolation': 'Does not know or invent the prior session name, trip day or destination.',
    'clear-marker-isolation': 'First turn acknowledges the supplied employee number; after clear, does not know or invent it.',
    'seq-3': 'Exactly the odd numbers 1,3,5,7,9 in comma-separated form; no even numbers.',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text_digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'Duplicate JSON key: ' + key)
            out[key] = value
        return out
    return json.loads(Path(path).read_text(), object_pairs_hook=unique,
                      parse_constant=lambda value: require(False, 'Nonfinite JSON: ' + value))


def bound(path):
    path = Path(path).absolute()
    return {'path': str(path), 'sha256': digest(path)}


def check(item):
    require(isinstance(item, dict) and isinstance(item.get('path'), str), 'Missing artifact binding')
    path = Path(item['path'])
    require(path.is_absolute() and path.is_file(), 'Artifact must be an existing absolute file: ' + str(path))
    require(isinstance(item.get('sha256'), str) and len(item['sha256']) == 64
            and digest(path) == item['sha256'], 'Artifact digest mismatch: ' + str(path))
    return path


def authenticated(path, sha):
    return load(check({'path': str(Path(path).absolute()), 'sha256': sha}))


def save(path, value):
    with Path(path).open('x') as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def validate_plan(plan):
    require(plan.get('bounds') == BOUNDS, 'Serving bounds changed')
    require(plan.get('chunks') == [dict(ids=ids, native_requests=n, oai_parity_requests=o)
                                   for ids, n, o in zip(CHUNKS, (3, 4, 3), (1, 0, 0))], 'Serving chunks changed')
    for key, name in [('suite', 'qwen-serving-suite.json'), ('harness', 'qwen-serving-harness.py'),
                      ('official_render', 'official-render.json')]:
        require(plan[key]['sha256'] == PINS[name], 'Plan source pin changed: ' + key)


def validate_runtime(runtime):
    require(runtime.get('schema') == 'trid-serving-runtime-v1' and runtime.get('commit') == '8ca4060'
            and runtime.get('cpu_only') is True, 'Explicit CPU 8ca4060 runtime required')
    for key in ('interpreter', 'server', 'helper'):
        check(runtime[key])
    require(Path(runtime['interpreter']['path']).absolute() == Path(sys.executable).absolute(),
            'Invoke with the exact bound interpreter path, including its venv')
    require(os.access(runtime['interpreter']['path'], os.X_OK) and os.access(runtime['server']['path'], os.X_OK),
            'Interpreter/server must be executable')
    require(runtime.get('libraries'), 'Reviewed native dependency closure required')
    paths = [str(check(item).resolve()) for item in runtime['libraries']]
    require(len(paths) == len(set(paths)), 'Duplicate runtime libraries')
    require(Path(runtime['helper']['path']).name == 'base_dialogue_runner.py', 'Wrong request helper')
    require(runtime['helper']['sha256'] == HELPER_SHA, 'Historical request helper source changed')


def relocate(source, repo, helper_dir, server):
    """Exact, auditable substitutions on a SHA-authenticated source; no execution."""
    replacements = [
        ("REPO = Path('/Users/eomgyuhyeon/사업/trid')", f'REPO = Path({str(repo)!r})'),
        ("sys.path.insert(0, str(REPO / 'scripts'))", f'sys.path.insert(0, {str(helper_dir)!r})'),
        ("SERVER_DIR = REPO / 'runs/tools/llama.cpp-8ca4060'", f'SERVER_DIR = Path({str(server.parent)!r})'),
        ("SERVER = SERVER_DIR / 'llama-server'", f'SERVER = Path({str(server)!r})'),
        ("row = dict(spec)", "row = dict(spec, server_pid=child.pid)"),
        ("'fits_context': fits, 'output': text,", "'fits_context': fits, 'output': text, 'raw_content': c['content'],"),
        ("return {'http_status': 200, 'output': text, 'finish_reason':", "return {'http_status': 200, 'output': text, 'raw_content': raw, 'finish_reason':"),
    ]
    for old, new in replacements:
        require(source.count(old) == 1, 'Relocation boundary changed: ' + old)
        source = source.replace(old, new, 1)
    return source, [{'original': old, 'replacement': new} for old, new in replacements]


def instrument_helper(source):
    old = "        return json.loads(r.read())"
    new = ("        if r.status != 200:\n"
           "            raise RuntimeError(f'Expected HTTP200, received {r.status}')\n"
           "        return json.loads(r.read())")
    require(source.count(old) == 1, 'Request helper boundary changed')
    return source.replace(old, new, 1), [{'original': old, 'replacement': new}]


def dev_source_bindings(repo):
    return {name: {'path': str(Path(repo).absolute() / 'data/qwen-specialization-003' / name), 'sha256': sha}
            for name, sha in DEV_PINS.items()}


def frozen_dev_population(bindings):
    """Authenticate fixed DEV membership/families; never generate or grade answers."""
    require(isinstance(bindings, dict) and set(bindings) == set(DEV_PINS), 'Missing frozen DEV sources')
    documents = {}
    for name, sha in DEV_PINS.items():
        require(bindings[name].get('sha256') == sha, 'Frozen DEV source pin changed')
        documents[name] = load(check(bindings[name]))
    domain = documents['domain-dev-scenarios-003.json']['results']
    behavior = documents['behavior-dev-scenarios.json']['results']
    rubrics = documents['behavior-dev-rubric.json']['items']
    rubric = {row['id']: row for row in rubrics}
    require(len(rubric) == len(rubrics) and set(rubric) == {row['id'] for row in behavior},
            'Frozen DEV rubric membership mismatch')
    population = {}
    for row in domain + behavior:
        sid, group = row['id'], row['group']
        require(sid not in population, 'Duplicate frozen DEV scenario')
        population[sid] = {'group': group}
        if group != 'domain':
            require(rubric[sid]['group'] == group, 'Frozen DEV rubric group mismatch')
            population[sid]['family'] = rubric[sid]['family']
    require(len(population) == 36 and all(sum(row['group'] == group for row in population.values()) == 12
                                        for group in ('domain', 'uncertainty', 'retention')), 'Invalid frozen DEV population')
    return population


def prepare(a):
    repo, sources, out = a.repo.resolve(strict=True), a.sources.resolve(strict=True), a.out.absolute()
    plan = authenticated(a.plan, a.plan_sha256)
    validate_plan(plan)
    runtime = authenticated(a.runtime, a.runtime_sha256)
    validate_runtime(runtime)
    dev_sources = dev_source_bindings(repo)
    frozen_dev_population(dev_sources)
    originals = {name: {'path': str(sources / name), 'sha256': sha} for name, sha in PINS.items()}
    for item in originals.values():
        check(item)
    source, changes = relocate((sources / 'qwen-serving-harness.py').read_text(), repo, out,
                               Path(runtime['server']['path']))
    helper_source, helper_changes = instrument_helper(check(runtime['helper']).read_text())
    out.mkdir(mode=0o700, parents=False, exist_ok=False)
    for name, item in originals.items():
        (out / name).write_bytes(check(item).read_bytes())
    (out / 'base_dialogue_runner.py').write_text(helper_source)
    (out / 'serving_harness.py').write_text(source)
    record = dict(schema='trid-serving-preparation-v1', state='PREPARED_NO_MODEL_EXECUTION', repo=str(repo),
                  driver=bound(__file__), plan=bound(a.plan), runtime_binding=bound(a.runtime), runtime=runtime,
                  original_sources=originals, dev_sources=dev_sources, replacements=changes, helper_replacements=helper_changes,
                  files={name: bound(out / name) for name in [*originals, 'base_dialogue_runner.py', 'serving_harness.py']})
    save(out / 'preparation.json', record)
    return record


def prepared(directory, expected_sha):
    record = authenticated(Path(directory) / 'preparation.json', expected_sha)
    require(record.get('schema') == 'trid-serving-preparation-v1'
            and record.get('state') == 'PREPARED_NO_MODEL_EXECUTION', 'Invalid preparation')
    require(record['driver'] == bound(__file__), 'Driver changed since preparation')
    validate_plan(load(check(record['plan'])))
    require(load(check(record['runtime_binding'])) == record['runtime'], 'Runtime manifest changed')
    validate_runtime(record['runtime'])
    frozen_dev_population(record.get('dev_sources'))
    require(set(record['original_sources']) == set(PINS), 'Incomplete source authentication')
    require(set(record['files']) == set(PINS) | {'base_dialogue_runner.py', 'serving_harness.py'}, 'Prepared file set changed')
    for name, sha in PINS.items():
        require(record['original_sources'][name]['sha256'] == sha, 'Original source pin changed')
        check(record['original_sources'][name])
        require(record['files'][name]['sha256'] == sha, 'Preserved source changed')
    for name, item in record['files'].items():
        require(check(item).resolve() == (Path(directory) / name).resolve(), 'Prepared source escaped directory')
    helper_source, helper_changes = instrument_helper(check(record['runtime']['helper']).read_text())
    require(check(record['files']['base_dialogue_runner.py']).read_text() == helper_source
            and record['helper_replacements'] == helper_changes, 'Unapproved helper derivative changes')
    expected, changes = relocate(check(record['original_sources']['qwen-serving-harness.py']).read_text(),
                                 Path(record['repo']), Path(directory).absolute(), Path(record['runtime']['server']['path']))
    require(check(record['files']['serving_harness.py']).read_text() == expected and record['replacements'] == changes,
            'Unapproved derivative changes')
    return record


def candidate_identity(candidate, step):
    return {'step': step, **{name + '_sha256': candidate[name]['sha256'] for name in
                            ('weights', 'config', 'training', 'conversion', 'adapter', 'base', 'full_training', 'pipeline_outcome')}}


def validate_candidate(candidate, selector, dev_sources):
    require(candidate.get('schema') == 'trid-serving-candidate-v1', 'Explicit selected candidate required')
    documents = {}
    for name in ('selection_input', 'selection', 'pipeline_outcome', 'full_training', 'training', 'weights', 'config', 'conversion', 'base', 'adapter'):
        path = check(candidate[name])
        if name not in ('weights', 'base', 'adapter'):
            documents[name] = load(path)
    selected = validate_candidate_records(candidate, documents, selector, dev_sources)
    require(sum(check(candidate[k]).stat().st_size for k in ('base', 'adapter')) <= 5 * 1024**3, 'Weights exceed harness bound')
    return selected


def validate_candidate_records(candidate, documents, selector, dev_sources):
    """Cross-receipt validation with independently pinned DEV membership."""
    selection, payload = documents['selection'], documents['selection_input']
    require(payload.get('scenarios') == frozen_dev_population(dev_sources),
            'Selection scenario IDs/groups/families differ from frozen DEV population')
    label = candidate.get('selected_label')
    require(isinstance(label, str) and label not in ('', 'NO_CANDIDATE') and selection.get('selected') == label,
            'Missing/unselected candidate')
    require(selection == selector.select(payload), 'Selection differs from original frozen calculator')
    require(sorted(c['step'] for c in payload['checkpoints']) == [8, 16, 23], 'DEV must compare checkpoints8/16/23')
    require(payload.get('pins', {}).get('baseline_preserved') is True, 'Frozen baseline preservation not attested')
    review = payload['pins'].get('independent_selection', {})
    require(review.get('independent') is True and isinstance(review.get('reviewer'), str)
            and review['reviewer'].strip(), 'Independent DEV selection evidence required')
    step = selection['selected_step']
    identities = payload['pins'].get('serving_candidates', {})
    require(set(identities) == {c['label'] for c in payload['checkpoints']}
            and all(identities[c['label']]['step'] == c['step'] for c in payload['checkpoints']), 'Missing blinded artifact bindings')
    require(identities[label] == candidate_identity(candidate, step), 'Selected artifact bindings differ')
    outcome = documents['pipeline_outcome']
    require(outcome.get('state') == 'COMPLETE_AWAITING_BLIND_SELECTION'
            and outcome.get('candidate_cases') == 108 and outcome.get('candidate_turns') == 138
            and outcome.get('baseline_regenerated') is False and outcome.get('final24_opened') is False
            and outcome.get('semantic_grading_performed') is False and outcome.get('selection_performed') is False
            and outcome.get('full_training_started') is True and outcome.get('native_generation_started') is True
            and outcome.get('export_state') != 'FAILED', 'Training/native pipeline failed or incomplete')
    full, trained, conversion = (documents[name] for name in ('full_training', 'training', 'conversion'))
    require(full.get('status') == 'complete' and full.get('steps') == 23 and full.get('planned_steps') == 23
            and [c['step'] for c in full.get('checkpoints', [])] == [8, 16, 23], 'Full training failed/incomplete')
    require(trained.get('status') == 'checkpoint' and trained.get('steps') == step
            and trained.get('planned_steps') == 23, 'Selected checkpoint failed/incomplete')
    row = next(c for c in full['checkpoints'] if c['step'] == step)
    require(trained.get('checkpoint') == row and row['adapter_weights_sha256'] == candidate['weights']['sha256'],
            'Checkpoint not retained by successful full training')
    require(trained.get('base_weights', {}).get('gguf_sha256') == candidate['base']['sha256'] == BASE_SHA,
            'Wrong research base; public default is not interchangeable')
    require(conversion.get('status') == 'complete' and conversion.get('returncode') == 0
            and conversion.get('checkpoint', {}).get('step') == step, 'Conversion failed/incomplete')
    require(conversion.get('converter_sha256') == CONVERTER_SHA, 'Unpinned converter')
    for name, location in [('weights', conversion['checkpoint']['weights']), ('config', conversion['checkpoint']['config']),
                           ('training', conversion['checkpoint']['training']), ('base', conversion['source_gguf']),
                           ('adapter', conversion['adapter_gguf'])]:
        require(candidate[name]['sha256'] == location['sha256'], 'Conversion binding mismatch: ' + name)
    cfg = documents['config']
    require(cfg.get('peft_type') == 'LORA' and cfg.get('r') == 16 and cfg.get('lora_alpha') == 32
            and set(cfg.get('target_modules', [])) == {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'}
            and not any(cfg.get(k) for k in ('use_dora', 'use_rslora', 'modules_to_save')), 'Unexpected adapter config')
    return {'label': label, 'identity': candidate_identity(candidate, step), 'bindings': candidate}


def admit(a):
    preparation = prepared(a.prepared, a.prepared_sha256)
    candidate = authenticated(a.candidate, a.candidate_sha256)
    selector = module(check(preparation['files']['select_native_development.py']), 'serving_selector')
    identity = validate_candidate(candidate, selector, preparation['dev_sources'])
    a.out.mkdir(mode=0o700, parents=False, exist_ok=False)
    record = dict(schema='trid-serving-admission-v1', state='ADMITTED_NO_MODEL_EXECUTION',
                  preparation=bound(a.prepared / 'preparation.json'), candidate=bound(a.candidate), selected=identity)
    save(a.out / 'admission.json', record)
    return preparation, candidate, record


def child_argv(preparation, candidate, native, log, index, port):
    """Single launch contract, also checked against persisted supervisor evidence."""
    return [preparation['runtime']['interpreter']['path'], '-B', preparation['files']['serving_harness.py']['path'],
            '--model-file', candidate['base']['path'], '--expected-sha256', candidate['base']['sha256'],
            '--lora', candidate['adapter']['path'], '--lora-sha256', candidate['adapter']['sha256'], '--arms', 'adapter',
            '--suite', preparation['files']['qwen-serving-suite.json']['path'],
            '--official-render', preparation['files']['official-render.json']['path'],
            '--receipt', str(Path(native).absolute()), '--server-log', str(Path(log).absolute()), '--port', str(port),
            '--context', '2048', '--max-new', '384', '--threads', '4', '--ready-timeout', '30',
            '--duration', '180', '--ids', *CHUNKS[index]]


def validate_launch(guard, preparation, candidate, native, log, index, port):
    require(guard.get('argv') == child_argv(preparation, candidate, native, log, index, port),
            'Supervisor child command differs from authenticated execution binding')


def finite(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, 'Invalid ' + label)


def validate_guard(guard):
    require(guard.get('harness_exit_code') == 0 and guard.get('reaped') is True
            and guard.get('cleanup_state') == 'completed' and guard.get('cleanup_errors') == []
            and guard.get('stop_reason') == 'harness_exit' and guard.get('first_soft_stop_reason') is None
            and guard.get('hard_alarm_fired') is False and guard.get('startup_released') is True
            and guard.get('duration_seconds') == 180, 'Guard did not complete cleanly')
    require(guard.get('group_samples') and guard.get('pressure')
            and all(row['value'] == 1 for row in guard['pressure']), 'Missing/failed pressure evidence')
    for sample in guard['group_samples']:
        require(isinstance(sample.get('members'), list), 'Missing group membership sample')
        for row in sample['members']:
            require(type(row.get('pid')) is int and row['pid'] > 0
                    and row.get('pgid') == guard['owned_pgid'], 'Invalid sampled process ownership')
            finite(row.get('rss_bytes'), 'sampled RSS')
        require(sum(row['rss_bytes'] for row in sample['members']) <= BOUNDS['rss_stop_bytes'], 'Supervisor RSS ceiling exceeded')
    require(any(row['pid'] == guard['harness_pid'] and row['rss_bytes'] > 0
                for sample in guard['group_samples'] for row in sample['members']), 'No live owned harness sampled')
    require('members_before_reap' in guard and all(row['status'].startswith('Z') for row in guard['members_before_reap']),
            'Owned children survived cleanup')
    finite(guard.get('elapsed_seconds'), 'guard elapsed')
    require(guard['elapsed_seconds'] <= 180, 'Guard exceeded the180s chunk bound')


def validate_chunk(record, guard, index, candidate, harness, suite, official):
    validate_guard(guard)
    require(record.get('status') == 'complete' and not record.get('error') and not record.get('stop_signal')
            and record.get('live_members_after_shutdown') == [] and not record.get('server_killed'), 'Native run incomplete/unclean')
    require(type(record.get('server_exit_code')) is int and record['server_exit_code'] in (0, -15), 'Server shutdown failed')
    require(record.get('harness_pid') == guard['harness_pid'] and record.get('pgid') == guard['owned_pgid'], 'Group ownership mismatch')
    require(record.get('arms') == ['adapter'] and record.get('ids') == sorted(CHUNKS[index]), 'Wrong chunk or base regeneration')
    for key, name in [('model', 'base'), ('lora', 'adapter')]:
        require(record.get(key + '_sha256') == candidate[name]['sha256']
                and Path(record[key + '_file']).resolve() == check(candidate[name]).resolve(), 'Wrong model/adapter identity')
    require(record.get('suite_sha256') == PINS['qwen-serving-suite.json'], 'Case contract changed')
    expected = dict(host='127.0.0.1', context=2048, max_new=384, threads=4, batch=256, ubatch=128,
                    n_gpu_layers=0, no_repack=True, no_context_shift=True, system_prompt=None,
                    decoding={'temperature': 0, 'seed': 42}, template_kwargs={'enable_thinking': False}, cache_prompt=False)
    require(all(key in record['binding'] and record['binding'][key] == value for key, value in expected.items()), 'Generation binding changed')
    port = record['binding'].get('port')
    require(type(port) is int and 1024 <= port <= 65535 and port not in PROTECTED_PORTS, 'Unsafe recorded port')
    require(record.get('argv') == harness.server_argv(Path(candidate['base']['path']).resolve(), port, 2048, 4,
                                                     Path(candidate['adapter']['path']).resolve()), 'Actual server command changed')
    require(all(record.get('env', {}).get(k) == v for k, v in
                {'DYLD_LIBRARY_PATH': str(harness.SERVER_DIR), 'LLAMA_ARG_DEVICE': 'none',
                 'LLAMA_ARG_N_GPU_LAYERS': '0', 'OMP_NUM_THREADS': '4'}.items()), 'Actual native environment changed')
    require(record.get('phases') and all(p['pressure'] == 1 for p in record['phases']), 'Phase pressure failed')
    adapters = record['server']['lora_adapters']
    require(len(adapters) == 1 and adapters[0].get('id') == 0 and adapters[0].get('scale') == 1
            and Path(adapters[0]['path']).resolve() == check(candidate['adapter']).resolve(), 'Active LoRA not exactly scale1')
    require(len(record['arm_switches']) == 1 and record['arm_switches'][0]['arm'] == 'adapter'
            and record['arm_switches'][0]['scale'] == 1, 'Unexpected arm switch')
    parity = record['render_parity']
    require(all(parity.get(k) is True for k in ('text_equal', 'token_ids_equal', 'no_think_suffix'))
            and parity['served_prompt'] == official['prompt'] and parity['served_token_ids'] == official['token_ids']
            and parity['sample_messages'] == official['messages'], 'Official render/token parity failed')
    require(record['timeouts']['duration'] == 180 and record['timeouts']['ready'] == 30
            and record['timeouts']['request_derived'] is True and record['timeouts']['shutdown_grace'] == 10,
            'Request window contract changed')
    finite(record['timeouts']['request'], 'request timeout', 30)
    finite(record['timeouts']['startup'], 'startup')
    require(record['timeouts']['window_needed'] <= 178, 'Window exceeds guard')
    finite(record.get('ready_seconds'), 'readiness')
    require(record['ready_seconds'] <= 30, 'Readiness exceeded30s')
    memory = record['memory']
    require(memory['samples'] > 0 and memory['ceiling_bytes'] == BOUNDS['rss_stop_bytes']
            and memory['under_ceiling'] is True and 0 < memory['group_rss_bytes'] <= BOUNDS['rss_stop_bytes'], 'Missing/failed RSS evidence')
    specs, parity_session = harness.expand(suite, set(CHUNKS[index]))
    rows = record['requests']
    require(len(rows) == len(specs) + bool(parity_session), 'Missing/extra native or OAI requests')
    startup, timeout = record['timeouts']['startup'], record['timeouts']['request']
    require(timeout == harness.derive_request_timeout(180, 30, 10, len(rows), startup)
            and record['timeouts']['window_needed'] == harness.budget(30, timeout, 10, len(rows), startup),
            'Reported request window differs from original bounded algorithm')
    histories, evidence = {}, []
    for position, row in enumerate(rows):
        is_oai = position == len(specs)
        spec = specs[0] if is_oai else specs[position]
        require(all(row.get(k) == spec[k] for k in ('session', 'group', 'turn', 'user'))
                and row.get('kind') == ('oai' if is_oai else 'native') and row.get('arm') == 'adapter', 'Request order/case changed')
        require(row.get('server_pid') == record['server_pid'] and type(record['server_pid']) is int and record['server_pid'] > 0,
                'Requests did not share server PID')
        require(row.get('http_status') == 200 and isinstance(row.get('raw_content'), str)
                and row['raw_content'].strip() and row.get('output') == row['raw_content'].strip()
                and not any(row.get(k) for k in ('error', 'transport_failure', 'contract_failure', 'interrupted')), 'Failed/empty request')
        require('<think>' not in row['raw_content'] and '</think>' not in row['raw_content'], 'Thinking-prefill leakage')
        finite(row.get('wall_seconds'), 'request wall time')
        require(isinstance(row.get('timings'), dict) and row['timings'], 'Native timing evidence missing')
        require(row.get('pressure') == 1 and 0 < row.get('group_rss_bytes', 0) <= BOUNDS['rss_stop_bytes'], 'Request resources failed')
        if is_oai:
            require(row['raw_content'] == rows[0]['raw_content'] and row.get('raw_content_equal') is True
                    and row.get('generated_text_equal') is True and row.get('no_think_prefill_echoed') is False
                    and not row.get('reasoning_content'), 'Raw native/OAI parity failed')
        else:
            require(row.get('reset') == spec['reset'], 'Reset marker changed')
            history = [] if spec['reset'] else histories.get(spec['session'], [])
            require(row.get('history_turns') == len(history) // 2, 'Wrong own-history length')
            messages = history + [('user', spec['user'])]
            prompt = ''.join(f'<|im_start|>{role}\n{text}<|im_end|>\n' for role, text in messages) + SUFFIX
            require(row.get('prompt') == prompt and row.get('rendered_prompt_sha256') == text_digest(prompt), 'Prompt/history/reset mismatch')
            tokens = row['prompt_token_ids']
            require(isinstance(tokens, list) and tokens and all(type(t) is int and t >= 0 for t in tokens)
                    and row.get('prompt_tokens') == len(tokens) and len(tokens) + 384 <= 2048
                    and row.get('fits_context') is True
                    and row.get('prompt_token_ids_sha256') == text_digest(json.dumps(tokens)), 'Context/token evidence failed')
            histories[spec['session']] = messages + [('assistant', row['output'])]
        require(row.get('surface') == harness.surface(spec, row['output']), 'Surface measurements disagree with raw output')
        evidence.append(dict(key=f'chunk{index + 1}:{row["kind"]}:{row["session"]}:{row["turn"]}',
                             output_sha256=text_digest(row['raw_content']), raw_content=row['raw_content'],
                             assertion=CONTENT_ASSERTIONS[row['session']]))
    return evidence


def harness_module(preparation):
    # Authenticated module import is stdlib-only; main() is never called here.
    harness = module(check(preparation['files']['serving_harness.py']), 'serving_contract')
    require(Path(harness.request.__code__.co_filename).resolve()
            == check(preparation['files']['base_dialogue_runner.py']).resolve(), 'Unbound cached HTTP helper')
    return harness


def run(a):
    require(sys.platform == 'darwin', 'Recovered supervisor requires Darwin')
    require(type(a.port) is int and 1024 <= a.port <= 65535 and a.port not in PROTECTED_PORTS, 'Unsafe loopback port')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', a.port))
    preparation, candidate, admission = admit(a)
    runtime = preparation['runtime']
    supervisor = module(check(preparation['files']['development_supervisor006.py']), 'serving_supervisor')
    harness = harness_module(preparation)
    suite = load(check(preparation['files']['qwen-serving-suite.json']))
    official = load(check(preparation['files']['official-render.json']))
    env = {'PATH': '/usr/bin:/bin:/usr/sbin:/sbin', 'HOME': str(Path.home()), 'TMPDIR': '/private/tmp',
           'LANG': 'C.UTF-8', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
           'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'OMP_NUM_THREADS': '4',
           'LLAMA_ARG_DEVICE': 'none', 'LLAMA_ARG_N_GPU_LAYERS': '0', 'LLAMA_ARG_LOG_VERBOSITY': '4',
           'DYLD_LIBRARY_PATH': str(Path(runtime['server']['path']).parent)}
    record = dict(schema='trid-serving-measurement-v1', state='FAILED', admission=bound(a.out / 'admission.json'),
                  preparation=admission['preparation'], candidate=admission['candidate'], chunks=[], requests=[])
    try:
        for index, ids in enumerate(CHUNKS):
            prepared(a.prepared, a.prepared_sha256)  # Reauthenticate before each model-bearing child; no retries.
            validate_candidate(candidate, module(check(preparation['files']['select_native_development.py']), 'serving_selector'),
                               preparation['dev_sources'])
            native, guard, log = [a.out.absolute() / f'chunk{index + 1}-{suffix}' for suffix in ('native.json', 'guard.json', 'server.log')]
            argv = child_argv(preparation, candidate, native, log, index, a.port)
            receipt = supervisor.supervise(argv, guard, absolute_deadline_unix=time.time() + 180,
                                            duration=180, pressure_required=1, env=env)
            validate_launch(receipt, preparation, candidate, native, log, index, a.port)
            evidence = validate_chunk(load(native), receipt, index, candidate, harness, suite, official)
            record['chunks'].append(dict(native=bound(native), guard=bound(guard), log=bound(log)))
            record['requests'].extend(evidence)
        record['state'] = 'MEASURED_AWAITING_CONTENT_REVIEW'
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        save(a.out / 'measurement.json', record)
    return record


def validate_review(measurement, measurement_sha, review):
    require(measurement.get('schema') == 'trid-serving-measurement-v1'
            and measurement.get('state') == 'MEASURED_AWAITING_CONTENT_REVIEW' and len(measurement['chunks']) == 3
            and len(measurement['requests']) == 11, 'Measurement incomplete; cannot qualify')
    require(review.get('schema') == 'trid-serving-content-review-v1' and review.get('measurement_sha256') == measurement_sha
            and review.get('independent') is True and isinstance(review.get('reviewer'), str) and review['reviewer'].strip(),
            'Independent hash-bound content review required')
    decisions = review.get('decisions', [])
    require(len(decisions) == 11 and len({d['key'] for d in decisions}) == 11, 'Every actual request needs exactly one review')
    by_key = {d['key']: d for d in decisions}
    require(set(by_key) == {r['key'] for r in measurement['requests']}, 'Review case set changed')
    for row in measurement['requests']:
        decision = by_key[row['key']]
        require(decision.get('pass') is True and decision.get('output_sha256') == row['output_sha256']
                and decision.get('assertion') == row['assertion'] and isinstance(decision.get('evidence'), str)
                and decision['evidence'].strip() and decision['evidence'] in row['raw_content'], 'Missing/failed semantic assertion: ' + row['key'])


def qualify(a):
    measured = authenticated(a.measurement, a.measurement_sha256)
    review = authenticated(a.review, a.review_sha256)
    validate_review(measured, a.measurement_sha256, review)
    preparation_path = check(measured['preparation'])
    preparation = prepared(preparation_path.parent, measured['preparation']['sha256'])
    candidate = load(check(measured['candidate']))
    admission = load(check(measured['admission']))
    require(admission.get('schema') == 'trid-serving-admission-v1'
            and admission.get('state') == 'ADMITTED_NO_MODEL_EXECUTION', 'Invalid/failed admission')
    require(admission['candidate'] == measured['candidate'] and admission['preparation'] == measured['preparation'], 'Admission binding changed')
    selected = validate_candidate(candidate, module(check(preparation['files']['select_native_development.py']), 'serving_selector'),
                                  preparation['dev_sources'])
    require(admission['selected'] == selected, 'Admitted selection changed')
    harness = harness_module(preparation)
    evidence = []
    for index, chunk in enumerate(measured['chunks']):
        check(chunk['log'])
        native, guard = load(check(chunk['native'])), load(check(chunk['guard']))
        validate_launch(guard, preparation, candidate, chunk['native']['path'], chunk['log']['path'],
                        index, native['binding']['port'])
        evidence.extend(validate_chunk(native, guard, index, candidate, harness,
                                       load(check(preparation['files']['qwen-serving-suite.json'])),
                                       load(check(preparation['files']['official-render.json']))))
    require(evidence == measured['requests'], 'Measurement differs from actual request receipts')
    record = dict(schema='trid-serving-qualification-v1', state='QUALIFIED_STANDALONE_NOT_DEPLOYED',
                  measurement=bound(a.measurement), review=bound(a.review), selected=selected,
                  native_requests=10, oai_requests=1,
                  limitation='Not integrated public-service memory admission, deployment, latency SLA, or final24 evaluation.')
    save(a.out, record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('repo', 'sources', 'plan', 'runtime', 'out'):
        prep.add_argument('--' + name, type=Path, required=True)
    for name in ('plan', 'runtime'):
        prep.add_argument('--' + name + '-sha256', required=True)
    for command in ('admit', 'run'):
        p = sub.add_parser(command)
        for name in ('prepared', 'candidate', 'out'):
            p.add_argument('--' + name, type=Path, required=True)
        p.add_argument('--candidate-sha256', required=True)
        p.add_argument('--prepared-sha256', required=True)
        if command == 'run':
            p.add_argument('--port', type=int, required=True)
    p = sub.add_parser('qualify')
    for name in ('measurement', 'review', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('measurement', 'review'):
        p.add_argument('--' + name + '-sha256', required=True)
    a = parser.parse_args(argv)
    sys.dont_write_bytecode = True
    result = {'prepare': prepare, 'admit': admit, 'run': run, 'qualify': qualify}[a.command](a)
    print(json.dumps({'state': result[2]['state'] if a.command == 'admit' else result['state']}))


if __name__ == '__main__':
    main()
