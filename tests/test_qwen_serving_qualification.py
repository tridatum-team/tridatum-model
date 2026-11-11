"""Deterministic boundary tests using only public source and temporary UNIT data.

Vendored algorithms are exact frozen source bytes, authenticated with production
pins. DEV membership, serving prompts/render tokens, artifact bytes and receipts
are synthetic, never candidate/model evidence. Only their fixture-specific trust
anchors are patched in test scope; qualification and selection logic stay real.
No model, server, subprocess, cloud, or network runs.
"""
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/fixtures/qwen_serving'
SPEC = importlib.util.spec_from_file_location('serving_qualification', ROOT / 'scripts/qwen_serving_qualification.py')
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def json_file(path, value):
    path.write_text(json.dumps(value))
    return Q.bound(path)


def guard_fixture():
    return dict(harness_exit_code=0, reaped=True, cleanup_state='completed', cleanup_errors=[],
                stop_reason='harness_exit', first_soft_stop_reason=None, hard_alarm_fired=False,
                startup_released=True, duration_seconds=180, harness_pid=701, owned_pgid=701,
                group_samples=[{'members': [{'pid': 701, 'pgid': 701, 'rss_bytes': 1024}]}],
                pressure=[{'value': 1}], members_before_reap=[{'status': 'Z'}], elapsed_seconds=12)


def selection_fixture(selector, dev_sources, directory):
    directory = directory / 'selection'
    directory.mkdir(exist_ok=True)
    candidate = {'schema': 'trid-serving-candidate-v1', 'selected_label': 'A'}
    for name in ('weights', 'base', 'adapter'):
        path = directory / (name + '.unit')
        path.write_bytes(b'unit-test bytes, NOT a model: ' + name.encode())
        candidate[name] = Q.bound(path)
    scenarios = Q.frozen_dev_population(dev_sources)
    observation = dict(verified=True, empty_output=False, stopped_at_eos=True)
    base_grades = {sid: ({'score': 1} if scenario['group'] == 'domain' else {'pass': True})
                   for sid, scenario in scenarios.items()}
    candidate_grades = {sid: {**({'score': 2} if scenario['group'] == 'domain' else {'pass': True}),
                              'severe': None, 'requested_language': True, 'persona_injection': False,
                              **({'refused_or_nonanswer': False} if scenario.get('family', '').startswith('ans.') else {})}
                        for sid, scenario in scenarios.items()}
    checkpoints = [dict(step=step, adapter_weights_sha256=candidate['weights']['sha256']) for step in (8, 16, 23)]
    full = dict(status='complete', steps=23, planned_steps=23, checkpoints=checkpoints)
    trained = dict(status='checkpoint', steps=8, planned_steps=23, checkpoint=checkpoints[0],
                   base_weights={'gguf_sha256': Q.BASE_SHA})
    config = dict(peft_type='LORA', r=16, lora_alpha=32,
                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
    for name, document in (('config', config), ('full_training', full), ('training', trained)):
        candidate[name] = json_file(directory / (name + '.json'), document)
    conversion = dict(status='complete', returncode=0, converter_sha256=Q.CONVERTER_SHA,
                      checkpoint={'step': 8, **{k: candidate[k] for k in ('weights', 'config', 'training')}},
                      source_gguf=candidate['base'], adapter_gguf=candidate['adapter'])
    outcome = dict(state='COMPLETE_AWAITING_BLIND_SELECTION', candidate_cases=108, candidate_turns=138,
                   baseline_regenerated=False, final24_opened=False, semantic_grading_performed=False,
                   selection_performed=False, full_training_started=True, native_generation_started=True)
    for name, document in (('conversion', conversion), ('pipeline_outcome', outcome)):
        candidate[name] = json_file(directory / (name + '.json'), document)
    payload = dict(rule={'sha256': selector.RULE_SHA256}, scenarios=scenarios,
                   baseline={'items': {sid: dict(observation) for sid in scenarios}},
                   checkpoints=[dict(label=label, step=step, dev_ce={key: 1.0 for key in selector.CE_FILES},
                                     items={sid: {**observation, 'parity_equal': True} for sid in scenarios})
                                for label, step in zip('ABC', (8, 16, 23))],
                   grades={'base': base_grades, 'candidates': {label: copy.deepcopy(candidate_grades) for label in 'ABC'}},
                   pins={'baseline_preserved': True,
                         'independent_selection': {'independent': True, 'reviewer': 'unit-fixture reviewer'},
                         'serving_candidates': {label: Q.candidate_identity(candidate, step) for label, step in zip('ABC', (8, 16, 23))}})
    docs = dict(selection_input=payload, selection=selector.select(payload), full_training=full,
                training=trained, conversion=conversion, config=config, pipeline_outcome=outcome)
    for name in ('selection_input', 'selection'):
        candidate[name] = json_file(directory / (name + '.json'), docs[name])
    return candidate, docs


class ServingQualificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.selector = Q.module(Q.check({'path': str(FIXTURES / 'select_native_development.py'),
                                        'sha256': Q.PINS['select_native_development.py']}),
                                'unit_frozen_selector')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.saved_path))
        self.sources = self.directory / 'sources'
        self.sources.mkdir()
        for name, sha in Q.PINS.items():
            if name.endswith('.py'):
                original = Q.check({'path': str(FIXTURES / name), 'sha256': sha})
                (self.sources / name).write_bytes(original.read_bytes())
        self.make_dev_sources()
        self.make_serving_specifications()
        helper = self.directory / 'base_dialogue_runner.py'
        helper.write_bytes((ROOT / 'scripts/base_dialogue_runner.py').read_bytes())
        source, _ = Q.relocate((self.sources / 'qwen-serving-harness.py').read_text(), self.directory,
                               self.directory, self.directory / 'native runtime/llama-server')
        harness_file = self.directory / 'harness.py'
        harness_file.write_text(source)
        self.harness = Q.module(harness_file, 'unit_relocated_harness')
        self.suite = Q.load(self.sources / 'qwen-serving-suite.json')
        self.official = Q.load(self.sources / 'official-render.json')
        self.candidate = {}
        for name in ('base', 'adapter'):
            path = self.directory / (name + '.gguf')
            path.write_bytes(b'unit-test bytes, NOT a model: ' + name.encode())
            self.candidate[name] = Q.bound(path)
        base_pin = patch.object(Q, 'BASE_SHA', self.candidate['base']['sha256'])
        base_pin.start()
        self.addCleanup(base_pin.stop)
        self.runtime_server = self.directory / 'unit-server'
        self.runtime_server.write_bytes(b'UNIT identity fixture; not a server; never executed\n')
        self.runtime_server.chmod(0o700)

    def make_dev_sources(self):
        """Invent membership metadata, not prompts, grades or frozen DEV contents."""
        directory = self.directory / 'data/qwen-specialization-003'
        directory.mkdir(parents=True)
        domain = [dict(id=f'unit-domain-{index:03}', group='domain') for index in range(1, 13)]
        behavior = [dict(id=f'beh-{prefix}-dev-{index:03}', group=group)
                    for prefix, group in (('unc', 'uncertainty'), ('ret', 'retention'))
                    for index in range(1, 13)]
        rubrics = [dict(row, family=('ans.unit' if row['id'] in ('beh-unc-dev-002', 'beh-unc-dev-006')
                                    else 'unc.unit' if row['group'] == 'uncertainty' else 'ret.unit'))
                   for row in behavior]
        documents = {'domain-dev-scenarios-003.json': {'results': domain},
                     'behavior-dev-scenarios.json': {'results': behavior},
                     'behavior-dev-rubric.json': {'items': rubrics}}
        pins = {name: json_file(directory / name, document)['sha256'] for name, document in documents.items()}
        dev_pins = patch.dict(Q.DEV_PINS, pins, clear=True)
        dev_pins.start()
        self.addCleanup(dev_pins.stop)
        self.dev_sources = Q.dev_source_bindings(self.directory)

    def make_serving_specifications(self):
        """Synthetic public test protocol; render token IDs are not tokenizer evidence."""
        def turn(text):
            return dict(user='UNIT TEST: ' + text, contains=['응답'], not_contains=['금지문구'], hangul=True)

        suite = dict(oai_parity_session='single', sessions=[
            dict(id='single', turns=[turn('출장 정산에 필요한 서류를 알려 주세요.')]),
            dict(id='seq-1', turns=[turn('회의록 항목 다섯 가지를 알려 주세요.')]),
            dict(id='seq-2', turns=[turn('공식 답변의 맺음말을 한 문장으로 써 주세요.')]),
            dict(id='history-correction', turns=[
                turn('가상 인물 박서연의 출장은 목요일 부산입니다.'),
                turn('금요일 세종으로 정정합니다.'),
                turn('이름, 출장일, 목적지를 다시 알려 주세요.')]),
            dict(id='new-session-isolation', turns=[turn('제 이름과 출장일과 목적지를 아시나요?')]),
            dict(id='clear-marker-isolation', turns=[
                turn('가상의 직원 번호는 UNIT-731입니다.'), '__CLEAR__', turn('제 직원 번호를 아시나요?')]),
            dict(id='seq-3', turns=[turn('1부터 9까지 홀수만 쉼표로 구분해 주세요.')]),
        ])
        render = dict(messages=[{'role': 'user', 'content': 'UNIT render fixture'}],
                      prompt='<|im_start|>user\nUNIT render fixture<|im_end|>\n' + Q.SUFFIX,
                      token_ids=[101, 102, 103])
        pins = {name: json_file(self.sources / name, document)['sha256']
                for name, document in (('qwen-serving-suite.json', suite), ('official-render.json', render))}
        source_pins = patch.dict(Q.PINS, pins)
        source_pins.start()
        self.addCleanup(source_pins.stop)
        # Deliberately independent of CHUNKS/BOUNDS so policy changes cannot silently
        # rewrite the expected protocol along with the code under test.
        plan = dict(bounds=dict(child_seconds=180, rss_stop_bytes=6 * 1024**3, pressure_required=1,
                               ready_timeout=30, max_new=384, context=2048, threads=4),
                    chunks=[dict(ids=['single', 'seq-1', 'seq-2'], native_requests=3, oai_parity_requests=1),
                            dict(ids=['history-correction', 'new-session-isolation'], native_requests=4, oai_parity_requests=0),
                            dict(ids=['clear-marker-isolation', 'seq-3'], native_requests=3, oai_parity_requests=0)],
                    suite=Q.bound(self.sources / 'qwen-serving-suite.json'),
                    harness=Q.bound(self.sources / 'qwen-serving-harness.py'),
                    official_render=Q.bound(self.sources / 'official-render.json'))
        self.plan = self.directory / 'serving-qualification-plan.json'
        json_file(self.plan, plan)

    def selection_fixture(self):
        return selection_fixture(self.selector, self.dev_sources, self.directory)

    def chunk(self, index):
        h, c = self.harness, self.candidate
        binding = dict(host='127.0.0.1', port=54123, context=2048, max_new=384, threads=4, batch=256,
                       ubatch=128, n_gpu_layers=0, no_repack=True, no_context_shift=True, system_prompt=None,
                       decoding={'temperature': 0, 'seed': 42}, template_kwargs={'enable_thinking': False}, cache_prompt=False)
        record = dict(status='complete', live_members_after_shutdown=[], harness_pid=701, pgid=701, server_pid=702, server_exit_code=0,
                      arms=['adapter'], ids=sorted(Q.CHUNKS[index]), model_file=c['base']['path'],
                      model_sha256=c['base']['sha256'], lora_file=c['adapter']['path'], lora_sha256=c['adapter']['sha256'],
                      suite_sha256=Q.PINS['qwen-serving-suite.json'], binding=binding,
                      argv=h.server_argv(Path(c['base']['path']).resolve(), 54123, 2048, 4, Path(c['adapter']['path']).resolve()),
                      env=dict(DYLD_LIBRARY_PATH=str(h.SERVER_DIR), LLAMA_ARG_DEVICE='none', LLAMA_ARG_N_GPU_LAYERS='0', OMP_NUM_THREADS='4'),
                      phases=[{'pressure': 1}],
                      server={'lora_adapters': [{'id': 0, 'scale': 1, 'path': c['adapter']['path']}]},
                      arm_switches=[{'arm': 'adapter', 'scale': 1}],
                      render_parity=dict(text_equal=True, token_ids_equal=True, no_think_suffix=True,
                                         served_prompt=self.official['prompt'], served_token_ids=self.official['token_ids'],
                                         sample_messages=self.official['messages']),
                      timeouts=dict(duration=180, ready=30, request_derived=True, shutdown_grace=10,
                                    request=34, startup=1, window_needed=177), ready_seconds=4,
                      memory=dict(samples=2, ceiling_bytes=Q.BOUNDS['rss_stop_bytes'], under_ceiling=True, group_rss_bytes=1024),
                      requests=[])
        specs, parity = h.expand(self.suite, set(Q.CHUNKS[index]))
        count = len(specs) + bool(parity)
        timeout = h.derive_request_timeout(180, 30, 10, count, 1)
        record['timeouts'].update(request=timeout, window_needed=h.budget(30, timeout, 10, count, 1))
        histories = {}
        for spec in specs:
            history = [] if spec['reset'] else histories.get(spec['session'], [])
            messages = history + [('user', spec['user'])]
            output = 'UNIT FIXTURE 응답 ' + spec['session'] + str(spec['turn'])
            prompt = ''.join(f'<|im_start|>{role}\n{text}<|im_end|>\n' for role, text in messages) + Q.SUFFIX
            tokens = [1, 2, 3]
            row = dict(spec, kind='native', arm='adapter', server_pid=702, http_status=200,
                       raw_content=output, output=output, wall_seconds=1, timings={'predicted_per_second': 2},
                       pressure=1, group_rss_bytes=1024, history_turns=len(history) // 2,
                       prompt=prompt, rendered_prompt_sha256=Q.text_digest(prompt), prompt_token_ids=tokens,
                       prompt_tokens=len(tokens), fits_context=True, prompt_token_ids_sha256=Q.text_digest(json.dumps(tokens)))
            row['surface'] = h.surface(spec, output)
            record['requests'].append(row)
            histories[spec['session']] = messages + [('assistant', output)]
        if parity:
            row = copy.deepcopy(record['requests'][0])
            row.update(kind='oai', raw_content_equal=True, generated_text_equal=True, no_think_prefill_echoed=False)
            record['requests'].append(row)
        # Match the real harness's resolved artifact paths, and prove the complete
        # baseline is admissible before any rejection test mutates its evidence.
        self.validate(record, index)
        return record

    def validate(self, record, index=0, guard=None):
        return Q.validate_chunk(record, guard or guard_fixture(), index, self.candidate,
                                self.harness, self.suite, self.official)

    def test_http_success_codes_are_not_fabricated_as_200(self):
        source, _ = Q.instrument_helper((ROOT / 'scripts/base_dialogue_runner.py').read_text())
        path = self.directory / 'instrumented_http.py'
        path.write_text(source)
        helper = Q.module(path, 'unit_instrumented_http')
        for status in (200, 201, 204, 206):
            response = io.BytesIO(b'{"content":"actual JSON body"}')
            response.status = status
            with self.subTest(status=status), patch.object(helper.urllib.request, 'urlopen', return_value=response):
                if status == 200:
                    self.assertEqual(helper.request('http://127.0.0.1:1', '/completion')['content'], 'actual JSON body')
                else:
                    with self.assertRaisesRegex(RuntimeError, 'Expected HTTP200'):
                        helper.request('http://127.0.0.1:1', '/completion')

    def test_relocation_uses_explicit_runtime_and_preserves_case_expansion(self):
        argv = self.harness.server_argv(Path('/model'), 54123, 2048, 4, Path('/adapter'))
        self.assertEqual(argv[0], str(self.directory / 'native runtime/llama-server'))
        self.assertEqual(argv[argv.index('-ngl') + 1], '0')
        specs, parity = self.harness.expand(self.suite, set(Q.CHUNKS[2]))
        self.assertEqual([(s['session'], s['turn'], s['reset']) for s in specs],
                         [('clear-marker-isolation', 0, False), ('clear-marker-isolation', 2, True), ('seq-3', 0, False)])
        self.assertIsNone(parity)
        self.assertEqual(self.harness.derive_request_timeout(180, 30, 10, 4, 1), 34)
        self.assertIsNone(self.harness.derive_request_timeout(180, 30, 10, 5, 1))

    def test_artifact_relocation_requires_same_bytes_not_old_path(self):
        original = self.directory / 'old'
        original.write_text('authenticated bytes')
        item = Q.bound(original)
        moved = self.directory / 'moved'
        original.rename(moved)
        relocated = {**item, 'path': str(moved)}
        self.assertEqual(Q.check(relocated), moved)
        with self.assertRaises(ValueError):
            Q.check(item)
        moved.write_text('replacement bytes')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            Q.check(relocated)

    def test_preparation_authenticates_external_anchor_before_runtime(self):
        directory = self.directory / 'prepared'
        directory.mkdir()
        path = directory / 'preparation.json'
        json_file(path, {'schema': 'trid-serving-preparation-v1'})
        sha = Q.digest(path)
        path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            Q.prepared(directory, sha)

    def test_prepare_is_model_free_and_rehashed_algorithm_changes_still_fail(self):
        library = self.directory / 'unit-library.dylib'
        library.write_bytes(b'unit dependency bytes; never loaded')
        # Identity-only prepare cannot claim this executable is a working server.
        # No run command is called: this exercises filesystem/authentication only.
        runtime = dict(schema='trid-serving-runtime-v1', commit='8ca4060', cpu_only=True,
                       interpreter=Q.bound(sys.executable), server=Q.bound(self.runtime_server),
                       helper=Q.bound(ROOT / 'scripts/base_dialogue_runner.py'), libraries=[Q.bound(library)])
        runtime_path = self.directory / 'runtime.json'
        json_file(runtime_path, runtime)
        plan = self.plan
        out = self.directory / 'prepared'
        args = SimpleNamespace(repo=self.directory, sources=self.sources, plan=plan, plan_sha256=Q.digest(plan),
                               runtime=runtime_path, runtime_sha256=Q.digest(runtime_path), out=out)
        result = Q.prepare(args)
        self.assertEqual(result['state'], 'PREPARED_NO_MODEL_EXECUTION')
        receipt = out / 'preparation.json'
        Q.prepared(out, Q.digest(receipt))
        with self.assertRaises(FileExistsError):
            Q.prepare(args)
        derivative = out / 'serving_harness.py'
        derivative.write_text(derivative.read_text().replace("'temperature': 0", "'temperature': 1"))
        result['files']['serving_harness.py'] = Q.bound(derivative)
        json_file(receipt, result)
        with self.assertRaisesRegex(ValueError, 'Unapproved derivative'):
            Q.prepared(out, Q.digest(receipt))

    def test_changed_runtime_library_rejected_before_preparation_writes(self):
        library = self.directory / 'unit-library.dylib'
        library.write_bytes(b'original dependency bytes')
        runtime = dict(schema='trid-serving-runtime-v1', commit='8ca4060', cpu_only=True,
                       interpreter=Q.bound(sys.executable), server=Q.bound(self.runtime_server),
                       helper=Q.bound(ROOT / 'scripts/base_dialogue_runner.py'), libraries=[Q.bound(library)])
        library.write_bytes(b'replaced dependency bytes')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            Q.validate_runtime(runtime)

    def test_same_binary_different_venv_path_is_not_bound_interpreter(self):
        alias = self.directory / 'different-venv-python'
        alias.symlink_to(sys.executable)
        library = self.directory / 'unit-library.dylib'
        library.write_bytes(b'unit dependency bytes; never loaded')
        runtime = dict(schema='trid-serving-runtime-v1', commit='8ca4060', cpu_only=True,
                       interpreter=Q.bound(alias), server=Q.bound(self.runtime_server),
                       helper=Q.bound(ROOT / 'scripts/base_dialogue_runner.py'), libraries=[Q.bound(library)])
        self.assertEqual(alias.resolve(), Path(sys.executable).resolve())
        with self.assertRaisesRegex(ValueError, 'exact bound interpreter'):
            Q.validate_runtime(runtime)

    def test_supervisor_receipt_must_bind_actual_child_and_all_chunk_arguments(self):
        preparation = {'runtime': {'interpreter': {'path': '/reviewed venv/bin/python'}},
                       'files': {name: {'path': str(self.directory / name)} for name in
                                 ('serving_harness.py', 'qwen-serving-suite.json', 'official-render.json')}}
        native, log = self.directory / 'native.json', self.directory / 'server.log'
        # Independent expected command, not output fed back from child_argv().
        argv = ['/reviewed venv/bin/python', '-B', str(self.directory / 'serving_harness.py'),
                '--model-file', self.candidate['base']['path'], '--expected-sha256', self.candidate['base']['sha256'],
                '--lora', self.candidate['adapter']['path'], '--lora-sha256', self.candidate['adapter']['sha256'],
                '--arms', 'adapter', '--suite', str(self.directory / 'qwen-serving-suite.json'),
                '--official-render', str(self.directory / 'official-render.json'),
                '--receipt', str(native), '--server-log', str(log), '--port', '54123',
                '--context', '2048', '--max-new', '384', '--threads', '4', '--ready-timeout', '30',
                '--duration', '180', '--ids', 'history-correction', 'new-session-isolation']
        Q.validate_launch({'argv': argv}, preparation, self.candidate, native, log, 1, 54123)
        for index, bad in [(0, '/different-venv/bin/python'), (2, '/old/tmp/harness.py'),
                           (argv.index('--lora') + 1, '/unselected-adapter.gguf'),
                           (argv.index('--duration') + 1, '900'),
                           (argv.index('--receipt') + 1, '/different/receipt.json'),
                           (argv.index('--ids') + 1, 'single')]:
            changed = list(argv)
            changed[index] = bad
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, 'execution binding'):
                Q.validate_launch({'argv': changed}, preparation, self.candidate, native, log, 1, 54123)
        with self.assertRaisesRegex(ValueError, 'execution binding'):
            Q.validate_launch({}, preparation, self.candidate, native, log, 1, 54123)

    def test_duplicate_json_keys_and_nonfinite_are_rejected(self):
        path = self.directory / 'ambiguous.json'
        for text in ('{"selected":"A","selected":"B"}', '{"x":NaN}'):
            path.write_text(text)
            with self.subTest(text=text), self.assertRaises(ValueError):
                Q.load(path)

    def test_changed_plan_chunk_or_bound_rejected(self):
        plan = Q.load(self.plan)
        Q.validate_plan(plan)
        for mutation in ('duration', 'order', 'requests'):
            changed = copy.deepcopy(plan)
            if mutation == 'duration':
                changed['bounds']['child_seconds'] = 900
            elif mutation == 'order':
                changed['chunks'].reverse()
            else:
                changed['chunks'][0]['oai_parity_requests'] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                Q.validate_plan(changed)

    def test_selected_identity_replays_actual_frozen_calculator(self):
        candidate, docs = self.selection_fixture()
        self.assertEqual(Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)['label'], 'A')
        docs['selection']['selected'] = 'B'
        candidate['selected_label'] = 'B'
        with self.assertRaisesRegex(ValueError, 'frozen calculator'):
            Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)

    def test_candidate_file_bindings_authenticate_every_synthetic_artifact(self):
        for name in ('selection_input', 'selection', 'pipeline_outcome', 'full_training',
                     'training', 'weights', 'config', 'conversion', 'base', 'adapter'):
            candidate, _ = self.selection_fixture()
            self.assertEqual(Q.validate_candidate(candidate, self.selector, self.dev_sources)['label'], 'A')
            path = Path(candidate[name]['path'])
            path.write_bytes(path.read_bytes() + b'\nUNIT TAMPER')
            with self.subTest(artifact=name), self.assertRaisesRegex(ValueError, 'digest mismatch'):
                Q.validate_candidate(candidate, self.selector, self.dev_sources)

    def test_selected_checkpoint_and_conversion_ancestry_remain_bound(self):
        for mutation in ('checkpoint-weight', 'training-base', 'conversion-weight', 'conversion-base'):
            candidate, docs = self.selection_fixture()
            Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)
            if mutation == 'checkpoint-weight':
                docs['full_training']['checkpoints'][0]['adapter_weights_sha256'] = '0' * 64
            elif mutation == 'training-base':
                docs['training']['base_weights']['gguf_sha256'] = '0' * 64
            elif mutation == 'conversion-weight':
                docs['conversion']['checkpoint']['weights'] = {'path': '/unit/unrelated', 'sha256': '0' * 64}
            else:
                docs['conversion']['source_gguf'] = {'path': '/unit/unrelated', 'sha256': '0' * 64}
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                    ValueError, 'Checkpoint not retained|Wrong research base|Conversion binding mismatch'):
                Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)

    def test_no_candidate_and_unselected_artifact_never_admit(self):
        candidate, docs = self.selection_fixture()
        candidate['selected_label'] = 'NO_CANDIDATE'
        with self.assertRaisesRegex(ValueError, 'unselected'):
            Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)
        candidate, docs = self.selection_fixture()
        candidate['adapter'] = {**candidate['adapter'], 'sha256': 'f' * 64}
        with self.assertRaisesRegex(ValueError, 'artifact bindings differ'):
            Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)

    def test_failed_or_partial_candidate_evidence_rejected(self):
        for document, key, bad in [('pipeline_outcome', 'state', 'FAILED_NO_RETRY'),
                                   ('pipeline_outcome', 'candidate_turns', 137),
                                   ('pipeline_outcome', 'baseline_regenerated', True),
                                   ('full_training', 'status', 'interrupted'),
                                   ('training', 'steps', 16), ('conversion', 'status', 'failed'),
                                   ('conversion', 'returncode', 1), ('conversion', 'converter_sha256', '0' * 64)]:
            candidate, docs = self.selection_fixture()
            docs[document][key] = bad
            with self.subTest(document=document, key=key), self.assertRaises(ValueError):
                Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)

    def test_real_calculator_no_improvement_cannot_admit(self):
        candidate, docs = self.selection_fixture()
        for grades in docs['selection_input']['grades']['candidates'].values():
            for sid, grade in grades.items():
                if docs['selection_input']['scenarios'][sid]['group'] == 'domain':
                    grade['score'] = 1
        docs['selection'] = self.selector.select(docs['selection_input'])
        self.assertEqual(docs['selection']['selected'], 'NO_CANDIDATE')
        with self.assertRaisesRegex(ValueError, 'unselected'):
            Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)

    def test_baseline_grades_or_selection_evidence_cannot_change(self):
        candidate, docs = self.selection_fixture()
        sid = next(sid for sid, row in docs['selection_input']['scenarios'].items() if row['group'] == 'domain')
        docs['selection_input']['grades']['base'][sid]['score'] = 2
        with self.assertRaisesRegex(ValueError, 'frozen calculator'):
            Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)

    def test_fixed_population_keeps_both_answerable_controls(self):
        candidate, docs = self.selection_fixture()
        result = Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)
        self.assertEqual(result['label'], 'A')
        for label in 'ABC':
            gate = docs['selection']['per_checkpoint'][label]['gates']['answerable_controls_kept']
            self.assertEqual(set(gate['answerable_valid_ids']), {'beh-unc-dev-002', 'beh-unc-dev-006'})

    def test_removed_relabelled_or_replaced_answerable_controls_are_rejected(self):
        for mutation in ('remove', 'replace-id', 'relabel-family', 'swap-groups'):
            candidate, docs = self.selection_fixture()
            Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)
            payload = docs['selection_input']
            sid = 'beh-unc-dev-002'
            if mutation in ('remove', 'replace-id'):
                collections = [payload['scenarios'], payload['baseline']['items'], payload['grades']['base'],
                               *(c['items'] for c in payload['checkpoints']),
                               *payload['grades']['candidates'].values()]
                for mapping in collections:
                    value = mapping.pop(sid)
                    if mutation == 'replace-id':
                        mapping['invented-control'] = value
            elif mutation == 'relabel-family':
                for row in payload['scenarios'].values():
                    if row.get('family', '').startswith('ans.'):
                        row['family'] = 'unc.unit'
                for grades in payload['grades']['candidates'].values():
                    for grade in grades.values():
                        grade.pop('refused_or_nonanswer', None)
            else:
                payload['scenarios'][sid], payload['scenarios']['beh-ret-dev-001'] = (
                    payload['scenarios']['beh-ret-dev-001'], payload['scenarios'][sid])
                for grades in payload['grades']['candidates'].values():
                    grades[sid].pop('refused_or_nonanswer')
                    grades['beh-ret-dev-001']['refused_or_nonanswer'] = False
            # The calculator alone accepts the altered population. Admission must
            # reject even a freshly replayed result, not just a stale result hash.
            docs['selection'] = self.selector.select(payload)
            self.assertEqual(docs['selection']['selected'], 'A')
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'frozen DEV population'):
                Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)

    def test_relocated_frozen_dev_source_requires_exact_pinned_bytes(self):
        bindings = copy.deepcopy(self.dev_sources)
        name = 'behavior-dev-rubric.json'
        relocated = self.directory / name
        relocated.write_bytes(Path(bindings[name]['path']).read_bytes())
        bindings[name]['path'] = str(relocated)
        self.assertEqual(Q.frozen_dev_population(bindings), Q.frozen_dev_population(self.dev_sources))
        changed = Q.load(relocated)
        next(row for row in changed['items'] if row['id'] == 'beh-unc-dev-002')['family'] = 'unc.unit'
        relocated.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            Q.frozen_dev_population(bindings)
        bindings[name] = Q.bound(relocated)
        with self.assertRaises(ValueError):
            Q.frozen_dev_population(bindings)

    def test_exact_three_chunks_yield_ten_native_and_one_oai(self):
        evidence = [row for i in range(3) for row in self.validate(self.chunk(i), i)]
        self.assertEqual(len(evidence), 11)
        self.assertEqual(sum(':native:' in row['key'] for row in evidence), 10)
        self.assertEqual(sum(':oai:' in row['key'] for row in evidence), 1)

    def test_native_command_uses_resolved_artifacts_even_with_manifest_aliases(self):
        for name in ('base', 'adapter'):
            alias = self.directory / (name + '-alias.gguf')
            alias.symlink_to(self.candidate[name]['path'])
            self.candidate[name] = Q.bound(alias)
        record = self.chunk(0)  # Validates the unchanged receipt before mutation.
        for flag, name in (('-m', 'base'), ('--lora', 'adapter')):
            index = record['argv'].index(flag) + 1
            self.assertEqual(record['argv'][index], str(Path(self.candidate[name]['path']).resolve()))
            changed = copy.deepcopy(record)
            changed['argv'][index] = self.candidate[name]['path']
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                self.validate(changed)

    def test_exit_zero_does_not_override_resource_or_reap_failure(self):
        for key, bad in [('pressure', [{'value': 2}]), ('members_before_reap', [{'status': 'S'}]),
                         ('hard_alarm_fired', True), ('cleanup_errors', ['failed']), ('elapsed_seconds', 181),
                         ('group_samples', [{'members': [{'pid': 701, 'pgid': 701, 'rss_bytes': 6 * 1024**3 + 1}]}])]:
            guard = guard_fixture()
            guard[key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(self.chunk(0), guard=guard)

    def test_guard_samples_cannot_hide_unknown_negative_or_foreign_memory(self):
        for member in ({'pid': 701, 'pgid': 701, 'rss_bytes': -1},
                       {'pid': 701, 'pgid': 701, 'rss_bytes': float('nan')},
                       {'pid': 701, 'pgid': 701, 'rss_bytes': True},
                       {'pid': 701, 'pgid': 999, 'rss_bytes': 1024},
                       {'pid': 999, 'pgid': 701, 'rss_bytes': 1024}):
            guard = guard_fixture()
            guard['group_samples'] = [{'members': [member]}]
            with self.subTest(member=member), self.assertRaises(ValueError):
                Q.validate_guard(guard)
        guard = guard_fixture()
        guard['group_samples'] = [{'members': []}]
        with self.assertRaisesRegex(ValueError, 'No live owned harness'):
            Q.validate_guard(guard)

    def test_missing_request_pid_switch_and_context_overflow_rejected(self):
        for mutation in ('missing', 'pid', 'context', 'empty'):
            record = self.chunk(0)
            if mutation == 'missing':
                record['requests'].pop()
            elif mutation == 'pid':
                record['requests'][1]['server_pid'] += 1
            elif mutation == 'context':
                record['requests'][0]['prompt_token_ids'] = [1] * 2048
            else:
                record['requests'][0].update(output='', raw_content='  ')
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(record)

    def test_exact_raw_oai_parity_not_stripped_or_flag_only(self):
        record = self.chunk(0)
        record['requests'][-1]['raw_content'] += '\n'
        with self.assertRaisesRegex(ValueError, 'Raw native/OAI parity'):
            self.validate(record)
        record = self.chunk(0)
        record['requests'][-1]['raw_content'] = '<think>\n\n</think>\n\n' + record['requests'][-1]['output']
        record['requests'][-1]['output'] = record['requests'][-1]['raw_content']
        with self.assertRaisesRegex(ValueError, 'Thinking-prefill'):
            self.validate(record)

    def test_history_correction_and_clear_validate_actual_prompt_not_count_flag(self):
        record = self.chunk(1)
        row = record['requests'][2]
        row['prompt'] = row['prompt'].replace(record['requests'][1]['output'], 'different assistant answer')
        row['rendered_prompt_sha256'] = Q.text_digest(row['prompt'])
        with self.assertRaisesRegex(ValueError, 'Prompt/history/reset'):
            self.validate(record, 1)
        record = self.chunk(2)
        row = record['requests'][1]
        row['prompt'] = record['requests'][0]['prompt'] + row['prompt']
        row['rendered_prompt_sha256'] = Q.text_digest(row['prompt'])
        with self.assertRaisesRegex(ValueError, 'Prompt/history/reset'):
            self.validate(record, 2)

    def test_official_parity_and_active_lora_must_be_actual(self):
        record = self.chunk(0)
        record['render_parity']['served_token_ids'] = [42]
        with self.assertRaisesRegex(ValueError, 'Official render/token'):
            self.validate(record)
        record = self.chunk(0)
        record['server']['lora_adapters'][0]['scale'] = 0
        with self.assertRaisesRegex(ValueError, 'scale1'):
            self.validate(record)

    def test_selection_without_independent_review_is_rejected_even_if_calculator_passes(self):
        candidate, docs = self.selection_fixture()
        del docs['selection_input']['pins']['independent_selection']
        docs['selection'] = self.selector.select(docs['selection_input'])
        with self.assertRaisesRegex(ValueError, 'Independent DEV'):
            Q.validate_candidate_records(candidate, docs, self.selector, self.dev_sources)

    def test_surface_flags_cannot_qualify_content_without_independent_review(self):
        evidence = [row for i in range(3) for row in self.validate(self.chunk(i), i)]
        measurement = dict(schema='trid-serving-measurement-v1', state='MEASURED_AWAITING_CONTENT_REVIEW',
                           chunks=[{}, {}, {}], requests=evidence)
        review = dict(schema='trid-serving-content-review-v1', measurement_sha256='a' * 64,
                      independent=True, reviewer='unit-test reviewer, not real model qualification',
                      decisions=[dict(key=row['key'], output_sha256=row['output_sha256'], **{'pass': True},
                                      assertion=row['assertion'], evidence='UNIT FIXTURE') for row in evidence])
        Q.validate_review(measurement, 'a' * 64, review)
        for mutation in ('fail', 'duplicate', 'different-output', 'no-review', 'measurement'):
            changed = copy.deepcopy(review)
            if mutation == 'fail':
                changed['decisions'][0]['pass'] = False
            elif mutation == 'duplicate':
                changed['decisions'][-1] = changed['decisions'][0]
            elif mutation == 'different-output':
                changed['decisions'][0]['output_sha256'] = '0' * 64
            elif mutation == 'no-review':
                changed['independent'] = False
            else:
                changed['measurement_sha256'] = 'b' * 64
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                Q.validate_review(measurement, 'a' * 64, changed)


if __name__ == '__main__':
    unittest.main()
