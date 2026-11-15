"""Consumer-visible fresh CUDA guards; no Torch/model/cloud execution."""
import copy
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
import fresh_eight_experiment as E
import fresh_eight_worker as F
import grounded_hf_worker as W
import grounded_hf_lifecycle as L


def history():
    inventory = {'torch.float32@cuda:0': dict(tensors=2, elements=8, bytes=32)}
    return [dict(step=s, loss=1.0, microbatch_losses=[0.25] * 4,
        applied_lr=1e-5 * s / 7, next_lr=1e-5 * (s + 1) / 7,
        numerical_health=dict(gradient_norm=1.0, gradients=inventory, optimizer=inventory,
            changed=True, changed_tensor='lora_B.weight', resident_moments=s > 1),
        resident_backward_checks=[] if s == 1 else [dict(resident_moments=True, gradient_norm=1.0)] * 4)
        for s in (1, 2)]


class CapacityAcceptanceTests(unittest.TestCase):
    def fixture(self):
        plan = dict(data={'policy': {'sha256': 'policy'}}, workload={'source_manifest': {'sha256': 'sources'}})
        source = dict(path='worker.py', bytes=1, sha256='source-sha')
        plan.update(identity={'run_id': E.RUN_ID}, sources={'worker.py': 'source-sha'}, bundle={'files': [source]})
        result = dict(schema='trid-fresh-eight-cuda-capacity-v1', status='complete',
            policy=plan['data']['policy'], source_manifest=plan['workload']['source_manifest'],
            runtime={'device': 'cuda'}, train_rows=600, dropped=0, sequence_tokens=3072,
            supervised_tokens=768, masked_prompt_tokens=2304, optimizer_updates=2,
            microbatches_per_update=4, includes_eos=True, reload_exact=True, released_before_reload=True,
            scratch_adapter_never_nominated=True, reload_loss=1.0, history=history())
        inventory = dict(base={'torch.bfloat16@cuda:0': dict(tensors=1, elements=8, bytes=16)},
                         adapter={'torch.float32@cuda:0': dict(tensors=2, elements=8, bytes=32)})
        result.update(dtype_inventory=inventory, reload_dtype_inventory=inventory)
        result['actual_source_files'] = [source]
        result['runtime'].update(backend='transformers-cuda', run_id=E.RUN_ID)
        return plan, result

    def validate(self, change=None):
        plan, result = self.fixture()
        if change:
            change(result)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'capacity').mkdir()
            (root / 'plan.json').write_text(json.dumps(plan))
            result['runtime']['plan_sha256'] = E.pin(root / 'plan.json')['sha256']
            (root / 'capacity/capacity-worker.json').write_text(json.dumps(result))
            return E.validate_capacity(root, plan)

    def test_exact_two_update_full_envelope_evidence_is_accepted(self):
        self.assertEqual(self.validate()['optimizer_updates'], 2)

    def test_short_reload_and_missing_resident_backward_are_rejected(self):
        for change in [lambda r: r.update(supervised_tokens=32),
                       lambda r: r.update(reload_exact=False),
                       lambda r: r.update(released_before_reload=False),
                       lambda r: r['history'][1].update(resident_backward_checks=[]),
                       lambda r: r['history'][1]['resident_backward_checks'][0].update(resident_moments=False)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.validate(change)

    def test_failed_numerics_never_become_capacity_acceptance(self):
        for key, value in [('gradient_norm', 0), ('gradient_norm', float('nan')), ('changed', False),
                           ('gradients', {}), ('optimizer', {})]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.validate(lambda r: r['history'][0]['numerical_health'].update({key: value}))
        with self.assertRaises(ValueError):
            self.validate(lambda r: r.update(reload_loss=float('nan')))
        with self.assertRaisesRegex(ValueError, 'dtype inventory'):
            self.validate(lambda r: r['dtype_inventory']['adapter'].update(
                {'torch.bfloat16@cuda:0': dict(tensors=1, elements=8, bytes=16)}))

    def test_scheduler_must_record_applied_not_next_lr(self):
        with self.assertRaisesRegex(ValueError, 'Scheduler'):
            self.validate(lambda r: r['history'][0].update(applied_lr=r['history'][0]['next_lr']))


class FreshBoundaryTests(unittest.TestCase):
    def test_portable_endpoint_paths_preserve_physical_row_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'train').mkdir()
            checkpoints = []
            coverage = [{'file': '/sealed/code/train-all.jsonl', 'source_line': 17, 'id': 'physical-row'}]
            for step in E.CHECKPOINT_STEPS:
                folder = root / f'train/checkpoints/step-{step:06d}'
                folder.mkdir(parents=True)
                entry = dict(step=step, path=str(folder / 'adapter'))
                checkpoints.append(entry)
                (folder / 'training.json').write_text(json.dumps(dict(checkpoint=entry, coverage=coverage)))
            payload = dict(checkpoints=checkpoints, last_step=dict(step=150, path=str(root / 'train/adapter')),
                           coverage=coverage)
            for name in ('training.json', 'checkpoints.json'):
                (root / 'train' / name).write_text(json.dumps(payload))
            F.portable_checkpoint_paths(W, root)
            report = json.loads((root / 'train/training.json').read_text())
            ledger = json.loads((root / 'train/checkpoints.json').read_text())
            self.assertEqual(report, ledger)
            self.assertEqual(report['last_step']['path'], 'train/adapter')
            self.assertEqual(report['checkpoints'][-1]['path'], 'train/checkpoints/step-000150/adapter')
            self.assertEqual(report['coverage'], coverage)

    def test_forbidden_phases_cannot_reach_native_sandbox(self):
        plan = dict(schema=E.SCHEMA, identity={'run_id': E.RUN_ID}, limits=dict(L.LIMITS,
            lifetime_seconds=7200, reserve_seconds=300, output_bytes=512 * 1024**2))
        gpu = dict(uuid='GPU-fixture', minor=0)
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text(json.dumps(plan))
            for phase in ('final_base', 'parent_selection', 'checkpoint-step-000050-dev', 'schema'):
                with self.subTest(phase=phase), self.assertRaises(RuntimeError):
                    W.stage_argv(plan, base, phase, 60, gpu, 1001, 1001)

    def test_acquisition_receipt_requires_exact_registered_file_metadata(self):
        registered = dict(path='model-00001-of-00001.safetensors', bytes=123, sha256='a' * 64,
                          url='https://huggingface.co/pinned-model')
        plan = dict(model={'files': [registered]})
        metadata = {key: registered[key] for key in ('path', 'bytes', 'sha256')}
        receipt = dict(status='authenticated', model_id=E.MODEL_ID, revision=E.MODEL_REVISION,
                       files=[metadata])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'model-acquisition.json'
            path.write_text(json.dumps(receipt))
            self.assertEqual(E.validate_acquisition(temp, plan), receipt)
            for field, value in [('path', 'different.safetensors'), ('bytes', 124), ('sha256', 'b' * 64)]:
                changed = dict(receipt, files=[dict(metadata, **{field: value})])
                path.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'base acquisition'):
                    E.validate_acquisition(temp, plan)
            path.write_text(json.dumps(dict(receipt, files=[dict(name=metadata['path'],
                bytes=metadata['bytes'], sha256=metadata['sha256'])])))
            with self.assertRaisesRegex(ValueError, 'base acquisition'):
                E.validate_acquisition(temp, plan)

    def test_fresh_egress_handoff_reserves_all_bootstrap_consumption(self):
        reservation = 16 * 1024**2
        for cap in (reservation + 1, 2 * 1024**3):
            plan = dict(schema=E.SCHEMA, network={'egress_byte_cap': cap})
            allowance = W.replacement_egress_allowance(plan)
            self.assertGreater(allowance, 0)
            self.assertEqual(allowance + reservation, cap)
        for cap in (-1, 0, reservation - 1, reservation):
            with self.subTest(cap=cap), self.assertRaisesRegex(RuntimeError, 'no post-bootstrap allowance'):
                W.replacement_egress_allowance(dict(schema=E.SCHEMA, network={'egress_byte_cap': cap}))
        self.assertEqual(W.replacement_egress_allowance(
            dict(schema='historical', network={'egress_byte_cap': reservation})), reservation)

    def test_failed_owned_cleanup_never_returns_acceptance(self):
        plan = dict(schema=E.SCHEMA, identity={'run_id': E.RUN_ID}, phase_seconds=E.PHASE_SECONDS)
        state = dict(LoadState='loaded', ActiveState='active', SubState='exited', Result='success',
            ExecMainStatus='0', InvocationID='invocation', Environment='TRID_OWNED_RUN=' + E.RUN_ID,
            MemoryMax=str(L.LIMITS['rss_bytes']), MemorySwapMax='0', DevicePolicy='closed', User='1001', Group='1001')
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'plan.json').write_text(json.dumps(plan))
            (base / 'release.json').write_text(json.dumps({'instance_id': 'i-owned'}))
            with patch.object(W, 'unit_state', side_effect=[{'LoadState': 'not-found'}, state]), \
                 patch.object(W, 'remaining', return_value=1000), patch.object(W, 'command'), \
                 patch.object(W, 'stage_argv', return_value=['registered']), \
                 patch.object(W, 'gpu_observation', side_effect=[{}, ({}, 0)]), \
                 patch.object(W, 'verify_gpu_observation'), patch.object(W, 'cgroup_empty', return_value=True), \
                 patch.object(W, 'stop_unit', side_effect=RuntimeError('owned child remains')):
                with self.assertRaisesRegex(RuntimeError, 'owned child remains'):
                    W.supervise_phase(plan, base, 'capacity', {}, 1001, 1001, 9999999999, [])

    def test_failed_evaluation_records_exact_phase_and_global_deadlines_without_extending_them(self):
        plan = dict(schema=E.SCHEMA, identity={'run_id': E.RUN_ID}, phase_seconds=E.PHASE_SECONDS)
        for available in (10000, 100):
            with self.subTest(available=available), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                W.save(base / 'plan.json', plan)
                attempt, units = {'phase': 'train', 'phase_deadline': 1}, []
                expected_seconds = min(E.PHASE_SECONDS['evaluate'], available - 15)
                with patch.object(W.time, 'time', return_value=1000), \
                     patch.object(W, 'unit_state', return_value={'LoadState': 'not-found'}), \
                     patch.object(W, 'remaining', side_effect=[available, RuntimeError('Absolute deadline exhausted')]), \
                     patch.object(W, 'gpu_observation', return_value={}), patch.object(W, 'verify_gpu_observation'), \
                     patch.object(W, 'stage_argv', return_value=['registered']), patch.object(W, 'command'), \
                     patch.object(W, 'stop_unit'):
                    with self.assertRaisesRegex(RuntimeError, 'Absolute deadline exhausted'):
                        W.supervise_phase(plan, base, 'evaluate', {}, 1001, 1001, 1000 + available, units,
                                          attempt=attempt)
                self.assertEqual(units, [])
                self.assertEqual(attempt, dict(phase='evaluate', unit=E.RUN_ID + '-evaluate',
                    work_deadline=1000 + available, started_at=1000, registered_seconds=3600,
                    effective_seconds=expected_seconds, phase_deadline=1000 + expected_seconds))

    def test_export_archive_overhead_is_inside_fresh_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            (base / 'output').mkdir()
            (base / 'output/runtime.log').write_text('')
            plan = dict(schema=E.SCHEMA, identity={'run_id': E.RUN_ID},
                limits={'output_bytes': 512 * 1024**2}, cost={'export_bytes': 10240},
                output_allowlist=['runtime.log'])
            (base / 'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(RuntimeError, 'overhead'):
                W.export_outputs(base, plan, {}, 9999999999)
            self.assertFalse((base / 'export.tar').exists())

    def test_supervisor_cannot_accept_a_different_worker_result(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'capacity').mkdir()
            plan = dict(identity={'run_id': E.RUN_ID}, data={'policy': {}}, workload={'source_manifest': {}})
            receipt = dict(schema='trid-fresh-eight-cuda-supervision-v1', phase='capacity', status='accepted',
                cleanup_verified=True, run_id=E.RUN_ID, policy={}, source_manifest={},
                worker_result={'path': 'runtime.json'}, unit={'state': 'complete', 'cleanup_verified': True})
            (root / 'capacity/supervisor-result.json').write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, 'different worker artifact'):
                E.validate_supervision(root, 'capacity', plan)

    def test_requested_greedy_cannot_hide_effective_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / 'evaluation'
            out.mkdir()
            (out / 'completion.json').write_text(json.dumps(dict(status='complete', rows_per_arm=200)))
            (out / 'run-manifest.json').write_text(json.dumps(dict(generation=E.GENERATION,
                grading_access=False, retries=False, resume=False, arm_order=['base', 'candidate'],
                selected_ids=[str(i) for i in range(200)])))
            (out / 'loaded-runtime.json').write_text(json.dumps(dict(
                generation_config=dict(do_sample=True, num_beams=1, max_new_tokens=768),
                generation_policy={})))
            with self.assertRaisesRegex(ValueError, 'Effective greedy decoder'):
                E.validate_evaluation(root, {})

    def test_partial_outputs_never_pass_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'mandatory artifacts'):
                E.validate_completion(temp, {})
        self.assertFalse(any('final-input' in name or 'checkpoint-selection' in name for name in E.output_members()))
        self.assertIn('evaluation/failure.json', E.output_members())


class FreshRecoveryTests(unittest.TestCase):
    def prepare(self, base):
        plan, capacity = CapacityAcceptanceTests().fixture()
        plan.update(schema=E.SCHEMA, model={'files': []},
                    limits={'output_bytes': 512 * 1024**2}, cost={'export_bytes': 512 * 1024**2},
                    output_allowlist=sorted(E.output_members()))
        plan['identity'].update(bucket='fixture', output_prefix='run/output/')
        plan['data'].update(manifest={'sha256': 'manifest'}, train={'sha256': 'rows'})
        (base / 'model').mkdir()
        (base / 'output').mkdir()
        (base / 'scratch/tmp/fresh-capacity').mkdir(parents=True)
        (base / 'scratch/cache').mkdir()
        W.save(base / 'plan.json', plan)
        capacity['runtime']['plan_sha256'] = W.digest(base / 'plan.json')
        report = dict(experimental_success=False, stages=[], run_id=E.RUN_ID,
                      plan_sha256=W.digest(base / 'plan.json'), instance_id='i-fixture')
        return plan, capacity, report

    def training(self, root, plan, capacity):
        inventory = capacity['dtype_inventory']['adapter']
        history_rows = []
        for step in range(1, 151):
            history_rows.append(dict(step=step, train_loss=1.0,
                applied_lr=1e-5 * (step / 7 if step <= 7 else
                    0.5 * (1 + math.cos(math.pi * (step - 8) / 143))),
                next_lr=1e-5 * ((step + 1) / 7 if step < 7 else
                    0.5 * (1 + math.cos(math.pi * (step - 7) / 143))),
                numerical_health=dict(gradient_norm=1.0, gradients=inventory, optimizer=inventory,
                    changed=True, changed_tensor='lora_B.weight', resident_moments=step > 1)))
        checkpoints = []
        for step in (*E.CHECKPOINT_STEPS, None):
            folder = root / ('train/adapter' if step is None else f'train/checkpoints/step-{step:06d}/adapter')
            folder.mkdir(parents=True)
            (folder / 'adapter_config.json').write_text('{}')
            (folder / 'adapter_model.safetensors').write_bytes(b'bounded fixture weights')
            row = dict(step=150 if step is None else step, path=folder.relative_to(root).as_posix(),
                adapter_config_sha256=W.digest(folder / 'adapter_config.json'),
                adapter_weights_sha256=W.digest(folder / 'adapter_model.safetensors'))
            if step is None:
                last = dict(row, equals_last_checkpoint=True)
            else:
                checkpoints.append(row)
        ledger = dict(run_status='complete', checkpoints=checkpoints, last_step=last,
                      selection={'candidate_step': 150})
        training = dict(status='complete', steps=150, train_rows=600, valid_rows=0,
            dropped_over_max_tokens=0, valid_dropped=0, coverage_complete=True, device='cuda',
            base_weights=dict(representation='hf-safetensors', dtype='torch.bfloat16'), init_adapter=None,
            manifest_sha256='manifest', dtype_inventory=capacity['dtype_inventory'],
            policy=plan['data']['policy'], runtime=capacity['runtime'],
            actual_source_files=capacity['actual_source_files'], history=history_rows,
            coverage=[dict(file='physical/train.jsonl', source_sha256='rows', source_line=i, id=str(i))
                      for i in range(1, 601)], checkpoints=checkpoints, last_step=last)
        W.save(root / 'train/training.json', training)
        W.save(root / 'train/checkpoints.json', ledger)
        W.save(root / 'train/training-worker.json', dict(schema='trid-fresh-eight-cuda-training-v1',
            status='complete', coverage_complete=True, policy=plan['data']['policy'],
            source_manifest=plan['workload']['source_manifest'], report=E.pin(root / 'train/training.json', root)))

    def client(self, objects):
        def put(*args):
            key = args[args.index('--key') + 1]
            if '--if-none-match' in args and key in objects:
                raise RuntimeError('Object already exists')
            objects[key] = Path(args[args.index('--body') + 1]).read_bytes()
        return put

    def test_interrupted_library_scratch_does_not_poison_registered_export(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            plan, _, report = self.prepare(base)
            (base / 'output/runtime.json').write_text('{"known":"artifact"}')
            child = subprocess.Popen([sys.executable, '-c',
                'import tempfile,pathlib,time; d=tempfile.TemporaryDirectory(); '
                'p=pathlib.Path(d.name)/"_remote_module_non_scriptable.py"; '
                'p.write_text("interrupted library scratch"); print(p,flush=True); time.sleep(60)'],
                env=dict(os.environ, TMPDIR=str(base / 'scratch/tmp'), HF_HOME=str(base / 'scratch/cache')),
                stdout=subprocess.PIPE, text=True)
            try:
                leftover = Path(child.stdout.readline().strip())
                self.assertTrue(leftover.is_file())
                with self.assertRaisesRegex(RuntimeError, 'owned process remains'):
                    W.cleanup_fresh_scratch(base, ['owned-running-unit'], 9999999999)
                child.kill()
                child.wait(timeout=10)
                self.assertTrue(leftover.is_file())
                export = W.export_outputs(base, plan, report, 9999999999)
                self.assertEqual([r['path'] for r in export['manifest']['files']], ['runtime.json'])
                W.cleanup_fresh_scratch(base, [], 9999999999)
                self.assertFalse((base / 'scratch').exists())
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                child.stdout.close()

    def test_unknown_file_inside_output_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            plan, _, report = self.prepare(base)
            path = base / 'output/tmp/_remote_module_non_scriptable.py'
            path.parent.mkdir()
            path.write_text('not registered')
            with self.assertRaisesRegex(RuntimeError, 'Unregistered output member'):
                W.export_outputs(base, plan, report, 9999999999)
            self.assertFalse((base / 'export.tar').exists())

    def test_evaluation_failure_retains_acknowledged_accepted_training_snapshot(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            plan, capacity, report = self.prepare(base)
            objects = {}
            def phase(plan, base, phase, *args, **kwargs):
                if phase == 'evaluate':
                    self.assertIn('export', report)
                    raise RuntimeError('evaluation interrupted')
                folder = base / 'output' / phase
                folder.mkdir(exist_ok=True)
                if phase == 'capacity':
                    W.save(folder / 'capacity-worker.json', capacity)
                else:
                    self.training(base / 'output', plan, capacity)
                return dict(phase=phase, state='complete', cleanup_verified=True, invocation='fixture',
                    run_id=E.RUN_ID, instance_id='i-fixture', plan_sha256=W.digest(base / 'plan.json'),
                    stopped_state={'LoadState': 'not-found'})
            try:
                with patch.object(W, 'validate_model'), patch.object(W, 'supervise_phase', side_effect=phase), \
                     patch.object(os, 'chown'):
                    with self.assertRaisesRegex(RuntimeError, 'evaluation interrupted'):
                        F.run_experiment(W, plan, base, report, self.client(objects), {},
                            SimpleNamespace(pw_uid=1001, pw_gid=1001), 9999999999, 9999999999, [], set(), [])
                payload = objects[plan['identity']['output_prefix'] + 'export.tar']
                self.assertEqual(report['export']['sha256'], W.hashlib.sha256(payload).hexdigest())
                with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                    manifest = json.load(archive.extractfile('export-manifest.json'))
                    self.assertFalse(manifest['status']['experimental_success'])
                    self.assertIn('train/supervisor-result.json', archive.getnames())
                    self.assertNotIn('candidate.json', archive.getnames())
                E.validate_training(base / 'output', plan)
                E.validate_supervision(base / 'output', 'train', plan)
                with self.assertRaisesRegex(ValueError, 'mandatory artifacts'):
                    E.validate_completion(base / 'output', plan)
                self.assertFalse(report['experimental_success'])
            finally:
                # Native acceptance intentionally freezes directories; restore only this fixture.
                for path in (base / 'output').rglob('*'):
                    if path.is_dir():
                        path.chmod(0o755)

    def test_failed_final_export_preserves_pin_and_successful_replacement_uses_one_key(self):
        for failure in ('inventory', 'upload'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                base = Path(temp)
                plan, _, report = self.prepare(base)
                (base / 'output/runtime.json').write_text('{"known":"artifact"}')
                objects = {}
                put = self.client(objects)
                W.publish_outputs(base, plan, report, put, 9999999999)
                prior = copy.deepcopy(report['export'])
                original = dict(objects)
                report['error'] = 'evaluation interrupted'
                if failure == 'inventory':
                    (base / 'output/unknown').write_text('not registered')
                    failing = put
                else:
                    def failing(*args):
                        raise RuntimeError('upload not acknowledged')
                with self.assertRaises(RuntimeError):
                    W.publish_outputs(base, plan, report, failing, 9999999999)
                self.assertEqual(report['export'], prior)
                self.assertEqual(objects, original)
                (base / 'output/unknown').unlink(missing_ok=True)
                W.publish_outputs(base, plan, report, put, 9999999999)
                self.assertEqual(len(objects), 1)
                self.assertNotEqual(report['export'], prior)
                self.assertEqual(sum(len(blob) for blob in objects.values()), report['export']['bytes'])
                self.assertLessEqual(report['export']['bytes'], plan['cost']['export_bytes'])
                with tarfile.open(base / 'export.tar') as archive:
                    self.assertNotIn('export', json.load(archive.extractfile('export-manifest.json'))['status'])


if __name__ == '__main__':
    unittest.main()
