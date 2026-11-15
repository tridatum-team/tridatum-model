"""Consumer-visible fresh CUDA guards; no Torch/model/cloud execution."""
import json
from pathlib import Path
import sys
import tempfile
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


if __name__ == '__main__':
    unittest.main()
