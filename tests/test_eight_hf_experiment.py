"""Offline contract boundaries; fixtures never represent execution evidence."""
import copy
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/aws_eval'))
import eight_hf_experiment as E
import grounded_hf_lifecycle as L


def pin(name, sha='a' * 64):
    return dict(path=name, bytes=100, sha256=sha)


def fixture():
    identity = dict(run_id=E.RUN_ID, code_prefix=f'code/{E.RUN_ID}/',
                    output_prefix=f'output/{E.RUN_ID}/', bucket='synthetic')
    data = {k: pin(k + '.json') for k in ('manifest', 'preparation_manifest', 'train', 'dev')}
    data['preparation_manifest']['sha256'] = E.PREPARATION_SHA
    workload = dict(development_inputs=pin('dev-inputs.json'), development_case_ids=[f'd{i}' for i in range(8)],
        development_behaviors={f'd{i}': f'b{i // 2}' for i in range(8)},
        final_inputs=pin('private/final.json'), final_gate_metadata=pin('private/gate.json'),
        prompt_audit=pin('private/prompt-audit.json'),
        final_gate_amendment=pin('private/gate-amendment.json'),
        execution_amendment=pin('private/execution-amendment.json', E.EXECUTION_AMENDMENT_SHA),
        final_case_ids={k: [f'{k}{i}' for i in range(12)] for k in ('general', 'mois')},
        source_closure=pin('closure.json'), generation=copy.deepcopy(E.GENERATION),
        channel_keys=dict(archive=identity['output_prefix'] + 'dev-checkpoint.tar',
            receipt=identity['output_prefix'] + 'dev-checkpoint.json',
            decision=identity['code_prefix'] + 'checkpoint-selection.json',
            final_inputs=identity['code_prefix'] + 'final-inputs.json'))
    plan = dict(schema=E.SCHEMA, identity=identity, recipe=copy.deepcopy(E.RECIPE),
        phase_seconds=copy.deepcopy(E.PHASE_SECONDS), model=dict(id=E.MODEL_ID, revision=E.MODEL_REVISION),
        data=data, workload=workload, authorization=pin('authorization.json', E.AUTHORIZATION_SHA),
        sources={n: 'a' * 64 for n in L.SOURCES | E.SOURCE_NAMES})
    plan['output_allowlist'] = sorted(E.output_members(plan))
    bundled = {p['path']: p for p in [*data.values(), workload['development_inputs'],
               workload['source_closure'], plan['authorization']]}
    return plan, bundled


def selection(plan):
    hashes = {E.arm(s): str(i + 1) * 64 for i, s in enumerate(E.CHECKPOINT_STEPS)}
    base = dict(joint_pass_count=4, critical_errors=0, behavior_pass_counts={f'b{i}': 1 for i in range(4)})
    candidates = [dict(step=s, joint_pass_count=6, critical_errors=0,
        behavior_pass_counts=dict(b0=2, b1=2, b2=1, b3=1), teacher_forced_dev_loss=1.0)
        for s in E.CHECKPOINT_STEPS]
    decision = dict(schema='trid-eight-sft-selection-v1', run_id=E.RUN_ID, plan_sha256='a' * 64,
        dev_export_sha256='b' * 64, checkpoint_hashes=hashes,
        review_pins=[pin(f'review{i}.json', str(i + 1) * 64) for i in range(5)],
        aggregate=dict(base=base, candidates=candidates), selected_step=12)
    return decision, hashes


class EightAdmissionTests(unittest.TestCase):
    def test_predispatch_revocation_prevents_raw_lifecycle_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            L.require_no_dispatch_evidence(directory)
            (Path(directory) / 'parent-no-dispatch.json').write_text('{}')
            with self.assertRaises(FileExistsError):
                L.require_no_dispatch_evidence(directory)

    def test_historical_closed_source_bundle_imports_without_eight_modules(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            for name in L.SOURCES:
                target = Path(directory) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, target)
            script = (
                'import sys; sys.path.insert(0, sys.argv[1]); '
                'import grounded_hf_lifecycle as L; import grounded_hf_worker; '
                "assert L.success_field({'schema':'trid-grounded-hf-v1'}) == 'training_success'; "
                "assert L.success_field({'schema':L.FOLLOWUP_SCHEMA}) == 'inference_success'; "
                "assert 'eight_hf_experiment' not in sys.modules; "
                "assert 'eight_hf_worker' not in sys.modules")
            result = subprocess.run([sys.executable, '-I', '-c', script,
                str(Path(directory) / 'scripts/aws_eval')], capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_workload_accepts_metadata_only_dev_transport(self):
        plan, bundled = fixture()
        E.validate_workload(plan, bundled)
        self.assertEqual(L.success_field(plan), 'experimental_success')
        self.assertFalse(L.inference_mode(plan))
        self.assertLessEqual(E.DEV_EXPORT_MAX, 32 * 1024**2)
        self.assertFalse(any('/adapter/' in n for n in E.dev_output_members(plan)))
        self.assertEqual(sum(n.endswith('adapter_model.safetensors') for n in E.output_members(plan)), 4)
        self.assertEqual(E.RECIPE['context_tokens'], 3072)
        self.assertEqual(E.GENERATION['context_tokens'], 4096)
        self.assertEqual(E.PHASE_SECONDS['acquire'], 900)
        self.assertEqual(sum(E.PHASE_SECONDS.values()), 10500)

    def test_final_inputs_and_private_gate_cannot_enter_initial_bundle(self):
        for name in ('final_inputs', 'final_gate_metadata', 'final_gate_amendment', 'prompt_audit', 'execution_amendment'):
            plan, bundled = fixture()
            item = plan['workload'][name]
            bundled[item['path']] = item
            with self.assertRaises(RuntimeError):
                E.validate_workload(plan, bundled)

    def test_model_recipe_sources_and_approval_cannot_drift(self):
        mutations = [lambda p: p['model'].update(id='Qwen/Qwen3-14B'),
            lambda p: p['recipe'].update(steps=48),
            lambda p: p['authorization'].update(sha256='f' * 64),
            lambda p: p['sources'].update({'scripts/foreign.py': 'f' * 64}),
            lambda p: p['phase_seconds'].update(parent_selection=3600)]
        for mutate in mutations:
            plan, bundled = fixture()
            mutate(plan)
            with self.assertRaises(RuntimeError):
                E.validate_workload(plan, bundled)

    def test_deterministic_ties_and_ineligible_outcome(self):
        plan, _ = fixture()
        decision, hashes = selection(plan)
        self.assertEqual(E.validate_selection(plan, 'a' * 64, decision, 'b' * 64, hashes), 12)
        decision['aggregate']['candidates'][1]['teacher_forced_dev_loss'] = 0.9
        decision['selected_step'] = 24
        self.assertEqual(E.validate_selection(plan, 'a' * 64, decision, 'b' * 64, hashes), 24)
        for candidate in decision['aggregate']['candidates']:
            candidate['critical_errors'] = 1
        decision['selected_step'] = None
        self.assertIsNone(E.validate_selection(plan, 'a' * 64, decision, 'b' * 64, hashes))
        decision['selected_step'] = 24
        with self.assertRaises(RuntimeError):
            E.validate_selection(plan, 'a' * 64, decision, 'b' * 64, hashes)

    def test_behavior_regression_and_foreign_checkpoint_rejected(self):
        plan, _ = fixture()
        decision, hashes = selection(plan)
        decision['aggregate']['base']['behavior_pass_counts'] = dict(b0=0, b1=0, b2=2, b3=2)
        decision['selected_step'] = None
        self.assertIsNone(E.validate_selection(plan, 'a' * 64, decision, 'b' * 64, hashes))
        decision['checkpoint_hashes'] = dict(hashes, **{'step-000048': 'f' * 64})
        with self.assertRaises(RuntimeError):
            E.validate_selection(plan, 'a' * 64, decision, 'b' * 64, hashes)

    def test_parent_decision_bound_to_plan_and_export(self):
        plan, _ = fixture()
        decision, hashes = selection(plan)
        for key in ('run_id', 'plan_sha256', 'dev_export_sha256'):
            changed = copy.deepcopy(decision)
            changed[key] = 'foreign'
            with self.assertRaises(RuntimeError):
                E.validate_selection(plan, 'a' * 64, changed, 'b' * 64, hashes)

    def test_exact_worker_channel_policy(self):
        plan, _ = fixture()
        ident = plan['identity']
        arn = f"arn:aws:s3:::{ident['bucket']}/"
        statements = [dict(Effect='Allow', Action=['s3:GetObject'], Resource=[arn + ident['code_prefix'] + n
            for n in ('plan.json', 'bundle.tar', 'approval-consumed.json', 'termination.json',
                      'release.json', 'checkpoint-selection.json', 'final-inputs.json')]),
            dict(Effect='Allow', Action=['s3:PutObject'], Resource=[arn + ident['output_prefix'] + '*']),
            dict(Effect='Allow', Action=['s3:ListBucket'], Resource=[arn[:-1]],
                Condition={'StringEquals': {'s3:prefix': [ident['code_prefix'] + n for n in
                    ('termination.json', 'release.json', 'checkpoint-selection.json', 'final-inputs.json')]}})]
        plan['iam'] = {'instance_role': {'inline': {'exact': {'Statement': statements}}}}
        L.validate_worker_policy(plan)
        statements[0]['Resource'] = [arn + ident['code_prefix'] + '*']
        with self.assertRaises(RuntimeError):
            L.validate_worker_policy(plan)

    def test_output_membership_identity_and_terminal_metadata(self):
        plan, _ = fixture()
        _, hashes = selection(plan)
        messages = [{'role': 'system', 'content': 'Synthetic framing'},
                    {'role': 'user', 'content': 'Synthetic case'}]
        rows = [dict(id=case, cohort='dev', arm='base', adapter_weights_sha256=None,
            run_id=E.RUN_ID, plan_sha256='a' * 64, model_id=E.MODEL_ID, model_revision=E.MODEL_REVISION,
            input_sha256=plan['workload']['development_inputs']['sha256'], generation=E.GENERATION,
            messages=messages, prompt_sha256=L.object_sha(messages), rendered_prompt_sha256='c' * 64,
            prompt_token_ids=[1, 2], output_token_ids=[3], generated_tokens=1,
            raw_output='Synthetic', response_text='Synthetic', termination_reason='eos', terminal_stop_token_id=3)
            for case in plan['workload']['development_case_ids']]
        E.validate_outputs(plan, 'a' * 64, rows, 'dev', None, hashes)
        for mutate in (lambda r: r.pop(), lambda r: r[0].update(terminal_stop_token_id=4),
                       lambda r: r[0].update(termination_reason='length', terminal_stop_token_id=None),
                       lambda r: r[0].update(adapter_weights_sha256='f' * 64)):
            changed = copy.deepcopy(rows)
            mutate(changed)
            with self.assertRaises(RuntimeError):
                E.validate_outputs(plan, 'a' * 64, changed, 'dev', None, hashes)

    def test_malformed_completion_cannot_claim_success(self):
        plan, _ = fixture()
        with self.assertRaises(RuntimeError):
            E.validate_completion(plan, 'a' * 64, {'experimental_success': True}, Path('.'))

    def test_cost_cap_and_new_channel_envelope(self):
        plan, _ = fixture()
        plan['bundle'] = {'bytes': 100}
        extra = E.CHANNEL_BYTES
        plan['network'] = {'egress_byte_cap': L.LIMITS['output_bytes'] + extra}
        plan['cost'] = dict(cap_usd=20, retention_days=1, ancillary_bound_usd=5,
            get_requests=4608, put_requests=4608, control_requests=4608, download_requests=100,
            network_egress_bytes=2 * (L.LIMITS['output_bytes'] + extra),
            retained_bytes=100 + L.LIMITS['output_bytes'] + 8 * 1024**2 + extra,
            termination_grace_seconds=300,
            rates=dict(ec2_hour=1, ipv4_hour=.01, ebs_gb_month=.1, s3_gb_month=.03,
                       egress_gb=.1, s3_get=.000001, s3_put=.000005, scheduler_invocation=.000001))
        self.assertLess(L.cost_bound(plan)['additional_pretax_bound_usd'], 20)
        for mutate in (lambda c: c['rates'].update(ec2_hour=10),
                       lambda c: c.update(get_requests=512),
                       lambda c: c.update(put_requests=256),
                       lambda c: c.update(control_requests=2500),
                       lambda c: c.update(cap_usd=50),
                       lambda c: c.update(ancillary_bound_usd=4)):
            changed = copy.deepcopy(plan)
            mutate(changed['cost'])
            with self.assertRaises(RuntimeError):
                L.cost_bound(changed)


if __name__ == '__main__':
    unittest.main()
