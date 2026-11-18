"""Offline safety contracts. Synthetic receipts are test data, never GPU evidence."""
import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.aws_eval import public_evidence_experiment as P


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    P.write_json(path, value)
    return P.pin(path, root)


def public_rows(count=708):
    return [dict(id=f'evaluation-{i}', messages=[dict(role='system', content='Public system'),
        dict(role='user', content=f'Public question {i}')]) for i in range(count)]


def admission(train, inputs):
    return dict(schema='trid-public-evidence-cloud-admission-v1', profile=P.PROFILE,
        state='ADMITTED_FOR_RESEARCH', training_ready=True, semantic_evaluated=False, expert_certified=False,
        train=train, evaluation_inputs=inputs, recipe=dict(P.RECIPE), evaluation_rows=708,
        training_rows=630, general_training_rows=126, reviewers=['independent-A', 'independent-B'],
        live_trainer_token_proof=True, training_ids=[f'train-{i}' for i in range(630)],
        provenance_sha256=dict(manifest='a'*64, research_admission='b'*64, candidate_manifest='c'*64,
            annotation_packet='d'*64, reports=['e'*64, 'f'*64], token_proof='1'*64, source_pins='2'*64))


def safety_plan():
    return dict(schema=P.SCHEMA, identity=dict(run_id='public-evidence-test-run'),
        lifetime=dict(total_seconds=28800, termination_grace_seconds=900, automatic_extension=False),
        limits=dict(P.HOST_LIMITS, lifetime_seconds=27900),
        absolute_termination_deadline=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        instance_type='g6e.2xlarge', recipe=dict(P.RECIPE),
        cost=dict(cap_usd=100, termination_grace_seconds=900, contingency_usd=1.0),
        workload=dict(source_manifest=dict(path='sources.json', bytes=1, sha256='a'*64)))


def resources(root, plan):
    auth = dict(schema=P.AUTHORIZATION_SCHEMA, approved=True, run_id=plan['identity']['run_id'],
        workload_sha256=P.object_sha(plan['workload']),
        source_manifest_sha256=plan['workload']['source_manifest']['sha256'], execution_started=False,
        additional_gpu_executions=1, automatic_retry=False, automatic_extension=False, deployment=False,
        budget_usd=plan['cost']['cap_usd'], max_wall_seconds=28800,
        absolute_deadline_utc=plan['absolute_termination_deadline'])
    plan['authorization'] = save(root, 'authorization.json', auth)
    envelope = dict(schema='trid-public-evidence-resource-envelope-v1', identity=plan['identity'],
        authorization=plan['authorization'], instance_type='g6e.2xlarge', root_gib=120,
        root_encrypted=True, root_delete_on_termination=True, gpu_execution_lease_seconds_max=27900,
        export_reserve_seconds=300, termination_grace_seconds_max=900, total_billable_lifetime_seconds_max=28800,
        whole_host_egress_bytes_max=2*P.GIB, controller_retrieval_bytes_max=P.GIB, priced_egress_bytes=3*P.GIB,
        combined_retained_s3_bytes_max=2*P.GIB, output_archive_bytes_max=512*1024**2,
        bundle_bytes_max=128*1024**2, s3_expiration_days=30, multipart_abort_days=1,
        scheduler_invocations_max=10000, billing_hard_cap=False,
        absolute_termination_deadline=plan['absolute_termination_deadline'], automatic_retry=False, automatic_extension=False,
        budget_usd=plan['cost']['cap_usd'], tax_assumption='0.10', contingency_before_tax_usd='1',
        cost_evidence=save(root, 'observed-cost.json', {'test_fixture': True}))
    plan['resource_envelope'] = save(root, 'resources.json', envelope)
    return auth, envelope


def training_fixture(root):
    plan = safety_plan()
    plan['runtime'] = dict(packages={'torch': 'fixture'}, torch_cuda='fixture')
    plan['model'] = dict(id=P.MODEL_ID, revision=P.MODEL_REVISION, files=[])
    plan['data'] = dict(train=dict(path='train.jsonl', bytes=1, sha256='3'*64),
        manifest=dict(path='training-manifest.json', bytes=1, sha256='4'*64))
    plan['workload']['evaluation_inputs'] = dict(path='inputs.json', bytes=1, sha256='5'*64)
    save(root, 'plan.json', plan)
    runtime = save(root, 'runtime.json', dict(name='NVIDIA L40S', count=1, bf16=True,
        packages=plan['runtime']['packages'], cuda='fixture', run_id=plan['identity']['run_id'],
        plan_sha256=P.pin(root/'plan.json')['sha256'], instance_id='i-fixture'))
    admit = admission(plan['data']['train'], plan['workload']['evaluation_inputs'])
    save(root, 'admission.json', admit)
    history = []
    scale = lambda s: (s+1)/15 if s < 15 else 0.5*(1+math.cos(math.pi*(s-15)/300))
    for step in range(1, 316):
        history.append(dict(step=step, train_loss=1.0, applied_lr=1e-5*scale(step-1), next_lr=1e-5*scale(step)))
    report = dict(status='complete', steps=315, cumulative_steps=315, train_rows=630, valid_rows=0,
        dropped_over_max_tokens=0, valid_dropped=0, init_adapter=None, device='cuda',
        base_weights=dict(representation='hf-safetensors', dtype='torch.bfloat16'),
        manifest_sha256=plan['data']['manifest']['sha256'], base_model_id=P.MODEL_ID, base_revision=P.MODEL_REVISION,
        batch_size=2, micro_batch_size=1, lr=1e-5, seed=14, max_tokens=4096, max_answer_tokens=768,
        lora=dict(rank=8, alpha=16, dropout=0.0), history=history,
        coverage=[dict(id=ident, source_line=i, source_sha256=plan['data']['train']['sha256'], file='train.jsonl')
                  for i, ident in enumerate(admit['training_ids'], 1)])
    report_pin = save(root, 'training/training.json', report)
    config = save(root, 'training/adapter/adapter_config.json', dict(r=8, lora_alpha=16, lora_dropout=0.0,
        peft_type='LORA', task_type='CAUSAL_LM', bias='none', modules_to_save=None))
    weights = root/'training/adapter/adapter_model.safetensors'
    weights.write_bytes(b'fixture-not-real-weights')
    inventory = dict(base={'torch.bfloat16': dict(tensors=1, elements=2, bytes=4)},
        adapter={'torch.float32': dict(tensors=1, elements=2, bytes=8)})
    provenance = dict(schema='trid-public-evidence-adapter-v1', run_id=plan['identity']['run_id'],
        plan_sha256=P.pin(root/'plan.json')['sha256'], source_manifest=plan['workload']['source_manifest'],
        runtime=runtime, fresh=True, merged=False, training=report_pin,
        adapter=dict(config=config, weights=P.pin(weights, root), step=315), parameter_inventory=inventory,
        fresh_zero_b=True, observed_steps=315, coverage_exactly_once=True,
        masking=dict(rows=630, all_prompt_tokens_masked=True, all_answer_tokens_supervised=True,
            eos_supervised=True, max_context_tokens=100, max_answer_tokens=20))
    save(root, 'training/provenance.json', provenance)
    return plan, report, provenance


def output_fixture(root):
    from scripts.evaluate_evidence_conditioning import case_seed, generation_dispatch, SAMPLING_PROFILE
    plan, _, provenance = training_fixture(root)
    cases = public_rows()
    plan['workload']['evaluation_inputs'] = save(root, 'evaluation-inputs.json', cases)
    save(root, 'plan.json', plan)
    runtime = P.read_json(root/'runtime.json')
    runtime['plan_sha256'] = P.pin(root/'plan.json')['sha256']
    runtime_pin = save(root, 'runtime.json', runtime)
    save(root, 'baseline/loaded-runtime.json', dict(device='cuda', dtype='torch.bfloat16', frozen=True,
        adapter_merged=False, cpu_fallback=False, local_files_only=True, trust_remote_code=False,
        plan_sha256=runtime['plan_sha256'], model_files=[], gpu_uuid='GPU-test', instance_id='i-fixture',
        adapter=None, adapter_reload_equal=None, runtime=runtime_pin,
        parameter_inventory=dict(base=provenance['parameter_inventory']['base'], adapter={})))
    prompts = [dict(id=c['id'], prompt_sha256=P.object_sha(c['messages']),
        rendered_prompt_sha256='a'*64, prompt_token_ids=[1], input_tokens=1) for c in cases]
    save(root, 'baseline/prompt-preflight.json', dict(status='passed',
        all_cases_preflighted_before_weights=True, truncation=False, prompts=prompts))
    dispatch = generation_dispatch(SAMPLING_PROFILE)
    config = {k: v for k, v in dispatch.items() if k != 'use_model_defaults'}
    config.update({k: P.GENERATION[k] for k in ('do_sample', 'temperature', 'top_p', 'top_k', 'min_p',
        'num_beams', 'repetition_penalty', 'no_repeat_ngram_size', 'max_new_tokens')})
    config['eos_token_id'] = 9
    records = [dict(id=c['id'], prompt_sha256=P.object_sha(c['messages']),
        binding=P.generation_binding(plan, 'baseline'), generation_receipt='hf_native_v1',
        generation_profile=SAMPLING_PROFILE, mode='evaluation', seed=case_seed('evaluation', c['id']),
        context_limit=4096, context_truncated=False, rendered_prompt=None, rendered_prompt_sha256=None,
        prompt_token_ids=[], output_token_ids=[], input_tokens=0, output_tokens=0,
        response='', raw_output='', error={'type': 'RuntimeError', 'message': 'Test-only failure receipt'},
        finish_reason='exception', generation_config=config, dispatch_kwargs=dispatch,
        template_kwargs=dict(add_generation_prompt=True, enable_thinking=False), generation_policy=None,
        terminal_stop_token_id=None, truncated_output=False) for c in cases]
    path = root/'baseline/responses.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    save(root, 'baseline/completion.json', dict(status='complete', arm='baseline', rows=708,
        quality_accepted=False, outputs=P.pin(path, root), exception_rows=708, length_rows=0))
    return plan, records


class PublicEvidenceContractTests(unittest.TestCase):
    def test_exact_physical_epoch(self):
        self.assertEqual(P.RECIPE['rows']*P.RECIPE['epochs'], P.RECIPE['steps']*P.RECIPE['batch_size'])
        self.assertEqual((P.RECIPE['steps'], P.RECIPE['micro_batch_size']), (315, 1))

    def test_unreviewed_projection_refused_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save(root, 'candidate.json', dict(schema='trid-evidence-preparation-manifest-v1',
                profile=P.PROFILE, state='CANDIDATE_NEEDS_INDEPENDENT_REVIEW'))
            with self.assertRaisesRegex(ValueError, 'Candidate/unreviewed'):
                P.prepare_projection(root/'candidate.json', root/'projection', root=root,
                    sources=P.SOURCE_NAMES, acquisition_manifest=root/'not-opened.json')
            self.assertFalse((root/'projection').exists())

    def test_fresh_dynamic_budget_and_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan = safety_plan(); auth, envelope = resources(root, plan)
            self.assertEqual(P.validate_resources(root, plan), envelope)
            for field, value in [('approved', False), ('execution_started', True), ('run_id', 'different-run'),
                                 ('workload_sha256', '0'*64), ('budget_usd', '20'), ('max_wall_seconds', 14400)]:
                changed = dict(auth, **{field: value})
                plan['authorization'] = save(root, 'authorization.json', changed)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    P.validate_resources(root, plan)
            plan['authorization'] = save(root, 'authorization.json', auth)
            plan['absolute_termination_deadline'] = '2026-11-16T23:00:00Z'
            auth['absolute_deadline_utc'] = plan['absolute_termination_deadline']
            plan['authorization'] = save(root, 'authorization.json', auth)
            envelope.update(authorization=plan['authorization'], absolute_termination_deadline=plan['absolute_termination_deadline'])
            plan['resource_envelope'] = save(root, 'resources.json', envelope)
            with self.assertRaisesRegex(ValueError, 'expired'):
                P.validate_resources(root, plan)
            self.assertEqual(P.validate_resources(root, plan, active=False), envelope)

    def test_fractional_deadlines_are_refused_before_dispatch(self):
        whole = '2099-11-17T08:30:45Z'
        self.assertEqual(P.utc(whole).timestamp(), int(P.utc(whole).timestamp()))
        for suffix in ('.1Z', '.000001Z', '.0000001Z', '.000Z', ',5Z'):
            plan = safety_plan()
            plan['absolute_termination_deadline'] = whole[:-1] + suffix
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'Whole-second'):
                P.validate_limits(plan)
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, 'Whole-second'):
                    P.validate_resources(Path(tmp), plan)

    def test_budget_representation_matches_bootstrap_before_dispatch(self):
        for cap in (1, 100, 100.5):
            plan = safety_plan(); plan['cost']['cap_usd'] = cap
            P.validate_limits(plan)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _, envelope = resources(root, plan)
                self.assertEqual(P.validate_resources(root, plan), envelope)
        for cap in ('100', True, False, None, 0, -1, float('nan'), float('inf'), -float('inf')):
            plan = safety_plan(); plan['cost']['cap_usd'] = cap
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, 'finite positive JSON number'):
                P.validate_limits(plan)
            with self.assertRaisesRegex(ValueError, 'finite positive JSON number'):
                P.cost_bound(plan)
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, 'finite positive JSON number'):
                    P.validate_resources(Path(tmp), plan)

    def test_host_limit_and_budget_cannot_expand(self):
        plan = safety_plan(); P.validate_limits(plan)
        for key, value in [('output_bytes', 1024**3), ('instance_count', 2), ('reserve_seconds', 0)]:
            bad = copy.deepcopy(plan); bad['limits'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): P.validate_limits(bad)
        plan['lifetime']['total_seconds'] = 28801
        with self.assertRaises(ValueError): P.validate_limits(plan)

    def test_general_multiturn_messages_remain_native(self):
        messages = [dict(role='system', content='General system'), dict(role='user', content='Earlier user'),
                    dict(role='assistant', content='Earlier answer'), dict(role='user', content='Current user')]
        row = dict(id='general', messages=messages)
        snapshot = copy.deepcopy(row)
        self.assertEqual(P.validate_public_rows([row], 1), ['general'])
        self.assertEqual(row, snapshot)
        self.assertEqual(P.validate_public_rows([dict(id='no-system', messages=messages[1:])], 1), ['no-system'])
        with self.assertRaises(ValueError):
            P.validate_public_rows([dict(id='assistant-last', messages=messages[:-1])], 1)

    def test_public_rows_reject_gold_duplicates_subset(self):
        rows = public_rows(); self.assertEqual(len(P.validate_public_rows(rows)), 708)
        for bad in (rows[:-1], rows + [rows[0]], [rows[0]]*708):
            with self.assertRaises(ValueError): P.validate_public_rows(bad)
        rows[0]['reference'] = {'target': 'private'}
        with self.assertRaises(ValueError): P.validate_public_rows(rows)

    def test_cloud_admission_rejects_stale_review_and_private_fields(self):
        train = dict(path='train.jsonl', bytes=1, sha256='a'*64)
        inputs = dict(path='inputs.json', bytes=1, sha256='b'*64)
        good = admission(train, inputs); P.validate_admission(good, train, inputs)
        for field, value in [('state', 'CANDIDATE_NEEDS_INDEPENDENT_REVIEW'), ('training_rows', 600),
                             ('evaluation_rows', 200), ('general_training_rows', 125),
                             ('live_trainer_token_proof', False), ('reviewers', ['same', 'same'])]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                P.validate_admission(dict(good, **{field: value}), train, inputs)
        with self.assertRaises(ValueError): P.validate_admission(dict(good, references=['private']), train, inputs)
        with self.assertRaises(ValueError): P.validate_admission(good, dict(train, sha256='0'*64), inputs)

    def test_pins_reject_mutation_traversal_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); record = save(root, 'a.json', {'value': 1}); P.bound(root, record)
            (root/'a.json').write_text('{}')
            with self.assertRaises(ValueError): P.bound(root, record)
            with self.assertRaises(ValueError): P.bound(root, dict(record, path='../a.json'))
            (root/'alias.json').symlink_to(root/'a.json')
            with self.assertRaises(ValueError): P.safe_path(root, 'alias.json')

    def test_exact_once_training_and_numerical_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan, report, provenance = training_fixture(root)
            self.assertEqual(P.validate_training(root, plan)['steps'], 315)
            variants = []
            duplicate = copy.deepcopy(report); duplicate['coverage'][-1] = duplicate['coverage'][0]; variants.append(duplicate)
            reordered_identity = copy.deepcopy(report); reordered_identity['coverage'][0]['id'] = 'train-1'; variants.append(reordered_identity)
            truncated = copy.deepcopy(report); truncated['history'].pop(); variants.append(truncated)
            bad_lr = copy.deepcopy(report); bad_lr['history'][0]['applied_lr'] = 1e-5; variants.append(bad_lr)
            bad_loss = copy.deepcopy(report); bad_loss['history'][1]['train_loss'] = float('inf'); variants.append(bad_loss)
            for index, changed in enumerate(variants):
                # JSON allows Infinity here to exercise fail-closed numerical validation.
                (root/'training/training.json').write_text(json.dumps(changed))
                provenance['training'] = P.pin(root/'training/training.json', root)
                save(root, 'training/provenance.json', provenance)
                with self.subTest(index=index), self.assertRaises(ValueError): P.validate_training(root, plan)

    def test_training_rejects_masking_warmstart_and_adapter_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan, _, provenance = training_fixture(root)
            provenance['masking']['eos_supervised'] = False
            save(root, 'training/provenance.json', provenance)
            with self.assertRaises(ValueError): P.validate_training(root, plan)
            provenance['masking']['eos_supervised'] = True; provenance['fresh_zero_b'] = False
            save(root, 'training/provenance.json', provenance)
            with self.assertRaises(ValueError): P.validate_training(root, plan)
            provenance['fresh_zero_b'] = True; save(root, 'training/provenance.json', provenance)
            (root/'training/adapter/adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError): P.validate_training(root, plan)

    def test_bindings_cannot_cross_arm_or_checkpoint(self):
        plan = safety_plan(); plan['workload']['evaluation_inputs'] = {'sha256': 'b'*64}
        self.assertIsNone(P.generation_binding(plan, 'baseline')['adapter'])
        with self.assertRaises(ValueError): P.generation_binding(plan, 'adapter')
        with self.assertRaises(ValueError): P.generation_binding(plan, 'adapter', dict(config={}, weights={}, step=150))
        with self.assertRaises(ValueError): P.generation_binding(plan, 'baseline', dict(config={}, weights={}, step=315))

    def test_full_native_exception_receipts_stay_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan, records = output_fixture(root)
            result = P.validate_outputs(root, plan, 'baseline')
            self.assertEqual(len(result), 708)
            self.assertTrue(all(row['error'] is not None for row in result))
            for field, value in [('id', 'unknown'), ('seed', 0), ('generation_receipt', 'external_unverified'),
                                 ('context_truncated', True), ('output_tokens', 2)]:
                bad = copy.deepcopy(records); bad[0][field] = value
                (root/'baseline/responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    P.validate_outputs(root, plan, 'baseline')
            (root/'baseline/responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records[:-1]))
            with self.assertRaisesRegex(ValueError, 'All708'):
                P.validate_outputs(root, plan, 'baseline')

    def test_capacity_requires_resident_second_update_and_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan, _, provenance = training_fixture(root)
            receipt = dict(synthetic=True, discarded=True, microbatches=4, context_tokens=4096, answer_tokens=768,
                admitted_expanded_rows=630, finite_parameter_update=True, adapter_reload_equal=True,
                optimizer_updates=2, microbatches_per_update=2, optimizer_state_resident_before_second_update=True,
                gradient_checkpointing=True, losses=[1.0]*4, max_allocated_bytes=1, max_reserved_bytes=2,
                plan_sha256=provenance['plan_sha256'], source_manifest=provenance['source_manifest'],
                runtime=provenance['runtime'], parameter_inventory=provenance['parameter_inventory'])
            save(root, 'capacity-receipt.json', receipt); P.validate_capacity(root, plan)
            receipt['optimizer_state_resident_before_second_update'] = False
            save(root, 'capacity-receipt.json', receipt)
            with self.assertRaises(ValueError): P.validate_capacity(root, plan)
            receipt['optimizer_state_resident_before_second_update'] = True
            receipt['parameter_inventory']['adapter'] = {'torch.bfloat16': dict(tensors=1, elements=2, bytes=4)}
            save(root, 'capacity-receipt.json', receipt)
            with self.assertRaises(ValueError): P.validate_capacity(root, plan)

    def test_export_and_subset_do_not_masquerade_as_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan = safety_plan()
            with self.assertRaises(ValueError): P.validate_export(root, plan)
            with self.assertRaises(ValueError): P.validate_outputs(root, plan, 'baseline', smoke=True)

    def test_terminal_requires_all_owned_phases_reaped(self):
        plan = safety_plan(); sha = 'f'*64; instance = 'i-test'
        def stage(phase):
            return dict(phase=phase, state='complete', run_id=plan['identity']['run_id'], plan_sha256=sha,
                instance_id=instance, unit=plan['identity']['run_id']+'-'+phase, invocation='unique-'+phase,
                cleanup_verified=True, stopped_state={'ActiveState': 'inactive'}, gpu={'uuid': 'test'})
        completion = dict(public_evidence_success=True, stop_errors=[], completion_scope=P.COMPLETION_SCOPE,
            quality_accepted=False, runtime_unit=stage('runtime'), stages=[stage(p) for p in ('capacity','baseline','train','adapter')])
        P.validate_terminal(plan, sha, instance, completion)
        completion['stages'][-1]['cleanup_verified'] = False
        with self.assertRaises(ValueError): P.validate_terminal(plan, sha, instance, completion)


if __name__ == '__main__':
    unittest.main()
