"""Offline safety contracts. Synthetic receipts are test data, never GPU evidence."""
import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.aws_eval import public_evidence_experiment as P
from scripts.aws_eval import public_work_experiment as W


def save(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    P.write_json(path, value)
    return P.pin(path, root)


def public_rows(count=708):
    return [dict(id=f'evaluation-{i}', messages=[dict(role='system', content='Public system'),
        dict(role='user', content=f'Public question {i}')]) for i in range(count)]


def admission(train, inputs):
    return dict(schema=P.CLOUD_ADMISSION_SCHEMA, profile=P.PROFILE,
        state='ADMITTED_FOR_RESEARCH', training_ready=True, semantic_evaluated=False, expert_certified=False,
        train=train, evaluation_inputs=inputs, recipe=dict(P.RECIPE), evaluation_rows=708,
        training_rows=1230, general_training_rows=246, reviewers=['independent-A', 'independent-B'],
        live_trainer_token_proof=True, training_ids=[f'train-{i}' for i in range(1230)],
        provenance_sha256=dict(manifest='a'*64, research_admission='b'*64, candidate_manifest='c'*64,
            annotation_packet='d'*64, reports=['e'*64, 'f'*64], token_proof='1'*64, source_pins='2'*64))


def safety_plan():
    return dict(schema=P.SCHEMA, identity=dict(run_id='public-evidence-test-run'),
        lifetime=dict(total_seconds=28800, termination_grace_seconds=900, automatic_extension=False),
        limits=dict(P.HOST_LIMITS, lifetime_seconds=27900),
        absolute_termination_deadline=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(timespec='seconds').replace('+00:00', 'Z'),
        instance_type='g6e.2xlarge', recipe=dict(P.RECIPE),
        cost=dict(cap_usd=30, termination_grace_seconds=900, contingency_usd=1.0),
        data=dict(train=dict(path='train.jsonl', bytes=1, sha256='3'*64)),
        workload=dict(source_manifest=dict(path='sources.json', bytes=1, sha256='a'*64)))


def cost_plan():
    plan = safety_plan()
    plan['bundle'] = dict(bytes=1024)
    plan['network'] = dict(egress_byte_cap=P.GIB)
    plan['cost'].update(export_bytes=512*1024**2, controller_retrieval_bytes=P.GIB,
        network_egress_bytes=3*P.GIB, retained_bytes=P.GIB, retention_days=30,
        get_requests=4608, put_requests=4608, control_requests=4608,
        scheduler_invocations=10000, tax_rate=0.10, ancillary_bound_usd=1,
        rates=dict(ec2_hour=2, ipv4_hour=0.005, ebs_gb_month=0.08, s3_gb_month=0.025,
            egress_gb=0.1, s3_get=0.000001, s3_put=0.000005, scheduler_invocation=0.000001))
    return plan


def comparator_cost_plan():
    plan = cost_plan()
    plan['schema'] = W.SCHEMA
    plan['bundle']['bytes'] = 256 * 1024**2
    plan['workload']['prior_adapter'] = dict(run_id=W.PRIOR_RUN_ID, step=530,
        model=dict(id=W.MODEL_ID, revision=W.MODEL_REVISION),
        files={name: dict(path='prior/' + name,
            bytes=W.PRIOR_WEIGHTS_BYTES if name == W.PRIOR_WEIGHT_PATH else 1,
            sha256=W.PRIOR_WEIGHTS_SHA if name == W.PRIOR_WEIGHT_PATH else W.PRIOR_RECEIPT_SHA[name])
            for name in W.PRIOR_FILES})
    return plan


def resources(root, plan):
    auth = dict(schema=P.AUTHORIZATION_SCHEMA, approved=True, run_id=plan['identity']['run_id'],
        workload_sha256=P.object_sha(plan['workload']),
        data_sha256=P.object_sha(plan['data']), recipe_sha256=P.object_sha(plan['recipe']),
        source_manifest_sha256=plan['workload']['source_manifest']['sha256'], execution_started=False,
        additional_gpu_executions=1, automatic_retry=False, automatic_extension=False, deployment=False,
        budget_usd=plan['cost']['cap_usd'], max_wall_seconds=28800,
        absolute_deadline_utc=plan['absolute_termination_deadline'])
    plan['authorization'] = save(root, 'authorization.json', auth)
    envelope = dict(schema=P.RESOURCE_ENVELOPE_SCHEMA, identity=plan['identity'],
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


def data_fixture(root):
    """Synthetic files exercise the real pin, closure and row validators."""
    plan = safety_plan()
    rows = [dict(id=f'train-{i}', task='general_preservation' if i < 246 else 'evidence_conditioning',
        split='train', messages=[dict(role='user', content=f'Question {i}')], answer=f'Answer {i}')
        for i in range(1230)]
    path = root / 'train.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    train = P.pin(path, root)
    inputs = save(root, 'evaluation-inputs.json', public_rows())
    plan['data'] = dict(train=train, manifest=save(root, 'training-manifest.json',
        dict(id='synthetic-paired1230', profile=P.PROFILE, rows=1230, valid=[], max_tokens=4096,
            max_answer_tokens=768, train=[dict(train, adapter='grounded_messages', epochs=1)])),
        admission=save(root, 'admission.json', admission(train, inputs)))
    sources = [save(root, name, {'synthetic_source_fixture': name}) for name in sorted(P.SOURCE_NAMES)]
    acquisition = save(root, 'acquisition.json', dict(repo_id=P.MODEL_ID, revision=P.MODEL_REVISION, files=[]))
    plan['workload'] = dict(evaluation_inputs=inputs, acquisition_manifest=acquisition,
        source_manifest=save(root, 'source-manifest.json', dict(schema=P.SOURCE_MANIFEST_SCHEMA, files=sources)),
        generation=dict(P.GENERATION))
    plan['sources'] = {p['path']: p['sha256'] for p in sources}
    plan['model'] = dict(id=P.MODEL_ID, revision=P.MODEL_REVISION, trust_remote_code=False, files=[])
    plan['phase_seconds'] = dict(acquire=100, capacity=100, baseline=100, train=100, adapter=100)
    plan['runtime'] = dict(packages=P.RUNTIME_PACKAGES, torch_cuda='12.8',
        bootstrap=dict(admission=save(root, 'bootstrap.json', dict(evidence=[]))))
    resources(root, plan)
    plan['ancillary_evidence'] = save(root, 'ancillary.json', dict(
        bucket_policy_status_evidence=save(root, 'bucket-policy.json', {'synthetic': True}),
        ebs_kms_key_evidence=save(root, 'key.json', {'synthetic': True})))
    plan['output_allowlist'] = sorted(P.output_members())
    rebuild_bundle(root, plan)
    return plan, rows


def rebuild_bundle(root, plan):
    records = [*plan['data'].values(), *(plan['workload'][k] for k in
        ('evaluation_inputs', 'source_manifest', 'acquisition_manifest')),
        plan['authorization'], plan['resource_envelope'], plan['ancillary_evidence'],
        plan['runtime']['bootstrap']['admission']]
    records += P.load(root, plan['workload']['source_manifest'])['files']
    records += [P.load(root, plan['resource_envelope'])['cost_evidence']]
    ancillary = P.load(root, plan['ancillary_evidence'])
    records += [ancillary['bucket_policy_status_evidence'], ancillary['ebs_kms_key_evidence']]
    plan['bundle'] = dict(files=records)


def reseal_train(root, plan, rows):
    (root/'train.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    train = P.pin(root/'train.jsonl', root)
    plan['data']['train'] = train
    manifest = P.load(root, plan['data']['manifest'])
    manifest['train'] = [dict(train, adapter='grounded_messages', epochs=1)]
    plan['data']['manifest'] = save(root, 'training-manifest.json', manifest)
    receipt = P.load(root, plan['data']['admission'])
    receipt['train'] = train
    plan['data']['admission'] = save(root, 'admission.json', receipt)
    rebuild_bundle(root, plan)


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
    scale = lambda s: (s+1)/30 if s < 30 else 0.5*(1+math.cos(math.pi*(s-30)/585))
    for step in range(1, 616):
        history.append(dict(step=step, train_loss=1.0, applied_lr=1e-5*scale(step-1), next_lr=1e-5*scale(step)))
    report = dict(status='complete', steps=615, cumulative_steps=615, train_rows=1230, valid_rows=0,
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
    provenance = dict(schema=P.ADAPTER_SCHEMA, run_id=plan['identity']['run_id'],
        plan_sha256=P.pin(root/'plan.json')['sha256'], source_manifest=plan['workload']['source_manifest'],
        runtime=runtime, fresh=True, merged=False, training=report_pin,
        adapter=dict(config=config, weights=P.pin(weights, root), step=615), parameter_inventory=inventory,
        fresh_zero_b=True, observed_steps=615, coverage_exactly_once=True,
        masking=dict(rows=1230, all_prompt_tokens_masked=True, all_answer_tokens_supervised=True,
            eos_supervised=True, max_context_tokens=100, max_answer_tokens=20))
    save(root, 'training/provenance.json', provenance)
    return plan, report, provenance


def output_fixture(root, arm='baseline'):
    from scripts.evaluate_evidence_conditioning import case_seed, generation_dispatch, SAMPLING_PROFILE
    plan, _, provenance = training_fixture(root)
    cases = public_rows()
    plan['workload']['evaluation_inputs'] = save(root, 'evaluation-inputs.json', cases)
    save(root, 'plan.json', plan)
    runtime = P.read_json(root/'runtime.json')
    runtime['plan_sha256'] = P.pin(root/'plan.json')['sha256']
    runtime_pin = save(root, 'runtime.json', runtime)
    adapter = provenance['adapter'] if arm == 'adapter' else None
    save(root, arm+'/loaded-runtime.json', dict(device='cuda', dtype='torch.bfloat16', frozen=True,
        adapter_merged=False, cpu_fallback=False, local_files_only=True, trust_remote_code=False,
        plan_sha256=runtime['plan_sha256'], model_files=[], gpu_uuid='GPU-test', instance_id='i-fixture',
        adapter=adapter, adapter_reload_equal=True if adapter else None, runtime=runtime_pin,
        parameter_inventory=dict(base=provenance['parameter_inventory']['base'],
            adapter=provenance['parameter_inventory']['adapter'] if adapter else {})))
    prompts = [dict(id=c['id'], prompt_sha256=P.object_sha(c['messages']),
        rendered_prompt_sha256='a'*64, prompt_token_ids=[1], input_tokens=1) for c in cases]
    save(root, arm+'/prompt-preflight.json', dict(status='passed',
        all_cases_preflighted_before_weights=True, truncation=False, prompts=prompts))
    dispatch = generation_dispatch(SAMPLING_PROFILE)
    config = {k: v for k, v in dispatch.items() if k != 'use_model_defaults'}
    config.update({k: P.GENERATION[k] for k in ('do_sample', 'temperature', 'top_p', 'top_k', 'min_p',
        'num_beams', 'repetition_penalty', 'no_repeat_ngram_size', 'max_new_tokens')})
    config['eos_token_id'] = 9
    records = [dict(id=c['id'], prompt_sha256=P.object_sha(c['messages']),
        binding=P.generation_binding(plan, arm, adapter), generation_receipt='hf_native_v1',
        generation_profile=SAMPLING_PROFILE, mode='evaluation', seed=case_seed('evaluation', c['id']),
        context_limit=4096, context_truncated=False, rendered_prompt=None, rendered_prompt_sha256=None,
        prompt_token_ids=[], output_token_ids=[], input_tokens=0, output_tokens=0,
        response='', raw_output='', error={'type': 'RuntimeError', 'message': 'Test-only failure receipt'},
        finish_reason='exception', generation_config=config, dispatch_kwargs=dispatch,
        template_kwargs=dict(add_generation_prompt=True, enable_thinking=False), generation_policy=None,
        terminal_stop_token_id=None, truncated_output=False) for c in cases]
    path = root/arm/'responses.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    save(root, arm+'/completion.json', dict(status='complete', arm=arm, rows=708,
        quality_accepted=False, outputs=P.pin(path, root), exception_rows=708, length_rows=0))
    return plan, records


class PublicEvidenceContractTests(unittest.TestCase):
    def test_exact_physical_epoch(self):
        self.assertEqual(P.RECIPE['rows']*P.RECIPE['epochs'], P.RECIPE['steps']*P.RECIPE['batch_size'])
        self.assertEqual((P.RECIPE['rows'], P.RECIPE['steps'], P.RECIPE['micro_batch_size']), (1230, 615, 1))
        self.assertEqual((P.GENERAL_TRAINING_ROWS, P.EVALUATION_ROWS), (246, 708))

    def test_real_data_boundary_rejects_counts_order_and_identity_mutations(self):
        changes = ('short', 'extra', 'old630', 'general245', 'general247', 'order', 'duplicate',
                   'empty', 'nonstring', 'overlap')
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); plan, rows = data_fixture(root)
                with patch.object(P, 'ACQUISITION_SHA', plan['workload']['acquisition_manifest']['sha256']):
                    self.assertEqual(len(P.validate_data(root, plan)['rows']), 1230)
                    if change == 'short':
                        rows.pop()
                    elif change == 'extra':
                        rows.append(dict(rows[-1], id='extra'))
                    elif change == 'old630':
                        rows = rows[:630]
                    elif change == 'general245':
                        rows[0]['task'] = 'evidence_conditioning'
                    elif change == 'general247':
                        rows[246]['task'] = 'general_preservation'
                    elif change == 'order':
                        rows[0], rows[1] = rows[1], rows[0]
                    elif change == 'duplicate':
                        rows[-1]['id'] = rows[0]['id']
                    elif change == 'empty':
                        rows[0]['id'] = ''
                    elif change == 'nonstring':
                        rows[0]['id'] = 0
                    else:
                        rows[0]['id'] = 'evaluation-0'
                    reseal_train(root, plan, rows)
                    with self.assertRaises(ValueError):
                        P.validate_data(root, plan)

    def test_receipt_training_ids_must_be_unique_nonempty_string_list(self):
        good = admission({'sha256': 'a'*64}, {'sha256': 'b'*64})
        ids = good['training_ids']
        for bad in (tuple(ids), ids[:-1], ids + ['extra'], [ids[0]]*1230,
                    [''] + ids[1:], [' '] + ids[1:], [0] + ids[1:], [{}] + ids[1:]):
            with self.subTest(kind=type(bad).__name__), self.assertRaises(ValueError):
                P.validate_admission(dict(good, training_ids=bad), good['train'], good['evaluation_inputs'])

    def test_source_manifest_schema_exact_keys_and_full_bundle_records(self):
        for change in ('v1', 'unknown', 'extra_key', 'missing', 'duplicate', 'record_bytes',
                       'record_hash', 'record_extra', 'bundle_bytes', 'bundle_extra'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); plan, _ = data_fixture(root)
                source = P.load(root, plan['workload']['source_manifest'])
                if change in ('v1', 'unknown'):
                    source['schema'] = 'trid-public-evidence-source-manifest-v1' if change == 'v1' else 'unknown'
                elif change == 'extra_key':
                    source['extra'] = True
                elif change == 'missing':
                    source['files'].pop()
                elif change == 'duplicate':
                    source['files'][-1] = source['files'][0]
                elif change == 'record_bytes':
                    source['files'][0]['bytes'] += 1
                elif change == 'record_hash':
                    source['files'][0]['sha256'] = '0'*64
                elif change == 'record_extra':
                    source['files'][0]['extra'] = True
                plan['workload']['source_manifest'] = save(root, 'source-manifest.json', source)
                # Keep the real source bundle records unchanged unless testing their full binding.
                for record in plan['bundle']['files']:
                    if record['path'] == 'source-manifest.json':
                        record.update(plan['workload']['source_manifest'])
                    elif record['path'] == source['files'][0]['path']:
                        if change == 'bundle_bytes':
                            record['bytes'] += 1
                        elif change == 'bundle_extra':
                            record['extra'] = True
                with patch.object(P, 'ACQUISITION_SHA', plan['workload']['acquisition_manifest']['sha256']):
                    with self.assertRaises(ValueError):
                        P.validate_data(root, plan)

    def test_old_plan_and_altered_recipes_fail_consumer_boundaries(self):
        for recipe in (dict(P.RECIPE, rows=630, steps=315), dict(P.RECIPE, steps=315),
                       dict(P.RECIPE, epochs=2), dict(P.RECIPE, batch_size=4)):
            plan = safety_plan(); plan['recipe'] = recipe
            with self.subTest(recipe=recipe), self.assertRaises(ValueError):
                P.validate_workload(plan, {})
        plan = safety_plan(); plan['schema'] = 'trid-public-evidence-gpu-plan-v1'
        with self.assertRaises(ValueError):
            P.validate_plan(plan, Path('/unused'))
        with self.assertRaises(ValueError):
            P.generation_binding(plan, 'baseline')

    def test_authorization_binds_payload_not_just_workload(self):
        for field in ('train', 'recipe', 'missing_data', 'missing_recipe'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); plan = safety_plan(); auth, _ = resources(root, plan)
                if field == 'train':
                    plan['data']['train']['sha256'] = '0'*64
                elif field == 'recipe':
                    plan['recipe']['steps'] = 315
                else:
                    del auth['data_sha256' if field == 'missing_data' else 'recipe_sha256']
                    plan['authorization'] = save(root, 'authorization.json', auth)
                with self.assertRaises(ValueError):
                    P.validate_resources(root, plan)

    def test_v1_approval_envelope_and_adapter_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan = safety_plan(); _, envelope = resources(root, plan)
            approval = dict(schema=P.APPROVAL_SCHEMA, run_id=plan['identity']['run_id'], plan_sha256='f'*64,
                instance_limit=1, cap_usd='30', workload_sha256=P.object_sha(plan['workload']),
                authorization_sha256=plan['authorization']['sha256'], limits=plan['limits'],
                ebs_kms_key_arn='test-key', automatic_retry=False, automatic_extension=False,
                retraining=True, deployment=False, approved=True,
                absolute_termination_deadline=plan['absolute_termination_deadline'])
            P.validate_approval(plan, approval, 'f'*64, 'test-key')
            approval['schema'] = 'trid-public-evidence-gpu-approval-v1'
            with self.assertRaises(ValueError):
                P.validate_approval(plan, approval, 'f'*64, 'test-key')
            envelope['schema'] = 'trid-public-evidence-resource-envelope-v1'
            plan['resource_envelope'] = save(root, 'resources.json', envelope)
            with self.assertRaises(ValueError):
                P.validate_resources(root, plan)
            plan, _, provenance = training_fixture(root)
            provenance['schema'] = 'trid-public-evidence-adapter-v1'
            save(root, 'training/provenance.json', provenance)
            with self.assertRaises(ValueError):
                P.validate_training(root, plan)

    def test_malformed_admission_closure_leaves_no_projection(self):
        required = ('train.jsonl', 'training-metadata.json', 'evaluation-inputs.json',
            'evaluation-references.private.json', 'source-records.private.json', 'derived-matrix.private.json',
            'annotation-review-packet.private.json', 'token-budget.json', 'training-manifest.json',
            'evaluation-manifest.json', 'research-admission.json', 'admission-token-budget.json')
        for stale in (False, True):
            with self.subTest(stale=stale), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                artifacts = {name: save(root, name, {'synthetic': True}) for name in required}
                if stale:
                    (root/'train.jsonl').write_text('changed after pinning')
                else:
                    del artifacts['training-metadata.json']
                save(root, 'manifest.json', dict(schema='trid-evidence-admitted-manifest-v1',
                    profile=P.PROFILE, state='ADMITTED_FOR_RESEARCH', training_ready=True,
                    new_annotations_independently_reviewed=True, artifacts=artifacts))
                with self.assertRaises(ValueError):
                    P.prepare_projection(root/'manifest.json', root/'projection', root=root,
                        sources=P.SOURCE_NAMES, acquisition_manifest=root/'unopened.json')
                self.assertFalse((root/'projection').exists())

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
                                 ('schema', 'trid-public-evidence-gpu-authorization-v1'),
                                 ('data_sha256', '0'*64), ('recipe_sha256', '0'*64),
                                 ('source_manifest_sha256', '0'*64), ('additional_gpu_executions', 2),
                                 ('automatic_retry', True), ('automatic_extension', True), ('deployment', True),
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
        for cap in (1, 29.5, 30):
            plan = safety_plan(); plan['cost']['cap_usd'] = cap
            P.validate_limits(plan)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); _, envelope = resources(root, plan)
                self.assertEqual(P.validate_resources(root, plan), envelope)
        for cap in ('30', True, False, None, 0, -1, 30.01, 100, float('nan'), float('inf'), -float('inf')):
            plan = safety_plan(); plan['cost']['cap_usd'] = cap
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, 'finite positive JSON number'):
                P.validate_limits(plan)
            with self.assertRaisesRegex(ValueError, 'finite positive JSON number'):
                P.cost_bound(plan)
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError, 'finite positive JSON number'):
                    P.validate_resources(Path(tmp), plan)

    def test_cost_ceiling_still_requires_the_all_in_bound_to_fit(self):
        plan = cost_plan()
        result = P.cost_bound(plan)
        self.assertLessEqual(float(result['all_in_bound_usd']), 30)
        plan['cost']['cap_usd'] = 1
        with self.assertRaises(ValueError):
            P.cost_bound(plan)
        plan['cost']['cap_usd'] = 30
        plan['cost']['rates']['ec2_hour'] = 4
        with self.assertRaises(ValueError):
            P.cost_bound(plan)

    def test_evidence_costing_does_not_import_unbundled_public_work(self):
        original_import = __import__

        def without_public_work(name, globals=None, locals=None, fromlist=(), level=0):
            if name.endswith('public_work_experiment') or 'public_work_experiment' in (fromlist or ()):
                raise ModuleNotFoundError('public_work_experiment is absent from evidence bundle')
            return original_import(name, globals, locals, fromlist, level)

        plan = cost_plan()
        plan['bundle']['bytes'] = 128 * 1024**2
        comparator = comparator_cost_plan()
        with patch('builtins.__import__', side_effect=without_public_work):
            result = P.cost_bound(plan)
            self.assertLessEqual(float(result['all_in_bound_usd']), 30)
            plan['bundle']['bytes'] += 1
            with self.assertRaisesRegex(ValueError, 'storage/network/request/tax'):
                P.cost_bound(plan)
            with self.assertRaisesRegex(ModuleNotFoundError, 'absent from evidence bundle'):
                P.cost_bound(comparator)

    def test_bundle_cap_cannot_be_supplied_by_caller(self):
        plan = cost_plan()
        plan['bundle'].update(bytes=128 * 1024**2 + 1, bytes_max=P.GIB)
        plan['cost']['bundle_bytes_max'] = P.GIB
        plan['bundle_bytes_max'] = P.GIB
        with self.assertRaisesRegex(ValueError, 'storage/network/request/tax'):
            P.cost_bound(plan)
        with self.assertRaises(TypeError):
            P.cost_bound(plan, bundle_bytes_max=P.GIB)

    def test_bundle_cap_requires_exact_public_work_comparator(self):
        for schema in (P.SCHEMA, W.SCHEMA, W.OFFLINE_PROJECTION_SCHEMA,
                       'trid-public-work-prerequisites-v1', 'unknown'):
            plan = cost_plan()
            plan['schema'] = schema
            plan['bundle']['bytes'] = 128 * 1024**2
            with self.subTest(schema=schema):
                P.cost_bound(plan)
                plan['bundle']['bytes'] += 1
                with self.assertRaisesRegex(ValueError, 'storage/network/request/tax'):
                    P.cost_bound(plan)
        comparator = comparator_cost_plan()
        legacy = cost_plan()
        legacy['bundle']['bytes'] = 128 * 1024**2
        self.assertEqual(P.cost_bound(comparator), P.cost_bound(legacy))
        self.assertEqual(W.cost_bound(comparator), P.cost_bound(comparator))
        comparator['bundle']['bytes'] += 1
        with self.assertRaisesRegex(ValueError, 'storage/network/request/tax'):
            P.cost_bound(comparator)
        for schema in (P.SCHEMA, W.OFFLINE_PROJECTION_SCHEMA,
                       'trid-public-work-prerequisites-v1', 'unknown'):
            plan = comparator_cost_plan()
            plan['schema'] = schema
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, 'storage/network/request/tax'):
                P.cost_bound(plan)

    def test_comparator_bundle_allowance_rejects_tampered_original(self):
        mutations = [
            ('run_id', lambda spec: spec.update(run_id='different-run')),
            ('step', lambda spec: spec.update(step=531)),
            ('model', lambda spec: spec['model'].update(id='different-model')),
            ('revision', lambda spec: spec['model'].update(revision='0'*40)),
            ('inventory', lambda spec: spec['files'].pop(W.PRIOR_WEIGHT_PATH)),
            ('weight_bytes', lambda spec: spec['files'][W.PRIOR_WEIGHT_PATH].update(bytes=1)),
            ('weight_sha', lambda spec: spec['files'][W.PRIOR_WEIGHT_PATH].update(sha256='0'*64)),
        ]
        mutations += [(name, lambda spec, name=name: spec['files'][name].update(sha256='0'*64))
            for name in W.PRIOR_RECEIPT_SHA]
        for name, mutate in mutations:
            for bundle_bytes in (128 * 1024**2, 256 * 1024**2):
                plan = comparator_cost_plan()
                plan['bundle']['bytes'] = bundle_bytes
                mutate(plan['workload']['prior_adapter'])
                with self.subTest(field=name, bundle_bytes=bundle_bytes), self.assertRaisesRegex(
                        ValueError, 'Original first-adapter'):
                    P.cost_bound(plan)

    def test_comparator_bundle_keeps_storage_network_and_budget_guards(self):
        minimum_retained = (256 + 512 + 8) * 1024**2
        plan = comparator_cost_plan()
        plan['cost']['retained_bytes'] = minimum_retained
        P.cost_bound(plan)
        for section, key, value in (
            ('cost', 'retained_bytes', minimum_retained - 1),
            ('cost', 'retained_bytes', 2 * P.GIB + 1),
            ('cost', 'retention_days', 31),
            ('cost', 'network_egress_bytes', 2 * P.GIB),
            ('cost', 'controller_retrieval_bytes', (512 + 121) * 1024**2 - 1),
            ('cost', 'controller_retrieval_bytes', P.GIB + 1),
            ('cost', 'export_bytes', 512 * 1024**2 + 1),
            ('network', 'egress_byte_cap', (512 + 16) * 1024**2 - 1),
            ('network', 'egress_byte_cap', 2 * P.GIB + 1),
        ):
            plan = comparator_cost_plan()
            plan[section][key] = value
            with self.subTest(field=key, value=value), self.assertRaisesRegex(
                    ValueError, 'storage/network/request/tax'):
                P.cost_bound(plan)
        plan = comparator_cost_plan()
        plan['cost']['cap_usd'] = 1
        with self.assertRaisesRegex(ValueError, 'All-in bound exceeds'):
            P.cost_bound(plan)
        plan['cost']['cap_usd'] = 30.01
        with self.assertRaisesRegex(ValueError, 'finite positive JSON number'):
            P.cost_bound(plan)
        plan['cost']['cap_usd'] = 30
        plan['cost']['rates']['ec2_hour'] = 4
        with self.assertRaisesRegex(ValueError, 'All-in bound exceeds'):
            P.cost_bound(plan)

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
        for field, value in [('state', 'CANDIDATE_NEEDS_INDEPENDENT_REVIEW'), ('training_rows', 630),
                             ('schema', 'trid-public-evidence-cloud-admission-v1'),
                             ('recipe', dict(P.RECIPE, rows=630, steps=315)),
                             ('evaluation_rows', 200), ('general_training_rows', 245), ('general_training_rows', 247),
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
            self.assertEqual(P.validate_training(root, plan)['steps'], 615)
            shuffled = copy.deepcopy(report); shuffled['coverage'].reverse()
            provenance['training'] = save(root, 'training/training.json', shuffled)
            save(root, 'training/provenance.json', provenance)
            self.assertEqual(len(P.validate_training(root, plan)['coverage']), 1230)
            variants = []
            duplicate = copy.deepcopy(report); duplicate['coverage'][-1] = duplicate['coverage'][0]; variants.append(duplicate)
            reordered_identity = copy.deepcopy(report); reordered_identity['coverage'][0]['id'] = 'train-1'; variants.append(reordered_identity)
            truncated = copy.deepcopy(report); truncated['history'].pop(); variants.append(truncated)
            bad_lr = copy.deepcopy(report); bad_lr['history'][0]['applied_lr'] = 1e-5; variants.append(bad_lr)
            bad_loss = copy.deepcopy(report); bad_loss['history'][1]['train_loss'] = float('inf'); variants.append(bad_loss)
            old = copy.deepcopy(report); old.update(steps=315, cumulative_steps=315, train_rows=630); variants.append(old)
            foreign = copy.deepcopy(report); foreign['coverage'][0]['source_sha256'] = '0'*64; variants.append(foreign)
            omitted = copy.deepcopy(report); omitted['coverage'].pop(); variants.append(omitted)
            extra = copy.deepcopy(report); extra['coverage'].append(extra['coverage'][0]); variants.append(extra)
            out_of_range = copy.deepcopy(report); out_of_range['coverage'][0]['source_line'] = 1231; variants.append(out_of_range)
            old_history = copy.deepcopy(report); old_history['history'] = old_history['history'][:315]; variants.append(old_history)
            old_warmup = copy.deepcopy(report); old_warmup['history'][0]['applied_lr'] = 1e-5/15; variants.append(old_warmup)
            wrong_step = copy.deepcopy(report); wrong_step['history'][-1]['step'] = 614; variants.append(wrong_step)
            nonfinite_lr = copy.deepcopy(report); nonfinite_lr['history'][0]['next_lr'] = float('nan'); variants.append(nonfinite_lr)
            dropped = copy.deepcopy(report); dropped['dropped_over_max_tokens'] = 1; variants.append(dropped)
            warmstart = copy.deepcopy(report); warmstart['init_adapter'] = 'previous'; variants.append(warmstart)
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
        with self.assertRaises(ValueError): P.generation_binding(plan, 'baseline', dict(config={}, weights={}, step=615))
        with self.assertRaises(ValueError): P.generation_binding(plan, 'adapter', dict(config={}, weights={}, step=315))

    def test_full_native_exception_receipts_stay_in_denominator(self):
        for arm in ('baseline', 'adapter'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); plan, records = output_fixture(root, arm)
                result = P.validate_outputs(root, plan, arm)
                self.assertEqual(len(result), 708)
                self.assertTrue(all(row['error'] is not None for row in result))
                for field, value in [('id', 'unknown'), ('seed', 0), ('generation_receipt', 'external_unverified'),
                                     ('context_truncated', True), ('output_tokens', 2),
                                     ('binding', dict(records[0]['binding'], schema='trid-public-evidence-generation-binding-v1'))]:
                    bad = copy.deepcopy(records); bad[0][field] = value
                    (root/arm/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad))
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        P.validate_outputs(root, plan, arm)
                for bad in (records[:-1], records + [records[0]], list(reversed(records))):
                    (root/arm/'responses.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in bad))
                    with self.assertRaises(ValueError):
                        P.validate_outputs(root, plan, arm)

    def test_capacity_requires_resident_second_update_and_precision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); plan, _, provenance = training_fixture(root)
            receipt = dict(synthetic=True, discarded=True, microbatches=4, context_tokens=4096, answer_tokens=768,
                admitted_expanded_rows=1230, finite_parameter_update=True, adapter_reload_equal=True,
                optimizer_updates=2, microbatches_per_update=2, optimizer_state_resident_before_second_update=True,
                gradient_checkpointing=True, losses=[1.0]*4, max_allocated_bytes=1, max_reserved_bytes=2,
                plan_sha256=provenance['plan_sha256'], source_manifest=provenance['source_manifest'],
                runtime=provenance['runtime'], parameter_inventory=provenance['parameter_inventory'])
            save(root, 'capacity-receipt.json', receipt); P.validate_capacity(root, plan)
            receipt['admitted_expanded_rows'] = 630
            save(root, 'capacity-receipt.json', receipt)
            with self.assertRaises(ValueError): P.validate_capacity(root, plan)
            receipt['admitted_expanded_rows'] = 1230
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
