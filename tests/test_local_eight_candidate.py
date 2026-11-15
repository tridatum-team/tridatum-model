"""Synthetic byte/receipt contracts only; no model imports or runtime claims."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import local_eight_candidate as C
from scripts import assess_expanded_experiment as A
from scripts import evaluate_local_eight as E
from test_expanded_assessment import SyntheticExperiment, dump, load, file_pin, sha, vote


class CandidateFixture:
    def __init__(self, root):
        self.root = Path(root)
        self.protocol = dict(schema='trid-local-eight-review-protocol-v2', instructions='Synthetic prospective instructions.',
            paired_consistency=True, critical_errors_are_codes=True,
            critical_error_codes={'wrong_actor': 'A materially wrong actor.'})
        self.protocol_path = dump(self.root / 'protocol.json', self.protocol)
        train = self.root / 'train.jsonl'
        train.write_text(''.join(json.dumps({'id': str(n)}) + '\n' for n in range(600)))
        manifest = dump(self.root / 'training-manifest.json', dict(schema='trid-expanded-training-manifest-v1',
            state='ADMITTED', rows=600, valid=[], train=[dict(file_pin(train), epochs=1)]))
        acquisition = dump(self.root / 'acquisition.json', dict(repo_id=C.MODEL_ID, revision=C.REVISION))
        self.policy = dict(schema='trid-local-eight-training-policy-v1', experiment_id='synthetic-cycle',
            model=dict(acquisition_manifest=file_pin(acquisition)),
            data=dict(training_manifest=file_pin(manifest)),
            training=dict(steps=150, batch_size=4, micro_batch_size=1, lr=1e-5, seed=14,
                max_tokens=3072, max_answer_tokens=768, family='qwen3', rank=8, alpha=16, dropout=0.,
                targets=['q_proj', 'k_proj']),
            evaluation=dict(review_protocol=file_pin(self.protocol_path)))
        self.policy_path = dump(self.root / 'policy.json', self.policy)
        self.directory = self.root / 'worker' / 'adapter'
        config = dict(copy.deepcopy(C.FIXED_LORA_CONFIG),
                      base_model_name_or_path=C.MODEL_ID, target_modules=['q_proj', 'k_proj'])
        dump(self.directory / 'adapter_config.json', config)
        for name in ('adapter_model.safetensors', 'README.md'):
            (self.directory / name).write_text('synthetic bytes, not model weights')
        self.weights_sha = sha(self.directory / 'adapter_model.safetensors')
        self.config_sha = sha(self.directory / 'adapter_config.json')
        lineage = dict(base_model_id=C.MODEL_ID, base_revision=C.REVISION,
            base_weights=dict(representation='hf-safetensors', dtype='torch.bfloat16'), init_adapter=None,
            manifest_sha256=sha(manifest))
        checkpoints = [dict(step=n, adapter_weights_sha256=self.weights_sha,
                            adapter_config_sha256=self.config_sha) for n in (50, 100, 150)]
        last = dict(step=150, path=str(self.directory), adapter_weights_sha256=self.weights_sha,
                    adapter_config_sha256=self.config_sha, equals_last_checkpoint=True)
        self.training_path = dump(self.root / 'worker' / 'training.json', dict(lineage,
            status='complete', steps=150, cumulative_steps=150, train_rows=600, valid_rows=0,
            dropped_over_max_tokens=0, valid_dropped=0, device='mps', dtype='torch.bfloat16',
            policy=file_pin(self.policy_path), coverage_complete=True,
            coverage=[dict(file=str(train), source_sha256=sha(train), source_line=n, id=str(n-1)) for n in range(1,601)],
            dtype_inventory={group:{dtype+'@mps:0':dict(tensors=1,elements=1,bytes=4)}
                             for group,dtype in [('base','torch.bfloat16'),('adapter','torch.float32')]},
            **{k:self.policy['training'][k] for k in ('batch_size','micro_batch_size','lr','seed','max_tokens','max_answer_tokens','family')},
            lora={k:self.policy['training'][k] for k in ('rank','alpha','dropout','targets')},
            history=[dict(step=n) for n in range(1,151)], checkpoints=checkpoints, last_step=last))
        self.ledger_path = dump(self.root / 'worker' / 'checkpoints.json', dict(lineage,
            run_status='complete', planned_steps=150, checkpoint_every=50, checkpoints=checkpoints, last_step=last))
        source = dump(self.root / 'source.py', {'synthetic_source': True})
        sources = dump(self.root / 'sources.json', dict(schema='trid-local-mps-sources-v1', files=[file_pin(source)]))
        worker = dict(load(self.training_path), actual_source_files=[file_pin(source)])
        dump(self.training_path, worker)
        capacity_worker = dump(self.root / 'capacity-worker.json', dict(status='complete', phase='capacity',
            policy=file_pin(self.policy_path), actual_source_files=[file_pin(source)], optimizer_updates=2,
            microbatches_per_update=4, reload_exact=True, reload_loss=1.))
        common = dict(schema='trid-local-mps-supervised-v1', status='accepted', policy=file_pin(self.policy_path),
            sources=file_pin(sources), source_files=[file_pin(source)], actual_source_files=[file_pin(source)],
            cleanup=dict(accepted=True, leader_reaped=True, remaining_members=[]))
        capacity = dump(self.root / 'capacity-result.json', dict(common, phase='capacity', worker_result=file_pin(capacity_worker)))
        self.supervision_path = dump(self.root / 'result.json', dict(common, phase='train',
            worker_result=file_pin(self.training_path), ledger=file_pin(self.ledger_path), capacity_result=file_pin(capacity),
            adapter_files={name:file_pin(self.directory/name) for name in ('adapter_config.json','adapter_model.safetensors')}))
        self.registration_path = self.root / 'candidate.json'
        self.registration = dict(schema='trid-local-eight-candidate-v1', policy=file_pin(self.policy_path),
            candidate_id='synthetic-cycle', arm='candidate', step=150, supervision=file_pin(self.supervision_path),
            training=file_pin(self.training_path), checkpoint_ledger=file_pin(self.ledger_path),
            adapter=dict(directory=str(self.directory), files={p.name:file_pin(p) for p in self.directory.iterdir()}))
        self.repin()

    def repin(self):
        supervisor = load(self.supervision_path)
        supervisor.update(worker_result=file_pin(self.training_path), ledger=file_pin(self.ledger_path))
        dump(self.supervision_path, supervisor)
        self.registration.update(supervision=file_pin(self.supervision_path), training=file_pin(self.training_path),
                                 checkpoint_ledger=file_pin(self.ledger_path))
        dump(self.registration_path, self.registration)

    def resolve(self):
        with patch.object(C, 'POLICY_SHA', sha(self.policy_path)), patch.object(C, 'PROTOCOL_SHA', sha(self.protocol_path)):
            return C.resolve_candidate(self.root, self.registration_path, sha(self.registration_path))


class CandidateIdentityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = CandidateFixture(temporary.name)

    def test_final_registered_endpoint_and_no_private_reads(self):
        result = self.fixture.resolve()
        self.assertEqual(result['arms'], ['base', 'candidate'])
        self.assertEqual(result['adapter']['step'], 150)
        self.assertEqual(result['adapter_sha256'], self.fixture.weights_sha)
        original = C.load
        def public_only(path):
            self.assertNotEqual(Path(path), self.fixture.protocol_path)
            self.assertNotIn('reference', str(path))
            return original(path)
        with patch.object(C, 'load', side_effect=public_only):
            self.fixture.resolve()

    def test_stale_registration_hash(self):
        with self.assertRaisesRegex(ValueError, 'registration hash'):
            C.resolve_candidate(self.fixture.root, self.fixture.registration_path, '0'*64)

    def test_wrong_step_and_warm_start_and_configuration_rejected(self):
        original = load(self.fixture.training_path)
        for key, value in [('steps',100), ('init_adapter',{'path':'warm'}), ('lr',2e-5),
                           ('batch_size',8), ('train_rows',599), ('valid_rows',1), ('dtype','torch.float32')]:
            with self.subTest(key=key):
                changed = dict(original, **{key:value})
                dump(self.fixture.training_path, changed)
                self.fixture.repin()
                with self.assertRaises(ValueError):
                    self.fixture.resolve()
        dump(self.fixture.training_path, original)
        self.fixture.registration['step'] = 100
        self.fixture.repin()
        with self.assertRaisesRegex(ValueError, 'step150'):
            self.fixture.resolve()

    def test_guard_failed_or_cleanup_failed_cannot_nominate(self):
        original = load(self.fixture.supervision_path)
        for changed in (dict(original, status='failed'), dict(original, cleanup=dict(accepted=False,
                leader_reaped=True, remaining_members=[])), dict(original, cleanup=dict(accepted=True,
                leader_reaped=False, remaining_members=[123]))):
            dump(self.fixture.supervision_path, changed)
            self.fixture.repin()
            with self.assertRaises(ValueError):
                self.fixture.resolve()

    def test_changed_weights_config_manifest_or_protocol_rejected(self):
        paths = [self.fixture.directory/'adapter_model.safetensors', self.fixture.directory/'adapter_config.json',
                 self.fixture.root/'training-manifest.json', self.fixture.protocol_path, self.fixture.root/'source.py']
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(original+b' ')
                with self.assertRaises(ValueError):
                    self.fixture.resolve()
                path.write_bytes(original)

    def test_rebound_config_still_rejects_base_mutation(self):
        path = self.fixture.directory/'adapter_config.json'
        dump(path, dict(load(path), bias='all'))
        self.fixture.registration['adapter']['files'][path.name] = file_pin(path)
        self.fixture.repin()
        with self.assertRaisesRegex(ValueError, 'evidence hash/size'):
            self.fixture.resolve()

    def test_rebound_weights_must_match_accepted_final_ledger(self):
        path = self.fixture.directory/'adapter_model.safetensors'
        path.write_text('different synthetic weight bytes')
        self.fixture.registration['adapter']['files'][path.name] = file_pin(path)
        self.fixture.repin()
        with self.assertRaisesRegex(ValueError, 'evidence hash/size'):
            self.fixture.resolve()

    def test_rebound_config_cannot_bypass_ledger_with_rebound_parent_file_pin(self):
        path=self.fixture.directory/'adapter_config.json'
        dump(path,dict(load(path),alpha_pattern={'q_proj':32}))
        self.fixture.registration['adapter']['files'][path.name]=file_pin(path)
        supervisor=load(self.fixture.supervision_path)
        supervisor['adapter_files'][path.name]=file_pin(path)
        dump(self.fixture.supervision_path,supervisor)
        self.fixture.repin()
        with self.assertRaisesRegex(ValueError,'config/weights binding'):
            self.fixture.resolve()

    def test_unregistered_effective_peft_overrides_rejected_even_with_rebound_producer(self):
        path=self.fixture.directory/'adapter_config.json'
        original=load(path)
        overrides = [
            ('alpha_pattern',{'q_proj':32}), ('rank_pattern',{'q_proj':16}),
            ('layer_replication',[[0,1]]), ('init_lora_weights','pissa'),
            ('init_lora_weights','pissa_niter_4'), ('init_lora_weights','loftq'),
            ('init_lora_weights','olora'), ('init_lora_weights','corda'),
            ('init_lora_weights','eva'), ('use_rslora',True), ('use_dora',True),
            ('use_qalora',True), ('lora_bias',True), ('bias','all'),
            ('modules_to_save',['lm_head']), ('target_parameters',['embed_tokens.weight']),
            ('trainable_token_indices',[0]), ('exclude_modules',['q_proj']),
            ('layers_to_transform',[0]), ('layers_pattern','layers'),
            ('fan_in_fan_out',True), ('loftq_config',{'loftq_bits':4}),
            ('future_architecture_override',True)]
        for key,value in overrides:
            with self.subTest(override=key,value=value):
                dump(path,dict(original,**{key:value}))
                self.fixture.registration['adapter']['files'][path.name]=file_pin(path)
                for producer_path in (self.fixture.training_path,self.fixture.ledger_path):
                    producer=load(producer_path)
                    producer['last_step']['adapter_config_sha256']=sha(path)
                    producer['checkpoints'][-1]['adapter_config_sha256']=sha(path)
                    dump(producer_path,producer)
                supervisor=load(self.fixture.supervision_path)
                supervisor['adapter_files'][path.name]=file_pin(path)
                dump(self.fixture.supervision_path,supervisor)
                self.fixture.repin()
                with self.assertRaisesRegex(ValueError,'base isolation'):
                    self.fixture.resolve()

    def test_capacity_guard_failure_survives_repinning(self):
        path = self.fixture.root/'capacity-result.json'
        dump(path,dict(load(path),status='failed'))
        supervisor=load(self.fixture.supervision_path)
        supervisor['capacity_result']=file_pin(path)
        dump(self.fixture.supervision_path,supervisor)
        self.fixture.repin()
        with self.assertRaisesRegex(ValueError,'accept capacity'):
            self.fixture.resolve()

    def test_coverage_and_actual_dtype_cannot_be_claimed_by_counts(self):
        original=load(self.fixture.training_path)
        duplicate=copy.deepcopy(original)
        duplicate['coverage'][-1]=duplicate['coverage'][0]
        wrong_dtype=copy.deepcopy(original)
        wrong_dtype['dtype_inventory']['adapter']={'torch.bfloat16@mps:0':dict(tensors=1,elements=1,bytes=2)}
        for changed in (duplicate,wrong_dtype):
            dump(self.fixture.training_path,changed)
            self.fixture.repin()
            with self.assertRaises(ValueError):
                self.fixture.resolve()

    def test_candidate_answer_hash_never_leaks_to_base(self):
        contract = self.fixture.resolve()
        decoded = dict(response_text='synthetic', termination_reason='eos')
        with patch.object(E, 'decode_generation', return_value=decoded):
            for arm in contract['arms']:
                row = E.answer_record({}, arm, {}, {'eos_token_id':1}, [1], None, 0., {}, contract)
                self.assertEqual(row['adapter_weights_sha256'], None if arm == 'base' else self.fixture.weights_sha)


class CandidateAnonymousReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.fixture = SyntheticExperiment(self.root)
        self.packet_path, _ = self.fixture.packet()
        packet = load(self.packet_path)
        packet.update(schema='trid-expanded-blind-packet-v2', instructions='Synthetic paired instructions.',
            paired_consistency=True, critical_errors_are_codes=True,
            critical_error_codes={'wrong_actor':'The actor is materially wrong.'})
        pairs = {}
        for row in packet['rows']:
            row['pair_id'] = pairs.setdefault(A.object_sha(row['messages']), 'opaque-'+str(len(pairs)))
        dump(self.packet_path, packet)
        self.protocol_hash = A.object_sha({key:packet[key] for key in A.PROTOCOL_FIELDS})
        self.patcher = patch.object(A, 'ANONYMOUS_PROTOCOL_SHA', self.protocol_hash)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_shards_preserve_protocol_pairs_and_anonymity(self):
        packet = load(self.packet_path)
        out = self.root/'paired-shards'
        A.shard(self.root, self.packet_path, sha(self.packet_path), 7, out)
        index, _, children = A.shard_index(self.root, out/'shard-index.json', sha(out/'shard-index.json'))
        covered = []
        for _, child in children.values():
            self.assertTrue(all(child[key] == packet[key] for key in A.PROTOCOL_FIELDS))
            counts = {}
            for row in child['rows']:
                counts[row['pair_id']] = counts.get(row['pair_id'],0)+1
                self.assertFalse({'arm','step','candidate_id','elapsed_seconds'} & set(row))
                covered.append(row['sample_id'])
            self.assertEqual(set(counts.values()), {2})
        self.assertEqual(len(covered),len(set(covered)))
        self.assertEqual(len(covered),400)
        with self.assertRaisesRegex(ValueError, 'both responses'):
            A.shard(self.root, self.packet_path, sha(self.packet_path), 1, self.root/'too-small')

    def test_whole_pair_third_review_and_frozen_propagation(self):
        packet = load(self.packet_path)
        reports = self.fixture.reports(self.packet_path)
        changed = load(reports[1])
        sid = packet['rows'][0]['sample_id']
        changed['judgments'][sid] = vote(False, errors=['wrong_actor'])
        dump(reports[1], changed)
        _, _, _, disputes = A.review_pair(self.packet_path,reports,self.root)
        self.assertEqual(len(disputes),2)
        out = self.root/'adjudication'
        A.adjudication(self.root,self.packet_path,reports,out)
        adjudication_path=out/'adjudication-packet.json'
        adjudicated=load(adjudication_path)
        self.assertTrue(all(adjudicated[key]==packet[key] for key in A.PROTOCOL_FIELDS))
        A.validate_blind_packet(adjudicated,partial=True)
        third=dump(self.root/'third.json',dict(schema='trid-expanded-blind-review-v1',
            packet_sha256=sha(self.packet_path),adjudication_packet_sha256=sha(adjudication_path),
            reviewer_id='third-independent',independent=True,blinded=True,
            judgments={key:vote(False,errors=['wrong_actor']) for key in disputes}))
        frozen=self.fixture.freeze(self.packet_path,reports,third=third,adjudication_path=adjudication_path)
        result=load(frozen)
        self.assertFalse(result['identities_joined'])
        self.assertEqual(len(result['judgments']),400)
        self.assertEqual(result['disagreements'],disputes)
        self.assertEqual(len(result['reports']),3)

    def test_protocol_tamper_codes_and_missing_counterpart_rejected(self):
        packet=load(self.packet_path)
        changed=dict(packet,instructions='post-output changed rubric')
        with self.assertRaisesRegex(ValueError,'protocol changed'):
            A.validate_blind_packet(changed)
        changed=copy.deepcopy(packet)
        changed['rows'][0]['pair_id']='unpaired'
        with self.assertRaisesRegex(ValueError,'counterpart'):
            A.validate_blind_packet(changed)
        reports=self.fixture.reports(self.packet_path)
        report=load(reports[0])
        sid=packet['rows'][0]['sample_id']
        report['judgments'][sid]=vote(False,errors=['free-form description instead of code'])
        with self.assertRaisesRegex(ValueError,'critical-error code'):
            A.review_document(report,packet,sha(self.packet_path),root=self.root)


def producer_source_fixture(root):
    """Synthetic bytes at the producer's real repository-relative source names."""
    return {name: file_pin(dump(root / name, {'synthetic_source': name}))
            for name in A.CANDIDATE_PRODUCER_SOURCES}


class CandidateProducerSourceTests(unittest.TestCase):
    def test_both_receipts_require_every_mandatory_producer_dependency(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = producer_source_fixture(root)
            A.authenticate_producer_sources(root, sources, sources)
            for name in A.CANDIDATE_PRODUCER_SOURCES:
                missing = {key: entry for key, entry in sources.items() if key != name}
                for first, second in ((missing, sources), (sources, missing), (missing, missing)):
                    with self.subTest(omitted=name), self.assertRaisesRegex(ValueError, 'mandatory producer'):
                        A.authenticate_producer_sources(root, first, second)

    def test_unrelated_file_cannot_substitute_a_correct_source_key(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = producer_source_fixture(root)
            unrelated = file_pin(dump(root / 'unrelated.py', {'synthetic': True}))
            for name in A.CANDIDATE_PRODUCER_SOURCES:
                substituted = dict(sources, **{name: unrelated})
                with self.subTest(substituted=name), self.assertRaisesRegex(ValueError, 'path substitution'):
                    A.authenticate_producer_sources(root, substituted, substituted)
            with self.assertRaisesRegex(ValueError, 'mandatory producer'):
                A.authenticate_producer_sources(root, {'source.py': unrelated}, {'source.py': unrelated})

    def test_same_files_accept_unicode_or_directory_alias_spelling(self):
        import unicodedata
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / '사업'
            sources = producer_source_fixture(root)
            alias = parent / unicodedata.normalize('NFD', '사업')
            if not alias.exists():
                alias = parent / 'repository-alias'
                alias.symlink_to(root, target_is_directory=True)
            alternate = {name: dict(entry, path=str(alias / name)) for name, entry in sources.items()}
            A.authenticate_producer_sources(root, sources, alternate)
            A.authenticate_producer_sources(alias, alternate, sources)

    def test_source_key_and_runtime_closure_are_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = producer_source_fixture(root)
            extra = file_pin(dump(root / 'scripts' / 'extra.py', {'synthetic': True}))
            with self.assertRaisesRegex(ValueError, 'changed between runtime'):
                A.authenticate_producer_sources(root, dict(sources, **{'scripts/extra.py': extra}), sources)
            with self.assertRaisesRegex(ValueError, 'source key'):
                A.authenticate_producer_sources(root, dict(sources, **{'scripts/../extra.py': extra}), sources)


class CandidateAssessmentIntegrationTests(unittest.TestCase):
    def test_one_shared_candidate_pipeline_and_development_only_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SyntheticExperiment(root)
            directory = fixture.primary
            manifest = load(directory / 'run-manifest.json')
            adapter = dict(manifest['adapter'], step=150)
            registration = dump(root / 'registration.json', {'synthetic': True})
            policy = dump(root / 'policy.json', {'synthetic': True})
            protocol = dict(instructions='Synthetic paired protocol.', paired_consistency=True,
                critical_errors_are_codes=True, critical_error_codes={'wrong_actor':'Wrong actor.'})
            protocol_path = dump(root / 'protocol.json', protocol)
            source = dump(root / 'acquisition.json', {'synthetic': True})
            sources = producer_source_fixture(root)
            contract = dict(arms=['base','candidate'], adapter_sha256=fixture.design['model']['adapter_sha256'],
                adapter=adapter, adapter_directory=str(root/'adapter'), candidate_id='synthetic',
                registration=file_pin(registration), policy=file_pin(policy), review_protocol=file_pin(protocol_path),
                policy_document=dict(data=dict(frozen_dataset=file_pin(fixture.frozen_path)),
                    evaluation=dict(development_gate={'synthetic':'prospective'})))
            manifest.update(schema='trid-local-eight-paired-v2', arm_order=contract['arms'], adapter=adapter,
                evaluation_contract=C.public_contract(contract), code=sources)
            manifest['model'].update(directory=str(root/'model'), acquisition_manifest=file_pin(source))
            dump(directory/'run-manifest.json',manifest)
            loaded=load(directory/'loaded-runtime.json')
            loaded['code']=sources
            dump(directory/'loaded-runtime.json',loaded)
            for old_arm, arm in [('base','base'),('step-000024','candidate')]:
                rows=fixture.rows(directory,old_arm)
                for row in rows:
                    row.update(arm=arm, run_manifest_sha256=sha(directory/'run-manifest.json'),
                        loaded_runtime_sha256=sha(directory/'loaded-runtime.json'),
                        candidate_manifest_sha256=sha(registration))
                fixture.write_rows(directory,arm,rows,refresh=False)
            completion=load(directory/'completion.json')
            completion.update(run_manifest_sha256=sha(directory/'run-manifest.json'),
                outputs={arm:file_pin(directory/(arm+'-outputs.jsonl')) for arm in contract['arms']})
            dump(directory/'completion.json',completion)
            with patch.object(A,'resolve_candidate',return_value=contract), \
                 patch.object(A,'review_protocol',return_value=protocol), \
                 patch.object(A,'ANONYMOUS_PROTOCOL_SHA',A.object_sha(protocol)), \
                 patch('scripts.evaluate_local_eight.model_identity',return_value=manifest['model']):
                out=root/'candidate-blind'
                A.make_packet(root,fixture.frozen_path,sha(fixture.frozen_path),directory,out,
                    candidate_manifest=registration,candidate_sha=sha(registration))
                packet_path=out/'review-packet.json'
                packet=load(packet_path)
                self.assertEqual(packet['schema'],'trid-expanded-blind-packet-v2')
                self.assertEqual(len(packet['rows']),400)
                reports=fixture.reports(packet_path)
                frozen=fixture.freeze(packet_path,reports)
                result=fixture.aggregate(frozen,out/'private-mapping.json')
                self.assertEqual(result['schema'],'trid-expanded-assessment-v2')
                self.assertEqual(set(result['primary']['arms']),{'base','candidate'})
                self.assertFalse(result['development_gate']['passed'])
                self.assertFalse(result['deployment_qualified'])
                self.assertIsNone(result['selected_step'])
                self.assertNotIn('diagnostic_comparator_step',result)
                self.assertEqual(result['role'],'development_only_not_unseen_final')
                manifest['schema']='trid-local-eight-paired-v1'
                dump(directory/'run-manifest.json',manifest)
                with self.assertRaisesRegex(ValueError,'identity/mode'):
                    A.completed_run(root,directory,{r['id']:r for r in fixture.inputs},
                        fixture.design,'unconstrained',contract=contract)


class CudaExportIdentityTests(unittest.TestCase):
    """Byte-only cloud completion tests; no cloud/model execution."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.state = Path(self.temporary.name)
        self.output = self.state / 'fresh-export'
        self.output.mkdir()
        self.plan = dict(identity=dict(run_id='synthetic-cuda'),
                         output_allowlist=['candidate.json', 'plan.json'])
        dump(self.output / 'candidate.json', {'synthetic': True})
        dump(self.output / 'plan.json', self.plan)
        self.plan_sha = sha(self.output / 'plan.json')
        self.status = dict(run_id='synthetic-cuda', plan_sha256=self.plan_sha,
            experimental_success=True, stop_errors=[], instance_id='i-synthetic')
        members = [dict(file_pin(self.output / name), path=name) for name in self.plan['output_allowlist']]
        dump(self.output / 'export-manifest.json', dict(run_id='synthetic-cuda', plan_sha256=self.plan_sha,
            status=self.status, files=members, total_bytes=sum(p['bytes'] for p in members)))
        self.completion = dict(self.status, export=dict(sha256='a' * 64, bytes=1000))
        dump(self.state / 'completion.json', self.completion)
        self.controller = dict(state='complete', experimental_success=True,
            evidence=dict(native_pipeline_success=True, completion=self.completion, export_sha256='a' * 64),
            cleanup=dict(cleanup_state='complete', instance_terminated=True, instance_id='i-synthetic',
                root_volume_ids=['vol-root'], recorded_volume_ids=['vol-root'],
                absent_volume_ids=['vol-root'], remaining_volumes=[]))
        dump(self.state / 'controller-result.json', self.controller)

    def test_relocated_complete_export_is_accepted(self):
        C.validate_cuda_export(self.output, self.plan, self.plan_sha)

    def test_failure_partial_and_unclosed_cloud_never_qualify(self):
        for changed in (
            dict(self.controller, state='native_failed'),
            dict(self.controller, schedule_cleanup_error='still armed'),
            dict(self.controller, cleanup=dict(self.controller['cleanup'], instance_terminated=False)),
            dict(self.controller, cleanup=dict(self.controller['cleanup'], absent_volume_ids=[])),
            dict(self.controller, evidence=dict(self.controller['evidence'], export_sha256='b' * 64)),
        ):
            with self.subTest(changed=changed):
                dump(self.state / 'controller-result.json', changed)
                with self.assertRaises(ValueError):
                    C.validate_cuda_export(self.output, self.plan, self.plan_sha)
        dump(self.state / 'controller-result.json', self.controller)
        (self.output / 'candidate.json').unlink()
        with self.assertRaises(ValueError):
            C.validate_cuda_export(self.output, self.plan, self.plan_sha)

    def test_failure_artifact_or_rebound_failed_export_rejected(self):
        dump(self.output / 'failure.json', dict(status='failed'))
        with self.assertRaises(ValueError):
            C.validate_cuda_export(self.output, self.plan, self.plan_sha)
        (self.output / 'failure.json').unlink()
        path = self.output / 'export-manifest.json'
        export = load(path)
        export['status']['experimental_success'] = False
        dump(path, export)
        with self.assertRaisesRegex(ValueError, 'not successful'):
            C.validate_cuda_export(self.output, self.plan, self.plan_sha)


class CudaCandidateIdentityTests(unittest.TestCase):
    def setUp(self):
        from scripts.aws_eval import fresh_eight_experiment as F
        self.F = F
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = CandidateFixture(self.root)
        self.output = self.root / 'export'
        train = self.output / 'train'
        train.mkdir(parents=True)
        self.fixture.directory.rename(train / 'adapter')
        self.training = load(self.fixture.training_path)
        self.training.update(device='cuda', dtype_inventory={
            group: {dtype + '@cuda:0': dict(tensors=1, elements=1, bytes=4)}
            for group, dtype in [('base', 'torch.bfloat16'), ('adapter', 'torch.float32')]})
        dump(train / 'checkpoints.json', load(self.fixture.ledger_path))
        dump(train / 'training-worker.json', {'synthetic': True})
        dump(self.output / 'capacity/capacity-worker.json', {'synthetic': True})
        self.plan = dict(identity=dict(run_id=F.RUN_ID), data=dict(
            policy=file_pin(self.fixture.policy_path), manifest=file_pin(self.root / 'training-manifest.json')))
        plan_path = dump(self.output / 'plan.json', self.plan)
        self.registration = dump(self.output / 'candidate.json',
            dict(schema='trid-fresh-eight-cuda-candidate-v1', plan=dict(file_pin(plan_path), path='plan.json'),
                 candidate_id=F.RUN_ID, arm='candidate', step=150))
        self.addCleanup(patch.stopall)
        patch.object(C, 'PROTOCOL_SHA', sha(self.fixture.protocol_path)).start()
        patch.object(F, 'validate_data', return_value=dict(policy=self.fixture.policy)).start()
        patch.object(F, 'validate_capacity').start()
        patch.object(F, 'validate_training', side_effect=lambda *_: self.training).start()
        patch.object(F, 'validate_supervision', side_effect=lambda root, phase, plan:
            dict(worker_result=dict(file_pin(self.output / phase /
                ('capacity-worker.json' if phase == 'capacity' else 'training-worker.json')),
                path=phase + '/' + ('capacity-worker.json' if phase == 'capacity' else 'training-worker.json')))).start()
        self.completed = patch.object(F, 'validate_completion').start()
        self.export = patch.object(C, 'validate_cuda_export').start()

    def resolve(self, complete=True):
        return C.resolve_cuda_candidate(self.root, self.registration, sha(self.registration),
                                        require_completion=complete)

    def test_cuda_contract_is_distinct_and_final_consumers_require_completion(self):
        contract = self.resolve()
        self.assertEqual(contract['runtime'], 'cuda')
        self.assertEqual(contract['producer'], 'trid-fresh-eight-cuda-paired-v1')
        self.assertEqual(contract['registration']['path'], 'candidate.json')
        self.assertEqual(contract['arms'], ['base', 'candidate'])
        self.completed.assert_called_once()
        self.export.assert_called_once()
        self.completed.reset_mock()
        self.export.reset_mock()
        self.resolve(complete=False)
        self.completed.assert_not_called()
        self.export.assert_not_called()

    def test_mps_inventory_cannot_impersonate_cuda(self):
        self.training['dtype_inventory']['adapter'] = {
            'torch.float32@mps:0': dict(tensors=1, elements=1, bytes=4)}
        with self.assertRaisesRegex(ValueError, 'CUDA training dtype'):
            self.resolve()

    def test_final_configuration_and_ledger_are_bound(self):
        ledger = self.output / 'train/checkpoints.json'
        value = load(ledger)
        value['last_step']['adapter_config_sha256'] = '0' * 64
        dump(ledger, value)
        with self.assertRaisesRegex(ValueError, 'report/ledger'):
            self.resolve()

    def test_cuda_answer_receipt_does_not_claim_mps(self):
        contract = self.resolve(complete=False)
        contract['policy_document']['evaluation']['generation'] = dict(E.GENERATION, backend='transformers-cuda')
        with patch.object(E, 'decode_generation', return_value=dict(response_text='synthetic', termination_reason='eos')):
            row = E.answer_record({}, 'base', {}, {'eos_token_id': 1}, [1], None, 0., {}, contract)
        self.assertEqual(row['generation']['backend'], 'transformers-cuda')
        self.assertIn('cuda_memory', row)
        self.assertNotIn('mps_memory', row)
        self.assertIsNone(row['adapter_weights_sha256'])

    def test_cuda_requires_additional_executed_source_closure(self):
        sources = producer_source_fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'mandatory producer'):
            A.authenticate_producer_sources(self.root, sources, sources, A.CUDA_PRODUCER_SOURCES)


class IsolatedCudaProducerSourceTests(unittest.TestCase):
    def test_receipts_match_isolated_evaluation_imports_not_training_process(self):
        # A new -I interpreter avoids contamination from trainer/observer imports
        # in this test suite. These are the lightweight imports reached by the
        # supervised evaluate child before model loading; no torch/model runs.
        script = """
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / 'scripts'), str(root / 'scripts/aws_eval')]
import fresh_eight_worker
import evaluate_local_eight
from scripts import local_eight_candidate
from scripts.aws_eval import fresh_eight_experiment
print(json.dumps(evaluate_local_eight.code_identity()))
"""
        result = subprocess.run([sys.executable, '-I', '-c', script, str(E.ROOT)],
            check=True, capture_output=True, text=True, timeout=30)
        sources = json.loads(result.stdout)
        self.assertEqual(set(sources), A.CUDA_PRODUCER_SOURCES)
        self.assertNotIn('scripts/local_mps_training.py', sources)
        self.assertNotIn('scripts/train_open_weight_lora.py', sources)
        self.assertFalse(any(name.startswith('trid/') for name in sources))
        A.authenticate_producer_sources(E.ROOT, sources, sources, A.CUDA_PRODUCER_SOURCES)
        for name in sources:
            with self.subTest(missing=name):
                incomplete = {key: value for key, value in sources.items() if key != name}
                with self.assertRaisesRegex(ValueError, 'mandatory producer'):
                    A.authenticate_producer_sources(E.ROOT, incomplete, sources, A.CUDA_PRODUCER_SOURCES)


class CudaAssessmentIntegrationTests(unittest.TestCase):
    def test_cuda_primary200_keeps_the_same_blind_protocol_and_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = SyntheticExperiment(root)
            directory = fixture.primary
            manifest = load(directory / 'run-manifest.json')
            adapter = dict(manifest['adapter'], step=150)
            registration = dump(directory.parent / 'candidate.json', {'synthetic': True})
            policy_path = dump(root / 'policy.json', {'synthetic': True})
            protocol = dict(instructions='Synthetic paired protocol.', paired_consistency=True,
                critical_errors_are_codes=True, critical_error_codes={'wrong_actor': 'Wrong actor.'})
            protocol_path = dump(root / 'protocol.json', protocol)
            metadata_path = dump(root / 'acquisition.json', dict(files=[
                dict(path='synthetic.safetensors', bytes=3, sha256='a' * 64)]))
            sources = {name: dict(file_pin(dump(root / name, {'synthetic_source': name})), path=name)
                       for name in A.CUDA_PRODUCER_SOURCES}
            plan_path = dump(directory.parent / 'plan.json',
                dict(sources={name: entry['sha256'] for name, entry in sources.items()}))
            contract = dict(arms=['base', 'candidate'], adapter_sha256=fixture.design['model']['adapter_sha256'],
                adapter=adapter, adapter_directory=str(root / 'adapter'), candidate_id='synthetic',
                registration=dict(file_pin(registration), path='candidate.json'), policy=file_pin(policy_path),
                review_protocol=file_pin(protocol_path), runtime='cuda', producer='trid-fresh-eight-cuda-paired-v1',
                plan=dict(file_pin(plan_path), path='plan.json'),
                policy_document=dict(data=dict(frozen_dataset=file_pin(fixture.frozen_path)),
                    model=dict(acquisition_manifest=file_pin(metadata_path)),
                    evaluation=dict(development_gate={'synthetic': 'prospective'})))
            manifest.update(schema=contract['producer'], arm_order=contract['arms'], adapter=adapter,
                evaluation_contract=C.public_contract(contract), code=sources,
                generation=dict(manifest['generation'], backend='transformers-cuda'),
                runtime=dict(manifest['runtime'], device='cuda'))
            manifest['model'].update(directory='model', acquisition_manifest=file_pin(metadata_path),
                files={'synthetic.safetensors': dict(path='model/synthetic.safetensors', bytes=3, sha256='a' * 64)})
            dump(directory / 'run-manifest.json', manifest)
            loaded = load(directory / 'loaded-runtime.json')
            loaded['code'] = sources
            dump(directory / 'loaded-runtime.json', loaded)
            for old_arm, arm in [('base', 'base'), ('step-000024', 'candidate')]:
                rows = fixture.rows(directory, old_arm)
                for row in rows:
                    row.update(arm=arm, run_manifest_sha256=sha(directory / 'run-manifest.json'),
                        loaded_runtime_sha256=sha(directory / 'loaded-runtime.json'), generation=manifest['generation'],
                        candidate_manifest_sha256=sha(registration))
                fixture.write_rows(directory, arm, rows, refresh=False)
            completion = load(directory / 'completion.json')
            completion.update(run_manifest_sha256=sha(directory / 'run-manifest.json'),
                outputs={arm: dict(file_pin(directory / (arm + '-outputs.jsonl')),
                    path=directory.name + '/' + arm + '-outputs.jsonl') for arm in contract['arms']})
            dump(directory / 'completion.json', completion)
            with patch.object(A, 'resolve_candidate', return_value=contract), \
                 patch.object(A, 'review_protocol', return_value=protocol), \
                 patch.object(A, 'ANONYMOUS_PROTOCOL_SHA', A.object_sha(protocol)):
                out = root / 'cuda-blind'
                A.make_packet(root, fixture.frozen_path, sha(fixture.frozen_path), directory, out,
                    candidate_manifest=registration, candidate_sha=sha(registration))
                packet = out / 'review-packet.json'
                self.assertEqual(len(load(packet)['rows']), 400)
                frozen = fixture.freeze(packet, fixture.reports(packet))
                result = fixture.aggregate(frozen, out / 'private-mapping.json')
                self.assertFalse(result['deployment_qualified'])
                self.assertFalse(result['development_gate']['passed'])
                self.assertIsNone(result['selected_step'])
                self.assertEqual(result['role'], 'development_only_not_unseen_final')
                manifest['runtime']['device'] = 'mps'
                dump(directory / 'run-manifest.json', manifest)
                completion['run_manifest_sha256'] = sha(directory / 'run-manifest.json')
                dump(directory / 'completion.json', completion)
                with self.assertRaisesRegex(ValueError, 'runtime/device'):
                    A.completed_run(root, directory, {r['id']: r for r in fixture.inputs},
                        fixture.design, 'unconstrained', contract=contract)


if __name__ == '__main__':
    unittest.main()
