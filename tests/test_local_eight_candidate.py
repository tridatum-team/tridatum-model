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


def immutable_source_bundle(root):
    import tarfile
    source = root / 'source.py'
    archive = root / 'source-bundle.tar'
    with tarfile.open(archive, 'w', format=tarfile.USTAR_FORMAT) as stream:
        stream.add(source, arcname='source.py', recursive=False)
    identity = file_pin(archive)
    return dict(local_path='source-bundle.tar', bytes=identity['bytes'], sha256=identity['sha256'],
                unpacked_bytes=source.stat().st_size, files=[dict(file_pin(source), path='source.py')])


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
            training_success=True, stop_errors=[], instance_id='i-synthetic',
            completion_scope='cuda_capacity_train_export_only', quality_accepted=False, cloud_evaluation=False)
        members = [dict(file_pin(self.output / name), path=name) for name in self.plan['output_allowlist']]
        dump(self.output / 'export-manifest.json', dict(run_id='synthetic-cuda', plan_sha256=self.plan_sha,
            status=self.status, files=members, total_bytes=sum(p['bytes'] for p in members)))
        self.completion = dict(self.status, export=dict(sha256='a' * 64, bytes=1000))
        dump(self.state / 'completion.json', self.completion)
        self.controller = dict(state='complete', training_success=True,
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
            dict(self.controller, cleanup=dict(self.controller['cleanup'], instance_id='i-other')),
            dict(self.controller, cleanup=dict(self.controller['cleanup'], root_volume_ids=[])),
            dict(self.controller, cleanup=dict(self.controller['cleanup'], remaining_volumes=['vol-root'])),
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
        export['status']['training_success'] = False
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
        self.plan['bundle'] = immutable_source_bundle(self.root)
        plan_path = dump(self.output / 'plan.json', self.plan)
        self.registration = dump(self.output / 'candidate.json',
            dict(schema='trid-fresh-eight-cuda-training-candidate-v2', plan=dict(file_pin(plan_path), path='plan.json'),
                 candidate_id=F.RUN_ID, arm='candidate', step=150,
                 completion_scope='cuda_capacity_train_export_only', quality_accepted=False, cloud_evaluation=False))
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
        self.export = patch.object(C, 'validate_cuda_export', return_value={}).start()

    def resolve(self):
        return C.resolve_cuda_candidate(self.root, self.registration, sha(self.registration))

    def test_cuda_contract_is_distinct_and_final_consumers_require_completion(self):
        contract = self.resolve()
        self.assertEqual(contract['training_runtime'], 'cuda')
        self.assertEqual(contract['inference_runtime'], 'mps')
        self.assertNotIn('runtime', contract)
        self.assertEqual(contract['producer'], 'trid-fresh-eight-cuda-training-candidate-v2')
        self.assertEqual(contract['registration']['path'], str(self.registration.resolve()))
        self.assertEqual(contract['arms'], ['base', 'candidate'])
        self.completed.assert_called_once()
        self.export.assert_called_once()
        with self.assertRaises(TypeError):
            C.resolve_cuda_candidate(self.root, self.registration, sha(self.registration), require_completion=False)

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

    def test_cuda_training_does_not_relabel_local_answer_receipt(self):
        contract = self.resolve()
        with patch.object(E, 'decode_generation', return_value=dict(response_text='synthetic', termination_reason='eos')):
            row = E.answer_record({}, 'base', {}, {'eos_token_id': 1}, [1], None, 0., {}, contract)
        self.assertEqual(row['generation']['backend'], 'transformers-mps')
        self.assertIn('mps_memory', row)
        self.assertNotIn('cuda_memory', row)
        self.assertEqual(contract['training_runtime'], 'cuda')
        self.assertIsNone(row['adapter_weights_sha256'])

    def test_old_schema_and_claimed_cloud_quality_are_rejected(self):
        original = load(self.registration)
        for field, value in [('schema', 'trid-fresh-eight-cuda-candidate-v1'),
                             ('cloud_evaluation', True), ('quality_accepted', True),
                             ('completion_scope', 'cuda_train_and_evaluate')]:
            with self.subTest(field=field):
                dump(self.registration, dict(original, **{field: value}))
                with self.assertRaisesRegex(ValueError, 'registration differs'):
                    self.resolve()

    def test_missing_training_capacity_or_export_proof_is_rejected(self):
        for module, name in [(self.F, 'validate_capacity'), (self.F, 'validate_training'),
                             (self.F, 'validate_supervision'), (self.F, 'validate_completion'),
                             (C, 'validate_cuda_export')]:
            with self.subTest(proof=name), patch.object(module, name, side_effect=ValueError('missing proof')):
                with self.assertRaisesRegex(ValueError, 'missing proof'):
                    C.resolve_candidate(self.root, self.registration, sha(self.registration))


class TrainingOnlyConsumerBoundaryTests(unittest.TestCase):
    """Real capacity/training/supervisor/export validators over synthetic bytes."""
    def setUp(self):
        from scripts.aws_eval import fresh_eight_experiment as F
        import math
        self.F = F
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        fixture = CandidateFixture(self.root)
        self.output = self.root / 'export'
        self.output.mkdir()
        for name in F.output_members():
            dump(self.output / name, {'synthetic': True})
        adapter = self.output / 'train/adapter'
        for path in fixture.directory.iterdir():
            (adapter / path.name).write_bytes(path.read_bytes())
        source = dict(file_pin(self.root / 'source.py'), path='source.py')
        self.plan = dict(schema=F.SCHEMA, identity=dict(run_id=F.RUN_ID), workflow=F.WORKFLOW,
            cloud_evaluation=False, data=dict(policy=file_pin(fixture.policy_path),
                manifest=file_pin(self.root / 'training-manifest.json'), train=file_pin(self.root / 'train.jsonl')),
            workload=dict(source_manifest=file_pin(self.root / 'sources.json')),
            bundle=immutable_source_bundle(self.root), sources={'source.py': source['sha256']},
            model=dict(files=[]), output_allowlist=sorted(F.output_members()))
        dump(self.output / 'plan.json', self.plan)
        plan_pin = dict(file_pin(self.output / 'plan.json'), path='plan.json')
        runtime = dict(device='cuda', backend='transformers-cuda', run_id=F.RUN_ID,
                       plan_sha256=plan_pin['sha256'])
        inventory = {group: {dtype + '@cuda:0': dict(tensors=1, elements=8, bytes=size)}
                     for group, dtype, size in [('base', 'torch.bfloat16', 16), ('adapter', 'torch.float32', 32)]}
        moments = inventory['adapter']
        history = []
        for step in range(1, 151):
            history.append(dict(step=step, train_loss=1., loss=1., microbatch_losses=[1.] * 4,
                applied_lr=1e-5 * (step / 7 if step <= 7 else .5 * (1 + math.cos(math.pi * (step - 8) / 143))),
                next_lr=1e-5 * ((step + 1) / 7 if step < 7 else .5 * (1 + math.cos(math.pi * (step - 7) / 143))),
                numerical_health=dict(gradient_norm=1., changed=True, changed_tensor='lora_A',
                    resident_moments=step > 1, gradients=moments, optimizer=moments),
                resident_backward_checks=[dict(resident_moments=True, gradient_norm=1.)] * 4))
        common = dict(policy=self.plan['data']['policy'], source_manifest=self.plan['workload']['source_manifest'])
        capacity = dict(common, schema='trid-fresh-eight-cuda-capacity-v1', status='complete',
            runtime=runtime, actual_source_files=[source], train_rows=600, dropped=0, sequence_tokens=3072,
            supervised_tokens=768, masked_prompt_tokens=2304, optimizer_updates=2, microbatches_per_update=4,
            includes_eos=True, reload_exact=True, released_before_reload=True, scratch_adapter_never_nominated=True,
            reload_loss=1., dtype_inventory=inventory, reload_dtype_inventory=inventory, history=history[:2])
        dump(self.output / 'capacity/capacity-worker.json', capacity)
        report = load(fixture.training_path)
        report.update(device='cuda', runtime=runtime, actual_source_files=[source],
                      dtype_inventory=inventory, history=history)
        ledger = load(fixture.ledger_path)
        ledger['selection'] = dict(candidate_step=150)
        for checkpoint in ledger['checkpoints']:
            checkpoint['path'] = f'train/checkpoints/step-{checkpoint["step"]:06d}/adapter'
            for name in ('adapter_config.json', 'adapter_model.safetensors'):
                (self.output / checkpoint['path'] / name).write_bytes((adapter / name).read_bytes())
        ledger['last_step']['path'] = 'train/adapter'
        report.update(checkpoints=ledger['checkpoints'], last_step=ledger['last_step'])
        dump(self.output / 'train/checkpoints.json', ledger)
        dump(self.output / 'train/training.json', report)
        dump(self.output / 'train/training-worker.json', dict(common,
            schema='trid-fresh-eight-cuda-training-v1', status='complete', coverage_complete=True,
            report=dict(file_pin(self.output / 'train/training.json'), path='train/training.json')))
        for phase, name in [('capacity', 'capacity-worker.json'), ('train', 'training-worker.json')]:
            worker = dict(file_pin(self.output / phase / name), path=phase + '/' + name)
            unit = dict(state='complete', cleanup_verified=True, plan_sha256=plan_pin['sha256'],
                run_id=F.RUN_ID, phase=phase, invocation='synthetic-invocation', instance_id='i-synthetic',
                stopped_state=dict(ActiveState='inactive'))
            dump(self.output / phase / 'supervisor-result.json', dict(common,
                schema='trid-fresh-eight-cuda-supervision-v1', phase=phase, status='accepted',
                cleanup_verified=True, run_id=F.RUN_ID, worker_result=worker, unit=unit,
                plan_sha256=plan_pin['sha256']))
        dump(self.output / 'model-acquisition.json', dict(status='authenticated',
            model_id=C.MODEL_ID, revision=C.REVISION, files=[]))
        self.registration = dump(self.output / 'candidate.json', dict(schema=F.CANDIDATE_SCHEMA,
            plan=plan_pin, candidate_id=F.RUN_ID, arm='candidate', step=150,
            completion_scope=F.COMPLETION_SCOPE, quality_accepted=False, cloud_evaluation=False))
        self.status = dict(run_id=F.RUN_ID, plan_sha256=plan_pin['sha256'], training_success=True,
            stop_errors=[], instance_id='i-synthetic', completion_scope=F.COMPLETION_SCOPE,
            quality_accepted=False, cloud_evaluation=False)
        completion = dict(self.status, export=dict(sha256='a' * 64, bytes=1000))
        dump(self.root / 'completion.json', completion)
        dump(self.root / 'controller-result.json', dict(state='complete', training_success=True,
            evidence=dict(native_pipeline_success=True, completion=completion, export_sha256='a' * 64),
            cleanup=dict(cleanup_state='complete', instance_terminated=True, instance_id='i-synthetic',
                root_volume_ids=['vol-root'], recorded_volume_ids=['vol-root'],
                absent_volume_ids=['vol-root'], remaining_volumes=[])))
        # Only the immutable upstream registration is replaced; all consumer
        # numerical/endpoint/cleanup/export checks below run their real code.
        data_patch = patch.object(F, 'validate_data', return_value=dict(policy=fixture.policy))
        data_patch.start()
        self.addCleanup(data_patch.stop)
        protocol_patch = patch.object(C, 'PROTOCOL_SHA', sha(fixture.protocol_path))
        protocol_patch.start()
        self.addCleanup(protocol_patch.stop)
        self.seal_export()

    def seal_export(self):
        for phase, name in [('capacity', 'capacity-worker.json'), ('train', 'training-worker.json')]:
            supervisor_path = self.output / phase / 'supervisor-result.json'
            supervisor = load(supervisor_path)
            supervisor['worker_result'] = dict(file_pin(self.output / phase / name), path=phase + '/' + name)
            dump(supervisor_path, supervisor)
        members = [dict(file_pin(self.output / name), path=name) for name in sorted(self.F.output_members())]
        dump(self.output / 'export-manifest.json', dict(run_id=self.F.RUN_ID,
            plan_sha256=self.status['plan_sha256'], status=self.status, files=members,
            total_bytes=sum(entry['bytes'] for entry in members)))

    def resolve(self):
        return C.resolve_candidate(self.root, self.registration, sha(self.registration))

    def test_training_only_export_is_accepted_without_any_cuda_answers(self):
        contract = self.resolve()
        self.assertFalse((self.output / 'evaluation').exists())
        self.assertEqual(contract['training_runtime'], 'cuda')
        self.assertEqual(contract['inference_runtime'], 'mps')
        self.assertFalse(contract['quality_accepted'])
        self.assertNotIn('collection_recovery', contract)
        self.assertEqual(C.bound(self.root, contract['controller_result']), self.root / 'controller-result.json')
        self.assertEqual(C.bound(self.root, contract['remote_completion']), self.root / 'completion.json')

    def test_later_checkout_source_change_does_not_rewrite_cuda_provenance(self):
        original = self.resolve()
        (self.root / 'source.py').write_text('changed local inference implementation')
        later = self.resolve()
        self.assertEqual(original['training_bundle'], later['training_bundle'])
        self.assertEqual(later['training_runtime'], 'cuda')
        archive = self.root / self.plan['bundle']['local_path']
        archive.write_bytes(archive.read_bytes() + b'tampered')
        with self.assertRaisesRegex(RuntimeError, 'Bundle authentication'):
            self.resolve()

    def test_resealed_incomplete_capacity_coverage_and_source_proof_fail(self):
        for name, mutate in [
            ('capacity/capacity-worker.json', lambda row: row.update(released_before_reload=False)),
            ('capacity/capacity-worker.json', lambda row: row['history'][1].update(resident_backward_checks=[])),
            ('train/training.json', lambda row: row.update(coverage=row['coverage'][:-1])),
            ('train/training.json', lambda row: row.update(actual_source_files=[])),
            ('train/checkpoints.json', lambda row: row['last_step'].update(step=100)),
            ('train/supervisor-result.json', lambda row: row['unit'].update(cleanup_verified=False)),
        ]:
            with self.subTest(artifact=name):
                path = self.output / name
                original = load(path)
                changed = copy.deepcopy(original)
                mutate(changed)
                dump(path, changed)
                if name == 'train/training.json':
                    worker_path = self.output / 'train/training-worker.json'
                    worker = load(worker_path)
                    worker['report'] = dict(file_pin(path), path=name)
                    dump(worker_path, worker)
                self.seal_export()
                with self.assertRaises(ValueError):
                    self.resolve()
                dump(path, original)
                if name == 'train/training.json':
                    worker['report'] = dict(file_pin(path), path=name)
                    dump(worker_path, worker)
                self.seal_export()

    def enable_collection_recovery(self):
        """Seal offline recovery receipts; no client, model, or cloud execution."""
        import tarfile
        from scripts.aws_eval import fresh_eight_recover as R
        self.R = R
        state = self.root / 'cloud-state'
        state.mkdir()
        self.output.rename(state / 'recovered-fresh-export')
        self.output = state / 'recovered-fresh-export'
        self.registration = self.output / 'candidate.json'
        self.plan['identity'].update(bucket='synthetic-bucket', output_prefix='synthetic/output/',
                                     account='123456789012')
        self.plan.update(cost=dict(controller_retrieval_bytes=1024**3, get_requests=5000),
            limits=dict(output_bytes=1024**3),
            runtime=dict(gpu_catalog_mib=45776, gpu_nvml_min_mib=45776,
                gpu_nvml_max_mib=49152, gpu_name='NVIDIA L40S', driver_version='synthetic'))
        dump(self.root / 'plan.json', self.plan)
        dump(self.output / 'plan.json', self.plan)
        plan_pin = dict(file_pin(self.output / 'plan.json'), path='plan.json')
        instance, volume = 'i-0123456789abcdef0', 'vol-0123456789abcdef0'
        for name in ('capacity/capacity-worker.json', 'train/training.json'):
            path = self.output / name
            row = load(path)
            row['runtime']['plan_sha256'] = plan_pin['sha256']
            dump(path, row)
        worker_path = self.output / 'train/training-worker.json'
        worker = load(worker_path)
        worker['report'] = dict(file_pin(self.output / 'train/training.json'), path='train/training.json')
        dump(worker_path, worker)
        for phase in ('capacity', 'train'):
            path = self.output / phase / 'supervisor-result.json'
            row = load(path)
            row['plan_sha256'] = plan_pin['sha256']
            row['unit'].update(plan_sha256=plan_pin['sha256'], instance_id=instance)
            dump(path, row)
        candidate = load(self.registration)
        candidate['plan'] = plan_pin
        dump(self.registration, candidate)
        gpu = dict(uuid='GPU-12345678-1234-1234-1234-123456789abc', minor=0,
                   name='NVIDIA L40S', driver='synthetic', total_mib=46068)
        self.status.update(plan_sha256=plan_pin['sha256'], instance_id=instance,
            runtime_unit=dict(gpu=gpu),
            stages=[dict(phase=phase, state='complete', cleanup_verified=True,
                         run_id=self.F.RUN_ID, plan_sha256=plan_pin['sha256'], instance_id=instance,
                         unit='synthetic-unit', invocation='synthetic-invocation', gpu=gpu)
                    for phase in ('capacity', 'train')])
        self.seal_export()
        archive = state / 'recovered-export.tar'
        with tarfile.open(archive, 'w', format=tarfile.USTAR_FORMAT) as stream:
            for path in sorted(self.output.rglob('*')):
                if path.is_file():
                    stream.add(path, arcname=path.relative_to(self.output).as_posix(), recursive=False)
        export = {key: file_pin(archive)[key] for key in ('bytes', 'sha256')}
        dump(state / 'completion.json', dict(self.status, export=export))
        dump(state / 'controller-result.json', dict(state='released', training_success=False,
            dispatch_attempted=True, remote_intent_attempted=True, instance_id=instance,
            completion_scope=self.F.COMPLETION_SCOPE, quality_accepted=False, cloud_evaluation=False,
            evidence_error='Command get-object timed out after 45 seconds',
            cleanup=dict(cleanup_state='complete', instance_terminated=True, instance_id=instance,
                root_volume_ids=[volume], recorded_volume_ids=[volume], absent_volume_ids=[volume],
                remaining_volumes=[], errors=['transient read timeout preserved'])))
        ownership = dict(run_id=self.F.RUN_ID, plan_sha256=plan_pin['sha256'], instance_id=instance)
        dump(state / 'release.json', dict(ownership, gpu=gpu))
        dump(state / 'volume-identity.json', dict(ownership, root_volume_ids=[volume]))
        dump(state / 'export-retrieval-intent.json', dict(ownership, byte_limit=1024**3))
        (state / 'source-bundle.tar').write_bytes((self.root / 'source-bundle.tar').read_bytes())
        (state / 'export.tar').write_bytes(b'original incomplete download')
        for name in R.SOURCES:
            dump(self.root / name, dict(synthetic_recovery_source=name))
        context = dump(self.root / 'collection-recovery-context.json',
            dict(schema='trid-existing-training-collection-authorization-v1',
                plan=R.pin(self.root, self.root / 'plan.json'),
                completion=R.pin(self.root, state / 'completion.json'),
                original_controller=R.pin(self.root, state / 'controller-result.json'),
                original_partial_archive=R.pin(self.root, state / 'export.tar'),
                expected_export=export, instance_id=instance, root_volume_id=volume,
                controller_retrieval_total_bytes_max=1024**3,
                prior_activity_conservative_debit_bytes=R.PRIOR_BYTES,
                new_payload_bytes_max=R.PRIOR_BYTES, automatic_gpu_retry=False,
                preserve_original_failure=True, budget_change_usd=0))
        permission = dump(self.root / 'collection-recovery-permission.json',
            dict(schema=R.AUTHORIZATION_SCHEMA, run_id=self.F.RUN_ID, collection_only=True,
                 additional_gpu_executions=0, prior_retrieval_bytes=R.PRIOR_BYTES,
                 prior_requests=R.PRIOR_REQUESTS, context=R.pin(self.root, context)))
        # Deliberately historical: later MPS evaluation must not require a live
        # transfer authorization after recovery has already finished.
        with patch.object(R.time, 'time', return_value=1000.):
            registration = R.registration_spec(self.root, state, permission, 1100.)
        registration_path = dump(state / 'collection-recovery-registration.json', registration)
        requests = []
        object_identity = dict(ETag='"synthetic-etag"', ContentLength=export['bytes'])
        prefix = ['--bucket', registration['bucket'], '--key', registration['key']]
        operations = [('s3api', 'head-object', prefix, 1024**2)]
        for start in range(0, export['bytes'], 8 * 1024**2):
            size = min(8 * 1024**2, export['bytes'] - start)
            operations.append(('s3api', 'get-object',
                prefix + ['--range', f'bytes={start}-{start + size - 1}',
                    '--if-match', object_identity['ETag'], str(state / f'range-{start}.part')], size))
        operations.extend([
            ('ec2', 'describe-instances', ['--instance-ids', instance], 1024**2),
            ('ec2', 'describe-volumes', ['--volume-ids', volume], 1024**2)])
        for sequence, (service, operation, arguments, size) in enumerate(operations, 1):
            path = dump(state / 'collection-recovery-requests' / f'{sequence:04d}.json',
                dict(sequence=sequence, service=service, operation=operation, arguments=arguments,
                     reserved_bytes=size, reserved_at=1001.))
            requests.append(R.pin(self.root, path))
        cleanup = [
            dict(operation='describe-instances', identifier=instance, started_at=1002., observed_at=1003.,
                 response=dict(Reservations=[dict(Instances=[dict(InstanceId=instance,
                     State=dict(Name='terminated'))])])),
            dict(operation='describe-volumes', identifier=volume, started_at=1004., observed_at=1005.,
                 response=dict(error_code='InvalidVolume.NotFound'))]
        self.recovery_path = dump(state / 'collection-recovery.json',
            dict(schema=R.SCHEMA, state='accepted', collection_only=True, additional_gpu_executions=0,
                quality_accepted=False, cloud_evaluation=False, registration=R.pin(self.root, registration_path),
                originals=registration['originals'], sources=registration['sources'],
                archive=R.pin(self.root, archive), candidate=R.pin(self.root, self.registration),
                object_identity=object_identity,
                validation=R.validate_export(self.root, state, self.plan, registration),
                cleanup=cleanup, requests=requests,
                accounting=R.validate_ledger(self.root, state, registration, requests, object_identity),
                accepted_at=1006.))
        return state

    def test_registered_recovery_preserves_failure_and_public_provenance_after_deadline(self):
        state = self.enable_collection_recovery()
        original = (state / 'controller-result.json').read_bytes()
        partial = (state / 'export.tar').read_bytes()
        contract = self.resolve()
        public = C.public_contract(contract)
        for key, path in [('collection_recovery', self.recovery_path),
                          ('controller_result', state / 'controller-result.json'),
                          ('remote_completion', state / 'completion.json'),
                          ('recovery_registration', state / 'collection-recovery-registration.json'),
                          ('recovered_archive', state / 'recovered-export.tar')]:
            self.assertEqual(C.bound(self.root, public[key]), path.resolve())
        self.assertEqual(public['training_runtime'], 'cuda')
        self.assertEqual(public['inference_runtime'], 'mps')
        self.assertEqual(contract['adapter']['step'], 150)
        self.assertEqual((state / 'controller-result.json').read_bytes(), original)
        self.assertEqual((state / 'export.tar').read_bytes(), partial)
        self.assertFalse(load(state / 'controller-result.json')['training_success'])
        self.assertTrue(load(state / 'controller-result.json')['cleanup']['errors'])

    def test_recovery_tampered_receipt_cleanup_and_error_sidecars_rejected(self):
        self.enable_collection_recovery()
        original = load(self.recovery_path)
        mutations = [
            lambda row: row.update(state='failed'),
            lambda row: row.update(error='download interrupted'),
            lambda row: row.update(evidence_error='partial archive'),
            lambda row: row['registration'].update(sha256='0' * 64),
            lambda row: row['originals']['controller-result.json'].update(sha256='0' * 64),
            lambda row: row.update(cleanup=[]),
            lambda row: row['cleanup'][0].update(identifier='i-fffffffffffffffff'),
            lambda row: row['cleanup'][0]['response']['Reservations'][0]['Instances'][0].update(
                InstanceId='i-fffffffffffffffff'),
            lambda row: row['cleanup'][1].update(identifier='vol-fffffffffffffffff'),
            lambda row: row['cleanup'][1].update(response=dict(error_code='RequestTimeout')),
            lambda row: row['cleanup'][1].update(observed_at=999.),
            lambda row: row.update(accepted_at=1101.),
            lambda row: row.update(requests=[]),
            lambda row: row.update(validation={'training_success': True}),
        ]
        for mutate in mutations:
            changed = copy.deepcopy(original)
            mutate(changed)
            dump(self.recovery_path, changed)
            with self.subTest(receipt=changed), self.assertRaises((ValueError, RuntimeError)):
                self.resolve()
        dump(self.recovery_path, original)

    def test_recovery_partial_archive_and_modified_originals_rejected(self):
        state = self.enable_collection_recovery()
        for path in (state / 'recovered-export.tar', state / 'export.tar',
                     state / 'source-bundle.tar', state / 'controller-result.json',
                     state / 'completion.json', self.root / self.R.SOURCES[0]):
            original = path.read_bytes()
            path.write_bytes(original[:len(original) // 2] if path.name == 'recovered-export.tar'
                             else original + b' ')
            with self.subTest(path=path.name), self.assertRaises((ValueError, RuntimeError)):
                self.resolve()
            path.write_bytes(original)
        self.recovery_path.unlink()
        with self.assertRaises((ValueError, RuntimeError, FileNotFoundError)):
            self.resolve()

    def test_recovery_registration_cannot_mask_failed_training_or_unsupported_controller(self):
        state = self.enable_collection_recovery()
        controller_path = state / 'controller-result.json'
        original = load(controller_path)
        registration_path = state / 'collection-recovery-registration.json'
        registration = load(registration_path)
        receipt = load(self.recovery_path)
        for mutate in (
            lambda row: row.update(error='training failed'),
            lambda row: row.update(state='native_failed'),
            lambda row: row.update(dispatch_attempted=False),
            lambda row: row.update(evidence_error=None),
            lambda row: row.update(instance_id='i-fffffffffffffffff'),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            dump(controller_path, changed)
            registration['originals']['controller-result.json'] = self.R.pin(self.root, controller_path)
            context_path = self.root / 'collection-recovery-context.json'
            context = load(context_path)
            context['original_controller'] = self.R.pin(self.root, controller_path)
            dump(context_path, context)
            permission_path = self.root / 'collection-recovery-permission.json'
            permission = load(permission_path)
            permission['context'] = self.R.pin(self.root, context_path)
            dump(permission_path, permission)
            registration['authorization'] = self.R.pin(self.root, permission_path)
            dump(registration_path, registration)
            receipt['originals'] = registration['originals']
            receipt['registration'] = self.R.pin(self.root, registration_path)
            dump(self.recovery_path, receipt)
            with self.subTest(controller=changed), self.assertRaises((ValueError, RuntimeError)):
                self.resolve()

    def test_recovery_failed_remote_completion_and_failure_sidecar_rejected(self):
        state = self.enable_collection_recovery()
        failure = dump(state / 'collection-recovery-failure.json', dict(state='failed', error='interrupted'))
        with self.assertRaises((ValueError, RuntimeError)):
            self.resolve()
        failure.unlink()
        completion_path = state / 'completion.json'
        completion = load(completion_path)
        completion['training_success'] = False
        dump(completion_path, completion)
        with self.assertRaisesRegex(ValueError, 'completion differs'):
            self.resolve()

    def test_recovery_context_cannot_authorize_different_originals_or_budget(self):
        state = self.enable_collection_recovery()
        context_path = self.root / 'collection-recovery-context.json'
        original = load(context_path)
        permission_path = self.root / 'collection-recovery-permission.json'
        permission = load(permission_path)
        registration_path = state / 'collection-recovery-registration.json'
        registration = load(registration_path)
        receipt = load(self.recovery_path)
        for mutate in (
            lambda row: row['original_controller'].update(sha256='0' * 64),
            lambda row: row['original_partial_archive'].update(sha256='0' * 64),
            lambda row: row['expected_export'].update(sha256='0' * 64),
            lambda row: row.update(instance_id='i-fffffffffffffffff'),
            lambda row: row.update(root_volume_id='vol-fffffffffffffffff'),
            lambda row: row.update(new_payload_bytes_max=1024**3),
            lambda row: row.update(automatic_gpu_retry=True),
            lambda row: row.update(preserve_original_failure=False),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            dump(context_path, changed)
            permission['context'] = self.R.pin(self.root, context_path)
            dump(permission_path, permission)
            registration['authorization'] = self.R.pin(self.root, permission_path)
            dump(registration_path, registration)
            receipt['registration'] = self.R.pin(self.root, registration_path)
            dump(self.recovery_path, receipt)
            with self.subTest(context=changed), self.assertRaises((ValueError, RuntimeError)):
                self.resolve()

    def test_recovery_receipt_selects_versioned_registration_without_rewriting_original(self):
        state = self.enable_collection_recovery()
        original_path = state / 'collection-recovery-registration.json'
        original_bytes = original_path.read_bytes()
        versioned = dump(state / 'collection-recovery-registration-02.json', load(original_path))
        receipt = load(self.recovery_path)
        receipt['registration'] = self.R.pin(self.root, versioned)
        dump(self.recovery_path, receipt)
        contract = self.resolve()
        self.assertEqual(C.bound(self.root, contract['recovery_registration']), versioned)
        self.assertEqual(original_path.read_bytes(), original_bytes)

    def test_recovery_receipt_rejects_registration_from_foreign_state(self):
        state = self.enable_collection_recovery()
        registration = load(state / 'collection-recovery-registration.json')
        foreign = dump(self.root / 'foreign-state' / 'collection-recovery-registration-02.json', registration)
        receipt = load(self.recovery_path)
        receipt['registration'] = self.R.pin(self.root, foreign)
        dump(self.recovery_path, receipt)
        with self.assertRaises((ValueError, RuntimeError)):
            self.resolve()

    def enable_collection_finalization(self):
        """Authenticate an existing archive and predecessor; never fetch payload."""
        import tarfile
        state = self.enable_collection_recovery()
        R = self.R
        old_receipt = load(self.recovery_path)
        self.recovery_path.unlink()
        prior_path = state / 'collection-recovery-registration.json'
        prior = load(prior_path)
        dump(state / 'collection-recovery-started.json',
             dict(registration=R.pin(self.root, prior_path), started_at=1001.))
        failure_path = dump(state / 'collection-recovery-failure.json',
            dict(state='failed', error='Fresh exact cleanup observation rejected',
                 registration=R.pin(self.root, prior_path), failed_at=1006., additional_gpu_executions=0))
        source_archive = self.root / 'executed-recovery-sources.tar'
        with tarfile.open(source_archive, 'w', format=tarfile.USTAR_FORMAT) as stream:
            for name in R.SOURCES:
                stream.add(self.root / name, arcname=name, recursive=False)
        # The predecessor must authenticate its archived bytes, not relabel the
        # corrected finalizer source as the executable that downloaded the model.
        dump(self.root / R.SOURCES[0], dict(synthetic_finalizer_source=True))
        prior_permission = load(self.root / 'collection-recovery-permission.json')
        permission_path = dump(self.root / 'collection-finalization-permission.json',
            dict(schema='trid-fresh-eight-collection-finalization-authorization-v1',
                collection_only=True, additional_gpu_executions=0,
                prior_registration=R.pin(self.root, prior_path),
                prior_failure=R.pin(self.root, failure_path),
                source_archive=R.pin(self.root, source_archive),
                archive=R.pin(self.root, state / 'recovered-export.tar'),
                context=prior_permission['context'], max_metadata_requests=3,
                max_metadata_bytes=3 * 1024**2))
        with patch.object(R.time, 'time', return_value=2000.):
            registration = R.finalization_spec(self.root, state, prior_path, source_archive,
                                               permission_path, 2100.)
        registration_path = dump(state / 'collection-finalization-registration.json', registration)
        requests = []
        operations = [
            ('sts', 'get-caller-identity', []),
            ('ec2', 'describe-instances', ['--instance-ids', prior['instance_id']]),
            ('ec2', 'describe-volumes', ['--volume-ids', prior['root_volume_ids'][0]])]
        for index, (service, operation, arguments) in enumerate(operations, 1):
            path = dump(state / 'collection-finalization-requests' / f'{index:04d}.json',
                dict(sequence=index, service=service, operation=operation, arguments=arguments,
                     reserved_bytes=1024**2, reserved_at=2001.))
            requests.append(R.pin(self.root, path))
        cleanup = [
            dict(operation='describe-instances', identifier=prior['instance_id'],
                 started_at=2002., observed_at=2003., response=dict(Reservations=[])),
            dict(operation='describe-volumes', identifier=prior['root_volume_ids'][0],
                 started_at=2004., observed_at=2005., response=dict(Volumes=[]))]
        observations = []
        caller = dict(operation='get-caller-identity', identifier=registration['account'],
            started_at=2001., observed_at=2002., response=dict(Account=registration['account']))
        for index, row in enumerate([caller, *cleanup]):
            path = dump(state / f'collection-finalization-observation-{index}.json', row)
            observations.append(R.pin(self.root, path))
        self.finalization_path = dump(state / 'collection-finalization.json',
            dict(schema=R.FINALIZATION_SCHEMA, state='accepted', collection_only=True,
                additional_gpu_executions=0, quality_accepted=False, cloud_evaluation=False,
                registration=R.pin(self.root, registration_path),
                prior_registration=registration['prior_registration'],
                prior_failure=registration['prior_failure'], source_archive=registration['source_archive'],
                archive=registration['archive'], candidate=registration['candidate'],
                validation=old_receipt['validation'], cleanup=cleanup, observations=observations,
                requests=requests, accounting=R.finalization_accounting(self.root, state, registration, requests),
                accepted_at=2006.))
        return state

    def test_metadata_only_finalization_preserves_failed_attempt_and_archived_source_provenance(self):
        state = self.enable_collection_finalization()
        paths = [state / 'recovered-export.tar', state / 'collection-recovery-failure.json',
                 state / 'controller-result.json', state / 'collection-recovery-registration.json']
        originals = {path: path.read_bytes() for path in paths}
        with patch.object(self.R, 'RecoveryClient', side_effect=AssertionError('no payload transfer')), \
             patch.object(self.R, 'FinalizationClient', side_effect=AssertionError('no consumer cloud call')):
            contract = self.resolve()
        public = C.public_contract(contract)
        self.assertEqual(C.bound(self.root, public['collection_recovery']), self.finalization_path)
        self.assertEqual(C.bound(self.root, public['collection_finalization']), self.finalization_path)
        self.assertEqual(C.bound(self.root, public['collection_failure']), state / 'collection-recovery-failure.json')
        self.assertEqual(C.bound(self.root, public['predecessor_registration']),
                         state / 'collection-recovery-registration.json')
        self.assertEqual(C.bound(self.root, public['executed_recovery_sources']),
                         self.root / 'executed-recovery-sources.tar')
        self.assertEqual(load(self.finalization_path)['accounting']['new_reserved_bytes'], 3 * 1024**2)
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        self.assertEqual(contract['training_runtime'], 'cuda')
        self.assertEqual(contract['inference_runtime'], 'mps')

    def seal_finalization_sources(self, state):
        import tarfile
        archive = state / 'collection-finalization-executed-source.tar'
        with tarfile.open(archive, 'w', format=tarfile.USTAR_FORMAT) as stream:
            for name in self.R.SOURCES:
                stream.add(self.root / name, arcname=name, recursive=False)
        return dump(state / 'collection-finalization-source-archive.json',
            dict(schema='trid-finalization-source-archive-v1',
                 finalization_registration=self.R.pin(self.root, state / 'collection-finalization-registration.json'),
                 source_archive=self.R.pin(self.root, archive)))

    def test_finalization_archive_preserves_candidate_after_lifecycle_and_recovery_checkout_changes(self):
        state = self.enable_collection_finalization()
        custody_path = self.seal_finalization_sources(state)
        before = self.resolve()
        originals = {path: path.read_bytes() for path in
            (state / 'collection-finalization-registration.json', self.finalization_path,
             state / 'controller-result.json', self.root / 'collection-finalization-permission.json')}
        for name in self.R.SOURCES[:2]:
            (self.root / name).write_text('# later checkout, not historical execution\n')
        after = self.resolve()
        self.assertEqual(C.public_contract(before), C.public_contract(after))
        self.assertEqual(after['inference_runtime'], 'mps')
        lineage = C.cuda_evaluation_lineage(self.root, C.bound(self.root, after['registration']),
                                            after['registration']['sha256'])
        self.assertEqual(lineage['finalization_source_custody'], self.R.pin(self.root, custody_path))
        self.assertEqual(lineage['adapter_sha256'], after['adapter_sha256'])
        self.assertFalse(Path(lineage['candidate']['path']).is_absolute())
        self.assertEqual(lineage['provenance']['inference_runtime'], 'mps')
        for path, raw in originals.items():
            self.assertEqual(path.read_bytes(), raw)
        with patch.object(self.R.time, 'time', return_value=2005.), \
                self.assertRaisesRegex((ValueError, RuntimeError), 'registration or predecessor'):
            self.R.validate_finalization_registration(self.root,
                state / 'collection-finalization-registration.json', active=True)

    def test_finalization_source_custody_and_archive_tampering_are_rejected(self):
        state = self.enable_collection_finalization()
        custody_path = self.seal_finalization_sources(state)
        original = load(custody_path)
        for mutate in (
            lambda row: row.update(schema='unregistered'),
            lambda row: row['finalization_registration'].update(sha256='0' * 64),
            lambda row: row['source_archive'].update(sha256='0' * 64),
        ):
            changed = copy.deepcopy(original)
            mutate(changed)
            dump(custody_path, changed)
            with self.subTest(custody=changed), self.assertRaises((ValueError, RuntimeError)):
                self.resolve()
        dump(custody_path, original)
        archive = C.bound(self.root, original['source_archive'])
        archive.write_bytes(archive.read_bytes() + b'tampered')
        with self.assertRaises((ValueError, RuntimeError)):
            self.resolve()

    def test_resealed_later_sources_do_not_replace_original_finalization_source_pins(self):
        state = self.enable_collection_finalization()
        (self.root / self.R.SOURCES[1]).write_text('# later lifecycle, not the registered source\n')
        self.seal_finalization_sources(state)
        with self.assertRaisesRegex((ValueError, RuntimeError), 'historical recovery source|source bytes differ'):
            self.resolve()

    def test_finalization_without_custody_cannot_relabel_later_checkout_as_history(self):
        self.enable_collection_finalization()
        (self.root / self.R.SOURCES[1]).write_text('# later lifecycle\n')
        with self.assertRaises((ValueError, RuntimeError)):
            self.resolve()

    def test_finalization_rejects_foreign_incomplete_error_and_post_acceptance_observations(self):
        state = self.enable_collection_finalization()
        original = load(self.finalization_path)
        original_rows = [load(state / f'collection-finalization-observation-{i}.json') for i in (0, 1, 2)]
        for index, change in [
            (0, lambda row: row['response'].update(Account='999999999999')),
            (1, lambda row: row.update(identifier='i-fffffffffffffffff')),
            (1, lambda row: row.update(response={})),
            (1, lambda row: row.update(response=dict(error_type='TimeoutError', error='network failure'))),
            (2, lambda row: row.update(identifier='vol-fffffffffffffffff')),
            (2, lambda row: row.update(response=dict(error_code='RequestTimeout'))),
            (2, lambda row: row.update(observed_at=2007.)),
        ]:
            receipt = copy.deepcopy(original)
            row = copy.deepcopy(original_rows[index])
            change(row)
            path = dump(state / f'collection-finalization-observation-{index}.json', row)
            receipt['observations'][index] = self.R.pin(self.root, path)
            if index:
                receipt['cleanup'][index - 1] = row
            dump(self.finalization_path, receipt)
            with self.subTest(index=index, row=row), self.assertRaises((ValueError, RuntimeError)):
                self.resolve()
            dump(path, original_rows[index])
        dump(self.finalization_path, original)

    def test_finalization_rejects_modified_predecessor_payload_sources_and_own_failure(self):
        state = self.enable_collection_finalization()
        for path in [state / 'collection-recovery-failure.json',
                     state / 'collection-recovery-started.json',
                     state / 'collection-recovery-requests/0001.json',
                     self.root / 'executed-recovery-sources.tar',
                     state / 'recovered-export.tar']:
            original = path.read_bytes()
            path.write_bytes(original + b' ')
            with self.subTest(path=path.name), self.assertRaises((ValueError, RuntimeError)):
                self.resolve()
            path.write_bytes(original)
        dump(state / 'collection-finalization-failure.json', dict(state='failed', error='cleanup failed'))
        with self.assertRaises((ValueError, RuntimeError)):
            self.resolve()


class IsolatedTrainingConsumerSourceTests(unittest.TestCase):
    def test_receipts_require_local_inference_and_training_authentication_imports(self):
        # Isolated lightweight imports only: never load model/tokenizer/reference bodies.
        script = """
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
sys.path[:0] = [str(root), str(root / 'scripts'), str(root / 'scripts/aws_eval')]
import evaluate_local_eight
import local_mps_training
import train_open_weight_lora
from scripts import local_eight_candidate
from scripts.aws_eval import fresh_eight_experiment
print(json.dumps(evaluate_local_eight.code_identity()))
"""
        result = subprocess.run([sys.executable, '-I', '-c', script, str(E.ROOT)],
            check=True, capture_output=True, text=True, timeout=30)
        sources = json.loads(result.stdout)
        self.assertTrue(A.CUDA_TRAINED_MPS_SOURCES <= set(sources))
        self.assertIn('scripts/train_open_weight_lora.py', sources)
        self.assertNotIn('scripts/aws_eval/fresh_eight_worker.py', sources)
        A.authenticate_producer_sources(E.ROOT, sources, sources, A.CUDA_TRAINED_MPS_SOURCES)
        for name in A.CUDA_TRAINED_MPS_SOURCES:
            with self.subTest(missing=name):
                incomplete = {key: value for key, value in sources.items() if key != name}
                with self.assertRaisesRegex(ValueError, 'mandatory producer'):
                    A.authenticate_producer_sources(E.ROOT, incomplete, sources, A.CUDA_TRAINED_MPS_SOURCES)


class CudaTrainingMpsAssessmentIntegrationTests(unittest.TestCase):
    def test_local_primary200_keeps_cuda_training_and_same_blind_protocol_and_gates(self):
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
                       for name in A.CUDA_TRAINED_MPS_SOURCES}
            plan_path = dump(directory.parent / 'plan.json',
                dict(sources={name: entry['sha256'] for name, entry in sources.items()}))
            contract = dict(arms=['base', 'candidate'], adapter_sha256=fixture.design['model']['adapter_sha256'],
                adapter=adapter, adapter_directory=str(root / 'adapter'), candidate_id='synthetic',
                registration=file_pin(registration), policy=file_pin(policy_path),
                review_protocol=file_pin(protocol_path), training_runtime='cuda', inference_runtime='mps',
                producer='trid-fresh-eight-cuda-training-candidate-v2',
                plan=dict(file_pin(plan_path), path='plan.json'),
                policy_document=dict(data=dict(frozen_dataset=file_pin(fixture.frozen_path)),
                    model=dict(acquisition_manifest=file_pin(metadata_path)),
                    evaluation=dict(development_gate={'synthetic': 'prospective'})))
            manifest.update(schema='trid-local-eight-paired-v2', arm_order=contract['arms'], adapter=adapter,
                evaluation_contract=C.public_contract(contract), code=sources)
            local_receipt = dump(root / 'local-supervision.json', {'synthetic': True})
            manifest['local_registration'] = file_pin(local_receipt)
            manifest['model'].update(directory=str(root / 'model'), acquisition_manifest=file_pin(metadata_path))
            dump(directory / 'run-manifest.json', manifest)
            loaded = load(directory / 'loaded-runtime.json')
            loaded['code'] = sources
            loaded['parameter_dtypes'] = ['torch.bfloat16', 'torch.float32']
            loaded['parameter_inventory'] = {
                group: {dtype + '@mps:0': dict(tensors=1, elements=1, bytes=4)}
                for group, dtype in [('base', 'torch.bfloat16'), ('adapter', 'torch.float32')]}
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
                outputs={arm: file_pin(directory / (arm + '-outputs.jsonl')) for arm in contract['arms']})
            dump(directory / 'completion.json', completion)
            with patch.object(A, 'resolve_candidate', return_value=contract), \
                 patch.object(A, 'review_protocol', return_value=protocol), \
                 patch.object(A, 'ANONYMOUS_PROTOCOL_SHA', A.object_sha(protocol)), \
                 patch('scripts.evaluate_local_eight.model_identity', return_value=manifest['model']), \
                 patch('scripts.evaluate_local_eight.accepted_local_supervision', return_value=local_receipt):
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
                self.assertEqual(result['evaluation_contract']['training_runtime'], 'cuda')
                for group, wrong in [('base', 'torch.float32@mps:0'), ('adapter', 'torch.bfloat16@mps:0'),
                                     ('adapter', 'torch.float32@cuda:0')]:
                    changed = copy.deepcopy(loaded)
                    changed['parameter_inventory'][group] = {wrong: dict(tensors=1, elements=1, bytes=4)}
                    dump(directory / 'loaded-runtime.json', changed)
                    with self.subTest(inventory=wrong), self.assertRaises(ValueError):
                        A.completed_run(root, directory, {r['id']: r for r in fixture.inputs},
                            fixture.design, 'unconstrained', contract=contract)
                dump(directory / 'loaded-runtime.json', loaded)
                manifest['runtime']['device'] = 'cuda'
                dump(directory / 'run-manifest.json', manifest)
                completion['run_manifest_sha256'] = sha(directory / 'run-manifest.json')
                dump(directory / 'completion.json', completion)
                with self.assertRaisesRegex(ValueError, 'runtime/device'):
                    A.completed_run(root, directory, {r['id']: r for r in fixture.inputs},
                        fixture.design, 'unconstrained', contract=contract)


if __name__ == '__main__':
    unittest.main()
