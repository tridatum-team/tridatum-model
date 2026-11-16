"""Offline contract regressions; synthetic receipts never represent cloud evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/aws_eval'))
import fresh_eight_evaluation as E


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def prepared(index=0):
    messages = [dict(role='user', content='Public question ' + str(index))]
    return dict(id=E.SMOKE_ID if index == 0 else 'case-' + str(index), messages=messages,
        prompt_sha256=E.object_sha(messages), rendered_prompt_sha256='a' * 64, prompt_token_ids=[1, 2])


def config():
    return dict(do_sample=False, num_beams=1, max_new_tokens=768, return_dict_in_generate=True,
        use_cache=True, eos_token_id=[9], pad_token_id=9)


def answer(case, arm='base', mode='evaluation', binding=None):
    return dict(**case, **(binding or {}), arm=arm, mode=mode, generation=dict(E.GENERATION),
        generation_config=config(), adapter_weights_sha256=None if arm == 'base' else E.ADAPTER_SHA,
        exception=None, context_truncated=False, skipped_context=False,
        effective_generation_mode='greedy_search', output_token_ids=[8, 9], generated_tokens=2,
        terminal_stop_token_id=9, termination_reason='eos', truncated_output=False,
        response_text='Answer', raw_output='Answer<eos>', elapsed_seconds=0.5,
        cuda_memory=dict(device='cuda:0', allocated_bytes=1, reserved_bytes=2,
            peak_allocated_bytes=3, peak_reserved_bytes=4))


def mode_fixture(root, mode='evaluation'):
    """Complete synthetic paired-mode receipt, with actual immutable JSONL hashes."""
    rows = [prepared(i) for i in range(200 if mode == 'evaluation' else 1)]
    adapter = [dict(path='adapter/' + name, bytes=size, sha256=sha) for name, (size, sha) in E.ADAPTER_FILES.items()]
    lineage = dict(schema=E.LINEAGE_SCHEMA, candidate=dict(path='historical/candidate.json', bytes=414,
        sha256=E.CANDIDATE_SHA), adapter_sha256=E.ADAPTER_SHA, adapter_files=adapter,
        provenance=dict(registration=dict(sha256=E.CANDIDATE_SHA)))
    sources = [dict(path=name, bytes=1, sha256='b' * 64) for name in sorted(E.REQUIRED_SOURCES)]
    plan = dict(schema=E.SCHEMA, limits=dict(E.LIMITS),
        lifetime=dict(total_seconds=19800, termination_grace_seconds=900, automatic_extension=False),
        cost=dict(termination_grace_seconds=900), sources={p['path']: p['sha256'] for p in sources},
        model=dict(files=[dict(path='model.safetensors', bytes=2, sha256='c' * 64)]),
        workload=dict(evaluation_inputs=dict(path=E.INPUT_PATH, bytes=568597, sha256=E.INPUT_SHA), adapter_files=adapter))
    save(root / 'plan.json', plan)
    save(root / 'lineage-binding.json', lineage)
    save(root / 'source-manifest.json', dict(files=sources))
    folder = root / mode
    decoder = dict(profile='unconstrained', semantic_truth_guarantee=False)
    manifest = dict(schema=E.RUN_SCHEMA, mode=mode, generation=E.GENERATION, arm_order=list(E.ARMS),
        grading_access=False, retries=False, resume=False, plan=E.pin(root / 'plan.json', root),
        lineage_binding=E.pin(root / 'lineage-binding.json', root), source_manifest=E.pin(root / 'source-manifest.json', root),
        input=plan['workload']['evaluation_inputs'], evaluation_contract=lineage['provenance'],
        model=dict(id=E.MODEL_ID, revision=E.MODEL_REVISION, body_hashes_authenticated=True,
            representation='hf-safetensors-bfloat16', files=plan['model']['files']),
        adapter=dict(step=150, files={Path(p['path']).name: p for p in adapter}),
        runtime=dict(device='cuda', dtype='torch.bfloat16', cpu_fallback=False, cuda_equivalence_claim=False,
            packages=E.RUNTIME_PACKAGES, gpu_uuid='GPU-synthetic'), code={p['path']: p for p in sources},
        decoder_profile=decoder, selected_ids=[r['id'] for r in rows])
    policy = dict(generation_mode='greedy_search', effective_config_sha256=E.object_sha(config()),
        dispatch_kwargs=dict(use_model_defaults=False, do_sample=False, num_beams=1))
    loaded = dict(code=manifest['code'], gpu_uuid='GPU-synthetic', parameter_dtypes=['torch.bfloat16', 'torch.float32'],
        frozen=True, model_loads=1, adapter_merged=False, generation_config=config(), generation_policy=policy,
        parameter_inventory={group: {dtype + '@cuda:0': dict(tensors=1, elements=2, bytes=size * 2)}
            for group, dtype, size in [('base', 'torch.bfloat16', 2), ('adapter', 'torch.float32', 4)]})
    for name, value in [('run-manifest.json', manifest), ('loaded-runtime.json', loaded),
        ('decoder-profile.json', decoder), ('prompt-preflight.json', dict(status='passed', prompts=rows,
            all_selected_preflighted_before_weights=True, all_200_preflighted_before_weights=True, truncation=False))]:
        save(folder / name, value)
    binding = dict(run_manifest_sha256=E.pin(folder / 'run-manifest.json')['sha256'],
        loaded_runtime_sha256=E.pin(folder / 'loaded-runtime.json')['sha256'], model_id=E.MODEL_ID,
        model_revision=E.MODEL_REVISION, input_sha256=E.INPUT_SHA, nominated_adapter_weights_sha256=E.ADAPTER_SHA,
        decoder_profile='unconstrained', decoder_profile_sha256=E.pin(folder / 'decoder-profile.json')['sha256'],
        generation_policy_sha256=E.object_sha(policy), candidate_manifest_sha256=E.CANDIDATE_SHA)
    for arm in E.ARMS:
        (folder / (arm + '-outputs.jsonl')).write_text(''.join(json.dumps(answer(r, arm, mode, binding)) + '\n' for r in rows))
    (folder / 'attempts.jsonl').write_text(''.join(json.dumps(dict(id=r['id'], arm=arm, state='started')) + '\n'
        for r in rows for arm in E.ARMS))
    save(folder / 'completion.json', dict(status='complete', mode=mode, rows_per_arm=len(rows),
        run_manifest_sha256=binding['run_manifest_sha256'],
        outputs={arm: E.pin(folder / (arm + '-outputs.jsonl'), folder) for arm in E.ARMS}))
    return plan


class EvaluationContractTests(unittest.TestCase):
    def test_recipe_free_workload_and_exact_runtime_source_contract(self):
        record = lambda name: dict(path=name, bytes=1, sha256='a' * 64)
        w = dict(evaluation_inputs=dict(path=E.INPUT_PATH, bytes=568597, sha256=E.INPUT_SHA),
            acquisition_manifest=dict(path='acquisition.json', bytes=4003, sha256=E.ACQUISITION_SHA),
            lineage_binding=record('lineage.json'), source_manifest=record('sources.json'),
            generation=dict(E.GENERATION), smoke_id=E.SMOKE_ID,
            adapter_files=[dict(path='adapter/' + name, bytes=size, sha256=sha)
                           for name, (size, sha) in E.ADAPTER_FILES.items()])
        plan = dict(schema=E.SCHEMA, identity=dict(run_id=E.RUN_ID), limits=dict(E.LIMITS),
            lifetime=dict(total_seconds=19800, termination_grace_seconds=900, automatic_extension=False),
            cost=dict(termination_grace_seconds=900), workload=w,
            model=dict(id=E.MODEL_ID, revision=E.MODEL_REVISION),
            runtime=dict(packages=dict(E.RUNTIME_PACKAGES), torch_cuda='12.8',
                bootstrap=dict(admission=record('bootstrap.json'))),
            authorization=record('authorization.json'), resource_envelope=record('resources.json'),
            ancillary_evidence=record('ancillary.json'), sources={n: 'a' * 64 for n in E.REQUIRED_SOURCES},
            output_allowlist=sorted(E.output_members()))
        records = [w[k] for k in ('evaluation_inputs', 'acquisition_manifest', 'lineage_binding', 'source_manifest')]
        records += [*w['adapter_files'], plan['authorization'], plan['resource_envelope'],
                    plan['ancillary_evidence'], plan['runtime']['bootstrap']['admission']]
        bundled = {p['path']: p for p in records}
        bundled.update({name: record(name) for name in E.REQUIRED_SOURCES})
        E.validate_workload(plan, bundled)
        mutations = [
            lambda p: p.update(recipe={}),
            lambda p: p.update(schema='unknown'),
            lambda p: p['workload'].update(smoke_id='alternate-question'),
            lambda p: p['workload']['generation'].update(do_sample=True),
            lambda p: p['workload'].update(development_references=record('private.json')),
            lambda p: p['runtime']['packages'].update(transformers='4.58.0'),
            lambda p: p['sources'].update({'scripts/train_open_weight_lora.py': 'a' * 64}),
            lambda p: p['workload']['adapter_files'][0].update(sha256='0' * 64)]
        for mutate in mutations:
            changed = copy.deepcopy(plan)
            mutate(changed)
            with self.assertRaises(ValueError):
                E.validate_workload(changed, bundled)

    def test_new_limits_reject_old_grant_and_extended_lifetime(self):
        plan = dict(limits=dict(E.LIMITS), lifetime=dict(total_seconds=19800, termination_grace_seconds=900,
            automatic_extension=False), cost=dict(termination_grace_seconds=900))
        E.validate_limits(plan)
        for key, value in [('lifetime_seconds', 19800), ('reserve_seconds', 0), ('instance_count', 2), ('swap_bytes', 1)]:
            mutated = copy.deepcopy(plan)
            mutated['limits'][key] = value
            with self.assertRaises(ValueError):
                E.validate_limits(mutated)
        plan['cost']['termination_grace_seconds'] = 0
        with self.assertRaises(ValueError):
            E.validate_limits(plan)

    def test_empty_eos_and_length_are_full_denominator_not_smoke(self):
        case = prepared()
        row = answer(case)
        row['response_text'] = ''
        E.validate_answer(row, case, 'base', 'evaluation', {}, config())
        row['mode'] = 'smoke'
        with self.assertRaisesRegex(ValueError, 'Smoke'):
            E.validate_answer(row, case, 'base', 'smoke', {}, config())
        row.update(mode='evaluation', output_token_ids=[8] * 768, generated_tokens=768,
            terminal_stop_token_id=None, termination_reason='length', truncated_output=True)
        E.validate_answer(row, case, 'base', 'evaluation', {}, config())
        row['output_token_ids'].pop()
        row['generated_tokens'] -= 1
        with self.assertRaisesRegex(ValueError, 'termination'):
            E.validate_answer(row, case, 'base', 'evaluation', {}, config())

    def test_identity_sampling_dtype_and_false_memory_rejected(self):
        case = prepared()
        for field, value in [('adapter_weights_sha256', E.ADAPTER_SHA), ('effective_generation_mode', 'sample'),
                             ('mode', 'smoke'), ('exception', {'message': 'failure'}), ('elapsed_seconds', float('nan'))]:
            row = answer(case)
            row[field] = value
            with self.assertRaises(ValueError):
                E.validate_answer(row, case, 'base', 'evaluation', {}, config())
        row = answer(case)
        row['cuda_memory']['peak_reserved_bytes'] = E.LIMITS['allocator_bytes'] + 1
        with self.assertRaises(ValueError):
            E.validate_answer(row, case, 'base', 'evaluation', {}, config())
        with self.assertRaises(ValueError):
            E.parameter_inventory(dict(base={'torch.float32@cuda:0': dict(tensors=1, elements=2, bytes=8)}, adapter={}))

    def test_all200_mode_and_missing_duplicate_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = mode_fixture(root)
            E.validate_mode(root, 'evaluation', plan)
            folder = root / 'evaluation'
            path = folder / 'base-outputs.jsonl'
            original = path.read_text().splitlines()
            for lines in (original[:-1], original + original[:1], [original[1], *original[1:]]):
                path.write_text('\n'.join(lines) + '\n')
                completion = json.loads((folder / 'completion.json').read_bytes())
                completion['outputs']['base'] = E.pin(path, folder)
                save(folder / 'completion.json', completion)
                with self.assertRaisesRegex(ValueError, 'Missing, duplicate'):
                    E.validate_mode(root, 'evaluation', plan)

    def test_smoke_namespaces_and_no_failure_acceptance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = mode_fixture(root, 'smoke')
            E.validate_mode(root, 'smoke', plan)
            save(root / 'smoke/failure.json', dict(status='failed'))
            with self.assertRaisesRegex(ValueError, 'Failure evidence'):
                E.validate_mode(root, 'smoke', plan)
        self.assertFalse(any('train/' in p or 'checkpoint' in p for p in E.output_members()))
        with self.assertRaises(ValueError):
            E.validate_mode(Path('.'), 'capacity', {})

    def test_portable_pin_refuses_parent_traversal(self):
        with self.assertRaises(ValueError):
            E.bound('.', dict(path='../secret', bytes=1, sha256='a' * 64))


if __name__ == '__main__':
    unittest.main()
