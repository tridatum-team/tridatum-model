"""Pure admission and evidence validation for the separately authorized 8B SFT.

Structural evidence is not semantic assessment or historical base qualification.
"""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath

SCHEMA = 'trid-eight-sft-experiment-v1'
APPROVAL_SCHEMA = 'trid-eight-sft-approval-v1'
RUN_ID = 'mois-eight-sft-20261114-071851'
MODEL_ID = 'Qwen/Qwen3-8B'
MODEL_REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
PREPARATION_SHA = '8aa882deeb7bda1c618108663f60c956b11613dad17d943df5700a981910a4ea'
AUTHORIZATION_SHA = '9d5580ba42ca6086ddd59317dd0e3a719b10aae6fb2d1f02862981eeb37959cc'
EXECUTION_AMENDMENT_SHA = '618e38af447e64c64c70df17b00e33d560834e19f807fef50c22acb24a73c31a'
RECIPE = dict(rows=24, epochs=3, steps=36, batch_size=2, micro_batch_size=1,
              rank=8, alpha=16, dropout=0.0, lr=1e-5, seed=14,
              context_tokens=3072, answer_tokens=768, checkpoint_every=12, valid_every=12)
CHECKPOINT_STEPS = (12, 24, 36)
PHASE_SECONDS = dict(acquire=900, capacity=600, baseline_dev=600, train=2400, checkpoint_dev=1800,
                     parent_selection=1800, final_base=1200, final_adapter=1200)
GENERATION = dict(enable_thinking=False, do_sample=False, num_beams=1, seed=14,
                  max_new_tokens=768, context_tokens=4096, dtype='bfloat16', backend='transformers-cuda')
DEV_EXPORT_MAX = 32 * 1024**2
CHANNEL_BYTES = DEV_EXPORT_MAX + 3 * 1024**2
SOURCE_NAMES = {'scripts/aws_eval/eight_hf_experiment.py', 'scripts/aws_eval/eight_hf_worker.py'}


def core():
    # Deferred import keeps this shared by the lifecycle and its native worker.
    if __package__:
        from . import grounded_hf_lifecycle
    else:
        import grounded_hf_lifecycle
    return grounded_hf_lifecycle


def is_eight(plan):
    return plan.get('schema') == SCHEMA


def arm(step):
    return f'step-{step:06d}'


def output_members(plan):
    members = {'runtime.json', 'runtime.log', 'capacity.log', 'capacity-receipt.json',
               'baseline_dev.log', 'baseline-dev-outputs.jsonl', 'train.log',
               'checkpoint_dev-receipt.json',
               'parent_selection-receipt.json', 'checkpoint-selection.json',
               'final-input-receipt.json', 'final_base.log', 'final_adapter.log',
               'final-base-outputs.jsonl', 'final-adapter-outputs.jsonl',
               'training/training.json', 'training/checkpoints.json', 'training/exposure-receipt.json'}
    for prefix in ['training/adapter', *[f'training/checkpoints/{arm(s)}/adapter' for s in CHECKPOINT_STEPS]]:
        members.update(prefix + '/' + name for name in ('adapter_config.json', 'adapter_model.safetensors', 'README.md'))
    for step in CHECKPOINT_STEPS:
        members.update({f'training/checkpoints/{arm(step)}/training.json',
                        f'checkpoint-{arm(step)}-dev-outputs.jsonl', f'checkpoint-{arm(step)}-dev.log'})
    return members


def dev_output_members(plan):
    members = output_members(plan) - {'parent_selection-receipt.json', 'checkpoint-selection.json',
        'final-input-receipt.json', 'final_base.log', 'final_adapter.log',
        'final-base-outputs.jsonl', 'final-adapter-outputs.jsonl'}
    return {n for n in members if '/adapter/' not in n}


def validate_workload(plan, bundled):
    c = core()
    c.require(is_eight(plan) and plan['identity']['run_id'] == RUN_ID, 'Unapproved eight experiment identity')
    c.require(plan['recipe'] == RECIPE and plan['phase_seconds'] == PHASE_SECONDS,
              'Eight recipe and finite phase ceilings are frozen')
    c.require(plan['model']['id'] == MODEL_ID and plan['model']['revision'] == MODEL_REVISION,
              'Eight base identity differs')
    w = plan['workload']
    c.require(set(w) == {'development_inputs', 'development_case_ids', 'development_behaviors',
        'final_inputs', 'final_gate_metadata', 'final_gate_amendment', 'prompt_audit', 'final_case_ids',
        'execution_amendment', 'source_closure', 'channel_keys', 'generation'}, 'Eight workload fields differ')
    c.require(w['generation'] == GENERATION, 'Matched native nonthinking settings differ')
    ident = plan['identity']
    c.require(w['channel_keys'] == dict(archive=ident['output_prefix'] + 'dev-checkpoint.tar',
        receipt=ident['output_prefix'] + 'dev-checkpoint.json',
        decision=ident['code_prefix'] + 'checkpoint-selection.json',
        final_inputs=ident['code_prefix'] + 'final-inputs.json'), 'Eight exact-key channel differs')
    c.require(set(plan['data']) == {'manifest', 'preparation_manifest', 'train', 'dev'}, 'Eight data closure fields differ')
    initial = [*plan['data'].values(), w['development_inputs'], w['source_closure'], plan['authorization']]
    for pin in [*initial, w['final_inputs'], w['final_gate_metadata'], w['final_gate_amendment'],
                w['prompt_audit'], w['execution_amendment']]:
        c.require(set(pin) == {'path', 'bytes', 'sha256'}, 'Exact input pin required')
        c.pins([pin], 32 * 1024**2)
    c.require(all(bundled.get(p['path']) == p for p in initial), 'Eight initial data not sealed in bundle')
    c.require(plan['authorization']['sha256'] == AUTHORIZATION_SHA, 'Fresh user authorization bytes differ')
    c.require(w['execution_amendment']['sha256'] == EXECUTION_AMENDMENT_SHA,
              'Effective generation/acquisition amendment differs from original-design-bound freeze')
    private = [w['final_inputs'], w['final_gate_metadata'], w['final_gate_amendment'],
               w['prompt_audit'], w['execution_amendment']]
    c.require(all(p['path'] not in bundled and p['bytes'] <= 1024**2 for p in private)
              and len({p['path'] for p in initial + private}) == len(initial) + 5,
              'Final inputs and private gate must be separate and absent from initial bundle')
    c.require(plan['data']['preparation_manifest']['sha256'] == PREPARATION_SHA,
              'Original preparation manifest lineage changed')
    dev = w['development_case_ids']
    finals = w['final_case_ids']
    c.require(set(finals) == {'general', 'mois'} and len(dev) == 8
              and all(len(finals[k]) == 12 for k in finals), 'Frozen cohort counts differ')
    ids = dev + finals['general'] + finals['mois']
    c.require(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == 32, 'Cohort IDs overlap')
    behaviors = w['development_behaviors']
    c.require(set(behaviors) == set(dev) and len(set(behaviors.values())) == 4
              and all(list(behaviors.values()).count(v) == 2 for v in behaviors.values()), 'Four balanced dev behaviors required')
    c.require(set(plan['sources']) == c.SOURCES | SOURCE_NAMES,
              'Eight executable closure must be exact')
    c.require(not any('reference' in n.lower() for n in bundled), 'Private references cannot be staged')
    c.require(set(plan['output_allowlist']) == output_members(plan)
              and len(plan['output_allowlist']) == len(output_members(plan)), 'Eight output membership differs')


def validate_data(root, plan):
    c = core()
    root = Path(root)
    validate_workload(plan, {p['path']: p for p in plan['bundle']['files']})
    def load(pin):
        path = root / pin['path']
        c.require(not path.is_symlink() and path.stat().st_size == pin['bytes'], 'Eight data envelope differs')
        return c.read_bound(path, pin['sha256'])
    auth = load(plan['authorization'])
    c.require(auth['schema'] == 'trid-eight-sft-user-authorization-v1' and auth['run_id'] == RUN_ID
        and auth['status'] == 'AUTHORIZED_NOT_DISPATCHED' and auth['base_model'] == MODEL_ID
        and auth['base_revision'] == MODEL_REVISION and auth['instance_limit'] == 1
        and auth['additional_pretax_usd'] == 20 and auth['maximum_lease_seconds'] == 14400
        and all(auth[k] is False for k in ('automatic_additional_lease', 'automatic_extension',
                                         'deployment_authorized', 'prior_gpu_grants_reused')),
        'Fresh bounded user authorization differs')
    data, w = plan['data'], plan['workload']
    original = load(data['preparation_manifest'])
    manifest = load(data['manifest'])
    lineage = manifest['lineage']
    c.require(set(lineage) == {'preparation_manifest_sha256', 'preparation_manifest',
        'preparation_result', 'design', 'closure_reports', 'messages_and_answers_unchanged',
        'historical_dev_train_exposure'} and lineage['preparation_manifest_sha256'] == PREPARATION_SHA
        and lineage['messages_and_answers_unchanged'] is True and lineage['historical_dev_train_exposure'] is True
        and lineage['preparation_manifest'] == data['preparation_manifest']
        and lineage['preparation_result']['sha256'] == 'd30924ba32da03d5851f7749eeeba44a1bdc9618e96070b779baaddc0b622aac'
        and lineage['design']['sha256'] == 'ea3d169b8879d0b741eab846797dffeba7003e58b96bfe8229144b54562e62c4',
        'Derived manifest provenance differs')
    closure_hashes = {
        'code_review_and_tests': 'dd75d1e08b346829ed60da60dff5591724cc5c47f987b1215b3df139fd64f664',
        'configured_reservation_checks': 'c5484799e669f09c31a3de2801852a659d142668c8be353cd559ba4d50c9979a',
        'fixed_evaluation_fingerprint_check': '3d31396a592691b660bb8b5a6701a7c536c5c908d9a6a82c059d47d6f0951b6c',
        'native_input_and_random_cpu_smoke': '24341ca5eb4c6d61c5e498592d749577bfe1c6668b8468f234c94c5600f836bb',
        'semantic_review': 'ea6e21ebd25731d9303d822f17dc703b83964963a1b509543d06e48d85acfdf2',
        'source_and_reservation_admission': 'bbc51772f2f54303978fcd792f61276053180d2ee886d9fee8eb6c8148a8603d',
        'source_attribution': 'cc77e0a896bf6d85d9cb55ff8181b1ca4a62d342ac0c7003d335b43f1d7cdd04'}
    c.require({k: p['sha256'] for k, p in lineage['closure_reports'].items()} == closure_hashes,
              'Preparation closure evidence changed')
    c.pins([lineage['preparation_manifest'], lineage['preparation_result'], lineage['design'],
            *lineage['closure_reports'].values()], 32 * 1024**2)
    seen, rows_by_split = set(), {}
    for split, key, count, epochs in [('train', 'train', 24, 3), ('dev', 'valid', 8, 1)]:
        entries = manifest[key]
        c.require(len(entries) == 1, 'Exactly one train/dev entry required')
        entry = entries[0]
        path = (PurePosixPath(data['manifest']['path']).parent / c.relative(entry['path'])).as_posix()
        c.require(path == data[split]['path'] and entry['sha256'] == data[split]['sha256']
            and entry['adapter'] == 'grounded_messages' and entry['epochs'] == epochs
            and entry['split'] == split, 'Derived execution exposure contract differs')
        file = root / path
        c.require(not file.is_symlink() and file.stat().st_size == data[split]['bytes']
                  and c.digest(file) == data[split]['sha256'], 'Train/dev bytes differ')
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        c.require(len(rows) == count, 'Train/dev counts differ')
        for row in rows:
            c.require(row['id'] not in seen and row['split'] == split and isinstance(row['answer'], str)
                      and row['answer'].strip(), 'Duplicate/split/empty training target')
            c.blind_messages({'id': row['id'], 'messages': row['messages']})
            c.require([m['role'] for m in row['messages']] == ['system', 'user'], 'Native framing differs')
            seen.add(row['id'])
        rows_by_split[split] = rows
    dev = load(w['development_inputs'])
    c.require(dev == [{'id': r['id'], 'messages': r['messages']} for r in rows_by_split['dev']]
              and [r['id'] for r in dev] == w['development_case_ids'], 'Dev prompts/labels differ')
    c.require(not seen.intersection(w['final_case_ids']['general'] + w['final_case_ids']['mois']), 'Final/train overlap')
    closure = dict(schema='trid-eight-sft-source-closure-v1', run_id=RUN_ID,
        preparation_manifest_sha256=PREPARATION_SHA, manifest_sha256=data['manifest']['sha256'],
        train_sha256=data['train']['sha256'], dev_sha256=data['dev']['sha256'],
        development_inputs_sha256=w['development_inputs']['sha256'], final_inputs_sha256=w['final_inputs']['sha256'],
        private_references_in_bundle=False, final_inputs_in_bundle=False)
    c.require(load(w['source_closure']) == closure, 'Eight source closure differs')
    # Only existing bootstrap/cost evidence may accompany the closed data.
    bundled = {p['path']: p for p in plan['bundle']['files']}
    allowed = set(plan['sources']) | {p['path'] for p in data.values()} | {
        w['development_inputs']['path'], w['source_closure']['path'], plan['authorization']['path']}
    bootstrap_pin, cost_pin = plan['runtime']['bootstrap']['admission'], plan['ancillary_evidence']
    for pin in (bootstrap_pin, cost_pin):
        allowed.add(pin['path'])
    bootstrap = c.read_bound(root / bootstrap_pin['path'], bootstrap_pin['sha256'])
    cost = c.read_bound(root / cost_pin['path'], cost_pin['sha256'])
    def evidence_pins(value):
        if isinstance(value, dict):
            if {'path', 'sha256'} <= set(value):
                c.require(value['path'] in bundled and bundled[value['path']]['sha256'] == value['sha256'],
                          'Undeclared evidence pin')
                allowed.add(value['path'])
            else:
                for child in value.values():
                    evidence_pins(child)
        elif isinstance(value, list):
            for child in value:
                evidence_pins(child)
    evidence_pins(bootstrap['evidence'])
    evidence_pins(cost['bucket_policy_status_evidence'])
    evidence_pins(cost['ebs_kms_key_evidence'])
    c.require(set(bundled) == allowed, 'Unexpected initial bundle data outside closed evidence')
    for key, split in [('train', 'train'), ('valid', 'dev')]:
        c.require(len(original[key]) == 1 and original[key][0]['sha256'] == data[split]['sha256'],
                  'Execution train/dev bytes differ from immutable preparation')
    return {'development_inputs': dev}


def validate_selection(plan, plan_sha, decision, dev_export_sha256, checkpoint_hashes):
    c = core()
    c.require(set(decision) == {'schema', 'run_id', 'plan_sha256', 'dev_export_sha256',
        'checkpoint_hashes', 'review_pins', 'aggregate', 'selected_step'}, 'Selection fields differ')
    c.require(decision['schema'] == 'trid-eight-sft-selection-v1' and decision['run_id'] == plan['identity']['run_id']
        and decision['plan_sha256'] == plan_sha and decision['dev_export_sha256'] == dev_export_sha256
        and decision['checkpoint_hashes'] == checkpoint_hashes
        and set(checkpoint_hashes) == {arm(s) for s in CHECKPOINT_STEPS}
        and all(c.SHA.fullmatch(v) for v in checkpoint_hashes.values())
        and c.SHA.fullmatch(dev_export_sha256), 'Selection provenance differs')
    reviews = decision['review_pins']
    c.require(len(reviews) in (5, 6) and all(set(p) == {'path', 'bytes', 'sha256'} for p in reviews), 'Independent blinded review pins required')
    c.pins(reviews, 32 * 1024**2)
    c.require(len({r['sha256'] for r in reviews}) == len(reviews), 'Review evidence must be distinct')
    aggregate = decision['aggregate']
    c.require(set(aggregate) == {'base', 'candidates'}, 'Selection aggregate differs')
    candidates = aggregate['candidates']
    c.require([r['step'] for r in candidates] == list(CHECKPOINT_STEPS), 'Checkpoint candidate membership differs')
    behaviors = set(plan['workload']['development_behaviors'].values())
    for row in [aggregate['base'], *candidates]:
        fields = {'joint_pass_count', 'critical_errors', 'behavior_pass_counts'}
        if row is not aggregate['base']:
            fields |= {'step', 'teacher_forced_dev_loss'}
            loss = row['teacher_forced_dev_loss']
            c.require(type(loss) in (int, float) and math.isfinite(loss) and loss >= 0, 'Finite dev loss required')
        counts = row['behavior_pass_counts']
        c.require(set(row) == fields and set(counts) == behaviors
            and all(type(v) is int and 0 <= v <= 2 for v in counts.values())
            and type(row['joint_pass_count']) is int and row['joint_pass_count'] == sum(counts.values())
            and type(row['critical_errors']) is int and 0 <= row['critical_errors'] <= 8,
            'Dev aggregate counts differ')
    base = aggregate['base']['behavior_pass_counts']
    eligible = [r for r in candidates if r['joint_pass_count'] >= 6 and r['critical_errors'] == 0
                and all(r['behavior_pass_counts'][k] >= base[k] for k in behaviors)]
    chosen = min(eligible, key=lambda r: (-r['joint_pass_count'], r['teacher_forced_dev_loss'], r['step']))['step'] if eligible else None
    c.require(decision['selected_step'] == chosen and (chosen is None or type(decision['selected_step']) is int),
              'Selection differs from frozen eligibility/ranking')
    return chosen


def validate_outputs(plan, plan_sha, rows, cohort, step, checkpoint_hashes):
    c = core()
    w = plan['workload']
    ids = w['development_case_ids'] if cohort == 'dev' else w['final_case_ids']['general'] + w['final_case_ids']['mois']
    c.require([r['id'] for r in rows] == ids, 'Output cohort membership/order differs')
    input_sha = w['development_inputs' if cohort == 'dev' else 'final_inputs']['sha256']
    for row in rows:
        expected_cohort = 'dev' if cohort == 'dev' else ('general' if row['id'] in w['final_case_ids']['general'] else 'mois')
        c.require(row['run_id'] == plan['identity']['run_id'] and row['plan_sha256'] == plan_sha
            and row['cohort'] == expected_cohort and row['arm'] == ('base' if step is None else arm(step))
            and row['adapter_weights_sha256'] == (None if step is None else checkpoint_hashes[arm(step)])
            and row['model_id'] == MODEL_ID and row['model_revision'] == MODEL_REVISION
            and row['input_sha256'] == input_sha and row['generation'] == GENERATION
            and row['prompt_sha256'] == c.object_sha(row['messages'])
            and c.SHA.fullmatch(row['rendered_prompt_sha256']), 'Output execution identity differs')
        c.blind_messages({'id': row['id'], 'messages': row['messages']})
        c.require(isinstance(row['prompt_token_ids'], list) and 0 < len(row['prompt_token_ids']) < GENERATION['context_tokens']
            and isinstance(row['output_token_ids'], list) and 0 < len(row['output_token_ids']) <= 768
            and all(type(t) is int and t >= 0 for t in row['prompt_token_ids'] + row['output_token_ids'])
            and row['generated_tokens'] == len(row['output_token_ids'])
            and len(row['prompt_token_ids']) + row['generated_tokens'] <= GENERATION['context_tokens']
            and row['termination_reason'] in ('eos', 'length')
            and isinstance(row['raw_output'], str) and isinstance(row['response_text'], str),
            'Output token/finish metadata differs')
        c.require(len(row['prompt_token_ids']) + 768 <= GENERATION['context_tokens']
            and ((row['termination_reason'] == 'length' and row['generated_tokens'] == 768
                  and row['terminal_stop_token_id'] is None)
                 or (row['termination_reason'] == 'eos'
                     and type(row['terminal_stop_token_id']) is int
                     and row['terminal_stop_token_id'] == row['output_token_ids'][-1])),
            'Output context reservation/terminal stop metadata differs')


def checkpoint_evidence(plan, root, *, weights=True):
    c = core()
    root = Path(root)
    ledger = json.loads((root / 'training/checkpoints.json').read_text())
    c.require(ledger['run_status'] == 'complete' and ledger['planned_steps'] == 36
        and ledger['checkpoint_every'] == 12 and ledger['base_model_id'] == MODEL_ID
        and ledger['base_revision'] == MODEL_REVISION and ledger['manifest_sha256'] == plan['data']['manifest']['sha256']
        and ledger['init_adapter'] is None and ledger['selection']['status'] == 'unselected'
        and [r['step'] for r in ledger['checkpoints']] == list(CHECKPOINT_STEPS), 'Checkpoint ledger differs')
    hashes = {}
    for row in ledger['checkpoints']:
        step = row['step']
        prefix = root / f'training/checkpoints/{arm(step)}'
        sha = row['adapter_weights_sha256']
        c.require(c.SHA.fullmatch(sha), 'Checkpoint hash invalid')
        if weights:
            c.require(c.digest(prefix / 'adapter/adapter_model.safetensors') == sha, 'Checkpoint weights differ')
            config = json.loads((prefix / 'adapter/adapter_config.json').read_text())
            c.require(config['r'] == 8 and config['lora_alpha'] == 16 and config['lora_dropout'] == 0.0
                and config['peft_type'] == 'LORA'
                and set(config['target_modules']) == {'q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                      'gate_proj', 'up_proj', 'down_proj'},
                'Checkpoint adapter configuration differs')
        report = json.loads((prefix / 'training.json').read_text())
        c.require(report['steps'] == step and report['status'] == 'checkpoint'
            and report['checkpoint'] == row and report['manifest_sha256'] == ledger['manifest_sha256']
            and report['base_model_id'] == MODEL_ID and report['base_revision'] == MODEL_REVISION
            and report['train_rows'] == 72 and report['valid_rows'] == 8
            and report['dropped_over_max_tokens'] == 0 and report['valid_dropped'] == 0,
            'Checkpoint step/data report differs')
        c.require(all(report[k] == RECIPE[k] for k in ('batch_size', 'micro_batch_size', 'lr', 'seed'))
            and report['max_tokens'] == 3072 and report['max_answer_tokens'] == 768
            and report['lora']['rank'] == 8 and report['lora']['alpha'] == 16
            and report['lora']['dropout'] == 0.0 and report['init_adapter'] is None,
            'Checkpoint training recipe differs')
        losses = row['valid_loss_by_task']
        c.require(isinstance(losses, dict) and losses and all(type(v) in (int, float)
            and math.isfinite(v) and v >= 0 for v in losses.values()), 'Checkpoint dev loss missing')
        hashes[arm(step)] = sha
    c.require(ledger['last_step']['step'] == 36 and ledger['last_step']['equals_last_checkpoint'] is True
        and ledger['last_step']['adapter_weights_sha256'] == hashes[arm(36)], 'Last adapter ledger differs')
    if weights:
        c.require(hashes[arm(36)] == c.digest(root / 'training/adapter/adapter_model.safetensors'),
                  'Last adapter differs from checkpoint36')
    return hashes, ledger


def validate_dev_export(plan, plan_sha, receipt, root):
    c = core()
    c.require(set(receipt) == {'schema', 'run_id', 'plan_sha256', 'instance_id', 'archive', 'files', 'checkpoint_hashes'}
        and receipt['schema'] == 'trid-eight-sft-dev-export-v1' and receipt['run_id'] == plan['identity']['run_id']
        and receipt['plan_sha256'] == plan_sha, 'Dev export identity differs')
    c.require(set(receipt['archive']) == {'bytes', 'sha256'} and type(receipt['archive']['bytes']) is int
        and 0 < receipt['archive']['bytes'] <= DEV_EXPORT_MAX and c.SHA.fullmatch(receipt['archive']['sha256']), 'Dev archive bound differs')
    files = c.pins(receipt['files'], DEV_EXPORT_MAX)
    c.require(set(files) == dev_output_members(plan), 'Dev export membership differs')
    root = Path(root)
    for name, pin in files.items():
        path = root / name
        c.require(not path.is_symlink() and path.stat().st_size == pin['bytes'] and c.digest(path) == pin['sha256'], 'Dev export member changed')
    hashes, ledger = checkpoint_evidence(plan, root, weights=False)
    c.require(receipt['checkpoint_hashes'] == hashes, 'Dev receipt checkpoint hashes differ')
    for step in (None, *CHECKPOINT_STEPS):
        name = 'baseline-dev-outputs.jsonl' if step is None else f'checkpoint-{arm(step)}-dev-outputs.jsonl'
        rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
        validate_outputs(plan, plan_sha, rows, 'dev', step, hashes)
    return hashes


def validate_completion(plan, plan_sha, completion, root):
    c = core()
    root = Path(root)
    c.require(set(completion) == {'experimental_success', 'stop_errors', 'run_id', 'plan_sha256',
        'instance_id', 'experiment_outcome', 'dev_export_sha256', 'selection_sha256', 'selected_step',
        'runtime', 'runtime_unit', 'stages', 'export'}, 'Experimental completion fields differ')
    identity = dict(run_id=plan['identity']['run_id'], plan_sha256=plan_sha, instance_id=completion['instance_id'])
    runtime = json.loads((root / 'runtime.json').read_text())
    c.require(runtime == completion['runtime'] and all(runtime.get(k) == v for k, v in identity.items())
        and runtime['packages'] == {k.lower().replace('_', '-'): v for k, v in plan['runtime']['packages'].items()}
        and runtime['torch'] == plan['runtime']['packages']['torch'] and runtime['cuda'] == plan['runtime']['torch_cuda']
        and runtime['count'] == 1 and runtime['bf16'] is True and runtime['name'] == plan['runtime']['gpu_name'],
        'Eight runtime package/GPU proof differs')
    unit = completion['runtime_unit']
    c.require(all(unit.get(k) == v for k, v in identity.items()) and unit['phase'] == 'runtime'
              and unit['state'] == 'complete' and unit['unit'] == identity['run_id'] + '-runtime'
              and isinstance(unit['invocation'], str) and unit['invocation'], 'Runtime ownership differs')
    c.require(completion['experimental_success'] is True and completion['stop_errors'] == []
        and completion['run_id'] == plan['identity']['run_id'] and completion['plan_sha256'] == plan_sha
        and 'training_success' not in completion and 'inference_success' not in completion, 'Experimental completion differs')
    hashes, ledger = checkpoint_evidence(plan, root)
    exposure = json.loads((root / 'training/exposure-receipt.json').read_text())
    train_items, dev_items = exposure['training_item_hashes'], exposure['development_item_hashes']
    c.require(exposure['run_id'] == plan['identity']['run_id'] and exposure['plan_sha256'] == plan_sha
        and exposure['manifest_sha256'] == plan['data']['manifest']['sha256']
        and exposure['checkpoint_hashes'] == hashes and exposure['training_exposures'] == 72
        and exposure['unique_training_rows'] == 24 and exposure['exposures_per_row'] == 3
        and exposure['development_rows'] == 8 and exposure['dropped_rows'] == 0
        and len(train_items) == 72 and len(set(train_items)) == 24
        and all(train_items.count(value) == 3 for value in set(train_items))
        and len(dev_items) == len(set(dev_items)) == 8 and not set(train_items).intersection(dev_items)
        and all(c.SHA.fullmatch(value) for value in train_items + dev_items), 'Actual native exposure evidence differs')
    selection_path = root / 'checkpoint-selection.json'
    decision = json.loads(selection_path.read_text())
    step = validate_selection(plan, plan_sha, decision, completion['dev_export_sha256'], hashes)
    c.require(c.digest(selection_path) == completion['selection_sha256'] and completion['selected_step'] == step
        and completion['experiment_outcome'] == ('selected' if step is not None else 'no_eligible_checkpoint'), 'Completion selection differs')
    for candidate, checkpoint in zip(decision['aggregate']['candidates'], ledger['checkpoints']):
        losses = checkpoint['valid_loss_by_task']
        c.require(candidate['teacher_forced_dev_loss'] == sum(losses.values()) / len(losses), 'Selection dev loss differs from checkpoint ledger')
    phases = [name for name in PHASE_SECONDS if name != 'acquire']
    if step is None:
        phases = phases[:5]
    c.require([s['phase'] for s in completion['stages']] == phases
        and all(s['state'] == 'complete' and s['run_id'] == plan['identity']['run_id']
                and s['plan_sha256'] == plan_sha for s in completion['stages']), 'Experimental stages differ')
    for stage in completion['stages']:
        c.require(all(stage.get(k) == v for k, v in identity.items())
            and all(type(stage[k]) in (int, float) and math.isfinite(stage[k])
                    for k in ('started_epoch', 'finished_epoch'))
            and 0 <= stage['finished_epoch'] - stage['started_epoch'] <= PHASE_SECONDS[stage['phase']] + 1,
            'Experimental phase timing/owner differs')
    for phase in ('checkpoint_dev', 'parent_selection'):
        stage = next(s for s in completion['stages'] if s['phase'] == phase)
        c.require(json.loads((root / (phase + '-receipt.json')).read_text()) == stage,
                  'Exported development/selection stage differs')
    expected = output_members(plan) - {'final-input-receipt.json', 'final_base.log', 'final_adapter.log',
                                     'final-base-outputs.jsonl', 'final-adapter-outputs.jsonl'}
    if step is not None:
        expected = output_members(plan)
        receipt = json.loads((root / 'final-input-receipt.json').read_text())
        c.require(receipt == dict(run_id=plan['identity']['run_id'], plan_sha256=plan_sha,
            sha256=plan['workload']['final_inputs']['sha256'],
            bytes=plan['workload']['final_inputs']['bytes'], selection_sha256=completion['selection_sha256']), 'Delayed final input receipt differs')
        final_rows = []
        for name, arm_step in [('final-base-outputs.jsonl', None), ('final-adapter-outputs.jsonl', step)]:
            rows = [json.loads(line) for line in (root / name).read_text().splitlines()]
            validate_outputs(plan, plan_sha, rows, 'final', arm_step, hashes)
            final_rows.append(rows)
        c.require(all(a['messages'] == b['messages'] and a['prompt_token_ids'] == b['prompt_token_ids']
            and a['rendered_prompt_sha256'] == b['rendered_prompt_sha256'] for a, b in zip(*final_rows)),
            'Original/adapter final framing differs')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()} - {'export-manifest.json'}
    c.require(actual == expected, 'Conditional completion output membership differs')
