"""Fresh CUDA phases; imports Torch only after root-owned native admission.

The root orchestrator accepts receipts only after the existing systemd/cgroup
supervisor has stopped and reaped the nonroot phase. No grading wait occurs here.
"""
import gc
import json
import math
from pathlib import Path
import shutil
import sys
import time

import fresh_eight_experiment as E


def phase_kind(phase):
    return phase if phase in ('runtime', 'capacity', 'train', 'evaluate') else None


def runtime_record(core, plan, base):
    return dict(device='cuda', backend='transformers-cuda', run_id=plan['identity']['run_id'],
                plan_sha256=core.digest(Path(base) / 'plan.json'),
                observed=json.loads((Path(base) / 'output/runtime.json').read_bytes()))


def observer_class(base):
    # Reuse the established numerical and source-row observer without importing
    # any MPS runtime, allocator, supervisor, or historical policy contract.
    sys.path.insert(0, str(Path(base) / 'code/scripts'))
    import local_mps_training as numerical

    class CudaObserver(numerical.Observer):
        def telemetry(self):
            import torch
            torch.cuda.synchronize()
            return dict(monotonic=time.monotonic(), allocator_bytes=torch.cuda.memory_allocated(),
                        reserved_bytes=torch.cuda.memory_reserved(), max_allocated_bytes=torch.cuda.max_memory_allocated(),
                        max_reserved_bytes=torch.cuda.max_memory_reserved())

        def __call__(self, event, **data):
            import torch
            if event == 'model':
                self.inventory = numerical.parameter_contract(data['model'], device='cuda')
                E.require(all(not bool(value.detach().count_nonzero()) for name, value in data['model'].named_parameters()
                              if 'lora_B' in name), 'Fresh adapter B matrices must start at zero')
                self.telemetry()
                return None
            if event == 'report':
                return dict(policy=self.policy_pin, actual_source_files=self.files,
                    coverage_complete=self.complete, dtype_inventory=self.inventory, runtime=self.runtime,
                    endpoint_selection='final150 only; checkpoints50/100 diagnostics; no validation-loss selection')
            return super().__call__(event, **data)

    return CudaObserver, numerical


def fresh_model(plan, base):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    # Same HF BF16 load and PEFT promotion path as the actual CUDA trainer.
    model = AutoModelForCausalLM.from_pretrained(Path(base) / 'model', torch_dtype=torch.bfloat16,
                                               local_files_only=True).to('cuda')
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        task_type='CAUSAL_LM'), autocast_adapter_dtype=True)
    return model


def capacity_stage(core, plan, base, torch, admitted):
    from transformers import AutoTokenizer
    root = Path(base) / 'output'
    out, scratch = root / 'capacity', Path(base) / 'scratch/tmp/fresh-capacity'
    out.mkdir(); scratch.mkdir()
    trainer = core.trainer_module(base)
    tok = AutoTokenizer.from_pretrained(Path(base) / 'model', local_files_only=True)
    manifest_path = Path(base) / 'code' / plan['data']['manifest']['path']
    train, dropped = trainer['build_items'](tok, admitted['manifest']['train'], manifest_path.parent,
                                           'qwen3', 3072, 768, True)
    Observer, numerical = observer_class(base)
    runtime = runtime_record(core, plan, base)
    observer = Observer(admitted['policy'], plan['data']['policy'], admitted['sources'], out, runtime)
    observer('admission', train=train, valid=[], dropped=dropped, valid_dropped=0)
    item = core.synthetic_capacity_item(tok, trainer, scratch, 3072, 768)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = fresh_model(plan, base)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=1e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: (s + 1) / 7 if s < 7 else
        0.5 * (1 + math.cos(math.pi * (s - 7) / 143)))
    observer('model', model=model)
    model.train()
    history = []
    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        resident_backwards = []
        for micro in range(4):
            loss = trainer['batch_loss'](model, [item], 'cuda', pad, 'qwen3') / 4
            E.require(bool(torch.isfinite(loss)), 'Nonfinite full-envelope capacity loss')
            losses.append(float(loss.detach()))
            loss.backward()
            observer('backward', model=model, optimizer=optimizer, step=step)
            if step == 2:
                resident_backwards.append(numerical.gradient_health(model, optimizer, resident=True))
        health = observer('before_step', model=model, optimizer=optimizer, step=step)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
        applied = optimizer.param_groups[0]['lr']
        optimizer.step(); schedule.step()
        health.update(observer('after_step', model=model, optimizer=optimizer, step=step))
        history.append(dict(step=step, loss=sum(losses), microbatch_losses=losses,
            applied_lr=applied, next_lr=schedule.get_last_lr()[0], numerical_health=health,
            resident_backward_checks=resident_backwards))
    adapter = scratch / 'scratch-adapter'
    model.save_pretrained(adapter)
    expected = {name: value.detach().cpu() for name, value in model.named_parameters() if value.requires_grad}
    scratch_weights = E.pin(adapter / 'adapter_model.safetensors', Path(base))
    # No graph/model/optimizer alias survives this release. CPU tensor snapshots
    # are retained solely for exact fresh-base reload comparison.
    del loss, optimizer, schedule, model
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    release_memory = observer.telemetry()
    model = fresh_model(plan, base)
    trainer['load_init_adapter'](model, adapter, 'cuda')
    numerical.verify_adapter_reload(model, expected)
    reload_inventory = numerical.parameter_contract(model, device='cuda')
    model.eval()
    with torch.no_grad():
        reload_loss = float(trainer['batch_loss'](model, [item], 'cuda', pad, 'qwen3'))
    E.require(math.isfinite(reload_loss), 'Nonfinite full-envelope reload loss')
    core.save(out / 'capacity-worker.json', dict(schema='trid-fresh-eight-cuda-capacity-v1',
        status='complete', policy=plan['data']['policy'], source_manifest=plan['workload']['source_manifest'],
        actual_source_files=admitted['sources'], runtime=runtime, train_rows=600, dropped=0,
        sequence_tokens=3072, supervised_tokens=768, masked_prompt_tokens=2304, includes_eos=True,
        optimizer_updates=2, microbatches_per_update=4, dtype_inventory=observer.inventory,
        reload_dtype_inventory=reload_inventory, history=history, reload_exact=True, reload_loss=reload_loss,
        scratch_adapter_never_nominated=True, released_before_reload=True, release_memory=release_memory,
        scratch_weights=scratch_weights, telemetry=observer.telemetry()))


def train_argv(plan, base):
    base = Path(base)
    return ['--model', str(base / 'model'), '--model-id', E.MODEL_ID, '--model-revision', E.MODEL_REVISION,
        '--family', 'qwen3', '--manifest', str(base / 'code' / plan['data']['manifest']['path']),
        '--out', str(base / 'output/train'), '--steps', '150', '--batch-size', '4', '--micro-batch-size', '1',
        '--lr', '1e-5', '--rank', '8', '--alpha', '16', '--dropout', '0', '--max-tokens', '3072',
        '--max-answer-tokens', '768', '--seed', '14', '--device', 'cuda', '--checkpoint-every', '50', '--valid-every', '0']


def portable_checkpoint_paths(core, root):
    """Relocate only artifact paths; physical training row identities stay intact."""
    root = Path(root)
    paths = [root / 'train/training.json', root / 'train/checkpoints.json',
             *[root / f'train/checkpoints/step-{step:06d}/training.json' for step in E.CHECKPOINT_STEPS]]
    for path in paths:
        payload = json.loads(path.read_bytes())
        rows = list(payload.get('checkpoints', []))
        rows.extend(payload[key] for key in ('last_step', 'checkpoint') if key in payload)
        for row in rows:
            expected = 'train/adapter' if row is payload.get('last_step') else f'train/checkpoints/step-{row["step"]:06d}/adapter'
            E.require(Path(row['path']) == root / expected, 'Trainer emitted an unexpected adapter location')
            row['path'] = expected
        core.save(path, payload)


def train_stage(core, plan, base, admitted):
    root = Path(base) / 'output'
    E.validate_capacity(root, plan); E.validate_supervision(root, 'capacity', plan)
    Observer, _ = observer_class(base)
    observer = Observer(admitted['policy'], plan['data']['policy'], admitted['sources'], root / 'train',
                        runtime_record(core, plan, base))
    trainer = core.trainer_module(base)
    trainer['main'](argv=train_argv(plan, base), observer=observer)
    E.require(observer.complete, 'Training observer did not accept all600 coverage')
    portable_checkpoint_paths(core, root)
    core.save(root / 'train/training-worker.json', dict(schema='trid-fresh-eight-cuda-training-v1',
        status='complete', policy=plan['data']['policy'], source_manifest=plan['workload']['source_manifest'],
        coverage_complete=True, report=E.pin(root / 'train/training.json', root)))
    E.validate_training(root, plan)


def child_stage(core, plan, base, phase):
    E.require(phase in ('capacity', 'train', 'evaluate'), 'Unknown fresh CUDA phase')
    admitted = E.validate_data(Path(base) / 'code', plan)
    core.validate_model(Path(base) / 'model', plan)
    torch = core.native_runtime(plan, base)
    if phase == 'capacity':
        capacity_stage(core, plan, base, torch, admitted)
    elif phase == 'train':
        train_stage(core, plan, base, admitted)
    else:
        root = Path(base) / 'output'
        sys.path.insert(0, str(Path(base) / 'code'))
        E.validate_training(root, plan); E.validate_supervision(root, 'train', plan)
        sys.path.insert(0, str(Path(base) / 'code/scripts'))
        from evaluate_local_eight import evaluate_cloud
        evaluate_cloud(plan, Path(base))


def run_experiment(core, plan, base, report, client, gpu, account, work_deadline,
                   acquisition_deadline, budget, hosts, units):
    base = Path(base)
    root = base / 'output'
    for row in plan['model']['files']:
        core.download(row, base / 'model', hosts, acquisition_deadline, budget)
    core.validate_model(base / 'model', plan)
    core.save(root / 'model-acquisition.json', dict(status='authenticated', model_id=E.MODEL_ID,
        revision=E.MODEL_REVISION, plan_sha256=core.digest(base / 'plan.json'),
        files=[{k: row[k] for k in ('path', 'bytes', 'sha256')} for row in plan['model']['files']]))
    for path in (base / 'model').iterdir():
        path.chmod(0o444)
    shutil.copyfile(base / 'plan.json', root / 'plan.json')
    (root / 'plan.json').chmod(0o444)
    for phase, result_name in [('capacity', 'capacity/capacity-worker.json'),
                               ('train', 'train/training-worker.json'), ('evaluate', 'evaluation/completion.json')]:
        if phase == 'evaluate':
            core.save(root / 'candidate.json', dict(schema='trid-fresh-eight-cuda-candidate-v1',
                plan=E.pin(root / 'plan.json', root), candidate_id=E.RUN_ID,
                arm='candidate', step=150))
            (root / 'candidate.json').chmod(0o444)
        unit = core.supervise_phase(plan, base, phase, gpu, account.pw_uid, account.pw_gid, work_deadline, units,
                                    attempt=report.setdefault('phase_attempt', {}))
        E.require(unit['cleanup_verified'] is True, 'Fresh phase has no owned-process cleanup proof')
        report['stages'].append(unit)
        if phase == 'capacity':
            E.validate_capacity(root, plan)
            scratch = base / 'scratch/tmp/fresh-capacity'
            E.require(scratch.is_dir() and not scratch.is_symlink(), 'Scratch capacity ownership differs')
            shutil.rmtree(scratch)
        elif phase == 'train':
            E.validate_training(root, plan)
        else:
            E.validate_evaluation(root, plan)
        folder = root / ('evaluation' if phase == 'evaluate' else phase)
        core.save(folder / 'supervisor-result.json', dict(schema='trid-fresh-eight-cuda-supervision-v1',
            phase=phase, status='accepted', cleanup_verified=True, run_id=plan['identity']['run_id'],
            plan_sha256=core.digest(base / 'plan.json'), policy=plan['data']['policy'],
            source_manifest=plan['workload']['source_manifest'], worker_result=E.pin(root / result_name, root), unit=unit))
        # Only the supervisor creates acceptance. Next phases cannot mutate it.
        for path in [folder, *folder.rglob('*')]:
            E.require(not path.is_symlink(), 'Unsafe phase output')
            import os
            os.chown(path, 0, 0)
            path.chmod(0o555 if path.is_dir() else 0o444)
        if phase == 'train':
            # Persist only accepted, reaped, root-frozen training artifacts.
            # This incomplete snapshot is recovery evidence, never quality success.
            E.require(not units, 'Training snapshot requires all owned units reaped')
            core.publish_outputs(base, plan, report, client, work_deadline)
    E.validate_completion(root, plan)
