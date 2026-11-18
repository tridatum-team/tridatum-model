"""Pinned serial local inference diagnostics. Launch through mlx_grounded_guard --phase inference."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import mlx_inference as inference
import mlx_grounded as worker
from mlx_grounded_guard import read_limits

SCHEMA = 'trid-offline-mlx-inference-v1'
RECEIPT_SCHEMA = 'trid-offline-mlx-inference-receipt-v1'
LIMIT_KEYS = {'rss_stop_bytes', 'mlx_stop_bytes', 'swap_growth_stop_bytes', 'pressure_required',
              'question_seconds', 'inference_seconds'}
SOURCE_NAMES = ('mlx_inference.py', 'diagnose_mlx_inference.py', 'mlx_grounded.py', 'mlx_grounded_guard.py')


def validate_directory(declaration, *, adapter=False):
    worker.require(isinstance(declaration, dict) and set(declaration) == {'path', 'files'},
                   'Directory declaration requires exact path/files')
    root = Path(declaration['path'])
    worker.require(root.is_absolute() and root.is_dir(), 'Existing absolute pinned directory required')
    pins = declaration['files']
    worker.require(isinstance(pins, list) and bool(pins), 'Nonempty file pins required')
    names = [pin['path'] for pin in pins]
    worker.require(len(set(names)) == len(names), 'Duplicate file pins')
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()}
    worker.require(set(names) == actual, 'Directory must contain exactly its pinned files')
    for pin in pins:
        path = worker.verify_pin(pin, root)
        worker.require(path.resolve().is_relative_to(root.resolve()), 'Pinned file escapes directory')
    if adapter:
        worker.require({'adapter_config.json', 'adapters.safetensors'} <= actual, 'Incomplete adapter pins')
    else:
        return inference.local_model_config(root)


def validate_plan(plan):
    required = {'schema', 'run_id', 'model', 'adapter', 'cases', 'profiles', 'limits', 'output_dir', 'source_pins'}
    worker.require(isinstance(plan, dict) and required <= set(plan) <= required | {'parity'},
                   'Inference plan requires exact registered fields')
    worker.require(plan['schema'] == SCHEMA and isinstance(plan['run_id'], str) and bool(plan['run_id']),
                   'Invalid inference plan identity')
    worker.require(type(plan.get('parity', False)) is bool, 'parity must be boolean')
    worker.require(isinstance(plan['limits'], dict) and set(plan['limits']) == LIMIT_KEYS,
                   'Inference limits require exact fields')
    read_limits(plan, 'inference')
    worker.require(plan['limits']['inference_seconds'] <= 1800, 'Inference deadline exceeds 1800 seconds')
    config = validate_directory(plan['model'])
    if plan['adapter'] is not None:
        validate_directory(plan['adapter'], adapter=True)
    worker.require(isinstance(plan['profiles'], list) and bool(plan['profiles']), 'Explicit profiles required')
    profiles = [inference.validate_profile(profile) for profile in plan['profiles']]
    worker.require(len({profile['name'] for profile in profiles}) == len(profiles), 'Duplicate profile names')
    context = config.get('text_config', config)['max_position_embeddings']
    worker.require(all(profile['context_tokens'] <= context for profile in profiles),
                   'Profile exceeds actual model context')
    if plan.get('parity'):
        worker.require(any(profile == inference.historical_profile() for profile in profiles),
                       'Parity requires the exact historical 768-token profile')
    case_path = worker.verify_pin(plan['cases'])
    cases = json.loads(case_path.read_text())
    worker.require(isinstance(cases, list) and bool(cases), 'Nonempty pinned cases required')
    for case in cases:
        worker.require(isinstance(case, dict) and set(case) == {'id', 'messages'}
                       and isinstance(case['id'], str) and bool(case['id']), 'Invalid pinned case')
        messages = case['messages']
        worker.require(isinstance(messages, list) and bool(messages), 'Nonempty case messages required')
        for message in messages:
            worker.require(isinstance(message, dict) and set(message) == {'role', 'content'}
                           and message['role'] in ('system', 'user', 'assistant')
                           and isinstance(message['content'], str) and bool(message['content'].strip()),
                           'Invalid pinned text message')
        worker.require(messages[-1]['role'] == 'user', 'Generation must follow user message')
    worker.require(len({case['id'] for case in cases}) == len(cases), 'Duplicate case IDs')
    output = Path(plan['output_dir'])
    worker.require(output.is_absolute(), 'Absolute output directory required')
    worker.require(not output.resolve().is_relative_to(Path(plan['model']['path']).resolve()),
                   'Output directory must not modify pinned model')
    if plan['adapter'] is not None:
        worker.require(not output.resolve().is_relative_to(Path(plan['adapter']['path']).resolve()),
                       'Output directory must not modify pinned adapter')
    pins = plan['source_pins']
    worker.require(isinstance(pins, list) and bool(pins), 'Source pins required')
    paths = [worker.verify_pin(pin).resolve() for pin in pins]
    worker.require(len(paths) == len(set(paths)), 'Duplicate source pins')
    expected = {Path(__file__).resolve().parent / name for name in SOURCE_NAMES}
    worker.require(expected <= set(paths), 'Inference runtime source pins incomplete')
    return config, cases


def direct_native_generation(model, prompt_ids, stops, profile, check, generated):
    """Independent native greedy reference, not stream_generate parity."""
    import mlx.core as mx
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(profile['seed'])
    iterator = generate_step(mx.array(prompt_ids, dtype=mx.int32), model,
                             sampler=make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0.0),
                             prompt_cache=make_prompt_cache(model), max_tokens=profile['max_new_tokens'],
                             prefill_step_size=profile['prefill_step_size'],
                             prompt_progress_callback=lambda n, total: check('prefill', n=n, total=total))
    try:
        for token, _logprobs in iterator:
            generated.append(int(token))
            check('token', generated_tokens=len(generated))
            if token in stops:
                break
    finally:
        try:
            iterator.close()
        finally:
            mx.synchronize()


def run_case(runtime, model, tokenizer, case, profile, stops, stream, *, reference=False):
    label = f"{profile['name']}:{case['id']}" + (':native-reference' if reference else '')
    started = time.monotonic()
    deadline = started + runtime.plan['limits']['question_seconds']
    generated = []
    runtime.partial = dict(id=case['id'], profile=profile['name'], output_token_ids=generated)
    runtime.emit('case_started', case_id=label, question_started_monotonic=started,
                 question_deadline_monotonic=deadline)
    with worker.Deadline(max(0, deadline - time.monotonic()), lambda: runtime.deadline('question_deadline')):
        text, ids = inference.render_prompt(tokenizer, case['messages'], profile)
        runtime.partial.update(prompt_token_ids=ids, rendered_prompt=text)
        callback = lambda event, **fields: runtime.check(event, case_id=label, **fields)
        generate = direct_native_generation if reference else inference.collect_generation
        generate(model, ids, stops, profile, callback, generated)
        decoded = inference.decode_generation(tokenizer, generated, stops, profile['max_new_tokens'], profile['thinking'])
        runtime.check('generation_finished', case_id=label)
        record = dict(id=case['id'], profile=profile['name'], profile_parameters=profile,
                      native_reference=reference, penalty_scope='generated_only',
                      prompt_sha256=hashlib.sha256(worker.canonical(case['messages'])).hexdigest(),
                      rendered_prompt_sha256=hashlib.sha256(text.encode()).hexdigest(), rendered_prompt=text,
                      prompt_token_ids=ids, output_token_ids=generated, **decoded,
                      elapsed_seconds=time.monotonic() - started, memory=dict(runtime.memory))
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
        runtime.check('case_finished', case_id=label, question_finished_monotonic=time.monotonic(),
                      termination_reason=decoded['termination_reason'])
        runtime.receipt['questions'].append(dict(id=case['id'], profile=profile['name'],
                                                 native_reference=reference,
                                                 elapsed_seconds=record['elapsed_seconds'], **runtime.memory))
    runtime.partial = None
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args(argv)
    worker.startup_gate()
    inference.configure_offline()
    worker.require(args.plan.is_absolute(), '--plan must be absolute')
    raw = args.plan.read_bytes()
    plan = json.loads(raw)
    runtime = None
    try:
        config, cases = validate_plan(plan)
        _, sources = worker.source_provenance()
        distribution = importlib.metadata.distribution('mlx-lm')
        extra = [Path(__file__).resolve(), Path(__file__).with_name('mlx_grounded_guard.py')]
        extra += [distribution.locate_file('mlx_lm/' + name)
                  for name in ('sample_utils.py', 'models/cache.py', 'models/qwen3_5.py')]
        sources += [dict(path=str(Path(path).resolve()), sha256=worker.sha256(path)) for path in extra]
        runtime = worker.Runtime(plan, 'inference', hashlib.sha256(raw).hexdigest(),
                                 receipt_schema=RECEIPT_SCHEMA, source_files=sources)
        runtime.receipt.update(worker_source_sha256=worker.sha256(__file__), profiles=plan['profiles'],
                               penalty_scope='generated_only', model=plan['model'], adapter=plan['adapter'],
                               cases=plan['cases'], source_pins=plan['source_pins'], questions=[], parity=[])
        with worker.Deadline(plan['limits']['inference_seconds'], lambda: runtime.deadline('phase_deadline')):
            import mlx.core as mx
            runtime.mx = mx
            mx.set_cache_limit(0)
            mx.set_memory_limit(plan['limits']['mlx_stop_bytes'])
            mx.set_wired_limit(plan['limits']['mlx_stop_bytes'])
            runtime.check('mlx_ready')
            model, tokenizer, loaded_config = inference.load_local(
                plan['model']['path'], None if plan['adapter'] is None else plan['adapter']['path'])
            runtime.check('model_loaded')
            stops = worker.eos_ids(tokenizer, loaded_config.get('text_config', loaded_config))
            # A top-level generation_config override must not be lost for nested architectures.
            if 'eos_token_id' in loaded_config:
                stops = sorted(set(stops + worker.eos_ids(tokenizer, loaded_config)))
            path = runtime.output / 'inference-outputs.jsonl'
            with path.open('x', encoding='utf-8') as stream:
                for profile in plan['profiles']:
                    for case in cases:
                        record = run_case(runtime, model, tokenizer, case, profile, stops, stream)
                        if plan.get('parity') and profile == inference.historical_profile():
                            reference = run_case(runtime, model, tokenizer, case, profile, stops, stream, reference=True)
                            equal = record['output_token_ids'] == reference['output_token_ids']
                            runtime.receipt['parity'].append(dict(id=case['id'], exact_tokens=equal,
                                                                  reference='native_generate_step_make_sampler'))
                            worker.require(equal, 'Historical/native generation token parity mismatch')
            runtime.receipt.update(artifacts=[worker.file_pin(path)], eos_token_ids=stops,
                                   case_ids=[case['id'] for case in cases], actual_context_tokens=inference.model_context(model))
            runtime.check('phase_complete')
            runtime.finish('complete')
        return 0
    except Exception as exc:
        if runtime is not None:
            runtime.finish('failed', stop_reason='mlx_memory_limit' if isinstance(exc, MemoryError) else 'error',
                           error=f'{type(exc).__name__}: {exc}')
        else:
            print(json.dumps(dict(phase='inference', event='failure', stop_reason='validation_error',
                                  error=f'{type(exc).__name__}: {exc}')), flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
