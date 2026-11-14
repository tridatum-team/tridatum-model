"""Offline paired BF16/MPS diagnostic for the nominated step-24 adapter, not selection.

Only a public JSON array of {id, messages} is accepted; reference/grade files are
never loaded. Both arms use one base+adapter instance, native nonthinking prompts,
and the unchanged cloud worker decoder. No CUDA equivalence is claimed. A smoke
run requires explicit IDs and its own fresh output directory. Full evaluation
uses every input in file order. Neither mode resumes or retries an answer.

Each attempt is fsynced before generation and each completed answer immediately
afterwards. An interruption leaves the immutable run identity, preflight, attempt
journal and completed JSONL rows intact; a failed answer stops the run. Use a new
output path for any deliberately authorized subsequent execution. Acquisition is
external: this command never downloads a model or installs a dependency.

--decoder-profile json-schema is a separately attributed format-control
diagnostic, never merged with the default unconstrained results. It requires
lm-format-enforcer==0.11.3 and only accepts the fixed public MOIS paragraph
contract; arbitrary JSON Schema features are rejected, not silently ignored.
It does not establish semantic truth, evidence entailment, or model selection.
The library's error-to-EOS fallback is refused. No post-generation repair occurs.
"""
import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
from grounded_hf_worker import decode_generation

MODEL_ID = 'Qwen/Qwen3-8B'
REVISION = 'b968826d9c46dd6066d109eabc6255188de91218'
ADAPTER_SHA = '8a400205116b6f13dcaa48d315c95c74f6e30104bc1db96629a93e21a37b1294'
EXPORT = ROOT / 'runs/mois-eight-sft-20261114T071851Z/eight-execution-state/eight-export'
GENERATION = dict(enable_thinking=False, do_sample=False, num_beams=1, seed=14,
                  max_new_tokens=768, context_tokens=4096, dtype='bfloat16',
                  backend='transformers-mps')
ARMS = ('base', 'step-000024')
METADATA_SHA = 'e0dea2b77e7d9108d58786af3d9210c6ba0600ee29733c144a8e7efd1ff55300'
LMFE_VERSION = '0.11.3'
GREEDY_DISPATCH = dict(use_model_defaults=False, do_sample=False, num_beams=1)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def object_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
            'Expected an ordinary, unshared local file: ' + str(path))
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=digest(path))


def save_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def append_row(stream, row):
    stream.write(canonical(row).decode() + '\n')
    stream.flush()
    os.fsync(stream.fileno())


def public_inputs(path, expected_sha):
    identity = pin(path)
    require(identity['sha256'] == expected_sha, 'Public input hash differs')
    rows = load_json(path)
    require(isinstance(rows, list) and rows, 'Expected nonempty public input array')
    ids = []
    for row in rows:
        require(isinstance(row, dict) and set(row) == {'id', 'messages'},
                'Only public id/messages fields are allowed; no reference fields')
        require(isinstance(row['id'], str) and row['id'], 'Input ID must be nonempty text')
        messages = row['messages']
        require(isinstance(messages, list) and messages, 'Messages must be nonempty')
        require(all(isinstance(m, dict) and set(m) == {'role', 'content'}
                    and m['role'] in ('system', 'user', 'assistant')
                    and isinstance(m['content'], str) for m in messages)
                and messages[-1]['role'] == 'user', 'Invalid public message contract')
        ids.append(row['id'])
    require(len(set(ids)) == len(ids), 'Duplicate public input IDs')
    return rows, identity


def select_rows(rows, mode, smoke_ids):
    require(mode in ('smoke', 'evaluation'), 'Unknown mode')
    if mode == 'evaluation':
        require(not smoke_ids, 'Full evaluation cannot select a subset')
        return rows
    require(smoke_ids and len(smoke_ids) == len(set(smoke_ids)),
            'Smoke requires explicit unique --smoke-id values')
    require(set(smoke_ids) <= {r['id'] for r in rows}, 'Unknown smoke ID')
    require(len(smoke_ids) < len(rows), 'Smoke must be a strict subset of the full inputs')
    return [r for r in rows if r['id'] in smoke_ids]


def preflight_prompts(tokenizer, rows):
    prepared = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(row['messages'], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
        require(isinstance(prompt, str) and prompt.endswith('<think>\n\n</think>\n\n'),
                'Native Qwen3 nonthinking prefix differs')
        encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
        ids = encoded['input_ids']
        require(isinstance(ids, list) and ids and all(type(t) is int and t >= 0 for t in ids),
                'Invalid tokenizer token IDs')
        require(len(ids) + GENERATION['max_new_tokens'] <= GENERATION['context_tokens'],
                f"{row['id']}: input exceeds matched context envelope; no truncation")
        prepared.append(dict(id=row['id'], messages=row['messages'],
                             prompt_sha256=object_sha(row['messages']),
                             rendered_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                             prompt_token_ids=ids))
    return prepared


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('Non-JSON numeric constant: ' + value)

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def public_answer_schema():
    return {
        'type': 'object', 'required': ['paragraphs'], 'additionalProperties': False,
        'properties': {'paragraphs': {
            'type': 'array', 'minItems': 1, 'items': {
                'type': 'object', 'required': ['text', 'sources'], 'additionalProperties': False,
                'properties': {
                    'text': {'type': 'string', 'minLength': 1},
                    'sources': {'type': 'array', 'minItems': 1, 'uniqueItems': True,
                                'items': {'type': 'string'}}}}}}}


def mois_schema(case):
    payload = strict_json(case['messages'][-1]['content'])
    require(isinstance(payload, dict) and isinstance(payload.get('question'), str)
            and payload['question'].strip(), 'JSON profile requires a public MOIS question')
    if set(payload) == {'question', 'source_data'}:
        packet = payload['source_data']
        require(isinstance(packet, dict) and set(packet) == {'documents', 'evidence', 'scope', 'warnings'}
                and isinstance(packet['documents'], list) and isinstance(packet['warnings'], list),
                'Unsupported source_data packet')
        sources = packet['evidence']
        required_fields = {'id', 'doc_id'}
    elif set(payload) == {'question', 'sources', 'answer_schema'}:
        require(object_sha(payload['answer_schema']) == object_sha(public_answer_schema()),
                'Unsupported answer_schema: only the fixed public paragraph schema is supported')
        sources = payload['sources']
        required_fields = {'id', 'title', 'version_date', 'text'}
    else:
        raise ValueError('JSON profile only supports source_data or sources+answer_schema MOIS inputs')
    require(isinstance(sources, list) and sources, 'No public citation sources')
    require(all(isinstance(s, dict) and required_fields <= set(s)
                and all(isinstance(s[k], str) for k in required_fields)
                and s['id'] and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in s['id'])
                for s in sources), 'Malformed public citation sources')
    ids = [s['id'] for s in sources]
    require(len(set(ids)) == len(ids), 'Duplicate public citation IDs')
    schema = public_answer_schema()
    paragraphs = schema['properties']['paragraphs']
    paragraphs['maxItems'] = 5
    properties = paragraphs['items']['properties']
    properties['text']['maxLength'] = 1200
    properties['sources']['maxItems'] = len(ids)
    properties['sources']['items']['enum'] = ids
    return schema


def validate_schema_finish(raw, schema):
    """EOS admission only; no mutation, JSON repair, or semantic/grounding verdict."""
    require(isinstance(raw, str) and len(raw) <= 12000 and '\ufffd' not in raw
            and not any(0xD800 <= ord(c) <= 0xDFFF for c in raw), 'Invalid complete JSON Unicode/length')
    value = strict_json(raw)
    require(isinstance(value, dict) and set(value) == {'paragraphs'}, 'Invalid root shape')
    paragraphs = value['paragraphs']
    require(isinstance(paragraphs, list) and 1 <= len(paragraphs) <= 5, 'Invalid paragraph count')
    allowed = schema['properties']['paragraphs']['items']['properties']['sources']['items']['enum']
    total = 0
    for paragraph in paragraphs:
        require(isinstance(paragraph, dict) and set(paragraph) == {'text', 'sources'},
                'Invalid paragraph shape')
        text, sources = paragraph['text'], paragraph['sources']
        require(isinstance(text, str) and text.strip() and len(text) <= 1200
                and not any(ord(c) < 32 or 0xD800 <= ord(c) <= 0xDFFF for c in text),
                'Invalid paragraph text')
        require(isinstance(sources, list) and sources
                and all(isinstance(s, str) and s in allowed for s in sources)
                and len(sources) == len(set(sources)), 'Invalid or repeated citation IDs')
        total += len(text)
    require(total <= 4000, 'Total paragraph text exceeds fixed production limit')


class SchemaDecoder:
    """Pinned library prefix API plus fail-closed EOS admission, never a grammar repair."""

    def __init__(self, tokenizer, rows):
        self.schemas = {row['id']: mois_schema(row) for row in rows}
        require(not any(k.startswith('LMFE_') for k in os.environ),
                'LMFE environment overrides are forbidden for the registered profile')
        require(importlib.metadata.version('lm-format-enforcer') == LMFE_VERSION,
                'JSON profile requires lm-format-enforcer==' + LMFE_VERSION)
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        self.tokenizer = tokenizer
        self.data = build_token_enforcer_tokenizer_data(tokenizer)
        distribution = importlib.metadata.distribution('lm-format-enforcer')
        sources = {str(p): digest(distribution.locate_file(p)) for p in distribution.files
                   if str(p).startswith('lmformatenforcer/') and str(p).endswith('.py')}
        self.identity = dict(profile='json-schema', package='lm-format-enforcer', version=LMFE_VERSION,
            package_source_sha256=sources,
            dependencies={name: importlib.metadata.version(name) for name in
                          ('pydantic', 'interegular', 'packaging', 'PyYAML')},
            integration='build_transformers_prefix_allowed_tokens_fn',
            schemas=self.schemas, schema_sha256={k: object_sha(v) for k, v in self.schemas.items()},
            unicode_policy='native tokenizer decode; reject replacement/surrogate corruption at EOS',
            parser_config=dict(alphabet_sha256=hashlib.sha256(self.data.tokenizer_alphabet.encode()).hexdigest(),
                               max_consecutive_whitespaces=sys.maxsize,
                               max_json_array_length=sys.maxsize, force_json_field_order=False),
            eos_only_checks=['uniqueItems (not implemented by LMFE 0.11.3)',
                             'duplicate keys', 'complete JSON', 'citation membership',
                             'nonblank/control-free text', 'total text length'],
            eos_violation_policy='fatal; never substitute EOS, repair, or retry',
            semantic_truth_guarantee=False, production_structural_validator='unchanged')
        self.preflight(rows)

    def prefix(self, case, eos_token_id):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
        from local_eight_schema import build_transformers_prefix_allowed_tokens_fn
        stops = [eos_token_id] if type(eos_token_id) is int else eos_token_id
        require(isinstance(stops, list) and stops and all(type(t) is int for t in stops),
                'JSON profile requires explicit generation EOS IDs')
        # The upstream builder otherwise uses tokenizer.eos_token_id, which can
        # omit a model generation-config stop. Preserve the exact registered set.
        self.data.eos_token_id = list(stops)
        schema = self.schemas[case['id']]
        config = CharacterLevelParserConfig(alphabet=self.data.tokenizer_alphabet,
                    max_consecutive_whitespaces=sys.maxsize, max_json_array_length=sys.maxsize,
                    force_json_field_order=False)
        parser = JsonSchemaParser(schema, config)
        library_prefix = build_transformers_prefix_allowed_tokens_fn(self.data, parser)
        # TokenEnforcer overwrites config during construction. Restore the public
        # config setter explicitly; do not inherit environment/default array caps.
        parser.config = config
        width = len(case['prompt_token_ids'])

        def allowed(batch_id, sent):
            ids = sent.tolist()
            require(batch_id == 0 and ids[:width] == case['prompt_token_ids'],
                    'JSON profile prompt/batch binding differs')
            result = library_prefix(batch_id, sent)
            state = library_prefix.token_enforcer.prefix_states[tuple(ids)]
            require(isinstance(state.parser, JsonSchemaParser),
                    'LMFE parser entered fallback/unsupported state; refusing forced EOS')
            if any(stop in result for stop in stops):
                require(state.parser.can_end(), 'LMFE returned premature fallback EOS')
                raw = self.tokenizer.decode(ids[width:], skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
                validate_schema_finish(raw, schema)
            return result

        return allowed

    def preflight(self, rows):
        """CPU tokenizer-path admission; no model execution or generated answer."""
        import torch
        probe = '한국어 조건과 예외를 확인합니다. 한글 똠뷁 뷁'
        self.identity['tokenizer_probe_text'] = probe
        checked = []
        for row in rows:
            ids = self.schemas[row['id']]['properties']['paragraphs']['items']['properties']['sources']['items']['enum']
            # Each ID is exercised independently; large evidence packets do not
            # manufacture an overlength multi-citation answer for this admission.
            for source_id in ids:
                prepared = dict(row, prompt_token_ids=[self.tokenizer.bos_token_id or self.tokenizer.eos_token_id])
                prefix = self.prefix(prepared, self.tokenizer.eos_token_id)
                raw = json.dumps({'paragraphs': [{'text': probe, 'sources': [source_id]}]},
                                 ensure_ascii=False, separators=(',', ':'))
                path = self.tokenizer.encode(raw, add_special_tokens=False)
                history = prepared['prompt_token_ids'].copy()
                for offset, token in enumerate(path):
                    permitted = prefix(0, torch.tensor(history, dtype=torch.long))
                    require(token in permitted,
                            f"LMFE/Qwen Unicode token path unsupported: {row['id']} citation {source_id!r}, "
                            f"token offset {offset}, token ID {token}; no fallback")
                    history.append(token)
                require(self.tokenizer.eos_token_id in prefix(0, torch.tensor(history, dtype=torch.long)),
                        'Valid schema probe cannot reach EOS')
            checked.append(dict(id=row['id'], citation_paths=len(ids)))
        self.identity['tokenizer_path_preflight'] = checked


def model_identity(directory, metadata_path):
    directory = Path(directory)
    metadata_pin = pin(metadata_path)
    require(metadata_pin['sha256'] == METADATA_SHA, 'Official acquisition manifest hash differs')
    metadata = load_json(metadata_path)
    require(metadata['schema'] == 'trid-immutable-model-metadata-v1'
            and metadata['repo_id'] == MODEL_ID and metadata['revision'] == REVISION
            and metadata['weight_shards'] == 5, 'Official base identity differs')
    expected = {entry['path']: entry for entry in metadata['files']}
    require(len(expected) == len(metadata['files'])
            and all(PurePosixPath(name).name == name and name not in ('.', '..') for name in expected),
            'Unsafe or duplicate model file names')
    require(directory.is_dir() and not directory.is_symlink()
            and {p.name for p in directory.iterdir()} == set(expected),
            'Model directory must contain exactly the eleven official pinned files')
    files = {}
    for name, entry in expected.items():
        actual = pin(directory / name)
        require(actual['bytes'] == entry['bytes'] and actual['sha256'] == entry['sha256'],
                'Model file body authentication failed: ' + name)
        files[name] = actual
    shards = {name for name in expected if name.endswith('.safetensors')}
    index = load_json(directory / 'model.safetensors.index.json')
    require(len(shards) == 5 and set(index['weight_map'].values()) == shards,
            'Official five-shard index differs')
    return dict(id=MODEL_ID, revision=REVISION, directory=str(directory.resolve()),
                acquisition_manifest=metadata_pin, files=files,
                body_hashes_authenticated=True, representation='hf-safetensors-bfloat16')


def adapter_identity(export):
    export = Path(export)
    ledger_path = export / 'training/checkpoints.json'
    ledger = load_json(ledger_path)
    require(ledger['base_model_id'] == MODEL_ID and ledger['base_revision'] == REVISION
            and ledger['base_weights'] == {'representation': 'hf-safetensors', 'dtype': 'torch.bfloat16'}
            and ledger['init_adapter'] is None, 'Checkpoint base lineage differs')
    candidates = [r for r in ledger['checkpoints'] if r['step'] == 24]
    require(len(candidates) == 1, 'Expected exactly one step-24 checkpoint')
    entry = candidates[0]
    require(PurePosixPath(entry['path']).parts[-4:] ==
            ('training', 'checkpoints', 'step-000024', 'adapter'), 'Checkpoint path differs')
    adapter = export / 'training/checkpoints/step-000024/adapter'
    files = {p.name: pin(p) for p in adapter.iterdir()}
    require(set(files) == {'adapter_config.json', 'adapter_model.safetensors', 'README.md'},
            'Unexpected checkpoint adapter membership')
    require(files['adapter_model.safetensors']['sha256'] == entry['adapter_weights_sha256'] == ADAPTER_SHA,
            'Nominated step-24 weights differ')
    config = load_json(adapter / 'adapter_config.json')
    require(config['r'] == 8 and config['lora_alpha'] == 16 and config['lora_dropout'] == 0
            and config['peft_type'] == 'LORA' and config['task_type'] == 'CAUSAL_LM'
            and config['bias'] == 'none' and config['modules_to_save'] is None,
            'Adapter must permit an unchanged disable_adapter base arm')
    return adapter, dict(status='nominated_diagnostic_comparator_not_selected', step=24,
                         files=files, checkpoint_ledger=pin(ledger_path), config=config)


def code_identity():
    paths = {Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT / 'scripts'):
                paths.add(path)
            elif path.suffix == '.py' and path.is_relative_to(ROOT / 'trid'):
                paths.add(path)
    return {str(p.relative_to(ROOT)): pin(p) for p in sorted(paths)}


def mps_memory(torch):
    # Values are real runtime observations. Missing APIs are null, never inferred.
    metrics = {}
    for name in ('current_allocated_memory', 'driver_allocated_memory',
                 'recommended_max_memory', 'max_memory_allocated', 'max_memory_reserved'):
        function = getattr(torch.mps, name, None)
        metrics[name] = int(function()) if callable(function) else None
    return dict(bytes=metrics, unsupported=[k for k, v in metrics.items() if v is None],
                peak_scope='runtime lifetime; only max_* APIs, when available, are peaks')


def generation_target(model):
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def assert_greedy_config(config):
    require(config.do_sample is False and config.num_beams == 1
            and config.max_new_tokens == GENERATION['max_new_tokens']
            and config.return_dict_in_generate is True
            and config.get_generation_mode().value == 'greedy_search',
            'Effective generation policy is not the registered greedy policy')


def resolve_generation_policy(model, requested):
    target = generation_target(model)
    effective, unused = target._prepare_generation_config(requested, **GREEDY_DISPATCH)
    require(not unused, 'Unexpected model kwargs during generation-policy resolution')
    assert_greedy_config(effective)
    implementation = Path(inspect.getsourcefile(target._prepare_generation_config))
    policy = dict(dispatch_kwargs=dict(GREEDY_DISPATCH),
                  generation_mode=effective.get_generation_mode().value,
                  effective_config_sha256=object_sha(effective.to_dict()),
                  preparation_implementation=dict(path=str(implementation.resolve()),
                      bytes=implementation.stat().st_size, sha256=digest(implementation)))
    return effective, policy


@contextlib.contextmanager
def observe_generation_policy(model, expected_config):
    """Assert the actual HF prepare call before its resolved mode is dispatched.

    A scoped instance hook observes the underlying Qwen model used by PEFT;
    it never changes the returned config, logits, RNG, or generation algorithm.
    Restoration also occurs after an exception. Generation is single-threaded.
    """
    target = generation_target(model)
    original = target._prepare_generation_config
    sentinel = object()
    previous = target.__dict__.get('_prepare_generation_config', sentinel)
    observed = {}
    expected_sha = object_sha(expected_config.to_dict())

    def observe(requested, use_model_defaults=None, **kwargs):
        require(use_model_defaults is False and kwargs.get('do_sample') is False
                and kwargs.get('num_beams') == 1, 'Explicit greedy dispatch controls missing')
        effective, model_kwargs = original(requested, use_model_defaults=use_model_defaults, **kwargs)
        assert_greedy_config(effective)
        require(object_sha(effective.to_dict()) == expected_sha,
                'Actual effective generation configuration differs from bound runtime receipt')
        observed['generation_mode'] = effective.get_generation_mode().value
        observed['effective_config_sha256'] = expected_sha
        return effective, model_kwargs

    target._prepare_generation_config = observe
    try:
        yield observed
        require(observed, 'Model generate bypassed the observed generation-policy resolver')
    finally:
        if previous is sentinel:
            delattr(target, '_prepare_generation_config')
        else:
            target._prepare_generation_config = previous


def answer_record(case, arm, binding, config, tokens, tokenizer, elapsed, memory):
    decoded = decode_generation(tokenizer, tokens, config['eos_token_id'], GENERATION['max_new_tokens'])
    return dict(**case, arm=arm, **binding, generation=dict(GENERATION),
                adapter_weights_sha256=None if arm == 'base' else ADAPTER_SHA,
                generation_config=config, output_token_ids=tokens, **decoded,
                exception=None, context_truncated=False, skipped_context=False,
                truncated_output=decoded['termination_reason'] == 'length',
                elapsed_seconds=elapsed, mps_memory=memory)


def run_pairs(torch, model, tokenizer, prepared, config, binding, out, decoder=None):
    with contextlib.ExitStack() as stack:
        streams = {arm: stack.enter_context((out / f'{arm}-outputs.jsonl').open('x', encoding='utf-8'))
                   for arm in ARMS}
        attempts = stack.enter_context((out / 'attempts.jsonl').open('x', encoding='utf-8'))
        for case in prepared:
            for arm in ARMS:
                append_row(attempts, dict(id=case['id'], arm=arm, state='started',
                    timestamp=dt.datetime.now(dt.timezone.utc).isoformat()))
                started = time.perf_counter()
                tokens = []
                try:
                    generation_kwargs = dict(GREEDY_DISPATCH)
                    if decoder is not None:
                        generation_kwargs['prefix_allowed_tokens_fn'] = decoder.prefix(
                            case, config.eos_token_id)
                    torch.manual_seed(GENERATION['seed'])
                    torch.mps.manual_seed(GENERATION['seed'])
                    ids = torch.tensor([case['prompt_token_ids']], dtype=torch.long, device='mps')
                    attention = torch.ones_like(ids)
                    torch.mps.synchronize()
                    with (model.disable_adapter() if arm == 'base' else contextlib.nullcontext()):
                        with torch.inference_mode(), observe_generation_policy(model, config) as observed:
                            result = model.generate(input_ids=ids, attention_mask=attention,
                                                    generation_config=config, **generation_kwargs)
                    torch.mps.synchronize()
                    tokens = result.sequences[0, len(case['prompt_token_ids']):].tolist()
                    elapsed = time.perf_counter() - started
                    row = answer_record(case, arm, binding, config.to_dict(), tokens,
                                        tokenizer, elapsed, mps_memory(torch))
                    row['effective_generation_mode'] = observed['generation_mode']
                    append_row(streams[arm], row)
                    print(json.dumps(dict(event='answer_complete', id=case['id'], arm=arm,
                        generated_tokens=len(tokens), termination_reason=row['termination_reason'],
                        elapsed_seconds=elapsed)), flush=True)
                    del result, ids, attention
                except BaseException as error:
                    append_row(streams[arm], dict(**case, arm=arm, **binding,
                        adapter_weights_sha256=None if arm == 'base' else ADAPTER_SHA,
                        generation=dict(GENERATION), generation_config=config.to_dict(),
                        output_token_ids=tokens, generated_tokens=len(tokens), raw_output=None,
                        response_text=None, termination_reason='error', terminal_stop_token_id=None,
                        context_truncated=False, skipped_context=False, truncated_output=False,
                        elapsed_seconds=time.perf_counter() - started,
                        exception=dict(type=type(error).__name__, message=str(error))))
                    raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-directory', required=True, type=Path)
    p.add_argument('--model-metadata', required=True, type=Path)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--inputs-sha256', required=True)
    p.add_argument('--export', type=Path, default=EXPORT)
    p.add_argument('--mode', required=True, choices=('smoke', 'evaluation'))
    p.add_argument('--smoke-id', action='append', default=[])
    p.add_argument('--decoder-profile', choices=('unconstrained', 'json-schema'),
                   default='unconstrained')
    p.add_argument('--out', required=True, type=Path)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    # A path is claimed once, before any expensive work. Existing paths are refused.
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0',
                'MPS CPU fallback is forbidden')
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
            os.environ[key] = '1'
        rows, input_pin = public_inputs(args.inputs, args.inputs_sha256)
        selected = select_rows(rows, args.mode, args.smoke_id)
        model_pin = model_identity(args.model_directory, args.model_metadata)
        adapter, adapter_pin = adapter_identity(args.export)
        import torch
        from transformers import AutoTokenizer, GenerationConfig
        from peft import PeftModel
        from train_open_weight_lora import load_qwen3_pread
        require(torch.backends.mps.is_available(), 'MPS is required; no backend fallback')
        require(platform.system() == 'Darwin', 'This diagnostic requires native macOS MPS')
        tokenizer = AutoTokenizer.from_pretrained(args.model_directory, local_files_only=True,
                                                  trust_remote_code=False)
        prepared = preflight_prompts(tokenizer, selected)
        decoder = None
        decoder_identity = dict(profile=args.decoder_profile, semantic_truth_guarantee=False)
        if args.decoder_profile == 'json-schema':
            decoder = SchemaDecoder(tokenizer, selected)
            decoder_identity = decoder.identity
        save_new(args.out / 'decoder-profile.json', decoder_identity)
        runtime = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
            machine=platform.machine(), device='mps', dtype='torch.bfloat16',
            packages={name: importlib.metadata.version(name) for name in
                      ('torch', 'transformers', 'peft', 'accelerate', 'safetensors', 'tokenizers', 'numpy')},
            torch_threads=torch.get_num_threads(), cpu_fallback=False,
            mps_environment={k: v for k, v in sorted(os.environ.items()) if k.startswith('PYTORCH_MPS_')},
            cuda_equivalence_claim=False)
        identity = dict(schema='trid-local-eight-paired-v1', mode=args.mode,
            model=model_pin, adapter=adapter_pin, input=input_pin,
            selected_ids=[r['id'] for r in selected], generation=GENERATION,
            decoder_profile=decoder_identity,
            runtime=runtime, code=code_identity(), arm_order=list(ARMS),
            paired_order='each input: base with adapter disabled, then step24 enabled',
            grading_access=False, retries=False, resume=False)
        save_new(args.out / 'run-manifest.json', identity)
        save_new(args.out / 'prompt-preflight.json', dict(status='passed', prompts=prepared,
            all_selected_preflighted_before_weights=True, truncation=False))
        print(json.dumps(dict(event='preflight_complete_loading_weights', rows=len(prepared),
                              mode=args.mode, decoder_profile=args.decoder_profile)), flush=True)
        load_started = time.perf_counter()
        base, loader_report = load_qwen3_pread(args.model_directory, torch.bfloat16, 'mps')
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False,
                                         local_files_only=True, autocast_adapter_dtype=False)
        model.eval()
        require(all(p.device.type == 'mps' and p.dtype == torch.bfloat16 for p in model.parameters()),
                'Loaded parameter device/dtype differs; no casting fallback')
        require(not any(p.requires_grad for p in model.parameters()), 'Inference parameters must be frozen')
        config = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=768,
            use_cache=True, eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, return_dict_in_generate=True)
        requested_config = config.to_dict()
        config, policy = resolve_generation_policy(model, config)
        torch.mps.synchronize()
        loaded = dict(loader=loader_report, elapsed_seconds=time.perf_counter() - load_started,
            attention_implementation=model.config._attn_implementation,
            generation_config=config.to_dict(), mps_memory=mps_memory(torch),
            requested_generation_config=requested_config, generation_policy=policy,
            parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
            buffer_dtypes=sorted({str(b.dtype) for b in model.buffers()}),
            code=code_identity())
        save_new(args.out / 'loaded-runtime.json', loaded)
        print(json.dumps(dict(event='weights_loaded', elapsed_seconds=loaded['elapsed_seconds'],
                              mps_memory=loaded['mps_memory'])), flush=True)
        binding = dict(run_manifest_sha256=digest(args.out / 'run-manifest.json'),
            loaded_runtime_sha256=digest(args.out / 'loaded-runtime.json'), model_id=MODEL_ID,
            model_revision=REVISION, input_sha256=input_pin['sha256'],
            nominated_adapter_weights_sha256=ADAPTER_SHA, mode=args.mode)
        binding['decoder_profile'] = args.decoder_profile
        binding['decoder_profile_sha256'] = digest(args.out / 'decoder-profile.json')
        binding['generation_policy_sha256'] = object_sha(policy)
        run_pairs(torch, model, tokenizer, prepared, config, binding, args.out, decoder=decoder)
        save_new(args.out / 'completion.json', dict(status='complete', mode=args.mode,
            elapsed_seconds=time.perf_counter() - started, rows_per_arm=len(prepared),
            outputs={arm: pin(args.out / f'{arm}-outputs.jsonl') for arm in ARMS},
            mps_memory=mps_memory(torch), run_manifest_sha256=binding['run_manifest_sha256']))
        print(json.dumps(dict(event='complete', rows_per_arm=len(prepared),
                              elapsed_seconds=time.perf_counter() - started)), flush=True)
    except BaseException as error:
        save_new(args.out / 'failure.json', dict(status='failed', type=type(error).__name__,
            message=str(error), elapsed_seconds=time.perf_counter() - started,
            completed_rows_retained=True, automatic_retry=False))
        print(json.dumps(dict(event='failed', type=type(error).__name__, message=str(error))), flush=True)
        raise


if __name__ == '__main__':
    main()
