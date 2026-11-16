"""Offline paired BF16 development diagnostics, never deployment selection.

Only a public JSON array of {id, messages} is accepted; reference/grade files are
never loaded. Both arms use one base+adapter instance, native nonthinking prompts,
and the unchanged cloud worker decoder. No CUDA equivalence is claimed. A smoke
run requires explicit IDs and its own fresh output directory. Full evaluation
uses every input in file order. Neither mode resumes or retries an answer.

With neither candidate option, preserve the authenticated historical step24
replay. Paired --candidate-manifest/--candidate-sha256 instead authenticate the
prospectively fixed final step150 and accepted parent supervision. Candidate
mode uses base/candidate arms, forbids export overrides and secondary decoder
search, and only hashes the opaque review-protocol bytes (never parses them).

The local CLI is MPS-only. Fresh CUDA training-only candidates are accepted only
after full capacity/training/export and independent instance/root-volume cleanup.
Their training runtime/source seal stays distinct from local inference receipts.
Local evaluation loads the same authenticated official BF16 base and FP32 LoRA,
then produces all400 primary answers; training success is never quality acceptance.
Legacy local candidates and historical step24 keep their existing BF16 inference.

CUDA-trained candidate execution also requires --local-registration FILE and
--local-registration-sha256 SHA, issued after cloud cleanup. The immutable local
registration binds the complete evaluator command (without these registration
flags), candidate/input/model pins, exact generation, executable source closure,
worker environment, distinct fresh output/supervision directories, and fixed
LOCAL_LIMITS. The supervisor reuses local_mps_training's host probes, allocator
cap, per-user exclusion lock and owned-process-group gate/cleanup conventions.
Its 43200-second deadline is independent of the CUDA lease; no retry or fallback.
Only accepted parent cleanup plus complete paired receipts can enter grading.

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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
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


def digest(path, *, uncached=False):
    with Path(path).open('rb') as stream:
        if uncached and sys.platform == 'darwin':
            import fcntl
            fcntl.fcntl(stream.fileno(), fcntl.F_NOCACHE, 1)
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pin(path, *, uncached=False):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1,
            'Expected an ordinary, unshared local file: ' + str(path))
    return dict(path=str(path.resolve()), bytes=path.stat().st_size,
                sha256=digest(path, uncached=uncached))


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
            result = None
            try:
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
            except Exception as error:
                # generate() has not returned its sequence on callback failure.
                # Preserve the actual attempted prefix, separately from answers,
                # so a failure can be reproduced without another model call.
                state = library_prefix.token_enforcer.prefix_states.get(tuple(ids))
                active = state.parser if state is not None else None
                error.decoder_diagnostic = dict(
                    id=case['id'], generated_prefix_token_ids=ids[width:],
                    decoded_prefix=self.tokenizer.decode(ids[width:], skip_special_tokens=False,
                                                         clean_up_tokenization_spaces=False),
                    parser_type=type(active).__name__ if active is not None else None,
                    parser_can_end=active.can_end() if active is not None else None,
                    parser_stack=[type(p).__name__ for p in getattr(active, 'object_stack', [])],
                    offered_eos_ids=[stop for stop in stops if result is not None and stop in result])
                raise

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
        actual = pin(directory / name, uncached=name.endswith('.safetensors'))
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


def inference_parameter_inventory(model, torch, fp32_adapter):
    """Observe actual MPS parameters; CUDA training provenance is not a device hint."""
    inventory = {'base': {}, 'adapter': {}}
    for name, parameter in model.named_parameters():
        group = 'adapter' if '.lora_A.' in name or '.lora_B.' in name else 'base'
        expected = torch.float32 if group == 'adapter' and fp32_adapter else torch.bfloat16
        require(parameter.device.type == 'mps' and parameter.dtype == expected,
                'Loaded parameter device/dtype differs; no casting fallback')
        key = str(parameter.dtype) + '@' + str(parameter.device)
        entry = inventory[group].setdefault(key, dict(tensors=0, elements=0, bytes=0))
        entry['tensors'] += 1
        entry['elements'] += parameter.numel()
        entry['bytes'] += parameter.numel() * parameter.element_size()
    require(all(inventory.values()), 'Loaded base/adapter inventory missing')
    return inventory


def answer_record(case, arm, binding, config, tokens, tokenizer, elapsed, memory, contract=None):
    decoded = decode_generation(tokenizer, tokens, config['eos_token_id'], GENERATION['max_new_tokens'])
    return dict(**case, arm=arm, **binding, generation=dict(GENERATION),
                adapter_weights_sha256=None if arm == 'base' else (contract['adapter_sha256'] if contract else ADAPTER_SHA),
                generation_config=config, output_token_ids=tokens, **decoded,
                exception=None, context_truncated=False, skipped_context=False,
                truncated_output=decoded['termination_reason'] == 'length',
                elapsed_seconds=elapsed, mps_memory=memory)


def run_pairs(torch, model, tokenizer, prepared, config, binding, out, decoder=None, contract=None):
    arms = contract['arms'] if contract else ARMS
    device = 'mps'
    backend = torch.mps
    generation = GENERATION
    with contextlib.ExitStack() as stack:
        streams = {arm: stack.enter_context((out / f'{arm}-outputs.jsonl').open('x', encoding='utf-8'))
                   for arm in arms}
        attempts = stack.enter_context((out / 'attempts.jsonl').open('x', encoding='utf-8'))
        for case in prepared:
            for arm in arms:
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
                    backend.manual_seed(GENERATION['seed'])
                    ids = torch.tensor([case['prompt_token_ids']], dtype=torch.long, device=device)
                    attention = torch.ones_like(ids)
                    backend.synchronize()
                    with (model.disable_adapter() if arm == 'base' else contextlib.nullcontext()):
                        with torch.inference_mode(), observe_generation_policy(model, config) as observed:
                            result = model.generate(input_ids=ids, attention_mask=attention,
                                                    generation_config=config, **generation_kwargs)
                    backend.synchronize()
                    tokens = result.sequences[0, len(case['prompt_token_ids']):].tolist()
                    elapsed = time.perf_counter() - started
                    row = answer_record(case, arm, binding, config.to_dict(), tokens,
                                        tokenizer, elapsed, mps_memory(torch), contract=contract)
                    row['effective_generation_mode'] = observed['generation_mode']
                    append_row(streams[arm], row)
                    print(json.dumps(dict(event='answer_complete', id=case['id'], arm=arm,
                        generated_tokens=len(tokens), termination_reason=row['termination_reason'],
                        elapsed_seconds=elapsed)), flush=True)
                    del result, ids, attention
                except BaseException as error:
                    error_detail = dict(type=type(error).__name__, message=str(error))
                    if hasattr(error, 'decoder_diagnostic'):
                        error_detail['decoder_prefix'] = error.decoder_diagnostic
                    append_row(streams[arm], dict(**case, arm=arm, **binding,
                        adapter_weights_sha256=None if arm == 'base' else (contract['adapter_sha256'] if contract else ADAPTER_SHA),
                        generation=dict(generation), generation_config=config.to_dict(),
                        output_token_ids=tokens, generated_tokens=len(tokens), raw_output=None,
                        response_text=None, termination_reason='error', terminal_stop_token_id=None,
                        context_truncated=False, skipped_context=False, truncated_output=False,
                        elapsed_seconds=time.perf_counter() - started,
                        exception=error_detail))
                    raise


LOCAL_LIMITS = dict(evaluation_deadline_seconds=43200, pressure_required=1,
    minimum_pressure_free_percent=10, require_ac_power=False, swap_growth_limit_bytes=1024 ** 3,
    process_group_rss_limit_bytes=24 * 1024 ** 3, process_group_footprint_limit_bytes=24 * 1024 ** 3,
    mps_allocator_limit_bytes=20 * 1024 ** 3, mps_driver_limit_bytes=22 * 1024 ** 3,
    sample_seconds=1, term_grace_seconds=3, kill_cleanup_seconds=10)
LOCAL_SOURCES = {
    'scripts/evaluate_local_eight.py', 'scripts/local_eight_candidate.py', 'scripts/local_mps_training.py',
    'scripts/train_open_weight_lora.py', 'scripts/aws_eval/grounded_hf_worker.py',
    'scripts/aws_eval/grounded_hf_lifecycle.py', 'scripts/aws_eval/qwen_cuda_smoke.py',
    'scripts/aws_eval/qwen_cpu_smoke.py', 'scripts/aws_eval/fresh_eight_experiment.py',
    'scripts/aws_eval/fresh_eight_recover.py',
    'trid/prompts.py', 'trid/text.py', 'trid/case_prompts.py',
}


def evaluation_command(args):
    return [sys.executable, '-u', '-B', str(Path(__file__).resolve()),
        '--model-directory', str(args.model_directory.resolve()),
        '--model-metadata', str(args.model_metadata.resolve()),
        '--inputs', str(args.inputs.resolve()), '--inputs-sha256', args.inputs_sha256,
        '--candidate-manifest', str(args.candidate_manifest.resolve()),
        '--candidate-sha256', args.candidate_sha256, '--mode', args.mode,
        '--decoder-profile', args.decoder_profile, '--out', str(args.out.resolve())] + [
        value for identifier in args.smoke_id for value in ('--smoke-id', identifier)]


def local_registration(path, expected_sha, args=None):
    """Authenticate local execution authorization without importing model code."""
    import local_mps_training as guard
    from local_eight_candidate import bound, file_pin, load, resolve_candidate
    registration_pin = file_pin(path)
    require(registration_pin['sha256'] == expected_sha, 'Local registration hash differs')
    registration = load(path)
    require(set(registration) == {'schema', 'candidate', 'inputs', 'model_metadata', 'model_directory',
            'generation', 'sources', 'supervision', 'output', 'supervision_output', 'command', 'environment'}
            and registration['schema'] == 'trid-local-eight-evaluation-registration-v1',
            'Local evaluation registration differs')
    require(canonical(registration['supervision']) == canonical(LOCAL_LIMITS)
            and registration['generation'] == GENERATION and registration['environment'] == guard.WORKER_ENV,
            'Local registered supervision/generation/environment differs')
    for field in ('output', 'supervision_output', 'model_directory'):
        value = Path(registration[field])
        require(value.is_absolute() and str(value.resolve()) == str(value), 'Local registered path differs')
    output, supervision = Path(registration['output']), Path(registration['supervision_output'])
    require(output != supervision and not output.is_relative_to(supervision)
            and not supervision.is_relative_to(output), 'Local output/supervision directories overlap')
    candidate = bound(ROOT, registration['candidate'])
    inputs = bound(ROOT, registration['inputs'])
    metadata = bound(ROOT, registration['model_metadata'])
    contract = resolve_candidate(ROOT, candidate, registration['candidate']['sha256'])
    require(contract.get('training_runtime') == 'cuda' and contract.get('inference_runtime') == 'mps',
            'Local supervision requires accepted CUDA training-only candidate')
    policy = contract['policy_document']
    require(registration['inputs']['sha256'] == policy['data']['evaluation_inputs']['sha256']
            and registration['model_metadata']['sha256'] == policy['model']['acquisition_manifest']['sha256']
            and registration['model_directory'] == str((ROOT / policy['model']['directory']).resolve())
            and registration['generation'] == policy['evaluation']['generation'],
            'Local registered candidate/input/model policy differs')
    sources = registration['sources']
    require(isinstance(sources, dict) and LOCAL_SOURCES <= sources.keys(), 'Local source closure incomplete')
    for name, entry in sources.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts
                and relative.parts[0] in ('scripts', 'trid') and relative.suffix == '.py'
                and name == relative.as_posix() and bound(ROOT, entry) == (ROOT / relative).resolve(),
                'Local source path substitution')
    command = registration['command']
    require(isinstance(command, list) and all(isinstance(value, str) for value in command),
            'Local registered command differs')
    try:
        registered_args = parser().parse_args(command[4:])
    except SystemExit as error:
        raise ValueError('Local registered command differs') from error
    require(registered_args.candidate_manifest is not None
            and command == evaluation_command(registered_args),
            'Local registered command differs')
    if args is None:
        args = registered_args
    require(args.export is None and args.decoder_profile == 'unconstrained'
            and args.candidate_manifest is not None
            and (args.mode == 'evaluation' and not args.smoke_id
                 or args.mode == 'smoke' and len(args.smoke_id) == 1),
            'Guarded candidate requires primary200 evaluation or one explicit smoke ID')
    selected = select_rows(public_inputs(inputs, registration['inputs']['sha256'])[0],
                           args.mode, args.smoke_id)
    require(len(selected) == (1 if args.mode == 'smoke' else 200),
            'Guarded public input selection differs')
    require(command == evaluation_command(args)
            and candidate == args.candidate_manifest.resolve()
            and registration['candidate']['sha256'] == args.candidate_sha256
            and inputs == args.inputs.resolve() and registration['inputs']['sha256'] == args.inputs_sha256
            and metadata == args.model_metadata.resolve()
            and registration['model_directory'] == str(args.model_directory.resolve())
            and output == args.out.resolve(), 'Local registered command differs')
    return registration, registration_pin


def authenticate_local_sources(registration, sources):
    require(sources and all(name in registration['sources']
            and all(entry[field] == registration['sources'][name][field] for field in ('bytes', 'sha256'))
            for name, entry in sources.items()), 'Unregistered local inference source')


def supervised_evaluation(args, registration, registration_pin):
    """Use the training guard's Darwin probes, gate, lock, and owned-PGID rules."""
    import fcntl
    import signal
    import subprocess
    import local_mps_training as guard
    out = Path(registration['supervision_output'])
    out.mkdir(parents=True, exist_ok=False)
    require(not args.out.exists(), 'Local evaluation output already exists')
    result = dict(schema='trid-local-eight-evaluation-supervision-v1', status='failed',
        registration=registration_pin, startup_released=False, signals=[], samples=[],
        cleanup=dict(accepted=False, leader_reaped=False, remaining_members=None))
    guard.save(out / 'result.json', result)
    child = lock = log = None
    gate_read = gate_write = None
    handlers = {}
    limits = registration['supervision']
    started = time.monotonic()

    def interrupt(signum, frame):
        raise TimeoutError('Local evaluation deadline' if signum == signal.SIGALRM else 'Local evaluation interrupted')

    def send(sig):
        attempt = dict(signal=signal.Signals(sig).name, pgid=child.pid)
        result['signals'].append(attempt)
        try:
            os.killpg(child.pid, sig)
            attempt['result'] = 'sent'
        except ProcessLookupError:
            attempt['result'] = 'absent'
        except PermissionError:
            rows = guard.group_members(child.pid, footprint=False)
            require(len(rows) == 1 and rows[0]['pid'] == child.pid and rows[0]['status'].startswith('Z'),
                    'Cannot signal living owned process group')
            attempt['result'] = 'sole_unreaped_zombie'

    try:
        require(sys.platform == 'darwin', 'MPS supervision requires Darwin')
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
            handlers[sig] = signal.signal(sig, interrupt)
        deadline = started + limits['evaluation_deadline_seconds']
        signal.setitimer(signal.ITIMER_REAL, max(.001, deadline - time.monotonic()))
        lock_path = Path.home() / '.trid-local-mps-training.lock'
        lock = lock_path.open('a+')
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result['device_lock'] = str(lock_path)
        host = guard.host_probe()
        baseline = host['swap_bytes']
        result['swap_baseline_bytes'] = baseline
        require(guard.stop_reason(limits, host, [], baseline) is None, 'Host startup guard refused')
        gate_read, gate_write = os.pipe()
        env = guard.worker_environment()
        env.update(TRID_MPS_GATE_FD=str(gate_read), TRID_MPS_PARENT=str(os.getpid()))
        command = registration['command'] + ['--local-registration', str(args.local_registration.resolve()),
            '--local-registration-sha256', args.local_registration_sha256, '--supervised-worker']
        log = (out / 'worker.log').open('xb')
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log,
            stderr=subprocess.STDOUT, pass_fds=(gate_read,), start_new_session=True, cwd=ROOT)
        os.close(gate_read); gate_read = None
        require(os.getpgid(child.pid) == child.pid, 'Worker process group ownership')
        result['worker_pid'] = child.pid
        require(guard.stop_reason(limits, guard.host_probe(), guard.group_members(child.pid), baseline) is None,
                'Host startup gate refused')
        os.write(gate_write, b'G'); os.close(gate_write); gate_write = None
        result['startup_released'] = True
        guard.save(out / 'result.json', result)
        while True:
            host = guard.host_probe()
            rows = guard.group_members(child.pid)
            result['samples'].append(dict(elapsed_seconds=time.monotonic() - started, host=host, members=rows,
                mps=guard.telemetry_check(out / 'telemetry.json', limits)))
            require(guard.stop_reason(limits, host, rows, baseline) is None, 'External host/memory guard refused')
            require(time.monotonic() < deadline, 'Local evaluation deadline exceeded')
            leader = next((row for row in rows if row['pid'] == child.pid), None)
            if leader is None or leader['status'].startswith('Z'):
                result['stop_reason'] = 'worker_exit'
                break
            time.sleep(limits['sample_seconds'])
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        errors = []
        if child is not None:
            try:
                send(signal.SIGTERM)
                until = time.monotonic() + limits['term_grace_seconds']
                while time.monotonic() < until:
                    if not any(not row['status'].startswith('Z') for row in guard.group_members(child.pid, footprint=False)):
                        break
                    time.sleep(.05)
            except BaseException as error:
                errors.append(repr(error))
            try:
                send(signal.SIGKILL)
            except BaseException as error:
                errors.append(repr(error))
            try:
                until = time.monotonic() + limits['kill_cleanup_seconds']
                while True:
                    rows = guard.group_members(child.pid, footprint=False)
                    if all(row['pid'] == child.pid and row['status'].startswith('Z') for row in rows):
                        break
                    require(time.monotonic() < until, 'Owned evaluation descendants did not exit')
                    time.sleep(.05)
                result['cleanup']['members_before_reap'] = rows
                result['exit_code'] = child.wait(timeout=max(.01, until - time.monotonic()))
                result['cleanup']['leader_reaped'] = True
                result['cleanup']['remaining_members'] = guard.group_members(child.pid, footprint=False)
            except BaseException as error:
                errors.append(repr(error))
        else:
            result['cleanup']['remaining_members'] = []
        for fd in (gate_read, gate_write):
            if fd is not None:
                os.close(fd)
        if log:
            log.close()
        result['cleanup']['errors'] = errors
        result['cleanup']['accepted'] = bool(child is not None and not errors
            and result['cleanup']['leader_reaped'] and result['cleanup']['remaining_members'] == [])
        try:
            require('error' not in result and result.get('stop_reason') == 'worker_exit'
                    and result.get('exit_code') == 0 and result['cleanup']['accepted'],
                    'Local evaluation or owned-process cleanup failed')
            require(guard.telemetry_check(out / 'telemetry.json', limits) is not None, 'Missing MPS telemetry')
            completion = load_json(args.out / 'completion.json')
            selected_ids = [row['id'] for row in select_rows(
                public_inputs(args.inputs, args.inputs_sha256)[0], args.mode, args.smoke_id)]
            require(len(selected_ids) == (1 if args.mode == 'smoke' else 200)
                    and completion.get('status') == 'complete' and completion.get('mode') == args.mode
                    and completion.get('rows_per_arm') == len(selected_ids)
                    and set(completion.get('outputs', {})) == {'base', 'candidate'},
                    'Local paired completion is incomplete')
            for name in ('run-manifest.json', 'loaded-runtime.json'):
                receipt = load_json(args.out / name)
                authenticate_local_sources(registration, receipt['code'])
            manifest = load_json(args.out / 'run-manifest.json')
            require(manifest.get('local_registration') == registration_pin
                    and manifest.get('mode') == args.mode and manifest.get('selected_ids') == selected_ids
                    and manifest.get('arm_order') == ['base', 'candidate']
                    and manifest.get('decoder_profile', {}).get('profile') == 'unconstrained'
                    and completion.get('run_manifest_sha256') == digest(args.out / 'run-manifest.json'),
                    'Local worker registration/completion binding differs')
            for arm, entry in completion['outputs'].items():
                require(entry == pin(args.out / (arm + '-outputs.jsonl')), 'Local completed output pin differs')
                if args.mode == 'smoke':
                    with (args.out / (arm + '-outputs.jsonl')).open(encoding='utf-8') as stream:
                        answers = [strict_json(line) for line in stream]
                    require(len(answers) == 1 and isinstance(answers[0], dict),
                            'Local smoke requires one answer per arm')
                    answer = answers[0]
                    require(answer.get('id') == selected_ids[0] and answer.get('arm') == arm
                            and answer.get('mode') == 'smoke'
                            and answer.get('run_manifest_sha256') == completion['run_manifest_sha256']
                            and 'exception' in answer and answer['exception'] is None
                            and isinstance(answer.get('response_text'), str) and answer['response_text'].strip()
                            and answer.get('termination_reason') == 'eos'
                            and answer.get('truncated_output') is False,
                            'Local smoke answer is incomplete or its binding differs')
            result['completion'] = pin(args.out / 'completion.json')
            result['status'] = 'accepted'
        except BaseException as error:
            result['acceptance_error'] = repr(error)
        result['elapsed_seconds'] = time.monotonic() - started
        guard.save(out / 'result.json', result)
        if lock:
            lock.close()
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return result


def start_mps_telemetry(torch, registration):
    import threading
    import local_mps_training as guard
    stop = threading.Event()
    path = Path(registration['supervision_output']) / 'telemetry.json'

    def sample():
        value = dict(monotonic=time.monotonic(), allocator_bytes=int(torch.mps.current_allocated_memory()),
                     driver_bytes=int(torch.mps.driver_allocated_memory()))
        guard.save(path, value)
        guard.telemetry_check(path, registration['supervision'])

    def monitor():
        while not stop.wait(registration['supervision']['sample_seconds']):
            try:
                sample()
            except BaseException as error:
                guard.save(path, dict(allocator_bytes=0, driver_bytes=0, failure=repr(error)))
                return

    sample()
    thread = threading.Thread(target=monitor, name='mps-evaluation-telemetry', daemon=True)
    thread.start()
    return stop, thread


def accepted_local_supervision(registration_pin, directory):
    from local_eight_candidate import bound, load
    registration, _ = local_registration(bound(ROOT, registration_pin), registration_pin['sha256'])
    require(Path(registration['output']) == Path(directory).resolve(), 'Local supervised output differs')
    path = Path(registration['supervision_output']) / 'result.json'
    result = load(path)
    cleanup = result.get('cleanup', {})
    require(result.get('schema') == 'trid-local-eight-evaluation-supervision-v1'
            and result.get('registration') == registration_pin and result.get('status') == 'accepted'
            and result.get('startup_released') is True and result.get('exit_code') == 0
            and result.get('stop_reason') == 'worker_exit'
            and not any(key in result for key in ('error', 'acceptance_error'))
            and cleanup.get('accepted') is True and cleanup.get('leader_reaped') is True
            and cleanup.get('remaining_members') == [] and cleanup.get('errors') == [],
            'Local parent evaluation/cleanup was not accepted')
    require(bound(ROOT, result['completion']) == Path(directory).resolve() / 'completion.json',
            'Local accepted completion differs')
    manifest = load(Path(directory) / 'run-manifest.json')
    registered_args = parser().parse_args(registration['command'][4:])
    selected_ids = [row['id'] for row in select_rows(
        public_inputs(registered_args.inputs, registered_args.inputs_sha256)[0],
        registered_args.mode, registered_args.smoke_id)]
    completion = load(Path(directory) / 'completion.json')
    require(manifest.get('local_registration') == registration_pin
            and manifest.get('mode') == registered_args.mode
            and manifest.get('selected_ids') == selected_ids
            and completion.get('mode') == registered_args.mode
            and completion.get('run_manifest_sha256') == digest(Path(directory) / 'run-manifest.json')
            and manifest['evaluation_contract']['registration']['sha256'] == registration['candidate']['sha256']
            and manifest['input']['sha256'] == registration['inputs']['sha256']
            and manifest['generation'] == registration['generation']
            and manifest['model']['directory'] == registration['model_directory'],
            'Accepted local registration/worker binding differs')
    for name in ('run-manifest.json', 'loaded-runtime.json'):
        authenticate_local_sources(registration, load(Path(directory) / name)['code'])
    return path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-directory', required=True, type=Path)
    p.add_argument('--model-metadata', required=True, type=Path)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--inputs-sha256', required=True)
    p.add_argument('--export', type=Path)
    p.add_argument('--candidate-manifest', type=Path)
    p.add_argument('--candidate-sha256')
    p.add_argument('--mode', required=True, choices=('smoke', 'evaluation'))
    p.add_argument('--smoke-id', action='append', default=[])
    p.add_argument('--decoder-profile', choices=('unconstrained', 'json-schema'),
                   default='unconstrained')
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--local-registration', type=Path)
    p.add_argument('--local-registration-sha256')
    p.add_argument('--supervised-worker', action='store_true', help=argparse.SUPPRESS)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    require(bool(args.local_registration) == bool(args.local_registration_sha256),
            'Local registration and SHA must be paired')
    require(not args.supervised_worker or args.local_registration, 'Worker requires local registration')
    registration = registration_pin = None
    if args.local_registration:
        registration, registration_pin = local_registration(
            args.local_registration, args.local_registration_sha256, args)
        if not args.supervised_worker:
            result = supervised_evaluation(args, registration, registration_pin)
            require(result['status'] == 'accepted', 'Local evaluation supervision rejected: '
                    + result.get('error', result.get('acceptance_error', 'unaccepted completion')))
            return
        fd = int(os.environ['TRID_MPS_GATE_FD'])
        require(os.getppid() == int(os.environ['TRID_MPS_PARENT']), 'Local worker parent differs')
        try:
            require(os.read(fd, 1) == b'G', 'Parent did not release evaluation startup gate')
        finally:
            os.close(fd)
    # A path is claimed once, before any expensive work. Existing paths are refused.
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    telemetry = None
    try:
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0',
                'MPS CPU fallback is forbidden')
        for key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE'):
            os.environ[key] = '1'
        rows, input_pin = public_inputs(args.inputs, args.inputs_sha256)
        selected = select_rows(rows, args.mode, args.smoke_id)
        model_pin = model_identity(args.model_directory, args.model_metadata)
        require(bool(args.candidate_manifest) == bool(args.candidate_sha256),
                'candidate manifest and SHA must be paired')
        contract = None
        if args.candidate_manifest:
            from local_eight_candidate import resolve_candidate, public_contract
            require(args.export is None and args.decoder_profile == 'unconstrained',
                    'candidate registration forbids export overrides and secondary profile search')
            contract = resolve_candidate(ROOT, args.candidate_manifest, args.candidate_sha256)
            require(contract.get('inference_runtime', 'mps') == 'mps',
                    'Candidate inference must use native local MPS')
            require(contract.get('training_runtime') != 'cuda' or (registration and args.supervised_worker),
                    'CUDA-trained candidate requires guarded local evaluation entry')
            policy = contract['policy_document']
            require(input_pin['sha256'] == policy['data']['evaluation_inputs']['sha256']
                    and model_pin['acquisition_manifest']['sha256'] == policy['model']['acquisition_manifest']['sha256']
                    and model_pin['directory'] == str((ROOT / policy['model']['directory']).resolve())
                    and GENERATION == policy['evaluation']['generation'],
                    'candidate public inputs/base/generation differ')
            adapter, adapter_pin = Path(contract['adapter_directory']), contract['adapter']
        else:
            adapter, adapter_pin = adapter_identity(args.export or EXPORT)
        arms = contract['arms'] if contract else ARMS
        adapter_sha = contract['adapter_sha256'] if contract else ADAPTER_SHA
        import torch
        from transformers import AutoTokenizer, GenerationConfig
        from peft import PeftModel
        from train_open_weight_lora import load_qwen3_pread
        require(torch.backends.mps.is_available(), 'MPS is required; no backend fallback')
        require(platform.system() == 'Darwin', 'This diagnostic requires native macOS MPS')
        allocator = None
        if registration:
            import local_mps_training as guard
            allocator = guard.configure_allocator(torch, registration['supervision'])
            telemetry = start_mps_telemetry(torch, registration)
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
        if registration:
            runtime['allocator_policy'] = allocator
        identity = dict(schema='trid-local-eight-paired-v2' if contract else 'trid-local-eight-paired-v1', mode=args.mode,
            model=model_pin, adapter=adapter_pin, input=input_pin,
            selected_ids=[r['id'] for r in selected], generation=GENERATION,
            decoder_profile=decoder_identity,
            runtime=runtime, code=code_identity(), arm_order=list(arms),
            paired_order='each input: base with adapter disabled, then registered adapter enabled' if contract
                        else 'each input: base with adapter disabled, then step24 enabled',
            grading_access=False, retries=False, resume=False)
        if registration:
            authenticate_local_sources(registration, identity['code'])
            identity['local_registration'] = registration_pin
        if contract:
            identity['evaluation_contract'] = public_contract(contract)
        save_new(args.out / 'run-manifest.json', identity)
        save_new(args.out / 'prompt-preflight.json', dict(status='passed', prompts=prepared,
            all_selected_preflighted_before_weights=True, truncation=False))
        print(json.dumps(dict(event='preflight_complete_loading_weights', rows=len(prepared),
                              mode=args.mode, decoder_profile=args.decoder_profile)), flush=True)
        load_started = time.perf_counter()
        base, loader_report = load_qwen3_pread(args.model_directory, torch.bfloat16, 'mps')
        fp32_adapter = bool(contract and contract.get('training_runtime') == 'cuda')
        model = PeftModel.from_pretrained(base, adapter, is_trainable=False,
                                         local_files_only=True, autocast_adapter_dtype=fp32_adapter)
        model.eval()
        parameter_inventory = inference_parameter_inventory(model, torch, fp32_adapter)
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
            parameter_inventory=parameter_inventory,
            buffer_dtypes=sorted({str(b.dtype) for b in model.buffers()}),
            code=code_identity())
        if registration:
            authenticate_local_sources(registration, loaded['code'])
        save_new(args.out / 'loaded-runtime.json', loaded)
        print(json.dumps(dict(event='weights_loaded', elapsed_seconds=loaded['elapsed_seconds'],
                              mps_memory=loaded['mps_memory'])), flush=True)
        binding = dict(run_manifest_sha256=digest(args.out / 'run-manifest.json'),
            loaded_runtime_sha256=digest(args.out / 'loaded-runtime.json'), model_id=MODEL_ID,
            model_revision=REVISION, input_sha256=input_pin['sha256'],
            nominated_adapter_weights_sha256=adapter_sha, mode=args.mode)
        binding['decoder_profile'] = args.decoder_profile
        binding['decoder_profile_sha256'] = digest(args.out / 'decoder-profile.json')
        binding['generation_policy_sha256'] = object_sha(policy)
        if contract:
            binding['candidate_manifest_sha256'] = contract['registration']['sha256']
        run_pairs(torch, model, tokenizer, prepared, config, binding, args.out, decoder=decoder, contract=contract)
        save_new(args.out / 'completion.json', dict(status='complete', mode=args.mode,
            elapsed_seconds=time.perf_counter() - started, rows_per_arm=len(prepared),
            outputs={arm: pin(args.out / f'{arm}-outputs.jsonl') for arm in arms},
            mps_memory=mps_memory(torch), run_manifest_sha256=binding['run_manifest_sha256']))
        print(json.dumps(dict(event='complete', rows_per_arm=len(prepared),
                              elapsed_seconds=time.perf_counter() - started)), flush=True)
    except BaseException as error:
        save_new(args.out / 'failure.json', dict(status='failed', type=type(error).__name__,
            message=str(error), elapsed_seconds=time.perf_counter() - started,
            completed_rows_retained=True, automatic_retry=False))
        print(json.dumps(dict(event='failed', type=type(error).__name__, message=str(error))), flush=True)
        raise
    finally:
        if telemetry:
            telemetry[0].set()
            telemetry[1].join(timeout=2)




if __name__ == '__main__':
    main()
