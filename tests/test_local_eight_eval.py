"""CPU contracts only: synthetic fixtures never constitute model evaluation outputs.

Official tokenizer tests read existing pinned metadata, never download or load
weights. Pair-loop tests use explicit in-memory doubles to exercise journaling,
adapter state and fail-stop behavior, not to claim model inference correctness.
"""
import contextlib
import copy
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
import evaluate_local_eight as E
from eight_hf_assessment import structural_result


def public_row(identifier='synthetic-a'):
    return {'id': identifier, 'messages': [{'role': 'user', 'content': 'synthetic question'}]}


class LocalInputContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_inputs(self, rows):
        path = self.root / 'public.json'
        path.write_text(json.dumps(rows), encoding='utf-8')
        return path

    def test_public_input_identity_and_order_are_preserved(self):
        rows = [public_row('b'), public_row('a')]
        path = self.write_inputs(rows)
        actual, identity = E.public_inputs(path, E.digest(path))
        self.assertEqual(actual, rows)
        self.assertEqual(identity['sha256'], E.digest(path))
        with self.assertRaisesRegex(ValueError, 'hash differs'):
            E.public_inputs(path, '0' * 64)

    def test_references_and_malformed_messages_are_rejected(self):
        variants = []
        for field in ('answer', 'reference', 'rubric', 'target', 'score'):
            row = public_row()
            row[field] = 'forbidden'
            variants.append([row])
        row = public_row()
        row['messages'][0]['reference'] = 'forbidden'
        variants.extend([[row], [public_row(), public_row()], [], [{'id': 1, 'messages': []}]])
        row = public_row()
        row['messages'][-1]['role'] = 'assistant'
        variants.append([row])
        for rows in variants:
            with self.subTest(rows=rows):
                path = self.write_inputs(rows)
                with self.assertRaises(ValueError):
                    E.public_inputs(path, E.digest(path))

    def test_symlink_input_is_refused(self):
        path = self.write_inputs([public_row()])
        link = self.root / 'link.json'
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            E.public_inputs(link, E.digest(path))

    def test_smoke_is_explicit_strict_subset_full_cannot_filter(self):
        rows = [public_row('b'), public_row('a'), public_row('c')]
        self.assertEqual(E.select_rows(rows, 'smoke', ['a', 'b']), rows[:2])
        self.assertIs(E.select_rows(rows, 'evaluation', []), rows)
        for mode, ids in [('smoke', []), ('smoke', ['missing']), ('smoke', ['a', 'a']),
                          ('smoke', ['a', 'b', 'c']), ('evaluation', ['a'])]:
            with self.subTest(mode=mode, ids=ids), self.assertRaises(ValueError):
                E.select_rows(rows, mode, ids)

    def test_output_and_output_directory_cannot_be_reused(self):
        target = self.root / 'identity.json'
        E.save_new(target, {'synthetic': True})
        original = target.read_bytes()
        with self.assertRaises(FileExistsError):
            E.save_new(target, {'synthetic': False})
        self.assertEqual(target.read_bytes(), original)
        with self.assertRaises(FileExistsError):
            E.main(['--model-directory', str(self.root), '--model-metadata', str(target),
                    '--inputs', str(target), '--inputs-sha256', E.digest(target),
                    '--mode', 'evaluation', '--out', str(self.root)])

    def test_unpinned_metadata_cannot_trigger_model_file_reads(self):
        metadata = self.root / 'metadata.json'
        metadata.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'manifest hash differs'):
            E.model_identity(self.root / 'nonexistent', metadata)

    def test_five_shard_bodies_are_hashed_not_trusted_from_metadata(self):
        # Tiny fixture bytes are intentionally not model weights; no loader runs.
        from unittest.mock import patch
        model = self.root / 'synthetic-model'
        model.mkdir()
        shards = [f'model-{i:05d}-of-00005.safetensors' for i in range(1, 6)]
        for name in ['LICENSE', 'config.json', 'generation_config.json',
                     'tokenizer.json', 'tokenizer_config.json', *shards]:
            (model / name).write_bytes(b'synthetic pinned bytes')
        index = {'weight_map': {f'synthetic-{i}': name for i, name in enumerate(shards)}}
        (model / 'model.safetensors.index.json').write_text(json.dumps(index), encoding='utf-8')
        metadata = dict(schema='trid-immutable-model-metadata-v1', repo_id=E.MODEL_ID,
                        revision=E.REVISION, weight_shards=5,
                        files=[dict(path=p.name, bytes=p.stat().st_size, sha256=E.digest(p))
                               for p in model.iterdir()])
        manifest = self.root / 'synthetic-metadata.json'
        manifest.write_text(json.dumps(metadata), encoding='utf-8')
        with patch.object(E, 'METADATA_SHA', E.digest(manifest)):
            identity = E.model_identity(model, manifest)
            self.assertEqual(len(identity['files']), 11)
            self.assertTrue(identity['body_hashes_authenticated'])
            (model / shards[0]).write_bytes(b'tampered pinned bytes!')
            with self.assertRaisesRegex(ValueError, 'body authentication failed'):
                E.model_identity(model, manifest)

    def test_supported_memory_metrics_are_actual_not_estimated(self):
        torch = types.SimpleNamespace(mps=types.SimpleNamespace(
            current_allocated_memory=lambda: 123, driver_allocated_memory=lambda: 456))
        metrics = E.mps_memory(torch)
        self.assertEqual(metrics['bytes']['current_allocated_memory'], 123)
        self.assertEqual(metrics['bytes']['driver_allocated_memory'], 456)
        self.assertIsNone(metrics['bytes']['max_memory_allocated'])
        self.assertIn('max_memory_allocated', metrics['unsupported'])


class BudgetContracts(unittest.TestCase):
    def test_exact_boundary_and_overflow_are_not_truncated(self):
        calls = []

        class BoundaryTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                calls.append(kwargs)
                return '<think>\n\n</think>\n\n'

            def __call__(self, prompt, **kwargs):
                calls.append(kwargs)
                return {'input_ids': [1] * self.width}

        tokenizer = BoundaryTokenizer()
        tokenizer.width = 4096 - 768
        prepared = E.preflight_prompts(tokenizer, [public_row()])
        self.assertEqual(len(prepared[0]['prompt_token_ids']), 3328)
        self.assertEqual(calls[0], dict(tokenize=False, add_generation_prompt=True, enable_thinking=False))
        self.assertEqual(calls[1], dict(add_special_tokens=False, truncation=False))
        tokenizer.width += 1
        with self.assertRaisesRegex(ValueError, 'no truncation'):
            E.preflight_prompts(tokenizer, [public_row()])


class OfficialTokenizerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT / 'runs/mois-ordered-models-20261113T062501Z/qwen8-native-metadata'
        if not directory.is_dir():
            raise unittest.SkipTest('Existing official Qwen3-8B tokenizer metadata is required')
        from transformers import AutoTokenizer, GenerationConfig
        cls.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        cls.config = GenerationConfig.from_pretrained(directory, local_files_only=True).to_dict()
        cls.stops = cls.config['eos_token_id']
        if isinstance(cls.stops, int):
            cls.stops = [cls.stops]

    def test_native_prompt_and_eos_match_assessment_structural_contract(self):
        case = E.preflight_prompts(self.tokenizer, [public_row()])[0]
        body = '  synthetic response\n'
        for stop in self.stops:
            tokens = self.tokenizer.encode(body, add_special_tokens=False) + [stop]
            row = E.answer_record(case, 'base', {}, self.config, tokens, self.tokenizer, 0.25, {})
            self.assertEqual(row['response_text'], body)
            self.assertEqual(row['output_token_ids'], tokens)
            self.assertEqual(row['elapsed_seconds'], 0.25)
            self.assertIsNone(row['adapter_weights_sha256'])
            self.assertEqual(row['termination_reason'], 'eos')
            self.assertFalse(row['truncated_output'])
            self.assertTrue(structural_result(public_row(), row, E.GENERATION)['pass'])
            bad = copy.deepcopy(row)
            bad['exception'] = {'type': 'synthetic'}
            self.assertFalse(structural_result(public_row(), bad, E.GENERATION)['pass'])

    def test_length_limit_preserves_tokens_and_is_structurally_invalid(self):
        case = E.preflight_prompts(self.tokenizer, [public_row()])[0]
        token = self.tokenizer.encode('x', add_special_tokens=False)[0]
        self.assertNotIn(token, self.stops)
        tokens = [token] * 768
        row = E.answer_record(case, 'step-000024', {}, self.config, tokens, self.tokenizer, 1.0, {})
        self.assertEqual(row['output_token_ids'], tokens)
        self.assertTrue(row['truncated_output'])
        self.assertEqual(row['adapter_weights_sha256'], E.ADAPTER_SHA)
        self.assertFalse(structural_result(public_row(), row, E.GENERATION)['pass'])
        # Consumers must receive no answer for an unexplained short stop; the
        # shared decoder's private exception class is not this CLI's contract.
        rejected = None
        with self.assertRaisesRegex(Exception, 'Unexplained generation termination'):
            rejected = E.answer_record(case, 'base', {}, self.config, tokens[:3], self.tokenizer, 0.1, {})
        self.assertIsNone(rejected)

    def test_internal_protocol_and_invalid_json_are_not_repaired(self):
        case = E.preflight_prompts(self.tokenizer, [public_row()])[0]
        body = ' \n```json\n<think>synthetic</think>{broken}\n```\t '
        tokens = self.tokenizer.encode(body, add_special_tokens=False) + [self.stops[0]]
        row = E.answer_record(case, 'base', {}, self.config, tokens, self.tokenizer, 0.1, {})
        self.assertEqual(row['response_text'], body)
        self.assertEqual(row['raw_output'], self.tokenizer.decode(
            tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False))
        with self.assertRaises(json.JSONDecodeError):
            json.loads(row['response_text'])


class PairJournalContracts(unittest.TestCase):
    def test_failed_adapter_stops_without_retry_and_retains_base_answer(self):
        calls = []

        class Tensor:
            def __getitem__(self, key):
                return self

            def tolist(self):
                return [11, 99]

        class Model:
            disabled = False

            @contextlib.contextmanager
            def disable_adapter(self):
                self.disabled = True
                try:
                    yield
                finally:
                    self.disabled = False

            def _prepare_generation_config(self, config, use_model_defaults=None, **kwargs):
                return config, {}

            def generate(self, **kwargs):
                self._prepare_generation_config(kwargs['generation_config'],
                    use_model_defaults=kwargs['use_model_defaults'],
                    do_sample=kwargs['do_sample'], num_beams=kwargs['num_beams'])
                calls.append(self.disabled)
                if not self.disabled:
                    raise RuntimeError('synthetic adapter failure')
                return types.SimpleNamespace(sequences=Tensor())

        class Tokenizer:
            def decode(self, ids, **kwargs):
                return 'synthetic' + ('<stop>' if 99 in ids else '')

        config = types.SimpleNamespace(to_dict=lambda: {'eos_token_id': [99]},
            do_sample=False, num_beams=1, max_new_tokens=768, return_dict_in_generate=True,
            get_generation_mode=lambda: types.SimpleNamespace(value='greedy_search'))
        torch = types.SimpleNamespace(long='synthetic-long', manual_seed=lambda seed: None,
            tensor=lambda *args, **kwargs: Tensor(), ones_like=lambda tensor: Tensor(),
            inference_mode=contextlib.nullcontext,
            mps=types.SimpleNamespace(manual_seed=lambda seed: None, synchronize=lambda: None))
        case = dict(**public_row(), prompt_token_ids=[7],
                    prompt_sha256=E.object_sha(public_row()['messages']), rendered_prompt_sha256='a' * 64)
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            model = Model()
            with self.assertRaisesRegex(RuntimeError, 'synthetic adapter failure'):
                E.run_pairs(torch, model, Tokenizer(), [case, dict(case, id='never-run')], config, {}, out)
            base = [json.loads(line) for line in (out / 'base-outputs.jsonl').read_text().splitlines()]
            adapter = [json.loads(line) for line in (out / 'step-000024-outputs.jsonl').read_text().splitlines()]
            attempts = [json.loads(line) for line in (out / 'attempts.jsonl').read_text().splitlines()]
            self.assertEqual(calls, [True, False])
            self.assertFalse(model.disabled)
            self.assertEqual(len(base), 1)
            self.assertEqual(base[0]['termination_reason'], 'eos')
            self.assertEqual(base[0]['output_token_ids'], [11, 99])
            self.assertEqual(len(adapter), 1)
            self.assertEqual(adapter[0]['termination_reason'], 'error')
            self.assertEqual(adapter[0]['exception']['type'], 'RuntimeError')
            self.assertEqual([a['arm'] for a in attempts], list(E.ARMS))
            with self.assertRaises(FileExistsError):
                E.run_pairs(torch, model, Tokenizer(), [case], config, {}, out)
            self.assertEqual(calls, [True, False])


def mois_row(source_ids=('E1', 'E2'), *, graph=False):
    if graph:
        payload = {'question': '조건은 무엇인가요?', 'source_data': {
            'documents': [{'doc_id': 'D1'}], 'scope': '', 'warnings': [],
            'evidence': [{'id': identifier, 'doc_id': 'D1', 'quote': '공개 근거'}
                         for identifier in source_ids]}}
    else:
        payload = {'question': '조건은 무엇인가요?', 'answer_schema': E.public_answer_schema(),
                   'sources': [{'id': identifier, 'title': '공개 자료', 'version_date': '',
                                'text': '공개 근거'} for identifier in source_ids]}
    return {'id': 'synthetic-mois', 'messages': [
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]}


class SchemaAdmissionContracts(unittest.TestCase):
    def test_both_public_formats_derive_exact_unmodified_citation_ids(self):
        source_ids = ['law:123:scope', 'admrul:83472:제2조제1항 자문단 정의']
        for graph in (False, True):
            row = mois_row(source_ids, graph=graph)
            before = copy.deepcopy(row)
            schema = E.mois_schema(row)
            paragraph = schema['properties']['paragraphs']
            self.assertEqual(paragraph['maxItems'], 5)
            self.assertEqual(paragraph['items']['properties']['text']['maxLength'], 1200)
            self.assertEqual(paragraph['items']['properties']['sources']['items']['enum'], source_ids)
            self.assertEqual(row, before)

    def test_general_inputs_and_unsupported_schema_keywords_are_refused(self):
        with self.assertRaises(ValueError):
            E.mois_schema(public_row())
        for keyword, value in [('patternProperties', {}), ('$ref', '#/unsupported'),
                               ('oneOf', []), ('description', 'not silently discarded')]:
            row = mois_row()
            payload = json.loads(row['messages'][0]['content'])
            payload['answer_schema'][keyword] = value
            row['messages'][0]['content'] = json.dumps(payload)
            with self.subTest(keyword=keyword), self.assertRaisesRegex(ValueError, 'Unsupported answer_schema'):
                E.mois_schema(row)
        row = mois_row()
        payload = json.loads(row['messages'][0]['content'])
        del payload['answer_schema']['properties']['paragraphs']['items']['properties']['sources']['uniqueItems']
        row['messages'][0]['content'] = json.dumps(payload)
        with self.assertRaisesRegex(ValueError, 'Unsupported answer_schema'):
            E.mois_schema(row)

    def test_eos_admission_rejects_duplicate_keys_duplicate_ids_and_foreign_ids(self):
        schema = E.mois_schema(mois_row())
        invalid = [
            '{"paragraphs":[],"paragraphs":[]}',
            '{"paragraphs":[{"text":"설명","text":"다른 설명","sources":["E1"]}]}',
            '{"paragraphs":[{"text":"설명","sources":["E1","E1"]}]}',
            '{"paragraphs":[{"text":"설명","sources":["invented"]}]}',
            '{"paragraphs":[{"text":" ","sources":["E1"]}]}',
            '{"paragraphs":[{"text":"설명","sources":[]}]}',
            '{"paragraphs":[{"text":"설명","sources":["E1"]}]',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                E.validate_schema_finish(raw, schema)
        raw = '{"paragraphs":[{"text":"한국어 설명","sources":["E1","E2"]}]}'
        self.assertIsNone(E.validate_schema_finish(raw, schema))
        self.assertEqual(raw, '{"paragraphs":[{"text":"한국어 설명","sources":["E1","E2"]}]}')

    def test_default_decoder_profile_keeps_native_generation_fields(self):
        args = E.parser().parse_args(['--model-directory', 'unused', '--model-metadata', 'unused',
            '--inputs', 'unused', '--inputs-sha256', '0' * 64, '--mode', 'evaluation', '--out', 'unused'])
        self.assertEqual(args.decoder_profile, 'unconstrained')
        self.assertNotIn('decoder_profile', E.GENERATION)
        self.assertEqual(E.GENERATION['max_new_tokens'], 768)
        self.assertEqual(E.GENERATION['context_tokens'], 4096)


class SchemaTokenizerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        if importlib.util.find_spec('lmformatenforcer') is None:
            raise unittest.SkipTest('Optional profile requires lm-format-enforcer==0.11.3; not a compatibility pass')
        from transformers import AutoTokenizer, GenerationConfig
        directory = ROOT / 'runs/mois-ordered-models-20261113T062501Z/qwen8-native-metadata'
        cls.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        cls.row = mois_row(['E1', 'E2', 'admrul:83472:제2조제1항 자문단 정의'])
        cls.decoder = E.SchemaDecoder(cls.tokenizer, [cls.row])
        cls.case = E.preflight_prompts(cls.tokenizer, [cls.row])[0]
        eos = GenerationConfig.from_pretrained(directory, local_files_only=True).eos_token_id
        cls.stops = [eos] if type(eos) is int else eos

    def walk(self, raw, *, tokens=None):
        import torch
        callback = self.decoder.prefix(self.case, self.stops)
        history = self.case['prompt_token_ids'].copy()
        if tokens is None:
            tokens = self.tokenizer.encode(raw, add_special_tokens=False)
        for offset, token in enumerate(tokens):
            allowed = callback(0, torch.tensor(history, dtype=torch.long))
            self.assertNotIn(self.stops[0], allowed, f'Premature EOS at token {offset}')
            self.assertIn(token, allowed, f'Unsupported real tokenizer path at {offset}: {token}')
            history.append(token)
        return callback(0, torch.tensor(history, dtype=torch.long)), tokens

    def test_korean_prose_byte_fragments_and_public_korean_citation_reach_all_configured_eos(self):
        text = '한국어 조건과 예외입니다. 한글 똠뷁'
        citation = 'admrul:83472:제2조제1항 자문단 정의'
        raw = json.dumps({'paragraphs': [{'text': text, 'sources': [citation]}]}, ensure_ascii=False)
        allowed, tokens = self.walk(raw)
        self.assertTrue(set(self.stops) <= set(allowed))
        self.assertTrue(any('\ufffd' in self.tokenizer.decode([token]) for token in tokens),
                        'Regression must exercise actual split UTF-8 token fragments')
        # encode() normalizes input via the pinned tokenizer's NFC normalizer;
        # decode() does not rewrite model output. Test those two facts separately.
        normalized = self.tokenizer.backend_tokenizer.normalizer.normalize_str(raw)
        decoded = self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False)
        self.assertEqual(decoded, normalized)
        self.assertNotEqual(decoded, raw)
        self.assertEqual(json.loads(decoded)['paragraphs'][0]['sources'], [citation])
        self.assertEqual(self.decoder.identity['version'], '0.11.3')

    def test_raw_model_token_path_preserves_nfd_jamo_without_input_normalization(self):
        from tokenizers import Tokenizer
        # Construct a possible model output from the actual pinned vocabulary.
        # Removing normalizer on this test-only clone does NOT alter inference
        # prompts/tokenizer; it avoids confusing encode normalization with decode.
        output_path_tokenizer = Tokenizer.from_str(self.tokenizer.backend_tokenizer.to_str())
        output_path_tokenizer.normalizer = None
        raw = '{"paragraphs":[{"text":"원문 한글 똠뷁","sources":["E1"]}]}'
        path = output_path_tokenizer.encode(raw, add_special_tokens=False).ids
        self.assertEqual(self.tokenizer.decode(path, clean_up_tokenization_spaces=False), raw)
        allowed, tokens = self.walk(raw, tokens=path)
        self.assertTrue(set(self.stops) <= set(allowed))
        self.assertEqual(tokens, path)
        self.assertEqual(self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False), raw)

    def test_json_unicode_escapes_and_field_order_are_preserved(self):
        raw = json.dumps({'paragraphs': [{'sources': ['E1'], 'text': '조건과 예외'}]}, ensure_ascii=True)
        allowed, tokens = self.walk(raw)
        self.assertIn(self.stops[0], allowed)
        self.assertEqual(self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False), raw)

    def test_combined_prose_and_json_boundary_tokens_are_not_pruned(self):
        raw = '{"paragraphs":[{"text":"푸른자료실은 화요일과 목요일에 문을 연다.","sources":["E1"]}]}'
        allowed, tokens = self.walk(raw)
        self.assertTrue(set(self.stops) <= set(allowed))
        self.assertIn('.","', [self.tokenizer.decode([token], clean_up_tokenization_spaces=False)
                              for token in tokens])
        self.assertEqual(self.tokenizer.decode(tokens, clean_up_tokenization_spaces=False), raw)

    def test_residual_boundary_index_is_shared_but_prefix_state_is_fresh(self):
        from local_eight_schema import build_transformers_prefix_allowed_tokens_fn
        from lmformatenforcer import JsonSchemaParser
        schema = self.decoder.schemas[self.case['id']]
        first = build_transformers_prefix_allowed_tokens_fn(self.decoder.data, JsonSchemaParser(schema))
        second = build_transformers_prefix_allowed_tokens_fn(self.decoder.data, JsonSchemaParser(schema))
        self.assertIs(first.token_enforcer.residual_tree, second.token_enforcer.residual_tree)
        self.assertIs(first.token_enforcer.tokenizer_tree, second.token_enforcer.tokenizer_tree)
        self.assertIsNot(first.token_enforcer.prefix_states, second.token_enforcer.prefix_states)

    def test_duplicate_object_key_is_blocked_not_repaired(self):
        import torch
        callback = self.decoder.prefix(self.case, self.stops)
        history = self.case['prompt_token_ids'].copy()
        raw = '{"paragraphs":[{"text":"설명","text":"다른 설명","sources":["E1"]}]}'
        blocked = False
        for token in self.tokenizer.encode(raw, add_special_tokens=False):
            try:
                allowed = callback(0, torch.tensor(history, dtype=torch.long))
            except ValueError:
                blocked = True
                break
            if token not in allowed:
                blocked = True
                break
            history.append(token)
        self.assertTrue(blocked, 'Duplicate JSON object key must never reach valid EOS')

    def test_library_unique_items_gap_is_rejected_at_eos_not_silently_accepted(self):
        raw = '{"paragraphs":[{"text":"한국어 설명","sources":["E1","E1"]}]}'
        with self.assertRaisesRegex(ValueError, 'repeated citation'):
            self.walk(raw)

    def test_upstream_fallback_eos_is_fatal_not_a_valid_finish(self):
        import torch
        from unittest.mock import patch
        from lmformatenforcer import JsonSchemaParser

        class FallbackPrefix:
            def __init__(self, parser):
                self.parser = parser
                self.token_enforcer = types.SimpleNamespace(prefix_states={})

            def __call__(self, batch_id, sent):
                self.token_enforcer.prefix_states[tuple(sent.tolist())] = types.SimpleNamespace(parser=self.parser)
                return [SchemaTokenizerContracts.stops[0]]

        schema = self.decoder.schemas[self.case['id']]
        with patch('local_eight_schema.build_transformers_prefix_allowed_tokens_fn',
                   return_value=FallbackPrefix(JsonSchemaParser(schema))):
            callback = self.decoder.prefix(self.case, self.stops)
            with self.assertRaisesRegex(ValueError, 'premature fallback EOS'):
                callback(0, torch.tensor(self.case['prompt_token_ids'], dtype=torch.long))


class EffectiveGreedyPolicyContracts(unittest.TestCase):
    def tiny_model(self):
        import torch
        from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(32)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, head_dim=8,
            max_position_embeddings=4096, eos_token_id=0, pad_token_id=1, bos_token_id=1,
            tie_word_embeddings=False)).eval()
        # Real HF forward/generate: tied zero logits make the greedy choice token
        # zero (EOS), independent of RNG, without a long synthetic generation.
        with torch.no_grad():
            model.lm_head.weight.zero_()
        model.generation_config = GenerationConfig(
            do_sample=True, temperature=0.6, top_k=20, top_p=0.95, num_beams=4,
            eos_token_id=0, pad_token_id=1, bos_token_id=1, transformers_version='4.57.6')
        requested = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=768,
            use_cache=True, eos_token_id=0, pad_token_id=1, bos_token_id=None,
            return_dict_in_generate=True)
        return model, requested

    def test_real_hf_sampling_defaults_cannot_override_explicit_greedy_dispatch(self):
        import torch
        model, requested = self.tiny_model()
        requested_before = requested.to_dict()
        config, policy = E.resolve_generation_policy(model, requested)
        self.assertEqual(requested.to_dict(), requested_before)
        self.assertFalse(config.do_sample)
        self.assertEqual(config.num_beams, 1)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 1.0)
        self.assertEqual(config.bos_token_id, 1)
        self.assertEqual(policy['generation_mode'], 'greedy_search')
        self.assertEqual(policy['effective_config_sha256'], E.object_sha(config.to_dict()))
        ids = torch.tensor([[1, 7]], dtype=torch.long)
        for seed in (11, 29):
            torch.manual_seed(seed)
            with torch.inference_mode(), E.observe_generation_policy(model, config) as observed:
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                        generation_config=config, **E.GREEDY_DISPATCH)
            self.assertEqual(result.sequences[0, ids.shape[1]:].tolist(), [0])
            self.assertEqual(observed['generation_mode'], 'greedy_search')
            self.assertEqual(observed['effective_config_sha256'], policy['effective_config_sha256'])
            self.assertNotIn('_prepare_generation_config', model.__dict__)
        self.assertTrue(model.generation_config.do_sample)
        self.assertEqual(model.generation_config.temperature, 0.6)

    def test_actual_prepare_observer_rejects_missing_dispatch_controls_before_generation(self):
        import torch
        model, requested = self.tiny_model()
        config, _ = E.resolve_generation_policy(model, requested)
        ids = torch.tensor([[1, 7]], dtype=torch.long)
        with self.assertRaisesRegex(ValueError, 'Explicit greedy dispatch controls missing'):
            with torch.inference_mode(), E.observe_generation_policy(model, config):
                model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config)
        self.assertNotIn('_prepare_generation_config', model.__dict__)

    def test_real_peft_wrapper_observes_the_underlying_hf_generation_call(self):
        import torch
        from peft import LoraConfig, get_peft_model
        base, requested = self.tiny_model()
        model = get_peft_model(base, LoraConfig(r=2, lora_alpha=2,
            target_modules=['q_proj', 'v_proj'], task_type='CAUSAL_LM', inference_mode=True)).eval()
        config, policy = E.resolve_generation_policy(model, requested)
        ids = torch.tensor([[1, 7]], dtype=torch.long)
        for disabled in (True, False):
            with (model.disable_adapter() if disabled else contextlib.nullcontext()):
                with torch.inference_mode(), E.observe_generation_policy(model, config) as observed:
                    result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                            generation_config=config, **E.GREEDY_DISPATCH)
            self.assertEqual(result.sequences[0, ids.shape[1]:].tolist(), [0])
            self.assertEqual(observed['effective_config_sha256'], policy['effective_config_sha256'])
            self.assertNotIn('_prepare_generation_config', base.__dict__)


if __name__ == '__main__':
    unittest.main()
