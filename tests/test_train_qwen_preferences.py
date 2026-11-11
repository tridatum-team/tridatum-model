"""Preference training primitives on a real tiny Qwen3: summed answer likelihood (shift, prompt/history masking,
padding, EOS, gradients), the DPO objective from a tied reference, the canonical-replay contract (manifest block,
source rows, full-epoch partition, combined mean gradient) and cache binding rejection. No downloads."""
import hashlib
import json
import math
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

from scripts import train_qwen_preferences as dpo

sys.path.insert(0, str(Path(__file__).parent))
from test_train_open_weight_lora import local_tokenizer  # noqa: E402


def tiny_qwen3(vocab):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    with torch.random.fork_rng():
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                             num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                                             tie_word_embeddings=True, attention_dropout=0.0))
    model.eval()  # deterministic; gradients still flow
    return model


def tiny_lora_qwen3(vocab):
    """A nonzero adapter near its initial state, isolated from other tests' random streams."""
    from peft import LoraConfig, get_peft_model
    with torch.random.fork_rng():
        torch.manual_seed(1)
        model = get_peft_model(tiny_qwen3(vocab), LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0,
                                                          target_modules=dpo.T.TARGETS, task_type='CAUSAL_LM'))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if 'lora_B' in name:
                    p.normal_(std=0.01)
    model.enable_input_require_grads()
    return model


def manual_logprob(model, item):
    """Full-width reference: sum of log-softmax at the shifted answer positions of one unpadded row."""
    ids = torch.tensor([item['ids']])
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0, :-1].float()
    logp = torch.log_softmax(logits, -1)
    labels = torch.tensor(item['labels'][1:])
    keep = labels != -100
    return logp[keep].gather(1, labels[keep][:, None]).sum(), int(keep.sum())


class PairLikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = local_tokenizer()

    def pair(self, chosen='자주색입니다.', rejected='은색'):
        return dpo.encode_pair(self.tok, dict(
            id='p1', task='color', system=None, user='색은?', chosen=chosen, rejected=rejected, file='pairs.jsonl',
            history=[{'user': '표시는 자주색으로 정해.', 'assistant': '자주색으로 기억합니다.'}]), 10000)

    def test_summed_logprob_covers_answer_plus_eos_only_and_matches_full_logits_with_padding(self):
        model = tiny_qwen3(len(self.tok))
        encoded = self.pair()
        self.assertGreater(len(encoded['chosen']['ids']), len(encoded['rejected']['ids']))  # rejected is right-padded
        for side in ('chosen', 'rejected'):
            self.assertEqual(encoded[side]['labels'][-1], self.tok.eos_token_id)
        chosen_ref, chosen_n = manual_logprob(model, encoded['chosen'])
        rejected_ref, rejected_n = manual_logprob(model, encoded['rejected'])
        answer_tokens = lambda text: len(self.tok(text + self.tok.eos_token, add_special_tokens=False)['input_ids'])
        self.assertEqual((chosen_n, rejected_n), (answer_tokens('자주색입니다.'), answer_tokens('은색')))
        self.addCleanup(setattr, dpo.T, 'LOGIT_CHUNK', dpo.T.LOGIT_CHUNK)
        dpo.T.LOGIT_CHUNK = 3  # several chunks and a ragged last one over the union of answer positions
        for family in ('qwen3', 'generic'):
            scores = dpo.pair_logprobs(model, encoded, 'cpu', self.tok.pad_token_id, family)
            self.assertEqual(tuple(scores.shape), (2,))
            self.assertTrue(torch.allclose(scores[0], chosen_ref, atol=1e-5), family)
            self.assertTrue(torch.allclose(scores[1], rejected_ref, atol=1e-5), family)
        # Not length-normalised: the longer answer sums more terms, so it is not the per-token mean.
        self.assertFalse(torch.allclose(scores[0], chosen_ref / chosen_n, atol=1e-3))

    def test_gradients_of_the_summed_likelihood_equal_the_full_logits_computation_and_reach_the_context(self):
        model = tiny_qwen3(len(self.tok))
        encoded = self.pair()
        context = []

        def keep_context(_module, _inputs, output):  # the embedded sequence the model actually read
            output.retain_grad()
            context.append(output)

        hook = model.get_input_embeddings().register_forward_hook(keep_context)
        self.addCleanup(hook.remove)
        model.zero_grad(set_to_none=True)
        (manual_logprob(model, encoded['chosen'])[0] - manual_logprob(model, encoded['rejected'])[0]).backward()
        want = {name: p.grad.clone() for name, p in model.named_parameters()}
        model.zero_grad(set_to_none=True)
        context.clear()
        scores = dpo.pair_logprobs(model, encoded, 'cpu', self.tok.pad_token_id)
        (scores[0] - scores[1]).backward()
        got = {name: p.grad.clone() for name, p in model.named_parameters()}
        self.assertEqual(set(got), set(want))
        for name in want:
            self.assertTrue(torch.allclose(got[name], want[name], atol=1e-5), name)
        embedded = context[-1].grad
        first_answer = next(i for i, label in enumerate(encoded['chosen']['labels']) if label != -100)
        self.assertGreater(embedded[0, :first_answer].abs().sum().item(), 0)  # prompt/history carry context gradient
        self.assertEqual(embedded[1, len(encoded['rejected']['ids']):].abs().sum().item(), 0)  # padding never does

    def test_sequence_losses_score_any_number_of_ragged_rows_as_if_alone_with_their_answer_token_counts(self):
        model = tiny_qwen3(len(self.tok))
        encoded = self.pair()
        extra = dpo.T.encode(self.tok, dict(id='r', task='t', system=None, user='이름이 뭐예요?', history=[], answer='트리드입니다.',
                                            file='train.jsonl'), 'qwen3', 10000)
        items = [encoded['rejected'], encoded['chosen'], extra]  # shortest first: the padded width is not row 0's
        self.assertGreater(len(set(len(x['ids']) for x in items)), 1)
        refs = [manual_logprob(model, x) for x in items]
        self.addCleanup(setattr, dpo.T, 'LOGIT_CHUNK', dpo.T.LOGIT_CHUNK)
        dpo.T.LOGIT_CHUNK = 3
        for family in ('qwen3', 'generic'):
            summed, counts = dpo.sequence_losses(model, items, 'cpu', self.tok.pad_token_id, family)
            self.assertEqual((tuple(summed.shape), tuple(counts.shape)), ((3,), (3,)))
            self.assertEqual(counts.tolist(), [n for _, n in refs])  # answer+EOS after the shift, never prompt or padding
            for i, (logprob, _) in enumerate(refs):
                self.assertTrue(torch.allclose(summed[i], -logprob, atol=1e-5), (family, i))
        single = dpo.sequence_losses(model, [items[2]], 'cpu', self.tok.pad_token_id)[0]
        self.assertTrue(torch.allclose(single[0], summed[2], atol=1e-5))  # the batch does not change a row's score
        self.assertTrue(torch.allclose(dpo.pair_logprobs(model, encoded, 'cpu', self.tok.pad_token_id), -summed[[1, 0]], atol=1e-5))


class DpoObjectiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = local_tokenizer()

    def test_tied_reference_starts_at_log2_and_one_update_favours_chosen_through_lora_only(self):
        model = tiny_lora_qwen3(len(self.tok))
        encoded = dpo.encode_pair(self.tok, dict(id='p', task='t', system=None, user='색은?', history=[],
                                                 chosen='자주색', rejected='은색', file='pairs.jsonl'), 10000)
        with torch.no_grad():
            reference = tuple(dpo.pair_logprobs(model, encoded, 'cpu', self.tok.pad_token_id).tolist())
        beta = 0.1
        policy = dpo.pair_logprobs(model, encoded, 'cpu', self.tok.pad_token_id)
        loss, margin = dpo.dpo_loss(policy, reference, beta)
        self.assertAlmostEqual(loss.item(), math.log(2), places=5)
        self.assertAlmostEqual(margin.item(), 0.0, places=5)
        loss.backward()
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(trainable and all('lora_' in name for name in trainable))
        with_grad = {name for name, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
        self.assertEqual(with_grad, trainable)  # connected to every adapter tensor, nothing on the frozen base
        # Expected gradient of the loss wrt the policy scores: -beta*(1-sigmoid(0)) on chosen, +beta/2 on rejected.
        scores = dpo.pair_logprobs(model, encoded, 'cpu', self.tok.pad_token_id)
        grad_chosen, grad_rejected = torch.autograd.grad(dpo.dpo_loss(scores, reference, beta)[0], scores)[0].tolist()
        self.assertAlmostEqual(grad_chosen, -beta / 2, places=6)
        self.assertAlmostEqual(grad_rejected, beta / 2, places=6)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.05)
        optimizer.step()
        with torch.no_grad():
            after = dpo.pair_logprobs(model, encoded, 'cpu', self.tok.pad_token_id)
            loss_after, margin_after = dpo.dpo_loss(after, reference, beta)
        self.assertGreater(margin_after.item(), 0)
        self.assertLess(loss_after.item(), math.log(2))


class CacheBindingTests(unittest.TestCase):
    def setUp(self):
        self.tok = local_tokenizer()
        self.digests = {pid: hashlib.sha256(pid.encode()).hexdigest() for pid in ('a', 'b', 'c')}
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.code_dir = Path(self.temp.name)
        self.code_sources = {}
        for name in ('train_qwen_preferences.py', 'train_open_weight_lora.py', 'q8_gguf_base.py'):
            source = self.code_dir / name
            source.write_text(f'# synthetic {name}\n', encoding='utf-8')
            self.code_sources[name] = source
        self.versions = dict(torch='2.9.1', transformers='4.57.6', peft='0.17.1')
        self.expected = dict(kind=dpo.CACHE_KIND, manifest_sha256='m' * 64,
                             reference=dict(adapter_weights_sha256='w' * 64, source_manifest=dict(sha256='s' * 64)),
                             base=dict(model_id='tiny/qwen3', revision='rev', gguf_sha256='g' * 64),
                             tokenizer=dict(files={'tokenizer.json': 't' * 64}, vocab_size=len(self.tok)),
                             family='qwen3', max_tokens=1024, objective=dpo.OBJECTIVE,
                             engine=self.current_engine())

    def current_engine(self):
        with unittest.mock.patch.object(dpo, '__file__', str(self.code_sources['train_qwen_preferences.py'])), \
                unittest.mock.patch.object(dpo.T, '__file__', str(self.code_sources['train_open_weight_lora.py'])):
            return dpo.engine_identity(self.versions, 'cpu', torch.float32)

    def shard(self, ids, **override):
        shard = dict(self.expected, status='complete', ids=list(ids),
                     scores={pid: dict(chosen=-3.5 - i, rejected=-7.25, pair_sha256=self.digests[pid]) for i, pid in enumerate(ids)})
        shard.update(override)
        return json.loads(json.dumps(shard))  # as read back from disk

    def test_exact_coverage_across_shards_passes_and_returns_scores(self):
        scores = dpo.verify_cache_shards([self.shard(['a', 'b']), self.shard(['c'])], self.expected, self.digests)
        self.assertEqual(scores, {'a': (-3.5, -7.25), 'b': (-4.5, -7.25), 'c': (-3.5, -7.25)})

    def test_cached_engine_without_current_head_marker_is_rejected(self):
        for marker in (None, 'previous_answer_head'):
            with self.subTest(marker=marker):
                shard = self.shard(['a', 'b', 'c'])
                if marker is None:
                    del shard['engine']['answer_head']
                else:
                    shard['engine']['answer_head'] = marker
                with self.assertRaisesRegex(ValueError, 'engine differs from the current run'):
                    dpo.verify_cache_shards([shard], self.expected, self.digests)

    def test_source_changes_invalidate_cached_reference_scores(self):
        cached = self.shard(['a', 'b', 'c'])
        for name, source in self.code_sources.items():
            with self.subTest(source=name):
                original = source.read_text(encoding='utf-8')
                try:
                    source.write_text(original + '# changed scoring source\n', encoding='utf-8')
                    current = dict(self.expected, engine=self.current_engine())
                    with self.assertRaisesRegex(ValueError, 'engine differs from the current run'):
                        dpo.verify_cache_shards([cached], current, self.digests)
                    refreshed = self.shard(['a', 'b', 'c'], engine=current['engine'])
                    self.assertEqual(dpo.verify_cache_shards([refreshed], current, self.digests),
                                     {'a': (-3.5, -7.25), 'b': (-4.5, -7.25), 'c': (-5.5, -7.25)})
                finally:
                    source.write_text(original, encoding='utf-8')

    def test_stale_or_foreign_bindings_are_rejected(self):
        complete = [self.shard(['a', 'b', 'c'])]
        stale_reference = dict(self.expected['reference'], adapter_weights_sha256='x' * 64)
        stale_tokenizer = dict(self.expected['tokenizer'], vocab_size=len(self.tok) + 1)
        other_encoding = dict(self.digests, b=hashlib.sha256(b'b-reencoded').hexdigest())
        cases = {
            'manifest': ([self.shard(['a', 'b', 'c'], manifest_sha256='n' * 64)], self.expected, self.digests),
            'reference': ([self.shard(['a', 'b', 'c'], reference=stale_reference)], self.expected, self.digests),
            'base': ([self.shard(['a', 'b', 'c'], base=dict(self.expected['base'], revision='other'))], self.expected, self.digests),
            'tokenizer': ([self.shard(['a', 'b', 'c'], tokenizer=stale_tokenizer)], self.expected, self.digests),
            'max_tokens': ([self.shard(['a', 'b', 'c'], max_tokens=512)], self.expected, self.digests),
            'engine code': ([self.shard(['a', 'b', 'c'], engine=dict(self.expected['engine'], code={'train_qwen_preferences.py': 'z' * 64}))],
                            self.expected, self.digests),
            'engine versions': ([self.shard(['a', 'b', 'c'], engine=dict(self.expected['engine'], versions=dict(torch='other')))],
                                self.expected, self.digests),
            'pair tokens': (complete, self.expected, other_encoding),
            'missing id': ([self.shard(['a', 'b'])], self.expected, self.digests),
            'duplicate id': ([self.shard(['a', 'b']), self.shard(['b', 'c'])], self.expected, self.digests),
            'unknown id': ([self.shard(['a', 'b', 'c']), dict(self.shard(['a']), ids=['z'], scores={'z': dict(chosen=-1.0, rejected=-2.0, pair_sha256='q')})],
                           self.expected, self.digests),
            'incomplete': ([self.shard(['a', 'b', 'c'], status='partial')], self.expected, self.digests),
        }
        for name, (shards, expected, digests) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                dpo.verify_cache_shards(shards, expected, digests)
        bad_score = self.shard(['a', 'b', 'c'])
        bad_score['scores']['a']['chosen'] = None
        with self.subTest('non-finite'), self.assertRaises(ValueError):
            dpo.verify_cache_shards([bad_score], self.expected, self.digests)

    def test_pair_digest_binds_exact_tokens_and_labels(self):
        encoded = dpo.encode_pair(self.tok, dict(id='p', task='t', system=None, user='색은?', history=[],
                                                 chosen='자주색', rejected='은색', file='pairs.jsonl'), 10000)
        digest = dpo.pair_digest(encoded)
        self.assertEqual(digest, dpo.pair_digest(json.loads(json.dumps(encoded))))
        changed = json.loads(json.dumps(encoded))
        changed['rejected']['labels'][-1] = -100  # same tokens, EOS no longer supervised
        self.assertNotEqual(digest, dpo.pair_digest(changed))


class InitialAgreementTests(unittest.TestCase):
    """The untouched policy must reproduce the reference cache before the first update."""

    @staticmethod
    def pair(pid, chosen, rejected, dc=0.0, dr=0.0):
        return dict(id=pid, policy=dict(chosen=chosen + dc, rejected=rejected + dr), reference=dict(chosen=chosen, rejected=rejected))

    def test_numerical_noise_passes_and_is_recorded(self):
        check = dpo.check_initial_agreement([self.pair('a', -12.5, -20.0, dc=2e-4), self.pair('b', -3.0, -9.0, dr=-5e-4)], 0.1)
        self.assertAlmostEqual(check['initial_cache_max_abs_error'], 5e-4)
        self.assertEqual(check['worst_pair'], 'b')
        self.assertEqual(check['tolerance'], dpo.INITIAL_CACHE_MAX_ABS_ERROR)
        self.assertAlmostEqual(check['initial_policy_margin'], (7.5 + 6.0) / 2, places=3)
        self.assertAlmostEqual(check['initial_preference_margin'], (2e-4 + 5e-4) / 2)
        self.assertAlmostEqual(check['initial_preference_loss'], math.log(2), places=3)
        self.assertEqual([r['id'] for r in check['pairs']], ['a', 'b'])

    def test_disagreement_or_nonfinite_scores_are_rejected(self):
        cases = {
            'above tolerance': [self.pair('a', -12.5, -20.0), self.pair('b', -3.0, -9.0, dc=2 * dpo.INITIAL_CACHE_MAX_ABS_ERROR)],
            'nan policy': [self.pair('a', -12.5, -20.0, dr=float('nan'))],
            'inf reference': [dict(id='a', policy=dict(chosen=-1.0, rejected=-2.0), reference=dict(chosen=-math.inf, rejected=-2.0))],
        }
        for name, pairs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                dpo.check_initial_agreement(pairs, 0.1)


class HyperparameterTests(unittest.TestCase):
    """--beta/--lr must be finite and positive and --valid-every >= 1, refused by the parser before anything is read."""

    def test_invalid_values_exit_before_reading_the_manifest(self):
        import contextlib, io, unittest.mock
        with tempfile.TemporaryDirectory() as temp:
            common = ['train_qwen_preferences.py', '--mode', 'train', '--model', temp, '--model-id', 'x', '--model-revision', 'r',
                      '--base-gguf', f'{temp}/base.gguf', '--base-gguf-sha256', 'g' * 64, '--manifest', f'{temp}/absent.json',
                      '--out', f'{temp}/run', '--cache', f'{temp}/absent-cache.json', '--cache-sha256', 'c' * 64,
                      '--checkpoint-every', '1']
            for name, extra in {'negative beta': ['--beta', '-0.1'], 'zero beta': ['--beta', '0'], 'nan lr': ['--lr', 'nan'],
                                'inf lr': ['--lr', 'inf'], 'zero lr': ['--lr', '0'], 'valid-every 0': ['--valid-every', '0'],
                                'replay-micro-batch 0': ['--replay-micro-batch', '0'],
                                'replay-micro-batch -1': ['--replay-micro-batch', '-1']}.items():
                with self.subTest(name), unittest.mock.patch.object(sys, 'argv', common + extra), \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stop:
                    dpo.main()
                self.assertEqual(stop.exception.code, 2)  # argparse error, not a FileNotFoundError from the absent manifest


class SourceMembershipTests(unittest.TestCase):
    def test_pairs_must_be_source_train_rows_with_the_source_answer(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            source = base / 'train.jsonl'
            source.write_text(json.dumps({'id': 'r1', 'task': 'color', 'prompt': '색은?', 'answer': '자주색'}, ensure_ascii=False) + '\n')
            entry = dict(path='train.jsonl', sha256=hashlib.sha256(source.read_bytes()).hexdigest(), adapter='prompt_answer')
            rows, ambiguous = dpo.source_rows({'train': [entry]}, base)
            self.assertEqual(set(rows), {'r1'}); self.assertEqual(ambiguous, set())

            def manifest_for(pairs, name):
                path = base / name
                path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in pairs))
                return {'train': [dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), adapter=dpo.PAIR_ADAPTER)]}

            good = {'id': 'r1', 'task': 'color', 'user': '색은?', 'chosen': '자주색', 'rejected': '은색', 'reviewer': 'x'}
            admitted, = dpo.load_pairs(manifest_for([good], 'ok.jsonl'), base, rows, ambiguous)
            self.assertEqual((admitted['id'], admitted['system'], admitted['history'], admitted['chosen']), ('r1', None, [], '자주색'))
            rejected = {
                'chosen differs': dict(good, chosen='자주색.'),
                'prompt differs': dict(good, user='색깔은?'),
                'unknown id': dict(good, id='r2'),
                'identical answers': dict(good, rejected='자주색'),
                'empty rejected': dict(good, rejected=''),
                'system injected': dict(good, system='다르게 답해.'),
            }
            for name, pair in rejected.items():
                with self.subTest(name), self.assertRaises(ValueError):
                    dpo.load_pairs(manifest_for([pair], f'{name}.jsonl'.replace(' ', '_')), base, rows, ambiguous)
            with self.subTest('duplicate id'), self.assertRaises(ValueError):
                dpo.load_pairs(manifest_for([good, dict(good, rejected='청록색')], 'dup.jsonl'), base, rows, ambiguous)


class ReplayContractTests(unittest.TestCase):
    """manifest.replay is required and must pin exactly the reference's source manifest, a finite positive weight and
    epochs == 1; every deviation is a ValueError from the pure check that main runs before any weight is loaded."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        (self.base / 'manifests').mkdir()
        self.source = self.base / 'manifests' / 'source.json'
        self.source.write_text('{"train": []}')
        self.sha = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.good = dict(source_manifest=dict(path='manifests/source.json', sha256=self.sha), weight=1.0, epochs=1)

    def check(self, replay):
        manifest = {'replay': replay} if replay is not None else {}
        return dpo.replay_contract(manifest, self.base, self.source, self.sha)

    def test_exact_block_is_normalised_to_the_resolved_pin(self):
        got = self.check(self.good)
        self.assertEqual(got, dict(source_manifest=dict(path=str(self.source.resolve()), sha256=self.sha), weight=1.0, epochs=1))
        self.assertEqual(self.check(dict(self.good, weight=2)), dict(got, weight=2.0))  # ints are fine, normalised to float
        absolute = dict(self.good, source_manifest=dict(path=str(self.source), sha256=self.sha))
        self.assertEqual(self.check(absolute), got)  # same file by absolute path

    def test_missing_or_deviating_blocks_are_rejected(self):
        other = self.base / 'manifests' / 'other.json'
        other.write_text('{"train": []}')  # same bytes as the source, so the same sha256: the PATH must still match
        cases = {
            'absent': None,
            'not an object': 'replay',
            'empty': {},
            'no source pin': dict(self.good, source_manifest=None),
            'other file same sha': dict(self.good, source_manifest=dict(path='manifests/other.json', sha256=self.sha)),
            'other sha same file': dict(self.good, source_manifest=dict(path='manifests/source.json', sha256='0' * 64)),
            'pin without sha': dict(self.good, source_manifest=dict(path='manifests/source.json')),
            'zero weight': dict(self.good, weight=0),
            'negative weight': dict(self.good, weight=-1.0),
            'nan weight': dict(self.good, weight=float('nan')),
            'inf weight': dict(self.good, weight=float('inf')),
            'string weight': dict(self.good, weight='1.0'),
            'bool weight': dict(self.good, weight=True),
            'absent weight': {k: v for k, v in self.good.items() if k != 'weight'},
            'two epochs': dict(self.good, epochs=2),
            'zero epochs': dict(self.good, epochs=0),
            'float epochs': dict(self.good, epochs=1.0),
            'bool epochs': dict(self.good, epochs=True),
            'absent epochs': {k: v for k, v in self.good.items() if k != 'epochs'},
        }
        for name, replay in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.check(replay)


class ReplayRowsTests(unittest.TestCase):
    """Replay is every canonical source TRAIN row exactly once in source order; duplicate ids are never collapsed."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def entry(self, name, rows, **extra):
        path = self.base / name
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), adapter='prompt_answer', **extra)

    def test_all_source_train_rows_once_in_manifest_order_with_provenance(self):
        a = self.entry('a.jsonl', [{'id': 'r1', 'task': 'color', 'prompt': '색은?', 'answer': '자주색'},
                                   {'id': 'r2', 'task': 'color', 'prompt': '다른 색은?', 'answer': '은색',
                                    'history': [{'user': '기억해.', 'assistant': '네.'}]}], epochs=3)
        b = self.entry('b.jsonl', [{'id': 'r3', 'task': 'size', 'prompt': '크기는?', 'answer': '크다'}])
        valid = self.entry('valid.jsonl', [{'id': 'v1', 'task': 'color', 'prompt': '검증?', 'answer': '검증'}])
        rows = dpo.replay_rows({'train': [a, b], 'valid': [valid]}, self.base)
        self.assertEqual([r['id'] for r in rows], ['r1', 'r2', 'r3'])  # source epochs=3 do not oversample; valid never joins
        self.assertEqual([r['file'] for r in rows], [str(self.base / 'a.jsonl')] * 2 + [str(self.base / 'b.jsonl')])
        self.assertEqual((rows[1]['system'], rows[1]['user'], rows[1]['history'], rows[1]['answer']),
                         (None, '다른 색은?', [{'user': '기억해.', 'assistant': '네.'}], '은색'))
        self.assertEqual(rows, [r for r in rows if r['id'] in dpo.source_rows({'train': [a, b]}, self.base)[0]])

    def test_duplicate_identical_or_differing_ids_and_missing_ids_are_rejected(self):
        row = {'id': 'r1', 'task': 'color', 'prompt': '색은?', 'answer': '자주색'}
        cases = {
            'identical twice in one file': [self.entry('same.jsonl', [row, row])],
            'identical across files': [self.entry('x.jsonl', [row]), self.entry('y.jsonl', [row])],
            'differing twice': [self.entry('diff.jsonl', [row, dict(row, answer='은색')])],
            'missing id': [self.entry('noid.jsonl', [{'task': 'color', 'prompt': '색은?', 'answer': '자주색'}])],
            'empty': [self.entry('empty.jsonl', [])],
        }
        for name, train in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                dpo.replay_rows({'train': train}, self.base)
        # The pair-side view tolerates identical repeats (it only flags differing ones); replay must not inherit that.
        rows, ambiguous = dpo.source_rows({'train': [self.entry('same2.jsonl', [row, row])]}, self.base)
        self.assertEqual((set(rows), ambiguous), ({'r1'}, set()))


class PartitionTests(unittest.TestCase):
    """Pairs: ceil(n / batch) consecutive slices, last partial. Replay: floor(i*N/S) bounds over the FULL epoch S;
    --max-steps only takes a prefix of those batches, so the partition is invariant under the smoke bound."""

    def test_91_pairs_in_batches_of_4_give_23_steps_with_a_final_partial_3(self):
        order = [f'p{i}' for i in range(91)]
        batches = dpo.pair_batches(order, 4)
        self.assertEqual(len(batches), math.ceil(91 / 4))
        self.assertEqual([len(b) for b in batches], [4] * 22 + [3])
        self.assertEqual([p for b in batches for p in b], order)
        self.assertEqual(dpo.pair_batches(order, 91), [order])
        self.assertEqual([len(b) for b in dpo.pair_batches(order, 100)], [91])
        with self.assertRaises(ValueError):
            dpo.pair_batches(order, 0)

    def test_288_rows_over_23_full_epoch_steps_cover_everything_once_and_a_2_step_smoke_takes_25(self):
        bounds = dpo.replay_partition(288, 23)
        self.assertEqual(len(bounds), 23)
        self.assertEqual(bounds[0][0], 0); self.assertEqual(bounds[-1][1], 288)
        self.assertTrue(all(stop > start for start, stop in bounds))  # nonempty
        self.assertTrue(all(bounds[i][1] == bounds[i + 1][0] for i in range(22)))  # contiguous, disjoint
        self.assertEqual(bounds, [(math.floor(i * 288 / 23), math.floor((i + 1) * 288 / 23)) for i in range(23)])
        sizes = [stop - start for start, stop in bounds]
        self.assertEqual(sum(sizes), 288); self.assertEqual(set(sizes), {12, 13})
        order = list(range(288))
        prefix = [order[start:stop] for start, stop in bounds][:2]
        self.assertEqual(sum(len(b) for b in prefix), 25)  # a 2-step smoke consumes rows 0..24, never all 288
        self.assertEqual(prefix, [order[start:stop] for start, stop in bounds[:2]])  # same batches as the full run
        self.assertNotEqual(bounds[:2], dpo.replay_partition(288, 2))  # a re-partition over 2 steps would eat everything

    def test_boundaries(self):
        self.assertEqual(dpo.replay_partition(5, 5), [(i, i + 1) for i in range(5)])  # N == S: singletons
        self.assertEqual(dpo.replay_partition(7, 1), [(0, 7)])  # one step carries every row
        self.assertEqual(dpo.replay_partition(7, 3), [(0, 2), (2, 4), (4, 7)])
        for n_rows, steps in ((4, 5), (0, 1), (22, 23)):
            with self.subTest(n_rows=n_rows, steps=steps), self.assertRaises(ValueError):
                dpo.replay_partition(n_rows, steps)  # fewer rows than full-epoch steps: an empty batch is never allowed
        with self.assertRaises(ValueError):
            dpo.replay_partition(5, 0)


class CombinedStepTests(unittest.TestCase):
    """One update = mean DPO loss over the pair batch + weight * mean over the replay rows of each row's answer+EOS
    token-mean cross-entropy, accumulated one pair, then one length-sorted replay micro-batch at a time: its gradient
    equals that of the single combined expression for any micro-batch size (1, 4, a ragged last one, one big batch)."""

    @classmethod
    def setUpClass(cls):
        cls.tok = local_tokenizer()


    def batches(self):
        pairs = [dpo.encode_pair(self.tok, dict(id=pid, task='t', system=None, user=user, history=[], chosen=chosen,
                                                rejected=rejected, file='pairs.jsonl'), 10000)
                 for pid, user, chosen, rejected in (('p1', '색은?', '자주색입니다.', '은색'), ('p2', '크기는?', '크다', '작다'))]
        replay = [dpo.T.encode(self.tok, dict(id=rid, task='t', system=None, user=user, history=history, answer=answer,
                                              file='train.jsonl'), 'qwen3', 10000)
                  for rid, user, history, answer in (('r1', '색은?', [], '자주색입니다.'),
                                                     ('r2', '다른 색은?', [{'user': '기억해.', 'assistant': '네.'}], '은색'),
                                                     ('r3', '크기는?', [], '크다'))]
        self.assertTrue(all(x is not None for x in replay))
        return pairs, replay

    def test_accumulated_gradient_equals_the_combined_mean_objective_and_the_numbers_are_separate(self):
        model = tiny_lora_qwen3(len(self.tok))
        pairs, replay = self.batches()
        beta, weight = 0.1, 0.7
        with torch.no_grad():  # tied reference, perturbed so the DPO term is not at its symmetric point
            shifts = [(0.5, 0.0), (0.0, 0.25)]  # pair 1 margin -0.5 (not preferred), pair 2 margin +0.25 (preferred)
            reference = [tuple(v + d for v, d in zip(dpo.pair_logprobs(model, e, 'cpu', self.tok.pad_token_id).tolist(), shift))
                         for e, shift in zip(pairs, shifts)]
        # Single-graph full-logits reference: mean DPO + weight * mean over rows of (-summed logprob / answer tokens).
        model.zero_grad(set_to_none=True)
        dpo_terms = []
        for e, ref in zip(pairs, reference):
            policy = torch.stack([manual_logprob(model, e['chosen'])[0], manual_logprob(model, e['rejected'])[0]])
            dpo_terms.append(dpo.dpo_loss(policy, ref, beta)[0])
        ce_terms = []
        for item in replay:
            logprob, n = manual_logprob(model, item)
            ce_terms.append(-logprob / n)
        want_pref, want_ce = torch.stack(dpo_terms).mean(), torch.stack(ce_terms).mean()
        (want_pref + weight * want_ce).backward()
        want = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        self.assertTrue(want)
        model.zero_grad(set_to_none=True)
        numbers = dpo.accumulate_step(model, [(f'p{i}', e, ref) for i, (e, ref) in enumerate(zip(pairs, reference), 1)],
                                      [(f'r{i}', item) for i, item in enumerate(replay, 1)], beta, weight, 'cpu', self.tok.pad_token_id)
        got = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        self.assertEqual(set(got), set(want))
        for name in want:
            self.assertTrue(torch.allclose(got[name], want[name], atol=1e-5), name)
        self.assertAlmostEqual(numbers['preference_loss'], want_pref.item(), places=5)
        self.assertAlmostEqual(numbers['replay_ce'], want_ce.item(), places=5)
        self.assertAlmostEqual(numbers['train_loss'], want_pref.item() + weight * want_ce.item(), places=5)
        self.assertNotAlmostEqual(numbers['preference_loss'], numbers['train_loss'], places=3)  # replay really contributes
        self.assertEqual((numbers['pairs'], numbers['replay_rows']), (2, 3))
        self.assertEqual(numbers['preferred_fraction'], 0.5)  # margins +0.5 and -0.25 under the tied-then-shifted reference
        self.assertAlmostEqual(numbers['reward_margin'], (-0.5 + 0.25) / 2, places=4)
        # The replay term is the token-MEAN cross-entropy, not the summed likelihood the DPO side uses.
        summed = sum(-manual_logprob(model, item)[0].item() for item in replay) / len(replay)
        self.assertNotAlmostEqual(numbers['replay_ce'], summed, places=2)

    def test_partial_batches_use_their_actual_counts_and_nonfinite_or_empty_batches_stop_before_further_work(self):
        model = tiny_lora_qwen3(len(self.tok))
        pairs, replay = self.batches()
        with torch.no_grad():
            reference = [tuple(dpo.pair_logprobs(model, e, 'cpu', self.tok.pad_token_id).tolist()) for e in pairs]
        model.zero_grad(set_to_none=True)
        one = dpo.accumulate_step(model, [('p1', pairs[0], reference[0])], [('r1', replay[0])], 0.1, 1.0, 'cpu', self.tok.pad_token_id)
        self.assertAlmostEqual(one['preference_loss'], math.log(2), places=5)  # a single tied pair: no /batch_pairs leftover
        _, n = manual_logprob(model, replay[0])
        self.assertAlmostEqual(one['replay_ce'], -manual_logprob(model, replay[0])[0].item() / n, places=5)
        for name, pair_batch, replay_batch in (('no pairs', [], [('r1', replay[0])]), ('no replay', [('p1', pairs[0], reference[0])], [])):
            with self.subTest(name), self.assertRaises(ValueError):
                dpo.accumulate_step(model, pair_batch, replay_batch, 0.1, 1.0, 'cpu', self.tok.pad_token_id)
        import unittest.mock
        real, calls = dpo.sequence_losses, []

        def nan_after_the_pair(model_, items, *args, **kwargs):  # the pair forward is real; every replay forward is NaN
            calls.append(len(items))
            if len(calls) == 1:
                return real(model_, items, *args, **kwargs)
            return torch.full((len(items),), float('nan'), requires_grad=True), torch.ones(len(items), dtype=torch.long)
        with unittest.mock.patch.object(dpo, 'sequence_losses', nan_after_the_pair), self.assertRaises(FloatingPointError):
            dpo.accumulate_step(model, [('p1', pairs[0], reference[0])], [('r1', replay[0]), ('r2', replay[1])], 0.1, 1.0, 'cpu',
                                self.tok.pad_token_id, replay_micro_batch=1)
        self.assertEqual(calls, [2, 1])  # one pair forward, then stopped at the first non-finite replay micro-batch
        with unittest.mock.patch.object(dpo, 'pair_logprobs', lambda *a, **k: torch.tensor([float('nan'), 0.0], requires_grad=True)), \
                self.assertRaises(FloatingPointError):
            dpo.accumulate_step(model, [('p1', pairs[0], reference[0])], [('r1', replay[0])], 0.1, 1.0, 'cpu', self.tok.pad_token_id)

    def ragged_replay(self):
        """Five rows of varying encoded length and answer length (history, a one-word and a long answer), given in an
        order that is NOT sorted by length, so micro-batches of 4 pad and the last one is ragged."""
        replay = [dpo.T.encode(self.tok, dict(id=rid, task='t', system=None, user=user, history=history, answer=answer,
                                              file='train.jsonl'), 'qwen3', 10000)
                  for rid, user, history, answer in (
                      ('r1', '색은?', [], '자주색입니다. 표시는 언제나 자주색으로 그려 주세요.'),
                      ('r2', '다른 색은?', [{'user': '기억해.', 'assistant': '네.'}], '은색'),
                      ('r3', '크기는?', [], '크다'),
                      ('r4', '모양은 어떻게 할까요?', [{'user': '동그랗게 해 줘.', 'assistant': '동그랗게 기억합니다.'}], '동그란 모양으로 합니다.'),
                      ('r5', '이름이 뭐예요?', [], '트리드입니다.'))]
        self.assertTrue(all(x is not None for x in replay))
        lengths = [len(x['ids']) for x in replay]
        counts = [sum(l != -100 for l in x['labels']) for x in replay]
        self.assertGreater(lengths[0], lengths[2]); self.assertNotEqual(lengths, sorted(lengths))  # the long answer precedes the short one
        self.assertGreater(counts[0], counts[2] + 3)  # answer+EOS counts really differ: a token mean would favour r1
        self.assertGreaterEqual(len(set(lengths)), 3)
        return [(f'r{i}', item) for i, item in enumerate(replay, 1)]

    def test_replay_micro_batches_are_length_sorted_chunks_of_the_update_only_and_sizes_below_1_are_rejected(self):
        replay = self.ragged_replay()
        for micro in (1, 2, 4, 5, 9):
            with self.subTest(micro=micro):
                chunks = dpo.replay_microbatches(replay, micro)
                flat = [rid for chunk in chunks for rid, _ in chunk]
                self.assertEqual(sorted(flat), sorted(rid for rid, _ in replay))  # exactly the update's rows, once each
                self.assertEqual([len(c) for c in chunks], [micro] * (5 // micro) + ([5 % micro] if 5 % micro else []))
                lengths = [len(item['ids']) for chunk in chunks for _, item in chunk]
                self.assertEqual(lengths, sorted(lengths))
        # Ties keep the assignment order (stable sort): two copies of the same row stay in their given order.
        tied = [('a', replay[0][1]), ('b', replay[0][1]), ('c', replay[2][1])]  # r3 is strictly shorter than r1
        self.assertEqual([rid for chunk in dpo.replay_microbatches(tied, 2) for rid, _ in chunk], ['c', 'a', 'b'])
        for micro in (0, -1):
            with self.subTest(micro=micro), self.assertRaises(ValueError):
                dpo.replay_microbatches(replay, micro)
        # accumulate_step refuses the size before any forward.
        import unittest.mock
        model = tiny_lora_qwen3(len(self.tok))
        pairs, _ = self.batches()
        forwards = []
        with unittest.mock.patch.object(dpo, 'sequence_losses', lambda *a, **k: forwards.append(1)), self.assertRaises(ValueError):
            dpo.accumulate_step(model, [('p1', pairs[0], (0.0, 0.0))], replay, 0.1, 1.0, 'cpu', self.tok.pad_token_id,
                                replay_micro_batch=0)
        self.assertEqual(forwards, [])

    def test_micro_batch_sizes_1_4_ragged_and_one_large_give_the_per_example_mean_objective_and_identical_gradients(self):
        model = tiny_lora_qwen3(len(self.tok))
        pairs, _ = self.batches()
        replay = self.ragged_replay()
        beta, weight = 0.1, 0.7
        with torch.no_grad():
            shifts = [(0.5, 0.0), (0.0, 0.25)]
            reference = [tuple(v + d for v, d in zip(dpo.pair_logprobs(model, e, 'cpu', self.tok.pad_token_id).tolist(), shift))
                         for e, shift in zip(pairs, shifts)]
        pair_batch = [(f'p{i}', e, ref) for i, (e, ref) in enumerate(zip(pairs, reference), 1)]
        # Full-logits, unpadded, one row at a time: mean DPO + weight * mean over rows of (-summed logprob / answer tokens).
        model.zero_grad(set_to_none=True)
        dpo_terms = [dpo.dpo_loss(torch.stack([manual_logprob(model, e['chosen'])[0], manual_logprob(model, e['rejected'])[0]]), ref, beta)[0]
                     for _, e, ref in pair_batch]
        per_example = []
        answer_counts = []
        for _, item in replay:
            logprob, n = manual_logprob(model, item)
            per_example.append(-logprob / n)
            answer_counts.append(float(n))
        want_pref, want_ce = torch.stack(dpo_terms).mean(), torch.stack(per_example).mean()
        (want_pref + weight * want_ce).backward()
        want = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        self.assertTrue(want)
        # The wrong objectives this must NOT be: equal weight per micro-batch (4 + 1 rows) or per token (long answers win).
        order = sorted(range(len(replay)), key=lambda i: len(replay[i][1]['ids']))
        ordered_losses = torch.stack([per_example[i] for i in order])
        chunk_mean = torch.stack([chunk.mean() for chunk in ordered_losses.split(4)]).mean()
        counts = torch.tensor(answer_counts)
        token_mean = (torch.stack(per_example) * counts).sum() / counts.sum()
        self.assertNotAlmostEqual(chunk_mean.item(), want_ce.item(), places=4)
        self.assertNotAlmostEqual(token_mean.item(), want_ce.item(), places=4)
        self.addCleanup(setattr, dpo.T, 'LOGIT_CHUNK', dpo.T.LOGIT_CHUNK)
        dpo.T.LOGIT_CHUNK = 5  # several chunks over the union of supervised positions, a ragged last one
        for micro in (1, 4, 2, 5, 9):  # 1: one row per forward; 4: padded 4 + ragged 1; 2: 2+2+1; 5 and 9: one batch
            with self.subTest(micro=micro):
                model.zero_grad(set_to_none=True)
                numbers = dpo.accumulate_step(model, pair_batch, replay, beta, weight, 'cpu', self.tok.pad_token_id,
                                              replay_micro_batch=micro)
                got = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                self.assertEqual(set(got), set(want))
                for name in want:
                    self.assertTrue(torch.allclose(got[name], want[name], atol=1e-5), name)
                self.assertAlmostEqual(numbers['preference_loss'], want_pref.item(), places=5)
                self.assertAlmostEqual(numbers['replay_ce'], want_ce.item(), places=5)
                self.assertAlmostEqual(numbers['train_loss'], want_pref.item() + weight * want_ce.item(), places=5)
                self.assertEqual((numbers['pairs'], numbers['replay_rows'], numbers['replay_micro_batches']),
                                 (2, 5, math.ceil(5 / micro)))
        # Generic (full-logits) family through the same batched path agrees too.
        model.zero_grad(set_to_none=True)
        generic = dpo.accumulate_step(model, pair_batch, replay, beta, weight, 'cpu', self.tok.pad_token_id, 'generic', 4)
        got = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
        for name in want:
            self.assertTrue(torch.allclose(got[name], want[name], atol=1e-5), name)
        self.assertAlmostEqual(generic['replay_ce'], want_ce.item(), places=5)


if __name__ == '__main__':
    unittest.main()
