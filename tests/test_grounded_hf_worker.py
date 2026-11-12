"""Behavioral worker guards; no downloads, cloud requests, or GPU execution."""
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import re
import runpy
import sys
import tempfile
from types import SimpleNamespace
import tarfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/aws_eval'))
import grounded_hf_worker as W


class RuntimeBoundaryTests(unittest.TestCase):
    def test_unapproved_transitive_distribution_cannot_produce_runtime_receipt(self):
        distribution = lambda name, version: SimpleNamespace(metadata={'Name': name}, version=version)
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'output').mkdir()
            plan = {'runtime': {'packages': {'approved': '1'}}}
            installed = [distribution('approved', '1'), distribution('unapproved', '2')]
            with patch.object(importlib.metadata, 'distributions', return_value=installed), \
                 patch.object(W, 'native_runtime', side_effect=AssertionError('Must not initialize CUDA')):
                with self.assertRaisesRegex(RuntimeError, 'package inventory differs'):
                    W.runtime_stage(plan, temp)
            self.assertFalse((Path(temp) / 'output/runtime.json').exists())

    def test_root_cannot_initialize_native_runtime(self):
        with patch.object(W.os, 'geteuid', return_value=0):
            with self.assertRaisesRegex(RuntimeError, 'nonroot without cloud credentials'):
                W.native_runtime({}, '.')

    def test_inherited_cloud_credentials_cannot_initialize_native_runtime(self):
        with patch.object(W.os, 'geteuid', return_value=1001), \
             patch.dict(W.os.environ, {'AWS_SECRET_ACCESS_KEY': 'synthetic-unit-test'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'nonroot without cloud credentials'):
                W.native_runtime({}, '.')

class BootstrapObservationTests(unittest.TestCase):
    def test_observed_command_hash_must_match_explicit_expectation(self):
        with tempfile.TemporaryDirectory() as temp:
            executable = Path(temp) / 'aws'
            executable.write_bytes(b'changed synthetic program; never executed')
            admission = {'expected_commands': {'aws': {'path': str(executable), 'probe_args': ['--version'],
                                                       'sha256': '0' * 64}}}
            with patch.object(W.shutil, 'which', return_value=str(executable)):
                with self.assertRaisesRegex(RuntimeError, 'Expected base-command hash differs'):
                    W.observe_prerequisites(admission, 1000)

    def test_missing_expected_command_is_not_postboot_proof(self):
        admission = {'expected_commands': {'dpkg-deb': {'path': '/expected/dpkg-deb', 'probe_args': ['--version']}}}
        with patch.object(W.shutil, 'which', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Expected base-command path differs'):
                W.observe_prerequisites(admission, 1000)

    def test_upstream_setuid_mode_is_not_preserved_when_copying_helper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            payload = b'synthetic helper bytes; never executed'
            archive_path = root / 'data.tar'
            with tarfile.open(archive_path, 'w') as archive:
                member = tarfile.TarInfo('./usr/bin/nvidia-modprobe')
                member.mode = 0o4755
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            pin = dict(max_unpacked_bytes=1024**2, member=dict(path='usr/bin/nvidia-modprobe',
                       bytes=len(payload), sha256=W.hashlib.sha256(payload).hexdigest()))
            target = W.extract_bootstrap_member(archive_path, pin, root / 'helper')
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(target.stat().st_mode & 0o7777, 0o555)



class FixtureTokenizer:
    """Deterministic tokenizer input double; the trainer's schema/encode code is real."""
    eos_token = '\x00'
    eos_token_id = 0

    def __init__(self, suffix):
        self.suffix = suffix

    def __call__(self, text, add_special_tokens=False):
        pieces = re.findall(r' x|[\s\S]', text)
        return {'input_ids': [0 if piece == self.eos_token else 257 if piece == ' x' else ord(piece) + 1
                              for piece in pieces]}

    def apply_chat_template(self, messages, **kwargs):
        return ''.join('<' + message['role'] + '>' + message['content'] for message in messages) + '<assistant>' + self.suffix


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('torch', 'transformers', 'tokenizers')),
                     'Real native trainer imports require installed torch/transformers/tokenizers; no installation performed')
class GroundedSyntheticCapacityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = runpy.run_path(str(ROOT / 'scripts/train_open_weight_lora.py'), run_name='capacity_contract_test')

    def test_real_grounded_loader_encoder_and_builder_preserve_exact_eos_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = FixtureTokenizer(self.trainer['QWEN3_NO_THINK'])
            item = W.synthetic_capacity_item(tokenizer, self.trainer, temp, 8192, 768)
            source = Path(temp) / 'synthetic.jsonl'
            row = json.loads(source.read_text())
            self.assertEqual(set(row), {'id', 'task', 'split', 'messages', 'answer'})
            self.assertEqual([message['role'] for message in row['messages']], ['system', 'user'])
            self.assertIn('fictional', row['messages'][0]['content'].lower())
            self.assertEqual(len(item['ids']), 8192)
            self.assertEqual(item['labels'][:7424], [-100] * 7424)
            self.assertEqual(item['labels'][7424:], item['ids'][7424:])
            self.assertEqual(sum(value != -100 for value in item['labels']), 768)
            self.assertEqual(item['labels'][-1], tokenizer.eos_token_id)
            self.assertEqual(item['file'], str(source))
            self.assertEqual(item['task'], 'synthetic-capacity')

    def test_schema_mutation_is_rejected_by_actual_training_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = FixtureTokenizer(self.trainer['QWEN3_NO_THINK'])
            W.synthetic_capacity_item(tokenizer, self.trainer, temp, 8192, 768)
            source = Path(temp) / 'synthetic.jsonl'
            row = json.loads(source.read_text())
            del row['messages'][0]
            source.write_text(json.dumps(row) + '\n')
            entry = dict(path=source.name, sha256=W.digest(source), adapter='grounded_messages', split='train')
            with self.assertRaisesRegex(ValueError, 'exactly system and user messages'):
                self.trainer['build_items'](tokenizer, [entry], Path(temp), 'qwen3', 8192, 768, False)

    def test_unrepresentable_envelope_fails_without_truncation(self):
        with tempfile.TemporaryDirectory() as temp:
            tokenizer = FixtureTokenizer(self.trainer['QWEN3_NO_THINK'])
            with self.assertRaisesRegex(RuntimeError, 'cannot fit token envelope'):
                W.synthetic_capacity_item(tokenizer, self.trainer, temp, 8, 2)
            self.assertFalse((Path(temp) / 'synthetic.jsonl').exists())


if __name__ == '__main__':
    unittest.main()
