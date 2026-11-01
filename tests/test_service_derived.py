import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from service_fixtures import make_corpus  # noqa: E402

from trid.service import derived
from trid.service.embed import load_dense, write_dense_f32
from trid.service.index import Index, write_documents_jsonl
from trid.service.ingest import ingest_dir


class DerivedIndexFilesTest(unittest.TestCase):
    """SITE-MEMORY-022/023: a derived file may change how an index is loaded, never what is loaded. A stale, partial
    or damaged copy or record must give the same vectors and documents as reading the original files."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.root = tmp / 'index'
        Index.build(ingest_dir(make_corpus(tmp / 'docs'))).save(self.root)
        rng = np.random.default_rng(1)
        np.save(self.root / 'dense.f16.npy', rng.standard_normal((7, 16)).astype(np.float16))
        self.query = rng.standard_normal(16).astype(np.float32)

    def tearDown(self):
        self.tmp.cleanup()

    def assertVectors(self, source):
        expected = np.load(source).astype(np.float32)
        vectors = load_dense(self.root)
        self.assertTrue(np.array_equal(vectors, expected))
        self.assertTrue(np.array_equal(vectors @ self.query, expected @ self.query))

    def assertDocuments(self):
        """Index.load here equals Index.load of the same files without any derived file: documents, order, passages."""
        bare = Path(self.tmp.name) / 'bare'
        for f in self.root.rglob('*'):
            if f.is_file() and not f.name.startswith(('dense.f32', 'documents.jsonl')):
                target = bare / f.relative_to(self.root)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(f.read_bytes())
        loaded, plain = Index.load(self.root), Index.load(bare)
        self.assertEqual(list(loaded.documents.values()), list(plain.documents.values()))
        self.assertEqual(loaded.passages, plain.passages)
        self.assertEqual(loaded.texts, plain.texts)

    def test_current_copies_load_the_same_vectors_and_documents(self):
        write_dense_f32(self.root)
        write_documents_jsonl(self.root)
        self.assertTrue(derived.current(self.root / 'dense.f16.npy', self.root / 'dense.f32.npy', self.root / 'dense.f32.json'))
        self.assertTrue(derived.current(self.root / 'documents.json', self.root / 'documents.jsonl',
                                        self.root / 'documents.jsonl.json'))
        self.assertVectors(self.root / 'dense.f16.npy')
        self.assertDocuments()

    def test_copies_are_not_used_after_the_source_changes(self):
        write_dense_f32(self.root)
        write_documents_jsonl(self.root)
        changed = np.load(self.root / 'dense.f16.npy')[::-1].copy()  # index rebuilt: same shape, other vectors
        np.save(self.root / 'dense.f16.npy', changed)
        documents = json.loads((self.root / 'documents.json').read_text(encoding='utf-8'))
        documents[0]['title'] = documents[0]['title'] + ' (개정)'
        (self.root / 'documents.json').write_text(json.dumps(documents, ensure_ascii=False), encoding='utf-8')
        self.assertVectors(self.root / 'dense.f16.npy')
        self.assertEqual(next(iter(Index.load(self.root).documents.values())).title, documents[0]['title'])

    def test_partial_or_damaged_copies_and_records_fall_back_to_the_source(self):
        write_dense_f32(self.root)
        write_documents_jsonl(self.root)
        for name in ('dense.f32.npy', 'documents.jsonl'):  # interrupted copy: half the bytes
            data = (self.root / name).read_bytes()
            (self.root / name).write_bytes(data[:len(data) // 2])
        self.assertVectors(self.root / 'dense.f16.npy')
        self.assertDocuments()
        write_dense_f32(self.root)
        write_documents_jsonl(self.root)
        for name in ('dense.f32.json', 'documents.jsonl.json'):  # damaged record
            (self.root / name).write_text('{"source_sha256": ')
        self.assertVectors(self.root / 'dense.f16.npy')
        self.assertDocuments()

    def test_producer_installs_nothing_when_the_source_changes_meanwhile(self):
        source = self.root / 'dense.f16.npy'

        def produce(path):
            path.write_bytes(b'copy of the old source')
            np.save(source, np.zeros((7, 16), np.float16))  # the index is rebuilt during the conversion
        with self.assertRaises(RuntimeError):
            derived.write(source, self.root / 'dense.f32.npy', self.root / 'dense.f32.json', produce)
        self.assertFalse((self.root / 'dense.f32.npy').exists() or (self.root / 'dense.f32.json').exists())
        self.assertVectors(source)


if __name__ == '__main__':
    unittest.main()
