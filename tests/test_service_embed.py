import tempfile
import unittest
from pathlib import Path

import numpy as np

from trid.service.embed import load_dense, write_dense_f32


class DenseCopyTest(unittest.TestCase):
    """SITE-MEMORY-022: the memory-mapped float32 copy must give the same vectors as converting the float16 index file,
    and must never be served once the float16 file it was made from has changed."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(1)
        np.save(self.root / 'dense.f16.npy', rng.standard_normal((7, 16)).astype(np.float16))

    def tearDown(self):
        self.tmp.cleanup()

    def test_copy_is_exact_and_dropped_when_the_source_changes(self):
        expected = np.load(self.root / 'dense.f16.npy').astype(np.float32)
        self.assertIsInstance(load_dense(self.root), np.ndarray)  # no copy yet: converted in memory
        write_dense_f32(self.root)
        mapped = load_dense(self.root)
        self.assertIsInstance(mapped, np.memmap)
        self.assertEqual(mapped.dtype, np.float32)
        self.assertTrue(np.array_equal(mapped, expected))
        query = np.ones(16, np.float32) / 4
        self.assertTrue(np.array_equal(mapped @ query, expected @ query))
        changed = expected[::-1].astype(np.float16)  # index rebuilt: same shape, different vectors
        np.save(self.root / 'dense.f16.npy', changed)
        reloaded = load_dense(self.root)
        self.assertNotIsInstance(reloaded, np.memmap)
        self.assertTrue(np.array_equal(reloaded, changed.astype(np.float32)))


if __name__ == '__main__':
    unittest.main()
