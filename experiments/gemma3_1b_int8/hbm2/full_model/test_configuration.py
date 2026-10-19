"""Check configured tiling against actual packed data and the saved baseline."""
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from config import Architecture
from workload import Graph

HERE = Path(__file__).resolve().parent


class ConfigurationChecks(unittest.TestCase):
    def test_invalid_sizes(self):
        for values in (dict(array_rows=0), dict(array_cols=True), dict(core_hz=-1),
                       dict(operand_staging_bytes=31), dict(vector_staging_bytes=1),
                       dict(array_rows=256, array_cols=256), dict(dataflow='is'),
                       dict(gemm_prefetch=1), dict(balance_k_tiles=1), dict(fuse_value_cache=1),
                       dict(gemm_prefetch=True), dict(fuse_value_cache=True,vector_staging_bytes=4096)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                Architecture(**values)

    def test_ragged_packed_reads_and_output_addresses(self):
        # Both K and N have partial final folds. The actual packed bytes are the
        # oracle; an incorrect packed-stride formula selects different weights.
        architecture = Architecture(array_rows=3, array_cols=5, operand_staging_bytes=64)
        stream = io.StringIO()
        graph = Graph(stream, steps=1, architecture=architecture)
        m, k, n = 2, 17, 13
        a = graph.tensor('a', m*k, 1)
        b = graph.tensor('b', k*n, 1)
        result = graph.gemm('ragged', a, m, k, n, b)
        weights = np.arange(k*n).reshape(k, n)
        packed = np.concatenate([weights[:, start:start+5].ravel() for start in range(0, n, 5)])
        reads, writes = [], []
        for line in stream.getvalue().splitlines():
            _, _, _, action, address, size, count, stride = line.split()
            address, size, count, stride = map(int, (address, size, count, stride))
            if action == 'R' and b.address <= address < b.address+b.size:
                self.assertEqual(count, 1)
                reads.append(packed[address-b.address:address-b.address+size])
            if action == 'W':
                for row in range(count):
                    writes.extend(range((address-result.address+row*stride)//4,
                                        (address-result.address+row*stride+size)//4))
        expected = [weights[k0:k0+12, n0:n0+5].ravel()
                    for n0 in range(0, n, 5) for k0 in range(0, k, 12)]
        self.assertEqual(len(reads), len(expected))
        for actual, reference in zip(reads, expected):
            np.testing.assert_array_equal(actual, reference)
        self.assertEqual(sorted(writes), list(range(m*n)))

    def test_baseline_commands_unchanged(self):
        baseline = HERE/'build/full_1ghz'
        if not (baseline/'commands.txt').exists():
            self.skipTest('Saved baseline is not available')
        numerical = json.loads((baseline/'numerical_result.json').read_text())
        old = json.loads((baseline/'manifest.json').read_text())
        with tempfile.TemporaryFile(mode='w+') as commands:
            manifest = Graph(commands, steps=numerical['decode_steps'],
                             context=numerical['first_decode_context'],
                             input_ids=numerical['decode_input_token_ids'],
                             architecture=Architecture.read(HERE/'configs/baseline_32x32.json')).run()
            self.assertEqual(manifest['ops'], old['ops'])
            commands.seek(0)
            self.assertEqual(commands.read(), (baseline/'commands.txt').read_text())


if __name__ == '__main__':
    unittest.main()
