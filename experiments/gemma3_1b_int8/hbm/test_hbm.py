#!/usr/bin/env python3
"""Focused correctness checks for HBM trace conversion and response accounting."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from run_hbm import (BURST_BYTES, PHYSICAL_BASES, REGION_BYTES, SOURCE_BASES,
                     STREAMS, WORD_BYTES, analyze_responses, capture_native, convert_rows)

HERE = Path(__file__).resolve().parent
FRONTEND = HERE / 'build' / 'ramulator_hbm'


def empty_capture():
    return {stream: [] for stream in STREAMS}


class ConversionChecks(unittest.TestCase):
    def test_shared_negative_origin_and_idle_gap(self):
        captured = empty_capture()
        captured['ifmap'] = [(-7, [0]), (21, [64])]
        captured['filter'] = [(-3, [SOURCE_BASES['filter']])]
        captured['ofmap'] = [(8, [SOURCE_BASES['ofmap']])]
        requests, manifest, origin = convert_rows(captured, dict(ifmap=128, filter=1, ofmap=1))
        self.assertEqual(origin, -7)
        self.assertEqual([r['arrival_core_cycle'] for r in requests], [0, 4, 15, 28])
        self.assertEqual([r['operation'] for r in requests], ['R', 'R', 'W', 'R'])
        self.assertEqual(len(manifest), 4)

    def test_null_rows_and_padding_never_generate_transactions(self):
        captured = empty_capture()
        captured['ifmap'] = [(-100, [-1, -1]), (-5, [0, -1]), (9, [-1])]
        captured['filter'] = [(30, [])]
        requests, manifest, origin = convert_rows(captured, dict(ifmap=1, filter=0, ofmap=0))
        self.assertEqual(origin, -5)  # Earliest actual traffic defines the memory origin.
        self.assertEqual(len(requests), 1)
        self.assertEqual([row['request_ids'] for row in manifest], [[], [0], [], []])
        self.assertEqual([row['requested_words'] for row in manifest], [0, 1, 0, 0])

    def test_duplicate_words_and_bursts_coalesce_only_within_row(self):
        captured = empty_capture()
        captured['ifmap'] = [(0, [0, 1, 1, 63, 64, -1]), (1, [0, 64])]
        requests, manifest, _ = convert_rows(captured, dict(ifmap=65, filter=0, ofmap=0))
        self.assertEqual([r['address_bytes'] for r in requests], [0, 64, 0, 64])
        self.assertEqual([row['requested_words'] for row in manifest], [5, 2])
        self.assertEqual([row['unique_requested_bytes'] for row in manifest], [4, 2])
        self.assertEqual([row['request_ids'] for row in manifest], [[0, 1], [2, 3]])

    def test_int8_int32_word_mapping_and_separate_regions(self):
        captured = empty_capture()
        captured['ifmap'] = [(0, [63, 64])]
        captured['filter'] = [(0, [SOURCE_BASES['filter'] + 63, SOURCE_BASES['filter'] + 64])]
        captured['ofmap'] = [(0, [SOURCE_BASES['ofmap'] + 15, SOURCE_BASES['ofmap'] + 16])]
        requests, manifest, _ = convert_rows(captured, dict(ifmap=65, filter=65, ofmap=17))
        self.assertEqual([r['address_bytes'] for r in requests],
                         [0, 64, PHYSICAL_BASES['filter'], PHYSICAL_BASES['filter'] + 64,
                          PHYSICAL_BASES['ofmap'], PHYSICAL_BASES['ofmap'] + 64])
        self.assertEqual([row['unique_requested_bytes'] for row in manifest], [2, 2, 8])
        self.assertEqual([r['operation'] for r in requests], ['R', 'R', 'R', 'R', 'W', 'W'])

    def test_word_spanning_smaller_burst_is_split(self):
        captured = empty_capture()
        captured['ofmap'] = [(0, [SOURCE_BASES['ofmap']])]
        requests, _, _ = convert_rows(captured, dict(ifmap=0, filter=0, ofmap=1), burst_bytes=2)
        self.assertEqual([r['address_bytes'] for r in requests],
                         [PHYSICAL_BASES['ofmap'], PHYSICAL_BASES['ofmap'] + 2])

    def test_ids_exactly_partition_requests_by_source_row(self):
        captured = empty_capture()
        captured['ifmap'] = [(20, [64, 0]), (-2, [1])]
        captured['filter'] = [(-3, [SOURCE_BASES['filter'] + 64])]
        requests, manifest, origin = convert_rows(captured, dict(ifmap=65, filter=65, ofmap=0))
        ids = [request_id for row in manifest for request_id in row['request_ids']]
        self.assertEqual(sorted(ids), list(range(len(requests))))
        self.assertEqual(origin, -3)
        for request in requests:
            row = manifest[request['row_index']]
            self.assertIn(request['id'], row['request_ids'])
            self.assertEqual(row['source_cycle'], request['source_cycle'])
        self.assertEqual([r['arrival_core_cycle'] for r in requests], [0, 1, 23, 23])

    def test_invalid_logical_addresses_fail_instead_of_aliasing(self):
        for stream in STREAMS:
            for address in [SOURCE_BASES[stream] - 2, SOURCE_BASES[stream] + 2]:
                with self.subTest(stream=stream, address=address):
                    captured = empty_capture()
                    captured[stream] = [(0, [address])]
                    with self.assertRaises(ValueError):
                        convert_rows(captured, dict(ifmap=2, filter=2, ofmap=2))

    def test_fractional_times_addresses_and_region_overflow_fail(self):
        for captured in [dict(ifmap=[(0.5, [0])], filter=[], ofmap=[]),
                         dict(ifmap=[(0, [0.5])], filter=[], ofmap=[])]:
            with self.assertRaises(ValueError):
                convert_rows(captured, dict(ifmap=1, filter=1, ofmap=1))
        for stream in STREAMS:
            elements = dict(ifmap=0, filter=0, ofmap=0)
            elements[stream] = REGION_BYTES // WORD_BYTES[stream] + 1
            with self.assertRaises(ValueError):
                convert_rows(empty_capture(), elements)
        for size in (0, -1, 3, 63):
            with self.assertRaises(ValueError):
                convert_rows(empty_capture(), dict(ifmap=0, filter=0, ofmap=0), size)

    def test_empty_traffic_preserves_manifest(self):
        captured = empty_capture()
        captured['ofmap'] = [(40, [-1])]
        requests, manifest, origin = convert_rows(captured, dict(ifmap=0, filter=0, ofmap=0))
        self.assertEqual(requests, [])
        self.assertEqual(origin, 0)
        self.assertEqual(manifest[0]['request_ids'], [])


class ResponseChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'responses.csv'
        captured = empty_capture()
        captured['ifmap'] = [(-3, [0, 64])]
        captured['filter'] = [(0, [SOURCE_BASES['filter']])]
        captured['ofmap'] = [(2, [SOURCE_BASES['ofmap']])]
        self.requests, self.manifest, _ = convert_rows(captured, dict(ifmap=65, filter=1, ofmap=1))
        self.responses = []
        for q, accepted, completed in zip(self.requests, [0, 2, 2, 3], [10, 15, 13, -1]):
            self.responses.append(dict(id=q['id'], arrival_core_cycle=q['arrival_core_cycle'],
                address_bytes=q['address_bytes'], operation=q['operation'],
                accepted_dram_cycle=accepted, completed_dram_cycle=completed))

    def analyze(self, responses=None, core_hz=1e9):
        responses = self.responses if responses is None else responses
        with self.path.open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.responses[0]))
            writer.writeheader()
            writer.writerows(responses)
        return analyze_responses(self.requests, self.manifest, self.path, 2.0, core_hz)

    def test_shuffled_response_ids_and_row_completion(self):
        stats = self.analyze(list(reversed(self.responses)))
        self.assertEqual(stats['ifmap']['transactions'], 2)
        self.assertEqual(stats['ifmap']['bus_bytes'], 2 * BURST_BYTES)
        self.assertEqual(stats['ifmap']['mean_admission_delay_ns'], 2.0)
        self.assertEqual(stats['ifmap']['mean_read_latency_ns'], 25.0)
        self.assertEqual(stats['filter']['mean_read_latency_ns'], 23.0)
        self.assertNotIn('mean_read_latency_ns', stats['ofmap'])
        self.assertEqual(self.manifest[0]['read_completed_core_cycle_from_origin'], 30)
        self.assertEqual(self.manifest[1]['read_completed_core_cycle_from_origin'], 26)
        self.assertNotIn('read_completed_core_cycle_from_origin', self.manifest[2])

    def test_duplicate_missing_and_extra_response_ids_fail(self):
        for responses in [self.responses[:-1], self.responses + [self.responses[0]],
                          self.responses + [dict(self.responses[0], id=999)]]:
            with self.assertRaises(ValueError):
                self.analyze(responses)

    def test_wrong_metadata_premature_arrival_or_completion_fail(self):
        cases = [(0, 'arrival_core_cycle', 1), (0, 'operation', 'W'),
                 (0, 'address_bytes', BURST_BYTES),
                 (2, 'accepted_dram_cycle', 1), (1, 'completed_dram_cycle', 1),
                 (3, 'completed_dram_cycle', 10)]
        for index, key, value in cases:
            with self.subTest(key=key):
                modified = deepcopy(self.responses)
                modified[index][key] = value
                with self.assertRaises(ValueError):
                    self.analyze(modified)

    def test_core_to_dram_clock_conversion(self):
        modified = deepcopy(self.responses)
        # At 500 MHz, source core cycles take 2 ns, as do HBM ticks.
        modified[2]['accepted_dram_cycle'] = 3
        modified[3]['accepted_dram_cycle'] = 5
        stats = self.analyze(modified, core_hz=5e8)
        self.assertEqual(stats['filter']['mean_admission_delay_ns'], 0)
        self.assertEqual(stats['filter']['mean_read_latency_ns'], 20)
        self.assertEqual(self.manifest[0]['read_completed_core_cycle_from_origin'], 15)


class NativeCaptureChecks(unittest.TestCase):
    def test_bounded_native_capture_and_patch_restoration(self):
        from scalesim.memory.read_port import read_port
        from scalesim.memory.write_port import write_port
        original_read, original_write = read_port.service_reads, write_port.service_writes
        args = SimpleNamespace(m=4, n=13, k=9, rows=4, cols=8)
        with tempfile.TemporaryDirectory() as tmp:
            captured, report = capture_native(args, Path(tmp))
        self.assertIs(read_port.service_reads, original_read)
        self.assertIs(write_port.service_writes, original_write)
        self.assertTrue(all(captured[stream] for stream in STREAMS))
        elements = dict(ifmap=args.m*args.k, filter=args.k*args.n, ofmap=args.m*args.n)
        requests, manifest, origin = convert_rows(captured, elements)
        self.assertGreater(len(requests), 0)
        self.assertLess(origin, 0)
        self.assertEqual(len(manifest), sum(len(v) for v in captured.values()))
        self.assertEqual(report[2], 0)  # Native CALC source schedule has no stalls.
        self.assertEqual(report[1], 3*2*(4+2*4+8-2)-1)
        self.assertEqual({q['operation'] for q in requests}, {'R', 'W'})


@unittest.skipUnless(FRONTEND.is_file(), 'Build the optional Ramulator HBM frontend first')
class FrontendChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.number = 0

    def replay(self, rows, channels=1, core_hz=1e9, expect_success=True):
        folder = self.folder / str(self.number)
        self.number += 1
        folder.mkdir()
        config = folder / 'HBM.cfg'
        config.write_text((HERE/'HBM-8ch.cfg').read_text().replace('channels = 8', f'channels = {channels}'))
        request_file, response_file, stats_file = (folder/'requests.csv', folder/'responses.csv', folder/'stats.txt')
        with request_file.open('w') as f:
            writer = csv.writer(f)
            writer.writerow(('id', 'arrival_core_cycle', 'address_bytes', 'operation'))
            writer.writerows(rows)
        result = subprocess.run([str(FRONTEND), str(config), str(request_file), str(response_file),
                                 str(stats_file), str(core_hz), '1000000', '64'],
                                cwd=folder, capture_output=True, text=True, timeout=30)
        if not expect_success:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            return result
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(Path(str(stats_file)+'.summary.json').read_text())
        with response_file.open() as f:
            responses = list(csv.DictReader(f))
        by_id = {int(response['id']): response for response in responses}
        self.assertEqual(len(by_id), len(rows))
        self.assertEqual(set(by_id), {int(row[0]) for row in rows})
        for request_id, arrival, address, operation in rows:
            response = by_id[request_id]
            self.assertEqual(int(response['arrival_core_cycle']), arrival)
            self.assertEqual(int(response['address_bytes']), address)
            self.assertEqual(response['operation'], operation)
            accepted = int(response['accepted_dram_cycle'])
            completed = int(response['completed_dram_cycle'])
            self.assertGreaterEqual(accepted*summary['tck_ns']+1e-8, arrival*1e9/core_hz)
            if operation == 'R':
                self.assertGreaterEqual(completed, accepted)
            else:
                self.assertEqual(completed, -1)
        self.assertEqual(summary['input_requests'], len(rows))
        return by_id, summary

    def test_saturated_trace_exercises_shared_queues_and_eight_channels(self):
        rows = [(i, 0, i*BURST_BYTES, 'R') for i in range(8192)]
        _, single = self.replay(rows, channels=1)
        _, eight = self.replay(rows, channels=8)
        self.assertEqual(single['completed_reads'], 8192)
        self.assertEqual(eight['completed_reads'], 8192)
        self.assertGreater(single['admission_retry_attempts'], 0)
        self.assertLess(eight['dram_cycles'], single['dram_cycles'])
        self.assertEqual(eight['capacity_bytes'], single['capacity_bytes']*8)

    def test_idle_arrivals_and_exact_clock_conversion(self):
        rows = [(0, 0, 0, 'R'), (1, 1001, BURST_BYTES, 'R')]
        fast, _ = self.replay(rows, core_hz=1e9)
        slow, _ = self.replay(rows, core_hz=5e8)
        self.assertEqual(int(fast[1]['accepted_dram_cycle']), 501)
        self.assertEqual(int(slow[1]['accepted_dram_cycle']), 1001)

    def test_write_forwarded_read_is_accounted_and_write_time_is_unknown(self):
        responses, summary = self.replay([(0, 0, 0, 'W'), (1, 0, 0, 'R')])
        self.assertEqual(int(responses[1]['completed_dram_cycle']), 1)
        self.assertEqual(int(responses[0]['completed_dram_cycle']), -1)
        self.assertEqual(summary['accepted_writes'], 1)
        self.assertEqual(summary['completed_reads'], 1)
        self.assertGreater(summary['post_queue_write_guard_ticks'], 0)
        self.assertGreater(summary['dram_cycles'], summary['queue_drain_cycle'])

    def test_mixed_stream_addresses_and_operations_share_replay(self):
        rows = []
        for i in range(100):
            for stream in STREAMS:
                rows.append((len(rows), i//2, PHYSICAL_BASES[stream]+i*64,
                             'W' if stream == 'ofmap' else 'R'))
        _, summary = self.replay(rows, channels=8)
        self.assertEqual(summary['accepted_reads'], 200)
        self.assertEqual(summary['completed_reads'], 200)
        self.assertEqual(summary['accepted_writes'], 100)

    def test_bad_alignment_capacity_id_or_sort_order_is_rejected(self):
        cases = [([(0, 0, 1, 'R')], 'Unaligned'),
                 ([(0, 0, 512*1024*1024, 'R')], 'capacity'),
                 ([(0, 0, 0, 'R'), (0, 0, 64, 'R')], 'Duplicate'),
                 ([(-1, 0, 0, 'R')], 'nonnegative'),
                 ([(0, 10, 0, 'R'), (1, 1, 64, 'R')], 'sorted')]
        for rows, diagnostic in cases:
            with self.subTest(diagnostic=diagnostic):
                result = self.replay(rows, expect_success=False)
                self.assertIn(diagnostic, result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
