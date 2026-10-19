#!/usr/bin/env python3
"""Independent checks for the F2 HBM2 profile and timestamped replay adapter."""
import csv
from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

from f2_config import make_f2_config
from run_hbm2 import shared

HERE = Path(__file__).resolve().parent
FRONTEND = HERE / "build" / "ramulator_hbm2"


class ProfileChecks(unittest.TestCase):
    def test_geometry_and_published_clock_capacity(self):
        config, profile = make_f2_config()
        controllers = config["memory_system"]["controllers"]
        self.assertEqual(len(controllers), 16)
        for controller in controllers:
            org = controller["dram"]["org"]
            self.assertEqual(org["count"], [1, 2, 2, 4, 4, 16384, 128])
            self.assertEqual(org["dq"], 64)
            self.assertEqual(math.prod(org["count"]) * org["dq"] // 8, 1024**3)
        self.assertEqual(profile["capacity_bytes"], 16 * 1024**3)
        self.assertEqual(profile["pseudochannels"], 32)
        self.assertEqual(profile["nominal_dram_clock_hz"], 900000000)
        self.assertEqual(profile["nominal_axi_clock_hz"], 450000000)
        self.assertEqual(profile["nominal_peak_GB_s"], 460.8)
        self.assertEqual(profile["transaction_bytes"], 32)

    def test_published_timing_values_survive_native_serialization(self):
        config, profile = make_f2_config()
        timing = profile["timing_cycles"]
        expected = dict(rate=1800, nBL=2, nCL=22, nCWL=7, nRC=43, nRAS=30,
                        nRCDRD=13, nRCDWR=9, nRRDL=4, nRRDS=4, nFAW=15,
                        nRTPL=5, nRP=13, nWR=15, nWTRL=8, nWTRS=4,
                        nRFC=315, nRFCpb=144, nRREFD=8, nREFI=3510,
                        nCCDS=2, nCCDL=4, nCCDR=3, nRTW=21,
                        nREFIpb=109, tCK_ps=1111)
        self.assertEqual(timing, expected)
        import ramulator
        native = config["memory_system"]["controllers"][0]["dram"]
        self.assertEqual(dict(zip(ramulator.dram.HBM2.timing_params, native["timing"])), expected)
        self.assertEqual(native["read_latency"], expected["nCL"] + expected["nBL"])
        self.assertEqual(native["channel_width"], 128)

    def test_reduced_profiles_and_invalid_channel_count(self):
        for channels in [1, 2, 4, 8, 16]:
            config, profile = make_f2_config(channels)
            self.assertEqual(len(config["memory_system"]["controllers"]), channels)
            self.assertEqual(profile["capacity_bytes"], channels * 1024**3)
            self.assertEqual(profile["full_f2_memory"], channels == 16)
        for channels in [0, 3, 32]:
            with self.assertRaises(ValueError):
                make_f2_config(channels)


class AdapterChecks(unittest.TestCase):
    def test_32byte_boundaries_and_repeated_rows(self):
        capture = {stream: [] for stream in shared.STREAMS}
        capture["ifmap"] = [(-9, [0, 31, 32, 32, -1]), (7, [0])]
        capture["ofmap"] = [(0, [shared.SOURCE_BASES["ofmap"] + 7,
                                  shared.SOURCE_BASES["ofmap"] + 8])]
        requests, manifest, origin = shared.convert_rows(
            capture, dict(ifmap=33, filter=0, ofmap=9), burst_bytes=32)
        self.assertEqual(origin, -9)
        self.assertEqual([r["address_bytes"] for r in requests],
                         [0, 32, shared.PHYSICAL_BASES["ofmap"],
                          shared.PHYSICAL_BASES["ofmap"] + 32, 0])
        self.assertEqual([r["arrival_core_cycle"] for r in requests], [0, 0, 9, 9, 16])
        self.assertEqual([r["unique_requested_bytes"] for r in manifest], [3, 1, 8])

    def test_32byte_response_accounting_and_ids(self):
        capture = {stream: [] for stream in shared.STREAMS}
        capture["ifmap"] = [(-2, [0, 32])]
        capture["ofmap"] = [(0, [shared.SOURCE_BASES["ofmap"]])]
        requests, manifest, _ = shared.convert_rows(
            capture, dict(ifmap=33, filter=0, ofmap=1), burst_bytes=32)
        responses = [dict(id=r["id"], arrival_core_cycle=r["arrival_core_cycle"],
                          address_bytes=r["address_bytes"], operation=r["operation"],
                          accepted_dram_cycle=10, completed_dram_cycle=(40 if r["operation"] == "R" else -1))
                     for r in requests]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "response.csv"
            def analyze(rows):
                with path.open("w") as f:
                    writer = csv.DictWriter(f, fieldnames=list(responses[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                return shared.analyze_responses(requests, deepcopy(manifest), path,
                                                1.111, 250000000, burst_bytes=32)
            stats = analyze(list(reversed(responses)))
            self.assertEqual(stats["ifmap"]["bus_bytes"], 64)
            self.assertEqual(stats["ofmap"]["bus_bytes"], 32)
            self.assertEqual(stats["filter"]["bus_bytes"], 0)
            self.assertAlmostEqual(stats["ifmap"]["mean_read_latency_ns"], 44.44)
            for rows in [responses[:-1], responses + [responses[0]],
                         [dict(responses[0], address_bytes=32)] + responses[1:]]:
                with self.assertRaises(ValueError):
                    analyze(rows)

    def test_native_capture_can_be_replayed_as_32byte_transactions(self):
        args = SimpleNamespace(m=4, n=13, k=9, rows=4, cols=8)
        with tempfile.TemporaryDirectory() as tmp:
            capture, report = shared.capture_native(args, Path(tmp))
        requests, manifest, origin = shared.convert_rows(
            capture, dict(ifmap=36, filter=117, ofmap=52), burst_bytes=32)
        self.assertLess(origin, 0)
        self.assertEqual(report[2], 0)
        self.assertTrue(all(r["address_bytes"] % 32 == 0 for r in requests))
        self.assertEqual(sorted(rid for row in manifest for rid in row["request_ids"]),
                         list(range(len(requests))))
        self.assertEqual({manifest[r["row_index"]]["stream"] for r in requests}, set(shared.STREAMS))


@unittest.skipUnless(FRONTEND.is_file(), "Build the optional Ramulator2 HBM2 frontend first")
class FrontendChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.number = 0

    def replay(self, rows, channels=1, core_hz=250000000, expect_success=True, config_mutator=None):
        folder = self.folder / str(self.number)
        self.number += 1
        folder.mkdir()
        config, _ = make_f2_config(channels)
        if config_mutator:
            config_mutator(config)
        config_path = folder / "config.json"
        config_path.write_text(json.dumps(config))
        request_path, response_path, stats_path = [folder / name for name in
                                                  ("requests.csv", "responses.csv", "stats.txt")]
        with request_path.open("w") as f:
            writer = csv.writer(f)
            writer.writerow(("id", "arrival_core_cycle", "address_bytes", "operation"))
            writer.writerows(rows)
        result = subprocess.run([str(FRONTEND), str(config_path), str(request_path),
                                 str(response_path), str(stats_path), str(core_hz), "1000000", "16"],
                                cwd=folder, capture_output=True, text=True, timeout=30)
        if not expect_success:
            self.assertNotEqual(result.returncode, 0, result.stdout)
            return result
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(Path(str(stats_path) + ".summary.json").read_text())
        with response_path.open() as f:
            responses = list(csv.DictReader(f))
        by_id = {int(r["id"]): r for r in responses}
        self.assertEqual(len(responses), len(rows))
        self.assertEqual(set(by_id), {r[0] for r in rows})
        for request_id, arrival, address, operation in rows:
            r = by_id[request_id]
            self.assertEqual(int(r["arrival_core_cycle"]), arrival)
            self.assertEqual(int(r["address_bytes"]), address)
            self.assertEqual(r["operation"], operation)
            accepted, completed, callback = [int(r[k]) for k in
                                            ("accepted_dram_cycle", "completed_dram_cycle", "callback_dram_cycle")]
            due = math.ceil(Fraction(arrival * 10**12, core_hz * summary["tck_ps"]))
            self.assertGreaterEqual(accepted, due)
            self.assertGreaterEqual(callback, accepted)
            if operation == "R":
                self.assertEqual(completed, callback)
                self.assertIn(r["callback_kind"], ("read_completed", "read_forwarded"))
            else:
                self.assertEqual(completed, -1)
                self.assertIn(r["callback_kind"], ("write_command_issued", "write_coalesced"))
        self.assertEqual(summary["input_requests"], len(rows))
        self.assertEqual(summary["completed_reads"] + summary["write_callbacks"], len(rows))
        self.assertEqual(summary["read_commands_issued"],
                         summary["completed_reads"] - summary["forwarded_reads"])
        self.assertEqual(summary["command_data_bytes"],
                         (summary["read_commands_issued"] + summary["write_commands_issued"]) * 32)
        self.assertEqual(summary["transaction_bytes"], 32)
        self.assertEqual(summary["capacity_bytes"], channels * 1024**3)
        self.assertEqual(summary["pseudochannels"], channels * 2)
        return by_id, summary

    def test_saturated_read_trace_scales_across_16_physical_channels(self):
        rows = [(i, 0, i * 32, "R") for i in range(8192)]
        _, single = self.replay(rows)
        _, full = self.replay(rows, channels=16)
        self.assertGreater(single["admission_retry_attempts"], 0)
        self.assertLess(full["dram_cycles"], single["dram_cycles"])
        self.assertEqual(full["completed_reads"], 8192)
        self.assertEqual(full["nominal_peak_GB_s"], 460.8)

    def test_clock_conversion_keeps_idle_gap(self):
        rows = [(0, 0, 0, "R"), (1, 1001, 32, "R")]
        for hz in [250000000, 1000000000]:
            responses, _ = self.replay(rows, core_hz=hz)
            due = math.ceil(Fraction(1001 * 10**12, hz * 1111))
            self.assertEqual(int(responses[1]["accepted_dram_cycle"]), due)

    def test_coalesced_write_forwarded_read_and_write_guard(self):
        responses, summary = self.replay([(0, 0, 0, "W"), (1, 0, 0, "W"), (2, 0, 0, "R")])
        self.assertEqual(responses[1]["callback_kind"], "write_coalesced")
        self.assertEqual(int(responses[1]["callback_dram_cycle"]), 0)
        self.assertEqual(responses[2]["callback_kind"], "read_forwarded")
        self.assertEqual(int(responses[2]["completed_dram_cycle"]), 1)
        self.assertEqual(summary["write_commands_issued"], 1)
        self.assertEqual(summary["writes_coalesced"], 1)
        self.assertEqual(summary["command_data_bytes"], 32)  # Three submitted transactions, one DRAM command.
        self.assertEqual(summary["post_queue_write_guard_ticks"], 7 + 2 + 15)
        self.assertGreaterEqual(summary["dram_cycles"],
                                int(responses[0]["callback_dram_cycle"]) + 7 + 2 + 15)

    def test_mixed_streams_and_last_valid_capacity_address(self):
        rows = []
        for i in range(64):
            for stream in shared.STREAMS:
                rows.append((len(rows), i // 2, shared.PHYSICAL_BASES[stream] + i * 32,
                             "W" if stream == "ofmap" else "R"))
        rows.append((len(rows), 100, 16 * 1024**3 - 32, "R"))
        _, summary = self.replay(rows, channels=16)
        self.assertEqual(summary["completed_reads"], 129)
        self.assertEqual(summary["write_callbacks"], 64)

    def test_invalid_alignment_capacity_identity_and_order(self):
        cases = [([(0, 0, 1, "R")], "Unaligned"),
                 ([(0, 0, 1024**3, "R")], "capacity"),
                 ([(0, 0, 0, "R"), (0, 0, 32, "R")], "Duplicate"),
                 ([(-1, 0, 0, "R")], "nonnegative"),
                 ([(0, 3, 0, "R"), (1, 2, 32, "R")], "sorted")]
        for rows, message in cases:
            with self.subTest(message=message):
                self.assertIn(message, self.replay(rows, expect_success=False).stderr)

    def test_malformed_geometry_and_unresolved_timing_are_rejected(self):
        def bad_geometry(config):
            config["memory_system"]["controllers"][0]["dram"]["org"]["count"][2] = 3
        def bad_timing(config):
            config["memory_system"]["controllers"][0]["dram"]["timing"][2] = -1
        for mutation in [bad_geometry, bad_timing]:
            self.replay([(0, 0, 0, "R")], expect_success=False, config_mutator=mutation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
