"""Regression tests for non-layout USER-mode read-buffer stall accounting."""

import unittest
from unittest.mock import patch

import numpy as np

from scalesim.memory.read_buffer import read_buffer


class TestReadBufferStallAccounting(unittest.TestCase):
    def service_single_miss(self, prefetch_completion_cycle):
        """Control one miss so stall-accounting can be tested independently."""
        buffer = read_buffer()
        buffer.active_buf_full_flag = True  # Skip initial prefetch; test miss handling.
        buffer.enable_layout_evaluation = False
        buffer.use_ramulator_trace = False
        buffer.hit_latency = 1
        buffer.last_prefetch_cycle = prefetch_completion_cycle

        # First active_buffer_hit call misses; after new_prefetch it hits.
        with patch.object(buffer, "active_buffer_hit", side_effect=[False, True]), \
             patch.object(buffer, "new_prefetch") as prefetch:
            serviced = buffer.service_reads(
                incoming_requests_arr_np=np.array([[42]]),
                incoming_cycles_arr=np.array([[100]]),
            )
            prefetch.assert_called_once_with()

        return int(serviced[0, 0])

    def test_positive_stall_charged_once(self):
        # Request cycle 100 + 1 cycle hit latency; ready at 111 => 10 stall cycles.
        # Original upstream returns 121, incorrectly counting the 10-cycle stall twice.
        self.assertEqual(self.service_single_miss(111), 111)

    def test_completed_prefetch_does_not_reduce_service_cycle(self):
        # Prefetch completed before the request arrives, so there is no stall.
        # Original upstream returns 90 due to an unconditional negative offset.
        self.assertEqual(self.service_single_miss(90), 101)


if __name__ == "__main__":
    unittest.main()
