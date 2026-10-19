#!/usr/bin/env python3
"""Bounded Gemma-3-1B hidden-width chunks: full K, one output-column tile."""
import json
from run_validation import RTL_ROOT, run_case

report = RTL_ROOT / 'build' / 'gemma_checks.json'
report.unlink(missing_ok=True)
results = [
    run_case(1000, (32, 32, 1, 1152, 32), False, 11451),
    run_case(1001, (32, 32, 512, 1152, 32), False, 11452, simulator="verilator"),
]
report.write_text(json.dumps({'cases': results, 'passed':len(results), 'commands':len(results)*2}, indent=2)+'\n')
print(f'PASS: {len(results)} Gemma chunks / {len(results)*2} commands; {report}')
