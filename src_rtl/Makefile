PYTHON ?= ../.venv/bin/python
RTL := $(wildcard rtl/*.sv)
.PHONY: test test-memory test-axi test-hbm2 check-all quick gemma lint synth synth-memory clean

test:
	env -u PYTHONPATH $(PYTHON) tests/run_validation.py
	env -u PYTHONPATH $(PYTHON) tests/review_checks.py
	env -u PYTHONPATH $(PYTHON) tests/wide_command_check.py

test-memory:
	env -u PYTHONPATH $(PYTHON) tests/sync_psum_checks.py
	env -u PYTHONPATH $(PYTHON) tests/banked_buffer_checks.py
	env -u PYTHONPATH $(PYTHON) tests/buffered_top_checks.py

test-axi:
	env -u PYTHONPATH $(PYTHON) tests/axi_top_checks.py

# Requires the pinned native library from experiments/gemma3_1b_int8/hbm2/setup_ramulator2.sh.
test-hbm2:
	env -u PYTHONPATH $(PYTHON) tests/hbm2_cosim_checks.py

check-all: test test-memory test-axi test-hbm2 gemma lint synth synth-memory

quick:
	env -u PYTHONPATH $(PYTHON) tests/run_validation.py --quick

gemma:
	env -u PYTHONPATH $(PYTHON) tests/gemma_checks.py

lint:
	verilator --lint-only -Wall --top-module scalesim_ws_top $(RTL)
	verilator --lint-only -Wall --top-module scalesim_ws_top -GSYNC_PSUM=1 $(RTL)
	verilator --lint-only -Wall --top-module scalesim_ws_buffered_top $(RTL)
	verilator --lint-only -Wall --top-module scalesim_ws_axi_top $(RTL)
	env -u PYTHONPATH $(PYTHON) tests/axi_top_checks.py --lint-only

# Generic RTL synthesis/check, preserving SRAMs. This is not a technology/PPA run.
synth:
	mkdir -p build
	yosys -Q -l build/synthesis.log -p 'read_verilog -sv $(RTL); hierarchy -check -top scalesim_ws_top; proc; opt; memory_collect; check -assert; stat; write_json build/scalesim_ws_top.json'

# Check inferred clocked SRAMs as well as generic synthesis of the buffered top.
synth-memory:
	env -u PYTHONPATH $(PYTHON) tests/buffered_synthesis_checks.py

clean:
	rm -rf build
