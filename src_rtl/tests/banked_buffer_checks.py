#!/usr/bin/env python3
"""Icarus functional/latency checks and Verilator lint for the banked SRAM.

Run from any directory: python3 src_rtl/tests/banked_buffer_checks.py
Artifacts are isolated under src_rtl/build/banked_buffer_checks. Tests compare
against a separate flat byte-memory model, not the DUT's internal bank arrays.
"""
from pathlib import Path
import json
import random
import subprocess

ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / 'src_rtl/rtl/scalesim_banked_operand_buffer.sv'
OUT = ROOT / 'src_rtl/build/banked_buffer_checks'

TB = r'''`timescale 1ns/1ps
module tb;
localparam LANES=@LANES@, BANKS=@BANKS@, PAGE_WORDS=@WORDS@;
reg clk=0; always #5 clk=~clk;
reg rst=1,wr_valid=0,wr_page=0,req_valid=0,req_page=0,rsp_ready=0;
reg [31:0] wr_addr=0;
reg [7:0] wr_data=0;
reg [LANES-1:0] req_mask=0;
reg [LANES*32-1:0] req_addr=0;
wire wr_ready,req_ready,rsp_valid,rsp_error;
wire [LANES*8-1:0] rsp_data;
reg [7:0] model [0:2*PAGE_WORDS-1];
integer transactions=0, aborted=0;
scalesim_banked_operand_buffer #(.LANES(LANES),.BANKS(BANKS),.PAGE_WORDS(PAGE_WORDS)) dut(.*);

task automatic write_word(input bit page,input reg[31:0] addr,input reg[7:0] value,input bit expected_ready);
begin
    wr_valid=1; wr_page=page; wr_addr=addr; wr_data=value;
    #1; if(wr_ready!==expected_ready) $fatal(1,"write readiness addr %h",addr);
    @(posedge clk); #1;
    if(expected_ready) model[page*PAGE_WORDS+addr]=value;
    @(negedge clk); wr_valid=0;
end
endtask

task automatic read_vector(input bit page,input reg[LANES-1:0] mask,
                           input reg[LANES*32-1:0] addresses,input integer stalls,input integer tag);
reg [LANES*8-1:0] expected_data;
reg expected_error;
reg [31:0] address;
integer count [0:BANKS-1];
integer maximum,elapsed,i;
begin
    expected_data=0; expected_error=0; maximum=0;
    for(i=0;i<BANKS;i=i+1) count[i]=0;
    for(i=0;i<LANES;i=i+1) begin
        address=addresses[i*32+:32];
        if(mask[i]) begin
            if(address<PAGE_WORDS) begin
                expected_data[i*8+:8]=model[page*PAGE_WORDS+address];
                count[address%BANKS]=count[address%BANKS]+1;
                if(count[address%BANKS]>maximum) maximum=count[address%BANKS];
            end else expected_error=1;
        end
    end
    req_valid=1; req_page=page; req_mask=mask; req_addr=addresses; rsp_ready=0;
    #1; if(!req_ready) $fatal(1,"request not ready tag %0d",tag);
    @(posedge clk); #1;
    if(req_ready || rsp_valid) $fatal(1,"request bypassed/register missing tag %0d",tag);
    @(negedge clk);
    // While busy, poison all live inputs and hold an additional request valid.
    req_valid=1; req_page=!page; req_mask='1; req_addr='1;
    elapsed=0;
    while(!rsp_valid) begin
        @(posedge clk); #1; elapsed=elapsed+1;
        if(req_ready) $fatal(1,"accepted second outstanding vector tag %0d",tag);
        if(elapsed>LANES+3) $fatal(1,"response timeout tag %0d",tag);
        @(negedge clk);
    end
    if(elapsed!=maximum+1) $fatal(1,"latency tag %0d expected %0d got %0d",tag,maximum+1,elapsed);
    if(rsp_data!==expected_data || rsp_error!==expected_error)
        $fatal(1,"response tag %0d expected %h error%0d got %h error%0d",tag,expected_data,expected_error,rsp_data,rsp_error);
    repeat(stalls+1) begin
        @(posedge clk); #1;
        if(!rsp_valid || req_ready || rsp_data!==expected_data || rsp_error!==expected_error)
            $fatal(1,"response unstable under backpressure tag %0d",tag);
        @(negedge clk);
    end
    req_valid=0; rsp_ready=1;
    @(posedge clk); #1;
    if(rsp_valid || !req_ready) $fatal(1,"response did not retire tag %0d",tag);
    @(negedge clk); rsp_ready=0;
    transactions=transactions+1;
end
endtask

task automatic abort_read(input integer phase);
integer watchdog;
begin
    req_valid=1; req_page=0; req_mask='1; req_addr=0;
    #1; if(!req_ready) $fatal(1,"abort request readiness");
    @(posedge clk); #1; @(negedge clk); req_valid=0;
    if(phase==1) begin @(posedge clk); #1; @(negedge clk); end
    if(phase==2) begin
        watchdog=0;
        while(!rsp_valid) begin
            @(posedge clk); #1; @(negedge clk); watchdog=watchdog+1;
            if(watchdog>LANES+3) $fatal(1,"abort wait timeout");
        end
        repeat(3) begin @(posedge clk); #1; @(negedge clk); end
    end
    rst=1; req_valid=1; rsp_ready=1; wr_valid=1;
    wr_page=0; wr_addr=0; wr_data=~model[0];
    #1;
    if(wr_ready || req_ready || rsp_valid) $fatal(1,"reset handshake suppression phase %0d",phase);
    repeat(2) begin
        @(posedge clk); #1;
        if(rsp_valid || rsp_data!==0 || rsp_error!==0) $fatal(1,"reset state phase %0d",phase);
        @(negedge clk);
    end
    wr_valid=0; req_valid=0; rsp_ready=0; rst=0;
    repeat(LANES+3) begin
        @(posedge clk); #1;
        if(rsp_valid || !req_ready) $fatal(1,"stale response after reset phase %0d",phase);
        @(negedge clk);
    end
    aborted=aborted+1;
    // Reset must preserve contents and must not accept the offered write.
    read_vector(0,'1,0,1,900+phase);
    read_vector(1,'1,0,1,910+phase);
end
endtask

task automatic read_first_collision;
reg [7:0] old_value;
reg [LANES*8-1:0] expected_data;
begin
    old_value=model[0]; expected_data=0; expected_data[7:0]=old_value;
    req_valid=1; req_page=0; req_mask=1; req_addr=0;
    @(posedge clk); #1; @(negedge clk); req_valid=0;
    wr_valid=1; wr_page=0; wr_addr=0; wr_data=8'he1;
    #1; if(!wr_ready) $fatal(1,"concurrent write not accepted");
    // This edge is the first real SRAM read, simultaneously with the write.
    @(posedge clk); #1;
    model[0]=8'he1;
    if(rsp_valid) $fatal(1,"read is not synchronous plus registered gather");
    @(negedge clk); wr_valid=0;
    @(posedge clk); #1;
    if(!rsp_valid || rsp_error || rsp_data!==expected_data) $fatal(1,"same-address read-first failure");
    @(negedge clk); rsp_ready=1;
    @(posedge clk); #1; @(negedge clk); rsp_ready=0;
    read_vector(0,1,0,2,800);
    read_vector(1,1,0,2,801);
end
endtask

// Same-address conflict with a write between the two reads makes lane priority
// externally observable: the lowest lane must get old data, the last lane new.
task automatic lowest_lane_priority;
reg [LANES*8-1:0] expected_data;
begin
    if(LANES>1) begin
        expected_data=0;
        expected_data[7:0]=model[0];
        expected_data[(LANES-1)*8+:8]=8'h3a;
        req_valid=1; req_page=0; req_mask=0;
        req_mask[0]=1; req_mask[LANES-1]=1; req_addr=0;
        @(posedge clk); #1; @(negedge clk); req_valid=0;
        wr_valid=1; wr_page=0; wr_addr=0; wr_data=8'h3a;
        @(posedge clk); #1; model[0]=8'h3a;
        if(rsp_valid) $fatal(1,"conflict prematurely completed first read");
        @(negedge clk); wr_valid=0;
        @(posedge clk); #1;
        if(rsp_valid) $fatal(1,"duplicate address was coalesced or gather missing");
        @(negedge clk); @(posedge clk); #1;
        if(!rsp_valid || rsp_error || rsp_data!==expected_data)
            $fatal(1,"bank arbitration did not serve lowest pending lane first");
        @(negedge clk); rsp_ready=1;
        @(posedge clk); #1; @(negedge clk); rsp_ready=0;
        transactions=transactions+1;
    end
end
endtask

integer p,a;
initial begin
    repeat(2) @(posedge clk);
    @(negedge clk); rst=0;
    for(p=0;p<2;p=p+1)
        for(a=0;a<PAGE_WORDS;a=a+1)
            write_word(p,a,(a*37+p*113+17)&255,1);
    write_word(0,PAGE_WORDS,8'hfe,0);
    write_word(1,32'hffffffff,8'hfe,0);
    @REQUESTS@
    read_first_collision();
    lowest_lane_priority();
    abort_read(0); abort_read(1); abort_read(2);
    $display("PASS banked buffer LANES=%0d BANKS=%0d PAGE_WORDS=%0d vectors=%0d resets=%0d",LANES,BANKS,PAGE_WORDS,transactions,aborted);
    $finish;
end
initial begin #10000000; $fatal(1,"global timeout"); end
endmodule
'''


def requests(lanes, banks, words):
    rng = random.Random(94137 + 100 * lanes + banks)
    cases = [
        (0, 0, [0xffffffff] * lanes, 0),
        (1, (1 << lanes) - 1, [words] * lanes, 3),
        (0, (1 << lanes) - 1, [0] * lanes, 2),
        (1, (1 << lanes) - 1, [(i * banks) % words for i in range(lanes)], 4),
        (0, (1 << lanes) - 1, [i % words for i in range(lanes)], 0),
        (1, (1 << lanes) - 1, [words-1] * lanes, 2),
    ]
    for index in range(90):
        mask = rng.randrange(1 << lanes)
        addresses = [rng.choice([rng.randrange(words), rng.randrange(words),
                                  words, words + 2, 0xffffffff]) for _ in range(lanes)]
        cases.append((index % 2, mask, addresses, rng.randrange(6)))
    commands = []
    for index, (page, mask, addresses, stalls) in enumerate(cases):
        packed = sum(address << (32 * lane) for lane, address in enumerate(addresses))
        commands.append(f"read_vector(1'b{page},{lanes}'h{mask:x},{lanes*32}'h{packed:x},{stalls},{index});")
    return '\n    '.join(commands)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / 'results.json'
    report_path.unlink(missing_ok=True)
    results = []
    # Include minimal dimensions, ragged vectors, non-power-of-two banking and
    # a full 32-lane/32-bank path. All have independently initialized two pages.
    for lanes, banks, words in [(1, 1, 1), (5, 1, 16), (5, 2, 16),
                                (7, 4, 32), (5, 3, 24), (32, 32, 128)]:
        folder = OUT / f'l{lanes}_b{banks}_w{words}'
        folder.mkdir(exist_ok=True)
        source = TB.replace('@LANES@', str(lanes)).replace('@BANKS@', str(banks)).replace('@WORDS@', str(words))
        source = source.replace('@REQUESTS@', requests(lanes, banks, words))
        (folder / 'tb.sv').write_text(source)
        subprocess.run(['iverilog', '-g2012', '-s', 'tb', '-o', str(folder / 'sim'),
                        str(RTL), str(folder / 'tb.sv')], check=True, capture_output=True, text=True)
        sim = subprocess.run(['vvp', str(folder / 'sim')], check=True, capture_output=True,
                             text=True, timeout=30)
        (folder / 'simulation.log').write_text(sim.stdout + sim.stderr)
        assert 'PASS banked buffer' in sim.stdout, sim.stdout
        print(sim.stdout.strip(), flush=True)
        lint = subprocess.run(['verilator', '--lint-only', '-Wall', '--top-module',
                               'scalesim_banked_operand_buffer', f'-GLANES={lanes}',
                               f'-GBANKS={banks}', f'-GPAGE_WORDS={words}', str(RTL)],
                              check=True, capture_output=True, text=True, timeout=60)
        (folder / 'lint.log').write_text(lint.stdout + lint.stderr)
        results.append(dict(lanes=lanes, banks=banks, page_words=words,
                            request_vectors=104 + int(lanes > 1), reset_cancellations=3,
                            simulation_passed=True, verilator_lint_passed=True))
    # Lint the production defaults as well as all functional parameter variants.
    lint = subprocess.run(['verilator', '--lint-only', '-Wall', '--top-module',
                           'scalesim_banked_operand_buffer', str(RTL)], check=True,
                          capture_output=True, text=True, timeout=60)
    (OUT / 'default_lint.log').write_text(lint.stdout + lint.stderr)
    report_path.write_text(json.dumps(dict(cases=results, default_lint_passed=True,
        all_passed=True, reproduction='python3 src_rtl/tests/banked_buffer_checks.py'), indent=2) + '\n')
    print('All banked-buffer simulations and lint configurations passed.', flush=True)


if __name__ == '__main__':
    main()
