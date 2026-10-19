// SPDX-License-Identifier: MIT
// Implements dense systolic_compute_ws.py demand rows without trace ROMs.
module scalesim_ws_scheduler #(
    parameter integer ROWS = 32,
    parameter integer COLS = 32,
    parameter integer MAX_M = 512,
    parameter integer ROW_W = (MAX_M > 1) ? $clog2(MAX_M) : 1
) (
    input wire clk, rst,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [31:0] cmd_m, cmd_k, cmd_n,
    input wire [31:0] cmd_a_base, cmd_b_base, cmd_c_base,
    input wire cycle_ready,
    output reg busy, done, error,
    output wire accepted, advance, load_weight, first_k, last_k,
    output reg [63:0] logical_cycle, elapsed_cycles, stall_cycles,
    output reg [ROWS-1:0] a_valid,
    output reg [ROWS*32-1:0] a_addr,
    output reg [COLS-1:0] b_valid,
    output reg [COLS*32-1:0] b_addr,
    output reg [COLS-1:0] c_valid,
    output reg [COLS*32-1:0] c_addr,
    output reg [COLS*ROW_W-1:0] c_row
);
    reg [63:0] m, k, n;
    reg [31:0] a_base, b_base, c_base;
    reg [63:0] k_start, n_start, fold_cycle;
    wire [63:0] fold_length = m + 2*64'(ROWS) + 64'(COLS) - 2;
    wire [63:0] a_limit = {32'b0,cmd_a_base} + ({32'b0,cmd_m} * {32'b0,cmd_k});
    wire [63:0] b_limit = {32'b0,cmd_b_base} + ({32'b0,cmd_k} * {32'b0,cmd_n});
    wire [63:0] c_limit = {32'b0,cmd_c_base} + ({32'b0,cmd_m} * {32'b0,cmd_n});
    wire bad_cmd = (cmd_m==0 || cmd_k==0 || cmd_n==0 || cmd_m>MAX_M ||
                   a_limit>64'h100000000 || b_limit>64'h100000000 || c_limit>64'h100000000);
    assign cmd_ready = !busy && !rst;
    assign accepted = cmd_valid && cmd_ready && !bad_cmd;
    assign advance = busy && cycle_ready && !rst;
    assign load_weight = busy && fold_cycle < 64'(ROWS);
    assign first_k = k_start == 0;
    assign last_k = k_start + 64'(ROWS) >= k;
    integer r, c;
    reg [63:0] sample_row;
    always @* begin
        a_valid = 0; a_addr = 0;
        b_valid = 0; b_addr = 0;
        c_valid = 0; c_addr = 0; c_row = 0;
        sample_row = 0;
        if (busy) begin
            for (r=0; r<ROWS; r=r+1) begin
                if (fold_cycle >= 64'(ROWS)+64'(r) && fold_cycle < 64'(ROWS)+64'(r)+m && k_start+64'(r) < k) begin
                    a_valid[r] = 1;
                    a_addr[r*32 +: 32] = 32'({32'b0,a_base} + (fold_cycle-64'(ROWS)-64'(r))*k + k_start+64'(r));
                end
            end
            for (c=0; c<COLS; c=c+1) begin
                if (fold_cycle < 64'(ROWS) && k_start+64'(ROWS)-1-fold_cycle < k && n_start+64'(c) < n) begin
                    b_valid[c] = 1;
                    b_addr[c*32 +: 32] = 32'({32'b0,b_base} + (k_start+64'(ROWS)-1-fold_cycle)*n+n_start+64'(c));
                end
                if (fold_cycle >= 2*64'(ROWS)-1+64'(c) && fold_cycle < 2*64'(ROWS)-1+64'(c)+m && n_start+64'(c) < n) begin
                    sample_row = fold_cycle-(2*64'(ROWS)-1+64'(c));
                    c_valid[c] = 1;
                    c_row[c*ROW_W +: ROW_W] = ROW_W'(sample_row);
                    c_addr[c*32 +: 32] = 32'({32'b0,c_base} + sample_row*n+n_start+64'(c));
                end
            end
        end
    end
    always @(posedge clk) begin
        if (rst) begin
            busy<=0; done<=0; error<=0;
            m<=0; k<=0; n<=0; a_base<=0; b_base<=0; c_base<=0;
            k_start<=0; n_start<=0; fold_cycle<=0;
            logical_cycle<=0; elapsed_cycles<=0; stall_cycles<=0;
        end else begin
            done <= 0;
            error <= 0;
            if (cmd_valid && cmd_ready) begin
                logical_cycle<=0; elapsed_cycles<=0; stall_cycles<=0;
                if (bad_cmd) begin
                    error<=1; done<=1;
                end else begin
                    busy<=1;
                    m<={32'b0,cmd_m}; k<={32'b0,cmd_k}; n<={32'b0,cmd_n};
                    a_base<=cmd_a_base; b_base<=cmd_b_base; c_base<=cmd_c_base;
                    k_start<=0; n_start<=0; fold_cycle<=0;
                end
            end else if (busy) begin
                elapsed_cycles <= elapsed_cycles+1;
                if (!cycle_ready) stall_cycles <= stall_cycles+1;
                if (advance) begin
                    logical_cycle <= logical_cycle+1;
                    if (fold_cycle+1 == fold_length) begin
                        fold_cycle<=0;
                        if (last_k) begin
                            k_start<=0;
                            if (n_start+64'(COLS) >= n) begin
                                busy<=0; done<=1;
                            end else n_start<=n_start+64'(COLS);
                        end else k_start<=k_start+64'(ROWS);
                    end else fold_cycle<=fold_cycle+1;
                end
            end
        end
    end
endmodule
