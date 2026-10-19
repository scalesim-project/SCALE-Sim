// SPDX-License-Identifier: MIT
// Dense WS engine: schedule + physical systolic array + banked K-fold accumulation.
module scalesim_ws_top #(
    parameter integer ROWS = 32,
    parameter integer COLS = 32,
    parameter integer MAX_M = 512,
    parameter integer SYNC_PSUM = 0
) (
    input wire clk, rst,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [31:0] cmd_m, cmd_k, cmd_n,
    input wire [31:0] cmd_a_base, cmd_b_base, cmd_c_base,
    input wire cycle_ready,
    output wire busy, done, error,
    output wire [63:0] logical_cycle, elapsed_cycles, stall_cycles,
    output wire [ROWS-1:0] a_valid,
    output wire [ROWS*32-1:0] a_addr,
    input wire [ROWS*8-1:0] a_data,
    output wire [COLS-1:0] b_valid,
    output wire [COLS*32-1:0] b_addr,
    input wire [COLS*8-1:0] b_data,
    output wire [COLS-1:0] c_valid,
    output wire [COLS*32-1:0] c_addr,
    output wire [COLS*32-1:0] c_data,
    output wire c_final,
    output wire cycle_fire
);
    localparam integer ROW_W = (MAX_M > 1) ? $clog2(MAX_M) : 1;
    wire accepted, load_weight, first_k;
    wire [COLS*ROW_W-1:0] c_row;
    wire [ROWS*8-1:0] a_masked;
    wire [COLS*8-1:0] b_masked;
    wire [COLS*32-1:0] partial;
    wire psum_ready;
    generate
        if (SYNC_PSUM != 0) begin: synchronous_psum_control
            reg read_valid;
            wire read_needed = busy && !first_k && (|c_valid);
            // A later-K output needs one clocked bank read. The proposed
            // demand row stays fixed until that read and external I/O are ready.
            always @(posedge clk) begin
                if (rst || accepted || cycle_fire) read_valid <= 0;
                else if (read_needed) read_valid <= 1;
            end
            assign psum_ready = !read_needed || read_valid;
        end else begin: asynchronous_psum_control
            assign psum_ready = 1'b1;
        end
    endgenerate
    scalesim_ws_scheduler #(.ROWS(ROWS), .COLS(COLS), .MAX_M(MAX_M)) scheduler (
        .clk(clk), .rst(rst), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_m(cmd_m), .cmd_k(cmd_k), .cmd_n(cmd_n),
        .cmd_a_base(cmd_a_base), .cmd_b_base(cmd_b_base), .cmd_c_base(cmd_c_base),
        .cycle_ready(cycle_ready && psum_ready), .busy(busy), .done(done), .error(error),
        .accepted(accepted), .advance(cycle_fire), .load_weight(load_weight),
        .first_k(first_k), .last_k(c_final),
        .logical_cycle(logical_cycle), .elapsed_cycles(elapsed_cycles), .stall_cycles(stall_cycles),
        .a_valid(a_valid), .a_addr(a_addr), .b_valid(b_valid), .b_addr(b_addr),
        .c_valid(c_valid), .c_addr(c_addr), .c_row(c_row)
    );
    genvar r, c;
    generate
        for (r=0; r<ROWS; r=r+1) begin: a_lane
            assign a_masked[r*8 +: 8] = a_valid[r] ? a_data[r*8 +: 8] : 8'b0;
        end
        for (c=0; c<COLS; c=c+1) begin: c_lane
            reg signed [31:0] accumulator [0:MAX_M-1];
            wire [ROW_W-1:0] row_index = c_row[c*ROW_W +: ROW_W];
            wire signed [31:0] previous;
            if (SYNC_PSUM != 0) begin: synchronous_read
                reg signed [31:0] read_data;
                // No reset on RAM/read data: first-K writes initialize each
                // location, and read_valid prevents stale data from committing.
                always @(posedge clk) begin
                    if (!rst && !psum_ready && c_valid[c]) read_data <= accumulator[row_index];
                end
                assign previous = first_k ? 32'sd0 : read_data;
            end else begin: asynchronous_read
                assign previous = first_k ? 32'sd0 : accumulator[row_index];
            end
            wire signed [31:0] sum = $signed(partial[c*32 +: 32]) + previous;
            assign b_masked[c*8 +: 8] = b_valid[c] ? b_data[c*8 +: 8] : 8'b0;
            assign c_data[c*32 +: 32] = c_valid[c] ? sum : 32'b0;
            // SRAM contents need no reset: first K fold overwrites before any read.
            always @(posedge clk) begin
                if (!rst && cycle_fire && c_valid[c]) accumulator[row_index] <= sum;
            end
        end
    endgenerate
    scalesim_ws_array #(.ROWS(ROWS), .COLS(COLS)) array (
        .clk(clk), .rst(rst), .clear(accepted), .ce(cycle_fire), .load_weight(load_weight),
        .activations(a_masked), .weights(b_masked), .results(partial)
    );
endmodule
