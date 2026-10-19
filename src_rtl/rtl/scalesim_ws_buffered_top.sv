// SPDX-License-Identifier: MIT
// Software-managed, two-page operand SRAM wrapper for the dense WS engine.
// Page-local byte writes preload A/B; command bases remain logical word bases.
// This concrete synchronous memory policy is not SCALE-Sim's CALC/USER model.
module scalesim_ws_buffered_top #(
    parameter integer ROWS = 32,
    parameter integer COLS = 32,
    parameter integer MAX_M = 512,
    parameter integer A_BANKS = ROWS,
    parameter integer B_BANKS = COLS,
    parameter integer A_PAGE_WORDS = 32768,
    parameter integer B_PAGE_WORDS = 32768,
    parameter integer OUTPUT_DEPTH = 2
) (
    input wire clk, rst,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [31:0] cmd_m, cmd_k, cmd_n,
    input wire [31:0] cmd_a_base, cmd_b_base, cmd_c_base,
    input wire cmd_page,
    output wire busy,
    output reg done, error,
    input wire a_load_valid, a_load_page,
    input wire [31:0] a_load_addr,
    input wire [7:0] a_load_data,
    output wire a_load_ready,
    input wire b_load_valid, b_load_page,
    input wire [31:0] b_load_addr,
    input wire [7:0] b_load_data,
    output wire b_load_ready,
    output wire out_valid,
    input wire out_ready,
    output wire [COLS-1:0] out_mask,
    output wire [COLS*32-1:0] out_addr, out_data,
    output wire out_final,
    output wire [1:0] page_busy,
    output wire [63:0] logical_cycle, core_elapsed_cycles, core_stall_cycles,
    output reg [63:0] transaction_cycles,
    output wire cycle_fire
);
    localparam [1:0] IDLE=2'd0, ACTIVE=2'd1, DRAIN=2'd2, ABORT=2'd3;
    localparam integer PTR_W = (OUTPUT_DEPTH > 1) ? $clog2(OUTPUT_DEPTH) : 1;
    localparam integer COUNT_W = $clog2(OUTPUT_DEPTH+1);
    initial begin
        if (OUTPUT_DEPTH < 1)
            $fatal(1, "OUTPUT_DEPTH must be positive");
    end
    reg [1:0] state;
    reg active_page, completion_error, abort_reset;
    reg [31:0] active_a_base, active_b_base;
    wire local_reset = rst || abort_reset;
    wire core_cmd_ready, core_busy, core_done, core_error;
    wire [ROWS-1:0] core_a_valid;
    wire [COLS-1:0] core_b_valid, core_c_valid;
    wire [ROWS*32-1:0] core_a_addr;
    wire [COLS*32-1:0] core_b_addr, core_c_addr, core_c_data;
    wire core_c_final;
    reg [ROWS*8-1:0] a_values;
    reg [COLS*8-1:0] b_values;
    reg a_sent, b_sent, a_complete, b_complete;
    wire [ROWS*32-1:0] a_local_addr;
    wire [COLS*32-1:0] b_local_addr;
    wire a_req_ready, b_req_ready, a_rsp_valid, b_rsp_valid;
    wire a_rsp_error, b_rsp_error;
    wire [ROWS*8-1:0] a_rsp_data;
    wire [COLS*8-1:0] b_rsp_data;
    wire a_wr_ready, b_wr_ready;

    // Small output packet FIFO. Packets include every intermediate K-fold write.
    reg [COLS-1:0] fifo_mask [0:OUTPUT_DEPTH-1];
    reg [COLS*32-1:0] fifo_addr [0:OUTPUT_DEPTH-1];
    reg [COLS*32-1:0] fifo_data [0:OUTPUT_DEPTH-1];
    reg fifo_final [0:OUTPUT_DEPTH-1];
    reg [PTR_W-1:0] read_ptr, write_ptr;
    reg [COUNT_W-1:0] fifo_count;
    assign out_valid = !local_reset && (fifo_count != 0);
    assign out_mask = out_valid ? fifo_mask[read_ptr] : {COLS{1'b0}};
    assign out_addr = out_valid ? fifo_addr[read_ptr] : {COLS*32{1'b0}};
    assign out_data = out_valid ? fifo_data[read_ptr] : {COLS*32{1'b0}};
    assign out_final = out_valid && fifo_final[read_ptr];
    wire pop = out_valid && out_ready;
    wire fifo_space = (fifo_count < COUNT_W'(OUTPUT_DEPTH)) || pop;
    wire push = cycle_fire && (|core_c_valid);

    assign cmd_ready = (state == IDLE) && core_cmd_ready && !local_reset;
    wire accept_command = cmd_valid && cmd_ready;
    wire [63:0] a_size = {32'b0,cmd_m} * {32'b0,cmd_k};
    wire [63:0] b_size = {32'b0,cmd_k} * {32'b0,cmd_n};
    wire capacity_error = (a_size > 64'(A_PAGE_WORDS)) ||
                          (b_size > 64'(B_PAGE_WORDS));
    // A rejected capacity follows the core's invalid-command completion path,
    // which resets its counters and never exposes a memory request.
    wire [31:0] checked_m = capacity_error ? 32'b0 : cmd_m;
    assign busy = (state != IDLE) && !rst;
    assign page_busy = !local_reset && (state != IDLE) ?
                       (active_page ? 2'b10 : 2'b01) : 2'b00;
    // Command acceptance owns the selected page before any coincident write.
    wire a_page_writable = !local_reset &&
        !((state != IDLE) && (active_page == a_load_page)) &&
        !(accept_command && (cmd_page == a_load_page));
    wire b_page_writable = !local_reset &&
        !((state != IDLE) && (active_page == b_load_page)) &&
        !(accept_command && (cmd_page == b_load_page));
    assign a_load_ready = a_page_writable && a_wr_ready;
    assign b_load_ready = b_page_writable && b_wr_ready;

    wire reading = (state == ACTIVE) && core_busy && !local_reset;
    wire a_req_valid = reading && !a_sent && !a_complete && (|core_a_valid);
    wire b_req_valid = reading && !b_sent && !b_complete && (|core_b_valid);
    wire a_rsp_ready = reading && !a_complete;
    wire b_rsp_ready = reading && !b_complete;
    wire fault_event = (a_rsp_valid && a_rsp_ready && a_rsp_error) ||
                       (b_rsp_valid && b_rsp_ready && b_rsp_error);
    wire operands_ready = ((~|core_a_valid) || a_complete) &&
                          ((~|core_b_valid) || b_complete);
    wire core_ready = reading && operands_ready &&
                      ((~|core_c_valid) || fifo_space) && !fault_event;

    genvar lane;
    generate
        for (lane=0; lane<ROWS; lane=lane+1) begin: a_address
            assign a_local_addr[lane*32 +: 32] =
                core_a_addr[lane*32 +: 32] - active_a_base;
        end
        for (lane=0; lane<COLS; lane=lane+1) begin: b_address
            assign b_local_addr[lane*32 +: 32] =
                core_b_addr[lane*32 +: 32] - active_b_base;
        end
    endgenerate

    scalesim_banked_operand_buffer #(
        .LANES(ROWS), .BANKS(A_BANKS), .PAGE_WORDS(A_PAGE_WORDS)
    ) a_buffer (
        .clk(clk), .rst(local_reset),
        .wr_valid(a_load_valid && a_page_writable), .wr_ready(a_wr_ready),
        .wr_page(a_load_page), .wr_addr(a_load_addr), .wr_data(a_load_data),
        .req_valid(a_req_valid), .req_ready(a_req_ready), .req_page(active_page),
        .req_mask(core_a_valid), .req_addr(a_local_addr),
        .rsp_valid(a_rsp_valid), .rsp_ready(a_rsp_ready),
        .rsp_data(a_rsp_data), .rsp_error(a_rsp_error)
    );
    scalesim_banked_operand_buffer #(
        .LANES(COLS), .BANKS(B_BANKS), .PAGE_WORDS(B_PAGE_WORDS)
    ) b_buffer (
        .clk(clk), .rst(local_reset),
        .wr_valid(b_load_valid && b_page_writable), .wr_ready(b_wr_ready),
        .wr_page(b_load_page), .wr_addr(b_load_addr), .wr_data(b_load_data),
        .req_valid(b_req_valid), .req_ready(b_req_ready), .req_page(active_page),
        .req_mask(core_b_valid), .req_addr(b_local_addr),
        .rsp_valid(b_rsp_valid), .rsp_ready(b_rsp_ready),
        .rsp_data(b_rsp_data), .rsp_error(b_rsp_error)
    );
    scalesim_ws_top #(.ROWS(ROWS), .COLS(COLS), .MAX_M(MAX_M), .SYNC_PSUM(1)) core (
        .clk(clk), .rst(local_reset),
        .cmd_valid(accept_command), .cmd_ready(core_cmd_ready),
        .cmd_m(checked_m), .cmd_k(cmd_k), .cmd_n(cmd_n),
        .cmd_a_base(cmd_a_base), .cmd_b_base(cmd_b_base), .cmd_c_base(cmd_c_base),
        .cycle_ready(core_ready), .busy(core_busy), .done(core_done), .error(core_error),
        .logical_cycle(logical_cycle), .elapsed_cycles(core_elapsed_cycles),
        .stall_cycles(core_stall_cycles), .cycle_fire(cycle_fire),
        .a_valid(core_a_valid), .a_addr(core_a_addr), .a_data(a_values),
        .b_valid(core_b_valid), .b_addr(core_b_addr), .b_data(b_values),
        .c_valid(core_c_valid), .c_addr(core_c_addr), .c_data(core_c_data), .c_final(core_c_final)
    );

    always @(posedge clk) begin
        if (local_reset) begin
            read_ptr <= 0; write_ptr <= 0; fifo_count <= 0;
        end else begin
            case ({push,pop})
                2'b10: fifo_count <= fifo_count + 1'b1;
                2'b01: fifo_count <= fifo_count - 1'b1;
                default: begin end
            endcase
            if (push) begin
                fifo_mask[write_ptr] <= core_c_valid;
                fifo_addr[write_ptr] <= core_c_addr;
                fifo_data[write_ptr] <= core_c_data;
                fifo_final[write_ptr] <= core_c_final;
                write_ptr <= (write_ptr == PTR_W'(OUTPUT_DEPTH-1)) ?
                             {PTR_W{1'b0}} : write_ptr + 1'b1;
            end
            if (pop)
                read_ptr <= (read_ptr == PTR_W'(OUTPUT_DEPTH-1)) ?
                            {PTR_W{1'b0}} : read_ptr + 1'b1;
        end
    end

    always @(posedge clk) begin
        if (local_reset || accept_command || cycle_fire) begin
            a_sent <= 0; b_sent <= 0; a_complete <= 0; b_complete <= 0;
            a_values <= 0; b_values <= 0;
        end else if (reading) begin
            if (a_req_valid && a_req_ready) a_sent <= 1;
            if (b_req_valid && b_req_ready) b_sent <= 1;
            if (a_rsp_valid && a_rsp_ready) begin
                a_values <= a_rsp_data; a_complete <= 1;
            end
            if (b_rsp_valid && b_rsp_ready) begin
                b_values <= b_rsp_data; b_complete <= 1;
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; active_page <= 0; active_a_base <= 0; active_b_base <= 0;
            done <= 0; error <= 0; completion_error <= 0; abort_reset <= 0;
            transaction_cycles <= 0;
        end else begin
            done <= 0; error <= 0; abort_reset <= 0;
            if (state != IDLE) transaction_cycles <= transaction_cycles + 1'b1;
            if (accept_command) begin
                state <= ACTIVE; active_page <= cmd_page;
                active_a_base <= cmd_a_base; active_b_base <= cmd_b_base;
                transaction_cycles <= 0; completion_error <= 0;
            end else if (fault_event) begin
                state <= ABORT; abort_reset <= 1; completion_error <= 1;
            end else if (state == ABORT) begin
                state <= IDLE; done <= 1; error <= 1;
            end else if ((state == ACTIVE) && core_done) begin
                completion_error <= core_error;
                if ((fifo_count == 0) || ((fifo_count == 1) && pop)) begin
                    state <= IDLE; done <= 1; error <= core_error;
                end else state <= DRAIN;
            end else if ((state == DRAIN) &&
                         ((fifo_count == 0) || ((fifo_count == 1) && pop))) begin
                state <= IDLE; done <= 1; error <= completion_error;
            end
        end
    end
endmodule
