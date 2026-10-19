// SPDX-License-Identifier: MIT
// AXI4 DMA integration for one dense GEMM tile. Full-width aligned reads may
// overfetch the first/last beat; full-width writes use four byte strobes per C
// word. One outstanding AXI transaction, one implicit ID, no exclusive accesses.
module scalesim_ws_axi_top #(
    parameter integer ROWS = 32,
    parameter integer COLS = 32,
    parameter integer MAX_M = 512,
    parameter integer A_BANKS = ROWS,
    parameter integer B_BANKS = COLS,
    parameter integer A_PAGE_WORDS = 32768,
    parameter integer B_PAGE_WORDS = 32768,
    parameter integer OUTPUT_DEPTH = 2,
    parameter integer AXI_DATA_WIDTH = 512,
    parameter integer MAX_BURST_BEATS = 16,
    parameter integer WRITE_INTERMEDIATES = 0
) (
    input wire clk, rst,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [31:0] cmd_m, cmd_k, cmd_n,
    input wire [63:0] cmd_a_addr, cmd_b_addr, cmd_c_addr,
    input wire cmd_page,
    output wire busy,
    output reg done, error,
    output reg [3:0] error_code,
    output wire [63:0] logical_cycle, core_elapsed_cycles, core_stall_cycles,
    output reg [63:0] total_cycles, operand_bytes, result_bytes,
    output wire cycle_fire,

    output wire [63:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    output wire m_axi_arvalid,
    input wire m_axi_arready,
    input wire [AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input wire [1:0] m_axi_rresp,
    input wire m_axi_rlast, m_axi_rvalid,
    output wire m_axi_rready,

    output wire [63:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output wire m_axi_awvalid,
    input wire m_axi_awready,
    output wire [AXI_DATA_WIDTH-1:0] m_axi_wdata,
    output wire [AXI_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output wire m_axi_wlast, m_axi_wvalid,
    input wire m_axi_wready,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output wire m_axi_bready
);
    localparam integer BUS_BYTES = AXI_DATA_WIDTH/8;
    localparam integer BUS_SHIFT = $clog2(BUS_BYTES);
    localparam [3:0] IDLE=0, READ_ADDRESS=1, READ_DATA=2, UNPACK=3,
                     START_CORE=4, RUN=5, SELECT_WRITE=6, WRITE=7, FAIL=8;
    localparam [3:0] BAD_COMMAND=1, READ_ERROR=2, READ_LAST_ERROR=3,
                     WRITE_ERROR=4, CORE_ERROR=5;
    initial begin
        if (AXI_DATA_WIDTH < 32 || AXI_DATA_WIDTH > 1024 ||
            (AXI_DATA_WIDTH & (AXI_DATA_WIDTH-1)) != 0)
            $fatal(1, "AXI_DATA_WIDTH must be a power of two from 32 to 1024");
        if (MAX_BURST_BEATS < 1 || MAX_BURST_BEATS > 256)
            $fatal(1, "MAX_BURST_BEATS must be from 1 to 256");
    end

    reg [3:0] state;
    reg [31:0] m_q, k_q, n_q, b_words_q;
    reg [63:0] b_base_q, c_base_q;
    reg page_q, loading_b, core_finished, core_failed;
    reg [63:0] read_addr;
    reg [31:0] beats_left, burst_left, useful_left, local_offset;
    reg [BUS_SHIFT:0] first_skip, byte_index, byte_end;
    reg [AXI_DATA_WIDTH-1:0] read_data;
    reg burst_failed;
    reg [COLS-1:0] packet_mask;
    reg [COLS*32-1:0] packet_addr, packet_data;
    reg [63:0] write_addr;
    reg [AXI_DATA_WIDTH-1:0] write_data;
    reg [BUS_BYTES-1:0] write_strb;
    reg aw_sent, w_sent;

    wire [63:0] a_words = {32'b0,cmd_m} * {32'b0,cmd_k};
    wire [63:0] b_words = {32'b0,cmd_k} * {32'b0,cmd_n};
    wire [63:0] c_words = {32'b0,cmd_m} * {32'b0,cmd_n};
    wire [66:0] a_end = {3'b0,cmd_a_addr} + {3'b0,a_words} - 67'd1;
    wire [66:0] b_end = {3'b0,cmd_b_addr} + {3'b0,b_words} - 67'd1;
    wire [66:0] c_end = {3'b0,cmd_c_addr} + {1'b0,c_words,2'b0} - 67'd1;
    wire invalid_command = cmd_m == 0 || cmd_k == 0 || cmd_n == 0 ||
        cmd_m > MAX_M || a_words > 64'(A_PAGE_WORDS) ||
        b_words > 64'(B_PAGE_WORDS) || c_words > 64'h100000000 ||
        (|a_end[66:64]) || (|b_end[66:64]) || (|c_end[66:64]) ||
        (|cmd_c_addr[1:0]);
    wire [63:0] bus_mask = 64'(BUS_BYTES-1);
    wire [31:0] boundary_beats = (32'd4096 - {20'b0,read_addr[11:0]}) >> BUS_SHIFT;
    wire [31:0] limited_beats = beats_left < 32'(MAX_BURST_BEATS) ?
                                beats_left : 32'(MAX_BURST_BEATS);
    wire [31:0] burst_beats = limited_beats < boundary_beats ? limited_beats : boundary_beats;
    wire read_fault = m_axi_rresp != 0;
    wire last_fault = m_axi_rlast != (burst_left == 1);

    assign cmd_ready = state == IDLE && !rst;
    assign busy = state != IDLE && !rst;
    assign m_axi_araddr = read_addr;
    assign m_axi_arlen = 8'(burst_beats-1);
    assign m_axi_arsize = 3'(BUS_SHIFT);
    assign m_axi_arburst = 2'b01;
    assign m_axi_arvalid = state == READ_ADDRESS && !rst;
    assign m_axi_rready = state == READ_DATA && !rst;
    assign m_axi_awaddr = write_addr;
    assign m_axi_awlen = 0;
    assign m_axi_awsize = 3'(BUS_SHIFT);
    assign m_axi_awburst = 2'b01;
    assign m_axi_awvalid = state == WRITE && !aw_sent && !rst;
    assign m_axi_wdata = write_data;
    assign m_axi_wstrb = write_strb;
    assign m_axi_wlast = 1'b1;
    assign m_axi_wvalid = state == WRITE && !w_sent && !rst;
    assign m_axi_bready = state == WRITE && aw_sent && w_sent && !rst;

    wire core_cmd_ready, core_busy, core_done, core_error;
    wire a_load_ready, b_load_ready, out_valid, out_final;
    wire [COLS-1:0] out_mask;
    wire [COLS*32-1:0] out_addr, out_data;
    wire [1:0] page_busy_unused;
    wire [63:0] transaction_cycles_unused;
    wire core_reset = rst || state == FAIL || (cmd_valid && cmd_ready);
    wire core_fault = core_failed || (core_done && core_error);
    wire out_ready = state == RUN && !core_fault && !rst;
    wire load_valid = state == UNPACK && !rst;
    wire load_ready = loading_b ? b_load_ready : a_load_ready;

    scalesim_ws_buffered_top #(
        .ROWS(ROWS), .COLS(COLS), .MAX_M(MAX_M), .A_BANKS(A_BANKS), .B_BANKS(B_BANKS),
        .A_PAGE_WORDS(A_PAGE_WORDS), .B_PAGE_WORDS(B_PAGE_WORDS), .OUTPUT_DEPTH(OUTPUT_DEPTH)
    ) engine (
        .clk(clk), .rst(core_reset), .cmd_valid(state == START_CORE && !rst),
        .cmd_ready(core_cmd_ready), .cmd_m(m_q), .cmd_k(k_q), .cmd_n(n_q),
        .cmd_a_base(32'b0), .cmd_b_base(32'b0), .cmd_c_base(32'b0), .cmd_page(page_q),
        .busy(core_busy), .done(core_done), .error(core_error),
        .a_load_valid(load_valid && !loading_b), .a_load_page(page_q),
        .a_load_addr(local_offset), .a_load_data(read_data[byte_index*8 +: 8]),
        .a_load_ready(a_load_ready), .b_load_valid(load_valid && loading_b),
        .b_load_page(page_q), .b_load_addr(local_offset),
        .b_load_data(read_data[byte_index*8 +: 8]), .b_load_ready(b_load_ready),
        .out_valid(out_valid), .out_ready(out_ready), .out_mask(out_mask),
        .out_addr(out_addr), .out_data(out_data), .out_final(out_final),
        .page_busy(page_busy_unused), .logical_cycle(logical_cycle),
        .core_elapsed_cycles(core_elapsed_cycles), .core_stall_cycles(core_stall_cycles),
        .transaction_cycles(transaction_cycles_unused), .cycle_fire(cycle_fire)
    );

    integer selected_lane;
    reg found_lane;
    always @* begin
        selected_lane = 0;
        found_lane = 0;
        for (integer lane=0; lane<COLS; lane=lane+1) begin
            if (packet_mask[lane] && !found_lane) begin
                selected_lane = lane;
                found_lane = 1;
            end
        end
    end
    wire [63:0] selected_addr = c_base_q + ({32'b0,packet_addr[selected_lane*32 +: 32]} << 2);
    wire [BUS_SHIFT-1:0] selected_offset = selected_addr[BUS_SHIFT-1:0];

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE; done <= 0; error <= 0; error_code <= 0;
            total_cycles <= 0; operand_bytes <= 0; result_bytes <= 0;
            m_q <= 0; k_q <= 0; n_q <= 0; b_words_q <= 0;
            b_base_q <= 0; c_base_q <= 0; page_q <= 0; loading_b <= 0;
            core_finished <= 0; core_failed <= 0;
            read_addr <= 0; beats_left <= 0; burst_left <= 0; useful_left <= 0;
            local_offset <= 0; first_skip <= 0; byte_index <= 0; byte_end <= 0;
            read_data <= 0; burst_failed <= 0;
            packet_mask <= 0; packet_addr <= 0; packet_data <= 0;
            write_addr <= 0; write_data <= 0; write_strb <= 0;
            aw_sent <= 0; w_sent <= 0;
        end else begin
            done <= 0; error <= 0;
            if (state != IDLE) total_cycles <= total_cycles + 1'b1;
            if (state != IDLE && core_done) begin
                core_finished <= 1;
                if (core_error) core_failed <= 1;
            end
            case (state)
                IDLE: if (cmd_valid) begin
                    total_cycles <= 0; operand_bytes <= 0; result_bytes <= 0;
                    error_code <= 0; core_finished <= 0; core_failed <= 0;
                    if (invalid_command) begin
                        done <= 1; error <= 1; error_code <= BAD_COMMAND;
                    end else begin
                        m_q <= cmd_m; k_q <= cmd_k; n_q <= cmd_n;
                        b_words_q <= b_words[31:0];
                        b_base_q <= cmd_b_addr; c_base_q <= cmd_c_addr; page_q <= cmd_page;
                        read_addr <= cmd_a_addr & ~bus_mask;
                        beats_left <= 32'(((cmd_a_addr & bus_mask) + a_words + bus_mask) >> BUS_SHIFT);
                        useful_left <= a_words[31:0]; local_offset <= 0;
                        first_skip <= (BUS_SHIFT+1)'(cmd_a_addr & bus_mask);
                        loading_b <= 0; state <= READ_ADDRESS;
                    end
                end
                READ_ADDRESS: if (m_axi_arready) begin
                    burst_left <= burst_beats; burst_failed <= 0;
                    state <= READ_DATA;
                end
                READ_DATA: if (m_axi_rvalid) begin
                    read_addr <= read_addr + 64'(BUS_BYTES);
                    beats_left <= beats_left - 1'b1;
                    burst_left <= burst_left - 1'b1;
                    if (read_fault || last_fault || burst_failed) begin
                        burst_failed <= 1;
                        if (!burst_failed) error_code <= read_fault ? READ_ERROR : READ_LAST_ERROR;
                        // Do not abandon a legal AXI burst after an RRESP error.
                        // Drain its requested beat count before completing failure.
                        if (burst_left == 1) state <= FAIL;
                    end else begin
                        read_data <= m_axi_rdata;
                        byte_index <= first_skip;
                        byte_end <= useful_left < 32'(BUS_BYTES-first_skip) ?
                                    first_skip + (BUS_SHIFT+1)'(useful_left) : (BUS_SHIFT+1)'(BUS_BYTES);
                        first_skip <= 0; state <= UNPACK;
                    end
                end
                UNPACK: if (load_ready) begin
                    local_offset <= local_offset + 1'b1;
                    useful_left <= useful_left - 1'b1;
                    operand_bytes <= operand_bytes + 1'b1;
                    byte_index <= byte_index + 1'b1;
                    if (byte_index + 1'b1 == byte_end) begin
                        if (burst_left != 0) state <= READ_DATA;
                        else if (beats_left != 0) state <= READ_ADDRESS;
                        else if (!loading_b) begin
                            loading_b <= 1;
                            read_addr <= b_base_q & ~bus_mask;
                            beats_left <= 32'(((b_base_q & bus_mask) + {32'b0,b_words_q} + bus_mask) >> BUS_SHIFT);
                            useful_left <= b_words_q; local_offset <= 0;
                            first_skip <= (BUS_SHIFT+1)'(b_base_q & bus_mask);
                            state <= READ_ADDRESS;
                        end else state <= START_CORE;
                    end
                end
                START_CORE: if (core_cmd_ready) state <= RUN;
                RUN: begin
                    if (core_fault) begin
                        state <= FAIL; error_code <= CORE_ERROR;
                    end else if (out_valid) begin
                        if (out_final || WRITE_INTERMEDIATES != 0) begin
                            packet_mask <= out_mask; packet_addr <= out_addr; packet_data <= out_data;
                            state <= SELECT_WRITE;
                        end
                    end else if (core_finished) begin
                        state <= IDLE; done <= 1;
                    end
                end
                SELECT_WRITE: begin
                    if (core_fault) begin
                        state <= FAIL; error_code <= CORE_ERROR;
                    end else if (found_lane) begin
                        write_addr <= selected_addr & ~bus_mask;
                        write_data <= AXI_DATA_WIDTH'(packet_data[selected_lane*32 +: 32]) << (selected_offset*8);
                        write_strb <= BUS_BYTES'(4'hf) << selected_offset;
                        packet_mask[selected_lane] <= 0;
                        aw_sent <= 0; w_sent <= 0; state <= WRITE;
                    end else state <= RUN;
                end
                WRITE: begin
                    if (m_axi_awvalid && m_axi_awready) aw_sent <= 1;
                    if (m_axi_wvalid && m_axi_wready) w_sent <= 1;
                    if (m_axi_bvalid && m_axi_bready) begin
                        if (m_axi_bresp != 0) begin
                            state <= FAIL; error_code <= WRITE_ERROR;
                        end else begin
                            result_bytes <= result_bytes + 64'd4;
                            if (core_fault) begin state <= FAIL; error_code <= CORE_ERROR; end
                            else state <= SELECT_WRITE;
                        end
                    end
                end
                FAIL: begin
                    state <= IDLE; done <= 1; error <= 1;
                end
                default: state <= IDLE;
            endcase
        end
    end

    // These are intentionally unexported integration status signals.
    wire unused_status = core_busy ^ (|page_busy_unused) ^
                         (|transaction_cycles_unused) ^ (|a_end[63:0]) ^
                         (|b_end[63:0]) ^ (|c_end[63:0]);
endmodule
