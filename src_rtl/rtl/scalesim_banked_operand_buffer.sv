`timescale 1ns/1ps
// Two pages of banked byte SRAM; the owner must prevent writes to its active page.
// One read vector is outstanding at a time. Each bank serves its lowest pending
// lane once per edge, including duplicate addresses (no read coalescing).
//
// An accepted request at edge n first reads SRAM at n+1. If L is the maximum
// number of valid active lanes assigned to any bank, L>0 produces rsp_valid just
// after edge n+L+1: the final synchronous read is gathered one edge later.
// Empty/all-invalid requests produce a response at n+1. Responses hold until
// accepted, and a new request can be accepted on the following edge (no bypass).
// Concurrent same-address read/write is read-first. Reset clears transaction
// state and suppresses handshakes immediately, but never clears SRAM contents.
module scalesim_banked_operand_buffer #(
    parameter integer LANES = 32,
    parameter integer BANKS = 32,
    parameter integer PAGE_WORDS = 32768
) (
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   wr_valid,
    output wire                   wr_ready,
    input  wire                   wr_page,
    input  wire [31:0]            wr_addr,
    input  wire [7:0]             wr_data,
    input  wire                   req_valid,
    output wire                   req_ready,
    input  wire                   req_page,
    input  wire [LANES-1:0]        req_mask,
    input  wire [LANES*32-1:0]     req_addr,
    output wire                   rsp_valid,
    input  wire                   rsp_ready,
    output reg  [LANES*8-1:0]      rsp_data,
    output reg                    rsp_error
);
    localparam integer PAGE_ROWS = PAGE_WORDS / BANKS;
    localparam integer BANK_WORDS = 2 * PAGE_ROWS;
    localparam integer ROW_BITS = (BANK_WORDS > 1) ? $clog2(BANK_WORDS) : 1;
    localparam integer LANE_BITS = (LANES > 1) ? $clog2(LANES) : 1;

    initial begin
        if (LANES < 1 || BANKS < 1 || PAGE_WORDS < 1 || PAGE_WORDS % BANKS != 0)
            $fatal(1, "LANES/BANKS/PAGE_WORDS must be positive; PAGE_WORDS must be a multiple of BANKS");
    end

    reg busy;
    reg response_valid;
    reg page_q;
    reg [LANES*32-1:0] addr_q;
    reg [LANES-1:0] pending;
    reg [LANES-1:0] accepted_mask;
    reg accepted_error;
    reg [LANES-1:0] selected_mask;
    reg [BANKS-1:0] bank_read_en;
    reg [ROW_BITS-1:0] bank_read_row [0:BANKS-1];
    reg [LANE_BITS-1:0] selected_lane [0:BANKS-1];
    reg [BANKS-1:0] gather_valid;
    reg [LANE_BITS-1:0] gather_lane [0:BANKS-1];
    wire [7:0] bank_read_data [0:BANKS-1];
    wire [31:0] write_bank = wr_addr % BANKS;
    wire [ROW_BITS-1:0] write_row = ROW_BITS'((wr_page ? PAGE_ROWS : 0) + wr_addr / BANKS);
    wire write_fire = wr_valid && wr_ready;

    assign wr_ready = !rst && wr_addr < PAGE_WORDS;
    assign req_ready = !rst && !busy && !response_valid;
    assign rsp_valid = !rst && response_valid;

    // Invalid active addresses are removed before arbitration, so no out-of-
    // range address can reach a bank read. They still contribute to rsp_error.
    always @* begin
        accepted_mask = {LANES{1'b0}};
        accepted_error = 1'b0;
        for (integer lane = 0; lane < LANES; lane = lane + 1) begin
            if (req_mask[lane]) begin
                if (req_addr[lane*32 +: 32] < PAGE_WORDS)
                    accepted_mask[lane] = 1'b1;
                else
                    accepted_error = 1'b1;
            end
        end
    end

    always @* begin
        selected_mask = {LANES{1'b0}};
        bank_read_en = {BANKS{1'b0}};
        for (integer bank = 0; bank < BANKS; bank = bank + 1) begin
            bank_read_row[bank] = {ROW_BITS{1'b0}};
            selected_lane[bank] = {LANE_BITS{1'b0}};
            for (integer lane = 0; lane < LANES; lane = lane + 1) begin
                if (!rst && busy && pending[lane] && !bank_read_en[bank] &&
                    addr_q[lane*32 +: 32] % BANKS == bank) begin
                    bank_read_en[bank] = 1'b1;
                    selected_mask[lane] = 1'b1;
                    selected_lane[bank] = LANE_BITS'(lane);
                    bank_read_row[bank] = ROW_BITS'((page_q ? PAGE_ROWS : 0) +
                                                        addr_q[lane*32 +: 32] / BANKS);
                end
            end
        end
    end

    // A separate 1R/1W SRAM per bank. Nonblocking assignments implement true
    // synchronous reads and read-first collision semantics without a memory reset.
    generate
        for (genvar bank = 0; bank < BANKS; bank = bank + 1) begin : banks
            reg [7:0] mem [0:BANK_WORDS-1];
            reg [7:0] read_q;
            assign bank_read_data[bank] = read_q;
            always @(posedge clk) begin
                if (write_fire && write_bank == bank)
                    mem[write_row] <= wr_data;
                if (bank_read_en[bank])
                    read_q <= mem[bank_read_row[bank]];
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            response_valid <= 1'b0;
            page_q <= 1'b0;
            addr_q <= {LANES*32{1'b0}};
            pending <= {LANES{1'b0}};
            gather_valid <= {BANKS{1'b0}};
            rsp_data <= {LANES*8{1'b0}};
            rsp_error <= 1'b0;
            for (integer bank = 0; bank < BANKS; bank = bank + 1)
                gather_lane[bank] <= {LANE_BITS{1'b0}};
        end else begin
            gather_valid <= bank_read_en;
            for (integer bank = 0; bank < BANKS; bank = bank + 1) begin
                if (bank_read_en[bank])
                    gather_lane[bank] <= selected_lane[bank];
                if (gather_valid[bank])
                    rsp_data[gather_lane[bank]*8 +: 8] <= bank_read_data[bank];
            end
            if (rsp_valid && rsp_ready)
                response_valid <= 1'b0;
            if (req_valid && req_ready) begin
                busy <= 1'b1;
                page_q <= req_page;
                addr_q <= req_addr;
                pending <= accepted_mask;
                rsp_data <= {LANES*8{1'b0}};
                rsp_error <= accepted_error;
            end else if (busy) begin
                pending <= pending & ~selected_mask;
                // When pending is empty, this edge gathers the final preceding
                // SRAM read (if any) and publishes the entire response together.
                if (pending == {LANES{1'b0}}) begin
                    busy <= 1'b0;
                    response_valid <= 1'b1;
                end
            end
        end
    end
endmodule
