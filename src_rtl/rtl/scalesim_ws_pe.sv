// SPDX-License-Identifier: MIT
// Original dense signed-integer WS processing element.
module scalesim_ws_pe #(
    parameter integer DATA_W = 8,
    parameter integer ACC_W = 32
) (
    input  wire clk, rst, clear, ce, load_weight,
    input  wire signed [DATA_W-1:0] weight_in,
    output wire signed [DATA_W-1:0] weight_out,
    input  wire signed [DATA_W-1:0] activation_in,
    output reg  signed [DATA_W-1:0] activation_out,
    input  wire signed [ACC_W-1:0] partial_in,
    output wire signed [ACC_W-1:0] partial_now,
    output reg  signed [ACC_W-1:0] partial_out
);
    reg signed [DATA_W-1:0] weight;
    wire signed [2*DATA_W-1:0] product = activation_in * weight;
    assign weight_out = weight;
    // Assignment intentionally wraps modulo 2**ACC_W.
    assign partial_now = partial_in + {{(ACC_W-2*DATA_W){product[2*DATA_W-1]}}, product};
    always @(posedge clk) begin
        if (rst || clear) begin
            weight <= 0;
            activation_out <= 0;
            partial_out <= 0;
        end else if (ce) begin
            if (load_weight) weight <= weight_in;
            activation_out <= activation_in;
            partial_out <= partial_now;
        end
    end
endmodule
