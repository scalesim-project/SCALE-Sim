// SPDX-License-Identifier: MIT
module scalesim_ws_array #(
    parameter integer ROWS = 32,
    parameter integer COLS = 32,
    parameter integer DATA_W = 8,
    parameter integer ACC_W = 32
) (
    input wire clk, rst, clear, ce, load_weight,
    input wire [ROWS*DATA_W-1:0] activations,
    input wire [COLS*DATA_W-1:0] weights,
    output wire [COLS*ACC_W-1:0] results
);
    wire signed [DATA_W-1:0] w [0:ROWS][0:COLS-1];
    wire signed [DATA_W-1:0] a [0:ROWS-1][0:COLS];
    wire signed [ACC_W-1:0] p [0:ROWS][0:COLS-1];
    wire signed [ACC_W-1:0] now_sum [0:ROWS-1][0:COLS-1];
    genvar r, c;
    generate
        for (c=0; c<COLS; c=c+1) begin: col_input
            assign w[0][c] = weights[c*DATA_W +: DATA_W];
            assign p[0][c] = 0;
            assign results[c*ACC_W +: ACC_W] = now_sum[ROWS-1][c];
        end
        for (r=0; r<ROWS; r=r+1) begin: row
            assign a[r][0] = activations[r*DATA_W +: DATA_W];
            for (c=0; c<COLS; c=c+1) begin: col
                scalesim_ws_pe #(.DATA_W(DATA_W), .ACC_W(ACC_W)) pe (
                    .clk(clk), .rst(rst), .clear(clear), .ce(ce), .load_weight(load_weight),
                    .weight_in(w[r][c]), .weight_out(w[r+1][c]),
                    .activation_in(a[r][c]), .activation_out(a[r][c+1]),
                    .partial_in(p[r][c]), .partial_now(now_sum[r][c]), .partial_out(p[r+1][c])
                );
            end
        end
    endgenerate
endmodule
