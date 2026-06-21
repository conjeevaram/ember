`timescale 1ns/1ps
//
// accumulator : centroid sums + min/max y for flame-base targeting (Option 3).
// Downstream computes: target_x = sum_x/count,
//                      target_y = max_y - (max_y - min_y)/8   (~12.5% up from base)
//
module accumulator (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid,
    input  wire        is_fire,
    input  wire [6:0]  x,
    input  wire [6:0]  y,
    input  wire        frame_done,
    output reg  [19:0] sum_x,
    output reg  [19:0] sum_y,
    output reg  [12:0] count,
    output reg  [6:0]  min_y,     // top of flame (smallest y)
    output reg  [6:0]  max_y,     // bottom of flame (largest y)
    output reg         result_valid
);
    reg [19:0] acc_x, acc_y;
    reg [12:0] acc_count;
    reg [6:0]  acc_min_y, acc_max_y;

    always @(posedge clk) begin
        if (rst) begin
            acc_x <= 0; acc_y <= 0; acc_count <= 0;
            acc_min_y <= 7'd127;   // start high so first fire pixel lowers it
            acc_max_y <= 7'd0;     // start low so first fire pixel raises it
            sum_x <= 0; sum_y <= 0; count <= 0;
            min_y <= 0; max_y <= 0;
            result_valid <= 0;
        end else begin
            result_valid <= 0;

            if (frame_done) begin
                sum_x <= acc_x;
                sum_y <= acc_y;
                count <= acc_count;
                min_y <= acc_min_y;
                max_y <= acc_max_y;
                result_valid <= 1;
                // reset for next frame
                acc_x <= 0; acc_y <= 0; acc_count <= 0;
                acc_min_y <= 7'd127;
                acc_max_y <= 7'd0;
            end else if (valid && is_fire) begin
                acc_x     <= acc_x + x;
                acc_y     <= acc_y + y;
                acc_count <= acc_count + 1;
                if (y < acc_min_y) acc_min_y <= y;
                if (y > acc_max_y) acc_max_y <= y;
            end
        end
    end
endmodule
