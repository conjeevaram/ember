`timescale 1ns/1ps
//
// accumulator : sums x and y over fire pixels, counts them.
// Latches totals at frame_done, then resets for the next frame.
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
    output reg         result_valid
);
    reg [19:0] acc_x, acc_y;
    reg [12:0] acc_count;

    always @(posedge clk) begin
        if (rst) begin
            acc_x <= 0; acc_y <= 0; acc_count <= 0;
            sum_x <= 0; sum_y <= 0; count <= 0;
            result_valid <= 0;
        end else begin
            result_valid <= 0;

            if (frame_done) begin
                // Latch this frame's totals
                sum_x <= acc_x;
                sum_y <= acc_y;
                count <= acc_count;
                result_valid <= 1;
                // Reset for next frame
                acc_x <= 0; acc_y <= 0; acc_count <= 0;
            end else if (valid && is_fire) begin
                acc_x     <= acc_x + x;
                acc_y     <= acc_y + y;
                acc_count <= acc_count + 1;
            end
        end
    end
endmodule
