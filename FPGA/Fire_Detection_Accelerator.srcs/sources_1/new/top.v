`timescale 1ns/1ps
module top #(
    parameter MEM_FILE = "image.mem"
)(
    input  wire clk,
    input  wire rst,
    output wire uart_tx
);
    wire [11:0] addr;
    wire [6:0]  x_raw, y_raw;
    wire        valid_raw, frame_done_raw;
    wire [23:0] pixel;
    wire        is_fire;
    wire [19:0] sum_x, sum_y;
    wire [12:0] count;
    wire [6:0]  min_y, max_y;
    wire        result_valid;

    addr_gen #(.WIDTH(64), .HEIGHT(64)) ag (
        .clk(clk), .rst(rst), .enable(1'b1),
        .addr(addr), .x(x_raw), .y(y_raw),
        .valid(valid_raw), .frame_done(frame_done_raw)
    );

    image_rom #(.MEM_FILE(MEM_FILE)) rom (
        .clk(clk), .addr(addr), .data(pixel)
    );

    reg [6:0] x_d, y_d;
    reg       valid_d, frame_done_d;
    always @(posedge clk) begin
        if (rst) begin
            x_d <= 0; y_d <= 0; valid_d <= 0; frame_done_d <= 0;
        end else begin
            x_d <= x_raw; y_d <= y_raw;
            valid_d <= valid_raw; frame_done_d <= frame_done_raw;
        end
    end

    fire_detect fd (.pixel(pixel), .is_fire(is_fire));

    accumulator acc (
        .clk(clk), .rst(rst),
        .valid(valid_d), .is_fire(is_fire),
        .x(x_d), .y(y_d), .frame_done(frame_done_d),
        .sum_x(sum_x), .sum_y(sum_y), .count(count),
        .min_y(min_y), .max_y(max_y),
        .result_valid(result_valid)
    );

    uart_tx_bytes utx (
        .clk(clk), .reset(rst),
        .start(result_valid),
        .sum_x(sum_x), .sum_y(sum_y), .count(count),
        .min_y(min_y), .max_y(max_y),
        .uart_tx(uart_tx), .busy()
    );
endmodule
