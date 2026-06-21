`timescale 1ns/1ps
//
// top : wires image_rom + addr_gen + fire_detect + accumulator.
// The 1-clock ROM read latency is compensated HERE (Option B):
// x/y/valid are delayed one clock so they align with the pixel they describe.
//
module top (
    input  wire        clk,
    input  wire        rst,
    output wire [19:0] sum_x,
    output wire [19:0] sum_y,
    output wire [12:0] count,
    output wire        result_valid
);
    // ---- addr_gen -> produces addr + raw coordinates ----
    wire [11:0] addr;
    wire [6:0]  x_raw, y_raw;
    wire        valid_raw, frame_done_raw;

    addr_gen #(.WIDTH(64), .HEIGHT(64)) ag (
        .clk(clk), .rst(rst), .enable(1'b1),
        .addr(addr), .x(x_raw), .y(y_raw),
        .valid(valid_raw), .frame_done(frame_done_raw)
    );

    // ---- image_rom -> pixel comes out 1 clock after addr ----
    wire [23:0] pixel;
    image_rom #(.MEM_FILE("pjeevy1.mem")) rom (
        .clk(clk), .addr(addr), .data(pixel)
    );

    // ---- ALIGNMENT: delay coords by 1 clock to match ROM latency ----
    reg [6:0] x_d, y_d;
    reg       valid_d, frame_done_d;
    always @(posedge clk) begin
        if (rst) begin
            x_d <= 0; y_d <= 0; valid_d <= 0; frame_done_d <= 0;
        end else begin
            x_d          <= x_raw;
            y_d          <= y_raw;
            valid_d      <= valid_raw;
            frame_done_d <= frame_done_raw;
        end
    end

    // ---- fire_detect on the (now 1-clock-late) pixel ----
    wire is_fire;
    fire_detect fd (.pixel(pixel), .is_fire(is_fire));

    // ---- accumulator: pixel + delayed coords now line up ----
    accumulator acc (
        .clk(clk), .rst(rst),
        .valid(valid_d), .is_fire(is_fire),
        .x(x_d), .y(y_d),
        .frame_done(frame_done_d),
        .sum_x(sum_x), .sum_y(sum_y),
        .count(count), .result_valid(result_valid)
    );
endmodule
