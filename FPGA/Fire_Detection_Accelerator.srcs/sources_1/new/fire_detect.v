`timescale 1ns/1ps
//
// fire_detect : combinational YCbCr fire-color threshold.
// Input pixel packed as {Y[7:0], Cb[7:0], Cr[7:0]}.
// One pixel in, one is_fire bit out. No clock, no state.
//
module fire_detect (
    input  wire [23:0] pixel,
    output wire        is_fire
);
    wire [7:0] Y  = pixel[23:16];
    wire [7:0] Cb = pixel[15:8];
    wire [7:0] Cr = pixel[7:0];

    // Classic YCbCr fire rules (Celik & Demirel)
    wire bright   = (Y  > 8'd120);
    wire red_high = (Cr > 8'd150);
    wire blue_low = (Cb < 8'd120);
    wire cr_gt_cb = (Cr > (Cb + 8'd20));

    assign is_fire = bright & red_high & blue_low & cr_gt_cb;
endmodule
