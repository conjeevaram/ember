`timescale 1ns/1ps
//
// addr_gen : raster-scan address generator.
// Walks addr 0 -> WIDTH*HEIGHT-1, emitting x, y, valid, frame_done.
// PURE counter: x/y are aligned to addr (the address being presented THIS cycle),
// NOT to the ROM's data (which lags one clock). Alignment is handled at top level.
//
module addr_gen #(
    parameter WIDTH   = 64,
    parameter HEIGHT  = 64,
    parameter ADDR_W  = 12,
    parameter COORD_W = 7      // 7 bits covers 0..63
)(
    input  wire               clk,
    input  wire               rst,
    input  wire               enable,
    output reg  [ADDR_W-1:0]  addr,
    output reg  [COORD_W-1:0] x,
    output reg  [COORD_W-1:0] y,
    output reg                valid,
    output reg                frame_done
);
    always @(posedge clk) begin
        if (rst) begin
            addr       <= 0;
            x          <= 0;
            y          <= 0;
            valid      <= 0;
            frame_done <= 0;
        end else if (enable) begin
            frame_done <= 0;     // default low, pulses high only on wrap
            valid      <= 1;

            if (x == WIDTH-1) begin
                x <= 0;
                if (y == HEIGHT-1) begin
                    y          <= 0;
                    addr       <= 0;
                    frame_done <= 1;   // finished the last pixel of the frame
                end else begin
                    y    <= y + 1;
                    addr <= addr + 1;
                end
            end else begin
                x    <= x + 1;
                addr <= addr + 1;
            end
        end else begin
            valid <= 0;
        end
    end
endmodule
