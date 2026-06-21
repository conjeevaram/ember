`timescale 1ns/1ps
//
// image_rom : test image baked into BRAM, read one pixel per clock.
// Synchronous read => data is valid ONE clock after addr is presented.
//
module image_rom #(
    parameter MEM_FILE = "image.mem",
    parameter ADDR_W   = 12,            // 64*64 = 4096 pixels
    parameter DATA_W   = 24             // {Y[7:0], Cb[7:0], Cr[7:0]}
)(
    input  wire              clk,
    input  wire [ADDR_W-1:0] addr,
    output reg  [DATA_W-1:0] data
);
    (* ram_style = "block" *) reg [DATA_W-1:0] mem [0:(1<<ADDR_W)-1];

    initial begin
        $readmemh(MEM_FILE, mem);
    end

    always @(posedge clk)
        data <= mem[addr];              // 1-cycle read latency
endmodule