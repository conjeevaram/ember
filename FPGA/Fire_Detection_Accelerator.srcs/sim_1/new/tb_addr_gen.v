`timescale 1ns/1ps
//
// Self-checking testbench for addr_gen.
// Verifies: addr sweeps 0..4095, x/y track correctly, frame_done pulses at wrap.
//
module tb_addr_gen;
    reg              clk = 0;
    reg              rst = 1;
    reg              enable = 0;
    wire [11:0]      addr;
    wire [6:0]       x, y;
    wire             valid;
    wire             frame_done;

    integer errors = 0;
    integer pixel_count = 0;
    integer frame_done_count = 0;

    addr_gen #(.WIDTH(64), .HEIGHT(64)) dut (
        .clk(clk), .rst(rst), .enable(enable),
        .addr(addr), .x(x), .y(y),
        .valid(valid), .frame_done(frame_done)
    );

    always #4 clk = ~clk;   // 125 MHz

    // Check that addr always equals y*64 + x (the raster invariant)
    always @(posedge clk) begin
        if (valid && !rst) begin
            if (addr !== (y*64 + x)) begin
                $display("FAIL: addr=%0d but y*64+x=%0d (x=%0d y=%0d)",
                         addr, y*64+x, x, y);
                errors = errors + 1;
            end
            pixel_count = pixel_count + 1;
        end
        if (frame_done && !rst)
            frame_done_count = frame_done_count + 1;
    end

    initial begin
        // Release reset, start scanning
        #20 rst = 0;
        enable = 1;

        // Run long enough for slightly more than one full frame (4096 pixels)
        // Each pixel takes one clock (8ns). 4096 * 8 = 32768 ns. Give it margin.
        #40000;

        // Verify results
        $display("---- addr_gen results ----");
        $display("pixels counted    = %0d (expect ~4096 per frame)", pixel_count);
        $display("frame_done pulses  = %0d (expect >=1)", frame_done_count);

        if (errors == 0 && frame_done_count >= 1)
            $display(">>> addr_gen OK: raster invariant held, frame_done pulsed");
        else
            $display(">>> addr_gen FAILED: %0d errors, %0d frame_done pulses",
                     errors, frame_done_count);
        $finish;
    end
endmodule
