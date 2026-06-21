`timescale 1ns/1ps
//
// Self-checking testbench for image_rom.
// Expected words are printed by make_image.py.
//
module tb_image_rom;
    reg         clk = 0;
    reg  [11:0] addr = 0;
    wire [23:0] data;
    integer     errors = 0;

    image_rom #(.MEM_FILE("image.mem")) dut (.clk(clk), .addr(addr), .data(data));

    always #4 clk = ~clk;               // 125 MHz

    task check_pixel(input [11:0] a, input [23:0] expected);
        begin
            @(negedge clk);
            addr = a;
            @(posedge clk);             // mem[addr] -> data on this edge
            #1;
            if (data === expected)
                $display("PASS  addr=%0d  data=%06h", a, data);
            else begin
                $display("FAIL  addr=%0d  data=%06h  expected=%06h", a, data, expected);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        check_pixel(12'd0,    24'h278c79);   // background pixel (x=0,  y=0)
        check_pixel(12'd3210, 24'h8e3ecd);   // distractor pixel (x=10, y=50)
        if (errors == 0) $display(">>> image_rom OK: all reads matched");
        else             $display(">>> image_rom FAILED: %0d mismatch(es)", errors);
        $finish;
    end
endmodule