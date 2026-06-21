`timescale 1ns/1ps
//
// Self-checking testbench for the full color-threshold pipeline.
// Golden numbers (raw color threshold, before morphology/Sobel):
//   count = 720, centroid approx (29, 32)
//
module tb_top;
    reg         clk = 0;
    reg         rst = 1;
    wire [19:0] sum_x, sum_y;
    wire [12:0] count;
    wire        result_valid;

    integer errors = 0;

    top dut (
        .clk(clk), .rst(rst),
        .sum_x(sum_x), .sum_y(sum_y),
        .count(count), .result_valid(result_valid)
    );

    always #4 clk = ~clk;   // 125 MHz

    initial begin
        #20 rst = 0;

        // Wait for the first frame to complete
        wait (result_valid == 1);
        #1;

        $display("---- color-threshold pipeline results ----");
        $display("count = %0d (expect 720)", count);
        $display("sum_x = %0d", sum_x);
        $display("sum_y = %0d", sum_y);
        if (count > 0)
            $display("centroid = (%0d, %0d) (expect approx (29, 32))",
                     sum_x / count, sum_y / count);

        if (count == 720)
            $display(">>> top OK: count matched golden 720");
        else begin
            $display(">>> top FAILED: count=%0d, expected 720", count);
            errors = errors + 1;
        end

        $finish;
    end
endmodule
