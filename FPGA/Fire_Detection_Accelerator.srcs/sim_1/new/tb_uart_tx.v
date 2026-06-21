`timescale 1ns/1ps
//
// Testbench for uart_tx_bytes: pulses start with known values, samples the
// serial line, decodes the bytes, and verifies the packet contents.
//
module tb_uart_tx;
    localparam CLK_HZ  = 125_000_000;
    localparam BAUD    = 115200;
    localparam integer DIVISOR = CLK_HZ / BAUD;       // 1085
    localparam CLK_PERIOD = 8;                         // 8 ns = 125 MHz

    reg         clk = 0;
    reg         reset = 1;
    reg         start = 0;
    reg  [19:0] sum_x = 20'd21224;
    reg  [19:0] sum_y = 20'd23285;
    reg  [12:0] count = 13'd720;
    reg  [6:0]  min_y = 7'd18;
    reg  [6:0]  max_y = 7'd46;
    wire        uart_tx;
    wire        busy;

    uart_tx_bytes #(.CLK_HZ(CLK_HZ), .BAUD(BAUD)) dut (
        .clk(clk), .reset(reset), .start(start),
        .sum_x(sum_x), .sum_y(sum_y), .count(count),
        .min_y(min_y), .max_y(max_y),
        .uart_tx(uart_tx), .busy(busy)
    );

    always #(CLK_PERIOD/2) clk = ~clk;

    // ---- expected packet ----
    reg [7:0] expected [0:11];
    integer   errors = 0;

    // ---- UART byte receiver task (samples uart_tx at mid-bit) ----
    reg [7:0] rxbyte;
    task uart_receive_byte(output [7:0] b);
        integer i;
        begin
            // wait for start bit (falling edge: line goes 1 -> 0)
            @(negedge uart_tx);
            // align to middle of start bit
            #(DIVISOR * CLK_PERIOD / 2);
            // sample 8 data bits, each one bit-period apart
            for (i = 0; i < 8; i = i + 1) begin
                #(DIVISOR * CLK_PERIOD);
                b[i] = uart_tx;
            end
            // step past stop bit
            #(DIVISOR * CLK_PERIOD);
        end
    endtask

    integer k;
    initial begin
        // build expected packet from the test values
        expected[0]  = 8'hFF;
        expected[1]  = 8'hFF;
        expected[2]  = {4'b0, sum_x[19:16]};
        expected[3]  = sum_x[15:8];
        expected[4]  = sum_x[7:0];
        expected[5]  = {4'b0, sum_y[19:16]};
        expected[6]  = sum_y[15:8];
        expected[7]  = sum_y[7:0];
        expected[8]  = {3'b0, count[12:8]};
        expected[9]  = count[7:0];
        expected[10] = {1'b0, min_y};
        expected[11] = {1'b0, max_y};

        // release reset, fire one packet
        #100 reset = 0;
        #100 start = 1;
        #(CLK_PERIOD) start = 0;

        // receive and check all 12 bytes
        for (k = 0; k < 12; k = k + 1) begin
            uart_receive_byte(rxbyte);
            if (rxbyte === expected[k])
                $display("PASS  byte %0d = %02h", k, rxbyte);
            else begin
                $display("FAIL  byte %0d = %02h  expected %02h", k, rxbyte, expected[k]);
                errors = errors + 1;
            end
        end

        // decode back to numbers for a human-readable sanity check
        begin : decode
            reg [19:0] dec_sum_x, dec_sum_y;
            reg [12:0] dec_count;
            reg [6:0]  dec_min_y, dec_max_y;
            dec_sum_x = {expected[2][3:0], expected[3], expected[4]};
            dec_sum_y = {expected[5][3:0], expected[6], expected[7]};
            dec_count = {expected[8][4:0], expected[9]};
            dec_min_y = expected[10][6:0];
            dec_max_y = expected[11][6:0];
            $display("");
            $display("decoded: sum_x=%0d sum_y=%0d count=%0d min_y=%0d max_y=%0d",
                     dec_sum_x, dec_sum_y, dec_count, dec_min_y, dec_max_y);
        end

        if (errors == 0) $display("\n>>> uart_tx OK: all 12 bytes matched");
        else             $display("\n>>> uart_tx FAILED: %0d mismatch(es)", errors);
        $finish;
    end
endmodule
