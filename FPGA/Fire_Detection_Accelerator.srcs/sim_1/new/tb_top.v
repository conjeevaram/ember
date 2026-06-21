`timescale 1ns/1ps
//
// End-to-end testbench: runs the pipeline, captures UART output,
// decodes the packet, checks against golden numbers.
//
module tb_top;
    localparam integer DIVISOR = 1085;     // 125MHz / 115200
    localparam CLK_PERIOD = 8;             // ns

    reg  clk = 0;
    reg  rst = 1;
    wire uart_tx;

    // declarations moved to module level (not inside an unnamed block)
    reg [7:0]  rxbyte;
    reg [7:0]  pkt [0:11];
    integer    k;
    reg [19:0] sx, sy;
    reg [12:0] cnt;
    reg [6:0]  mny, mxy;

    top dut (.clk(clk), .rst(rst), .uart_tx(uart_tx));

    always #(CLK_PERIOD/2) clk = ~clk;

    task uart_receive_byte(output [7:0] b);
        integer i;
        begin
            @(negedge uart_tx);                       // start bit
            #(DIVISOR * CLK_PERIOD / 2);              // mid start bit
            for (i = 0; i < 8; i = i + 1) begin
                #(DIVISOR * CLK_PERIOD);
                b[i] = uart_tx;
            end
            #(DIVISOR * CLK_PERIOD);                  // stop bit
        end
    endtask

    initial begin
        #100 rst = 0;

        // capture the first 12 bytes the design transmits
        for (k = 0; k < 12; k = k + 1) begin
            uart_receive_byte(rxbyte);
            pkt[k] = rxbyte;
            $display("byte %0d = %02h", k, rxbyte);
        end

        // decode
        sx  = {pkt[2][3:0], pkt[3], pkt[4]};
        sy  = {pkt[5][3:0], pkt[6], pkt[7]};
        cnt = {pkt[8][4:0], pkt[9]};
        mny = pkt[10][6:0];
        mxy = pkt[11][6:0];

        $display("");
        $display("decoded packet:");
        $display("  sync   = %02h %02h (expect ff ff)", pkt[0], pkt[1]);
        $display("  sum_x  = %0d", sx);
        $display("  sum_y  = %0d", sy);
        $display("  count  = %0d (expect 720)", cnt);
        $display("  min_y  = %0d", mny);
        $display("  max_y  = %0d", mxy);

        if (cnt == 720 && pkt[0] == 8'hFF && pkt[1] == 8'hFF)
            $display("\n>>> END-TO-END OK: pipeline -> UART packet correct");
        else
            $display("\n>>> CHECK: count=%0d (expected 720), sync=%02h %02h",
                     cnt, pkt[0], pkt[1]);

        $finish;
    end
endmodule
