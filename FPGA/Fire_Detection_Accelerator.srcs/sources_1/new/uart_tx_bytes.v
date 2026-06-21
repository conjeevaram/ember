`timescale 1ns / 1ps
//
// uart_tx_bytes : on `start`, transmits a 12-byte binary packet over uart_tx
// at 115200 baud (125 MHz clk). Packet: FF FF, sum_x[23:0], sum_y[23:0],
// count[15:0], min_y, max_y.  No divider, no string building.
//
module uart_tx_bytes #(
    parameter CLK_HZ = 125_000_000,
    parameter BAUD   = 115200
)(
    input  wire        clk,
    input  wire        reset,
    input  wire        start,            // pulse high 1 clk to send a packet
    input  wire [19:0] sum_x,
    input  wire [19:0] sum_y,
    input  wire [12:0] count,
    input  wire [6:0]  min_y,
    input  wire [6:0]  max_y,
    output reg         uart_tx = 1'b1,    // idle high
    output reg         busy = 1'b0
);
    localparam integer DIVISOR = CLK_HZ / BAUD;   // 1085
    localparam integer NBYTES  = 12;

    // ---- baud tick ----
    reg [15:0] baud_cnt = 0;
    reg        baud_tick = 0;
    always @(posedge clk) begin
        if (reset) begin
            baud_cnt <= 0; baud_tick <= 0;
        end else if (baud_cnt == DIVISOR-1) begin
            baud_cnt <= 0; baud_tick <= 1;
        end else begin
            baud_cnt <= baud_cnt + 1; baud_tick <= 0;
        end
    end

    // ---- packet buffer ----
    reg [7:0] pkt [0:NBYTES-1];

    // ---- transmit FSM ----
    localparam IDLE=0, SEND=1;
    reg        state = IDLE;
    reg [3:0]  byte_idx = 0;
    reg [3:0]  bit_idx  = 0;
    reg [7:0]  cur = 0;

    always @(posedge clk) begin
        if (reset) begin
            state    <= IDLE;
            uart_tx  <= 1'b1;
            busy     <= 1'b0;
            byte_idx <= 0;
            bit_idx  <= 0;
        end else begin
            case (state)
            IDLE: begin
                uart_tx <= 1'b1;
                busy    <= 1'b0;
                if (start) begin
                    pkt[0]  <= 8'hFF;                  // sync
                    pkt[1]  <= 8'hFF;                  // sync
                    pkt[2]  <= {4'b0, sum_x[19:16]};
                    pkt[3]  <= sum_x[15:8];
                    pkt[4]  <= sum_x[7:0];
                    pkt[5]  <= {4'b0, sum_y[19:16]};
                    pkt[6]  <= sum_y[15:8];
                    pkt[7]  <= sum_y[7:0];
                    pkt[8]  <= {3'b0, count[12:8]};
                    pkt[9]  <= count[7:0];
                    pkt[10] <= {1'b0, min_y};
                    pkt[11] <= {1'b0, max_y};
                    busy     <= 1'b1;
                    byte_idx <= 0;
                    bit_idx  <= 0;
                    cur      <= 8'hFF;                 // first byte to send
                    state    <= SEND;
                end
            end
            SEND: begin
                if (baud_tick) begin
                    case (bit_idx)
                        0: uart_tx <= 1'b0;            // start bit
                        1: uart_tx <= cur[0];
                        2: uart_tx <= cur[1];
                        3: uart_tx <= cur[2];
                        4: uart_tx <= cur[3];
                        5: uart_tx <= cur[4];
                        6: uart_tx <= cur[5];
                        7: uart_tx <= cur[6];
                        8: uart_tx <= cur[7];
                        9: uart_tx <= 1'b1;            // stop bit
                    endcase

                    if (bit_idx == 9) begin
                        bit_idx <= 0;
                        if (byte_idx == NBYTES-1) begin
                            state <= IDLE;
                            busy  <= 1'b0;
                        end else begin
                            byte_idx <= byte_idx + 1;
                            cur      <= pkt[byte_idx + 1];
                        end
                    end else begin
                        bit_idx <= bit_idx + 1;
                    end
                end
            end
            endcase
        end
    end
endmodule
