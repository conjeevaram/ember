import sys
import serial
import serial.tools.list_ports

# ---- config ----
BAUD = 115200
# auto-detect port, or hardcode e.g. PORT = "COM5"
PORT = None

def pick_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial ports found. Plug in the USB-UART adapter.")
        sys.exit(1)
    if len(ports) == 1:
        print(f"Using {ports[0].device} ({ports[0].description})")
        return ports[0].device
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  {p.description}")
    idx = int(input("Pick port number: "))
    return ports[idx].device

def main():
    port = PORT or pick_port()
    try:
        ser = serial.Serial(port, BAUD, timeout=1)
    except Exception as e:
        print(f"Could not open {port}: {e}")
        sys.exit(1)

    print(f"Listening on {port} at {BAUD} baud. Ctrl+C to stop.\n")

    state = 0          # how many 0xFF seen in a row
    buf = bytearray()

    try:
        while True:
            b = ser.read(1)
            if not b:
                continue
            byte = b[0]

            if state < 2:
                # looking for FF FF sync
                if byte == 0xFF:
                    state += 1
                else:
                    state = 0
                continue

            # we've seen FF FF; collect 10 payload bytes
            buf.append(byte)
            if len(buf) == 10:
                sum_x = (buf[0] << 16) | (buf[1] << 8) | buf[2]
                sum_y = (buf[3] << 16) | (buf[4] << 8) | buf[5]
                count = (buf[6] << 8) | buf[7]
                min_y = buf[8]
                max_y = buf[9]

                if count > 0:
                    cx = sum_x // count
                    cy = sum_y // count
                    flame_h = max_y - min_y if max_y >= min_y else 0
                    target_y = max_y - (flame_h // 8)   # ~12.5% up from base
                    print(f"count={count:4d}  centroid=({cx:2d},{cy:2d})  "
                          f"BASE_TARGET=({cx:2d},{target_y:2d})  "
                          f"[min_y={min_y} max_y={max_y}]")
                else:
                    print("count=0  (no fire)")

                # reset for next packet
                buf.clear()
                state = 0

    except KeyboardInterrupt:
        print("\nStopped.")
        ser.close()

if __name__ == "__main__":
    main()
