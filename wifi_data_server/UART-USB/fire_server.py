import threading
import time
import serial
import serial.tools.list_ports
from flask import Flask, jsonify, Response

# ---------------- config ----------------
BAUD = 115200
PORT = "COM6"
FLAME_SHIFT = 3
GRID_W, GRID_H = 64, 64     # image dimensions

# ---------------- shared state ----------------
state_lock = threading.Lock()
latest = {
    "count": 0, "sum_x": 0, "sum_y": 0,
    "centroid_x": 0, "centroid_y": 0,
    "min_y": 0, "max_y": 0,
    "target_x": 0, "target_y": 0,
    "fire_detected": False,
    "frames_seen": 0, "timestamp": 0.0,
}

# ---------------- serial reader thread ----------------
def pick_port():
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found.")
    if len(ports) == 1:
        print(f"Using {ports[0].device} ({ports[0].description})")
        return ports[0].device
    print("Available ports:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device}  {p.description}")
    return ports[int(input("Pick port number: "))].device

def serial_reader():
    port = PORT or pick_port()
    ser = serial.Serial(port, BAUD, timeout=1)
    print(f"Serial reader started on {port} @ {BAUD}")
    sync = 0
    buf = bytearray()
    frames = 0
    while True:
        b = ser.read(1)
        if not b:
            continue
        byte = b[0]
        if sync < 2:
            sync = sync + 1 if byte == 0xFF else 0
            continue
        buf.append(byte)
        if len(buf) == 10:
            sum_x = (buf[0] << 16) | (buf[1] << 8) | buf[2]
            sum_y = (buf[3] << 16) | (buf[4] << 8) | buf[5]
            count = (buf[6] << 8) | buf[7]
            min_y = buf[8]
            max_y = buf[9]
            frames += 1
            if count > 0:
                cx = sum_x // count
                cy = sum_y // count
                flame_h = max_y - min_y if max_y >= min_y else 0
                ty = max_y - (flame_h >> FLAME_SHIFT)
                fire = True
            else:
                cx = cy = ty = 0
                fire = False
            with state_lock:
                latest.update({
                    "count": count, "sum_x": sum_x, "sum_y": sum_y,
                    "centroid_x": cx, "centroid_y": cy,
                    "min_y": min_y, "max_y": max_y,
                    "target_x": cx, "target_y": ty,
                    "fire_detected": fire,
                    "frames_seen": frames, "timestamp": time.time(),
                })
            buf.clear()
            sync = 0

# ---------------- Flask app ----------------
app = Flask(__name__)

@app.route("/fire")
def fire():
    with state_lock:
        return jsonify(dict(latest))

@app.route("/target")
def target():
    with state_lock:
        return jsonify({
            "fire_detected": latest["fire_detected"],
            "x": latest["target_x"], "y": latest["target_y"],
            "count": latest["count"],
        })

# live visual + text page (auto-refreshes via JS fetch)
PAGE = """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fire Detector</title>
  <style>
    body { font-family: -apple-system, sans-serif; background:#111; color:#eee;
           margin:0; padding:16px; text-align:center; }
    h1 { font-size:20px; margin:8px 0; }
    #status { font-size:28px; font-weight:bold; margin:12px 0; }
    .fire { color:#ff5520; }
    .nofire { color:#4a9; }
    canvas { background:#000; border:2px solid #333; border-radius:8px;
             width:90vw; max-width:400px; height:90vw; max-height:400px; }
    table { margin:16px auto; border-collapse:collapse; font-size:16px; }
    td { padding:4px 14px; text-align:left; border-bottom:1px solid #222; }
    td.k { color:#888; }
  </style>
</head>
<body>
  <h1>🔥 FPGA Fire Detector</h1>
  <div id="status">—</div>
  <canvas id="grid" width="320" height="320"></canvas>
  <table id="data"></table>
  <script>
    const W = 64, H = 64;
    const cv = document.getElementById('grid');
    const ctx = cv.getContext('2d');
    const sx = cv.width / W, sy = cv.height / H;

    function draw(d) {
      ctx.fillStyle = '#000'; ctx.fillRect(0,0,cv.width,cv.height);
      // grid lines
      ctx.strokeStyle = '#1a1a1a'; ctx.lineWidth = 1;
      for (let i=0;i<=W;i+=8){ctx.beginPath();ctx.moveTo(i*sx,0);ctx.lineTo(i*sx,cv.height);ctx.stroke();}
      for (let j=0;j<=H;j+=8){ctx.beginPath();ctx.moveTo(0,j*sy);ctx.lineTo(cv.width,j*sy);ctx.stroke();}
      if (d.fire_detected) {
        // flame bounding band (min_y..max_y) faint
        ctx.fillStyle = 'rgba(255,120,30,0.15)';
        ctx.fillRect(0, d.min_y*sy, cv.width, (d.max_y-d.min_y+1)*sy);
        // centroid (green)
        drawCross(d.centroid_x, d.centroid_y, '#3f3', 10);
        // base target (yellow, big)
        drawCross(d.target_x, d.target_y, '#ff0', 16);
      }
    }
    function drawCross(x,y,color,r){
      const px=x*sx+sx/2, py=y*sy+sy/2;
      ctx.strokeStyle=color; ctx.lineWidth=3;
      ctx.beginPath();ctx.moveTo(px-r,py);ctx.lineTo(px+r,py);
      ctx.moveTo(px,py-r);ctx.lineTo(px,py+r);ctx.stroke();
    }

    async function tick() {
      try {
        const d = await (await fetch('/fire')).json();
        const s = document.getElementById('status');
        if (d.fire_detected) { s.textContent='🔥 FIRE DETECTED'; s.className='fire'; }
        else { s.textContent='no fire'; s.className='nofire'; }
        draw(d);
        document.getElementById('data').innerHTML = `
          <tr><td class=k>count</td><td>${d.count}</td></tr>
          <tr><td class=k>centroid</td><td>(${d.centroid_x}, ${d.centroid_y})</td></tr>
          <tr><td class=k>base target</td><td>(${d.target_x}, ${d.target_y})</td></tr>
          <tr><td class=k>min_y / max_y</td><td>${d.min_y} / ${d.max_y}</td></tr>
          <tr><td class=k>frames</td><td>${d.frames_seen}</td></tr>`;
      } catch(e) {
        document.getElementById('status').textContent = 'no connection';
      }
    }
    setInterval(tick, 200);   // refresh 5x/sec
    tick();
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")

if __name__ == "__main__":
    t = threading.Thread(target=serial_reader, daemon=True)
    t.start()
    time.sleep(0.5)
    app.run(host="0.0.0.0", port=5000, threaded=True)

