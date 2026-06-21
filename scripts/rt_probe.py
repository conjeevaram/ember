"""Measure sim-time vs wall-time ratio (real-time factor) under render load."""
from __future__ import annotations
import os, sys, time, threading, pathlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
import torch
torch.set_num_threads(1)
from ember import scenes
from ember.config import G1_MODEL_DIR, RENDER_W, RENDER_H, CAM_W, CAM_H, RENDER_FPS
from ember.sim import G1Sim
from ember.spec import from_json

spec = from_json(str(REPO / "scenes" / "specs" / "demo_scene.json"))
scenes.build(spec)
n12, n29 = scenes.spec_scene_names(spec.name)
sim = G1Sim(scene_path=str(G1_MODEL_DIR / n12), overlay_scene=str(G1_MODEL_DIR / n29), spec=spec)
threading.Thread(target=sim.run, daemon=True).start()
time.sleep(2.0)
sim.enable_wind(True)
sim.set_command(0.4, 0.0, 0.0)
t0 = time.time(); s0 = sim.get_state()["sim_time"]
DUR = 12.0
time.sleep(DUR)
wall = time.time() - t0; simt = sim.get_state()["sim_time"] - s0
print(f"main={RENDER_W}x{RENDER_H} cam={CAM_W}x{CAM_H} fps={RENDER_FPS} "
      f"-> sim {simt:.1f}s / wall {wall:.1f}s = {simt/wall:.2f}x realtime")
sim.stop()
