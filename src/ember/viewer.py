"""Headless web viewer for the G1 firefighter sim.

Discovers the named demo scenes plus every ``scenes/specs/*.json`` procedural
spec, runs the active :class:`~ember.sim.G1Sim` on its own physics + render
threads, and serves a browser UI (keyboard driving, a top-down nav map, and
the autonomous "approach fire" controller) over MJPEG + JSON. Scenes hot-swap
under a lock without restarting the server.

Run:
    python -m ember.viewer --scene obstacles
    python -m ember.viewer --spec scenes/specs/scene_000.json
    ember-walk --scene obstacles            # if installed
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import scenes
from .config import G1_MODEL_DIR, RENDER_FPS
from .sim import G1Sim, OVERLAY_SCENES, SCENES
from .spec import from_json
from .streaming import FrameBuffer, create_app, mjpeg_response, serve

_SPECS_DIR = Path(__file__).resolve().parent.parent.parent / "scenes" / "specs"
_NAMED_LABELS = {"flat": "flat", "obstacles": "obstacles", "fire": "fire"}


@dataclass(frozen=True)
class SceneEntry:
    key: str
    label: str
    kind: Literal["named", "proc"]
    scene: str | None = None
    spec_path: Path | None = None


def discover_scene_catalog() -> list[SceneEntry]:
    """Named demos plus every ``scenes/specs/*.json`` (sorted)."""
    entries = [
        SceneEntry(key=k, label=_NAMED_LABELS[k], kind="named", scene=k)
        for k in ("flat", "obstacles", "fire")
    ]
    if _SPECS_DIR.is_dir():
        for path in sorted(_SPECS_DIR.glob("*.json")):
            spec = from_json(path)
            entries.append(SceneEntry(
                key=spec.name, label=spec.name, kind="proc", spec_path=path))
    return entries


# Module-level handle on the live sim, set by SimManager on each (re)build.
_SIM: G1Sim | None = None


class SimManager:
    """Owns the active :class:`G1Sim` and hot-swaps scenes under a lock."""

    def __init__(self, initial_key: str, *, overlay=True, render=True,
                 loop=False, initial_cmd=None):
        self._lock = threading.Lock()
        self._overlay = overlay
        self._render = render
        self._loop = loop
        self._initial_cmd = initial_cmd
        self.catalog = discover_scene_catalog()
        self._by_key = {e.key: e for e in self.catalog}
        if initial_key not in self._by_key:
            raise ValueError(f"unknown scene key: {initial_key}")
        self._current_key = initial_key
        self._physics_thread: threading.Thread | None = None
        self._retired: list[G1Sim] = []
        self._switch_unlocked(initial_key, first=True)

    def current_key(self) -> str:
        with self._lock:
            return self._current_key

    def catalog_list(self) -> list[dict]:
        return [{"key": e.key, "label": e.label} for e in self.catalog]

    def sim(self) -> G1Sim:
        with self._lock:
            if _SIM is None:
                raise RuntimeError("no active sim")
            return _SIM

    def frames(self) -> FrameBuffer:
        return self.sim().frames

    def cam_frames(self) -> FrameBuffer | None:
        sim = self.sim()
        return sim.cam_frames if sim.has_camera else None

    def has_camera(self) -> bool:
        return self.sim().has_camera

    def switch(self, key: str) -> str:
        with self._lock:
            if key not in self._by_key:
                raise ValueError(f"unknown scene key: {key}")
            if key == self._current_key:
                return key
            return self._switch_unlocked(key)

    def shutdown(self):
        with self._lock:
            sim = _SIM
            if sim is not None:
                sim.stop()
            if self._physics_thread is not None:
                self._physics_thread.join(timeout=5.0)
                self._physics_thread = None

    def _build_sim(self, entry: SceneEntry) -> G1Sim:
        if entry.kind == "proc":
            spec = from_json(entry.spec_path)
            scenes.build(spec)
            name_12, name_29 = scenes.spec_scene_names(spec.name)
            scene_path = str(G1_MODEL_DIR / name_12)
            overlay = (str(G1_MODEL_DIR / name_29)
                       if self._overlay else None)
            return G1Sim(scene_path=scene_path, overlay_scene=overlay, spec=spec)
        scenes.ensure_scenes()
        scene_path = SCENES[entry.scene]
        overlay = (OVERLAY_SCENES.get(scene_path) if self._overlay else None)
        return G1Sim(scene_path=scene_path, overlay_scene=overlay)

    def _apply_policy(self, sim: G1Sim, entry: SceneEntry):
        if entry.kind == "proc":
            sim.auto_reset = False
            sim.heading_hold = False
            sim.set_command(0.0, 0.0, 0.0)
        elif entry.scene == "obstacles":
            sim.auto_reset = self._loop
            sim.heading_hold = self._loop
            sim.set_command(0.4, 0.0, 0.0)
        else:  # "flat" / "fire": walk forward, no auto-reset
            sim.auto_reset = False
            sim.heading_hold = False
            sim.set_command(0.5, 0.0, 0.0)

    def _switch_unlocked(self, key: str, first: bool = False) -> str:
        global _SIM
        retiring: G1Sim | None = None
        if self._physics_thread is not None:
            retiring = _SIM
            if retiring is not None:
                retiring.stop()
            self._physics_thread.join(timeout=5.0)
            self._physics_thread = None
            if retiring is not None:
                self._retired.append(retiring)
            time.sleep(0.15)

        entry = self._by_key[key]
        sim = self._build_sim(entry)
        _SIM = sim
        if first and self._initial_cmd is not None:
            sim.set_command(*self._initial_cmd)
            sim.auto_reset = self._loop
            sim.heading_hold = self._loop
        else:
            self._apply_policy(sim, entry)

        self._physics_thread = threading.Thread(
            target=lambda: sim.run(render=self._render),
            daemon=True, name=f"g1-physics-{key}")
        self._physics_thread.start()
        self._current_key = key
        self._retired.clear()
        return key


_PAGE = """<!doctype html><html><head><title>G1 Walk</title>
<style>body{background:#111;color:#ddd;font-family:monospace;margin:0;text-align:center}
img{max-width:100%;height:auto;background:#000}
#hud{position:fixed;top:8px;left:8px;background:rgba(0,0,0,.6);padding:8px;text-align:left;font-size:13px;border-radius:4px;z-index:2}
#ego{position:fixed;bottom:10px;right:10px;border:2px solid #555;border-radius:4px;background:#000;z-index:2;display:none}
#ego img{display:block;width:320px}
#ego .lbl{position:absolute;top:2px;left:6px;font-size:11px;color:#0f0;text-shadow:0 0 3px #000}
kbd{background:#333;padding:1px 5px;border-radius:3px}
#sceneSel{margin:4px 0;font-family:monospace;background:#222;color:#ddd;border:1px solid #555}
#navPanel{position:fixed;top:8px;right:8px;background:rgba(0,0,0,.75);padding:6px;border-radius:4px;z-index:2;text-align:left}
#navCanvas{background:#1a1a1a;border:1px solid #444;display:block}
.navRow{font-size:12px;margin:4px 0}
</style></head>
<body>
<div id="hud">
<b>G1 locomotion</b> (12-DOF walker + kinematic arm overlay)<br>
scene: <select id="sceneSel"></select><br>
<kbd>W</kbd>/<kbd>S</kbd> fwd/back &nbsp; <kbd>A</kbd>/<kbd>D</kbd> turn<br>
<kbd>Q</kbd>/<kbd>E</kbd> strafe &nbsp; <kbd>Space</kbd> stop<br>
<kbd>H</kbd> auto-steer (re-center) <span id="hh"></span><br>
<kbd>1</kbd> carry hose &nbsp; <kbd>2</kbd> aim &nbsp; <kbd>3</kbd> arms down &nbsp; arms: <span id="arm">?</span><br>
<kbd>F</kbd> spray water <span id="spray">[off]</span> &nbsp; <kbd>G</kbd> reignite<br>
fire: <span id="fire">100%</span> <span id="hit"></span><br>
cmd: <span id="cmd">0,0,0</span><br>
<span id="st"></span>
</div>
<div id="navPanel">
<div class="navRow"><b>Nav map</b></div>
<canvas id="navCanvas" width="280" height="220"></canvas>
<div class="navRow">
<button id="approachBtn">Approach fire</button>
<button id="resetBtn">Reset</button>
<span id="approachPhase"></span>
</div>
</div>
<img src="/stream"/>
<div id="ego"><div class="lbl">ROBOT CAM</div><img id="egoImg" src="/camera"/></div>
<script>
let vx=0,vy=0,yaw=0,sceneBusy=false,navOcc=null,navMeta=null,navHit=null;
function send(){fetch(`/cmd?vx=${vx}&vy=${vy}&yaw=${yaw}`).then(r=>r.json()).then(d=>{
  document.getElementById('cmd').textContent=d.cmd.map(x=>x.toFixed(2)).join(', ');});}
function setApproach(on){
  fetch('/approach?on='+(on?1:0)).then(r=>r.json()).then(d=>{
    document.getElementById('approachBtn').disabled=!!d.approach;});}
function resetPose(){fetch('/reset').then(()=>{setApproach(false);});}
function phaseLabel(p){
  if(!p)return '';
  if(p==='navigate')return 'NAVIGATING\u2026';
  if(p==='face')return 'FACING\u2026';
  if(p==='ready')return 'READY \u2713';
  return p;}
function drawNav(d){
  if(!d)return;
  const c=document.getElementById('navCanvas');
  const ctx=c.getContext('2d');
  const w=c.width,h=c.height;
  const ox=d.origin[0],oy=d.origin[1],res=d.res,nx=d.shape[0],ny=d.shape[1];
  const meta=ox+'|'+oy+'|'+res+'|'+nx+'|'+ny;
  if(navMeta!==meta){navMeta=meta;navOcc=d.occupied;}
  ctx.fillStyle='#111';ctx.fillRect(0,0,w,h);
  const sx=w/(nx*res),sy=h/(ny*res),sc=Math.min(sx,sy);
  const padX=(w-nx*res*sc)/2,padY=(h-ny*res*sc)/2;
  function toX(x){return padX+(x-ox)*sc;}
  function toY(y){return padY+(ny*res-(y-oy))*sc;}
  navHit={ox,oy,res,nx,ny,sc,padX,padY,fires:d.fires,targeted:d.targeted_fire};
  if(navOcc){
    for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){
      if(navOcc[j*nx+i]){
        const x0=toX(ox+i*res),y0=toY(oy+(j+1)*res);
        ctx.fillStyle='#555';ctx.fillRect(x0,y0,res*sc,res*sc);
      }
    }
  }
  if(d.home){ctx.fillStyle='#4af';const hx=toX(d.home[0]),hy=toY(d.home[1]);
    ctx.beginPath();ctx.arc(hx,hy,4,0,6.28);ctx.fill();}
  d.fires.forEach((f,i)=>{
    const col=f.health<=0?'#333':(i===d.targeted_fire?'#ff6600':'#ff3333');
    ctx.fillStyle=col;const fx=toX(f.pos[0]),fy=toY(f.pos[1]);
    ctx.beginPath();ctx.arc(fx,fy,5,0,6.28);ctx.fill();});
  if(d.path&&d.path.length>1){
    ctx.strokeStyle='#0f0';ctx.lineWidth=2;ctx.beginPath();
    ctx.moveTo(toX(d.path[0][0]),toY(d.path[0][1]));
    for(let k=1;k<d.path.length;k++)ctx.lineTo(toX(d.path[k][0]),toY(d.path[k][1]));
    ctx.stroke();
    const gl=d.path[d.path.length-1],gx=toX(gl[0]),gy=toY(gl[1]);
    ctx.strokeStyle='#0f0';ctx.lineWidth=2;ctx.beginPath();
    ctx.arc(gx,gy,7,0,6.28);ctx.stroke();
    const tf=d.fires[d.targeted_fire];
    if(tf&&tf.health>0){
      const ffx=toX(tf.pos[0]),ffy=toY(tf.pos[1]);
      ctx.setLineDash([4,3]);ctx.strokeStyle='#888';ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(gx,gy);ctx.lineTo(ffx,ffy);ctx.stroke();
      ctx.setLineDash([]);}}
  ctx.fillStyle='#ccc';ctx.font='11px monospace';
  ctx.fillText('target: fire '+d.targeted_fire,6,h-6);
  const rp=d.robot.pos,ryaw=d.robot.yaw;
  let rx=toX(rp[0]),ry=toY(rp[1]);
  const oob=(rx<2||rx>w-2||ry<2||ry>h-2);
  rx=Math.max(6,Math.min(w-6,rx));ry=Math.max(6,Math.min(h-6,ry));
  const rcol=oob?'#ff3333':'#00ffcc';
  ctx.fillStyle=rcol;ctx.beginPath();ctx.arc(rx,ry,5,0,6.28);ctx.fill();
  ctx.strokeStyle=rcol;ctx.beginPath();
  ctx.moveTo(rx,ry);ctx.lineTo(rx+12*Math.cos(ryaw),ry-12*Math.sin(ryaw));ctx.stroke();
  if(oob){ctx.fillStyle='#ff3333';ctx.font='10px monospace';ctx.fillText('off-map \u2192 Reset',6,12);}
  document.getElementById('approachBtn').disabled=!!d.approach;
  document.getElementById('approachPhase').textContent=phaseLabel(d.phase);}
document.getElementById('navCanvas').addEventListener('click',e=>{
  if(!navHit)return;
  const c=document.getElementById('navCanvas');
  const rect=c.getBoundingClientRect();
  const mx=e.clientX-rect.left,my=e.clientY-rect.top;
  const {ox,oy,res,nx,ny,sc,padX,padY,fires}=navHit;
  function toX(x){return padX+(x-ox)*sc;}
  function toY(y){return padY+(ny*res-(y-oy))*sc;}
  let best=-1,bestD=1e9;
  fires.forEach((f,i)=>{
    if(f.health<=0)return;
    const fx=toX(f.pos[0]),fy=toY(f.pos[1]);
    const dist=Math.hypot(mx-fx,my-fy);
    if(dist<8&&dist<bestD){bestD=dist;best=i;}});
  if(best>=0)fetch('/target?idx='+best);});
function setArm(p){fetch('/arm?pose='+p).then(r=>r.json()).then(d=>{
  document.getElementById('arm').textContent=d.arm_pose;});}
function setEgo(on){
  document.getElementById('ego').style.display=on?'block':'none';}
function loadScenes(){
  fetch('/scenes').then(r=>r.json()).then(d=>{
    const sel=document.getElementById('sceneSel');
    sel.innerHTML='';
    d.scenes.forEach(s=>{
      const o=document.createElement('option');
      o.value=s.key; o.textContent=s.label;
      if(s.key===d.current) o.selected=true;
      sel.appendChild(o);});
    setEgo(!!d.has_camera);});}
function switchScene(key){
  if(sceneBusy)return;
  sceneBusy=true;
  fetch('/scene?key='+encodeURIComponent(key)).then(r=>{
    if(!r.ok) throw new Error('switch failed');
    return r.json();}).then(d=>{
    setEgo(!!d.has_camera);
    vx=vy=yaw=0; send();
    navOcc=null;navMeta=null;setApproach(false);}).catch(()=>{
    loadScenes();}).finally(()=>{sceneBusy=false;});}
document.getElementById('sceneSel').addEventListener('change',e=>{
  switchScene(e.target.value);});
document.addEventListener('keydown',e=>{
  if(e.repeat)return;
  switch(e.key.toLowerCase()){
    case'w':vx=0.8;break; case's':vx=-0.4;break;
    case'a':yaw=0.6;break; case'd':yaw=-0.6;break;
    case'q':vy=0.3;break;  case'e':vy=-0.3;break;
    case' ':vx=vy=yaw=0;setApproach(false);break;
    case'h':fetch('/heading?on=1').then(r=>r.json()).then(d=>{
      document.getElementById('hh').textContent=d.heading_hold?'[ON]':'[off]';});return;
    case'1':setArm('carry');return; case'2':setArm('aim');return;
    case'3':setArm('down');return;
    case'f':fetch('/spray').then(r=>r.json()).then(d=>{
      document.getElementById('spray').textContent=d.spraying?'[ON]':'[off]';});return;
    case'g':fetch('/reignite');return;
    default:return;}
  send();});
document.addEventListener('keyup',e=>{
  switch(e.key.toLowerCase()){
    case'w':case's':vx=0;break;
    case'a':case'd':yaw=0;break;
    case'q':case'e':vy=0;break; default:return;}
  send();});
document.getElementById('approachBtn').addEventListener('click',()=>{setApproach(true);});
document.getElementById('resetBtn').addEventListener('click',resetPose);
setInterval(()=>{fetch('/nav').then(r=>r.json()).then(d=>{if(d&&!d.error)drawNav(d);});},500);
setInterval(()=>{fetch('/state').then(r=>r.json()).then(d=>{
  if(d.arm_pose)document.getElementById('arm').textContent=d.arm_pose;
  document.getElementById('spray').textContent=d.spraying?'[ON]':'[off]';
  document.getElementById('fire').textContent=Math.round(d.fire_health*100)+'%';
  document.getElementById('hit').textContent=
    d.fire_health<=0?'\u2713 OUT':(d.spraying?(d.hitting?'\u25c9 HIT':'\u2715 MISS'):'');
  document.getElementById('st').textContent=
    `pos ${d.pos.map(x=>x.toFixed(2))} ${d.fell?'\u26a0 FELL':'ok'}${d.blocked?' BLOCKED':''}`+
    (d.approach_phase?' '+phaseLabel(d.approach_phase):'');});},500);
loadScenes();
</script></body></html>"""


def _make_app(mgr: SimManager):
    from flask import jsonify, request

    def register(app):
        @app.route("/cmd")
        def cmd():
            def f(name):
                v = request.args.get(name)
                return float(v) if v is not None else None
            c = mgr.sim().set_command(vx=f("vx"), vy=f("vy"), yaw=f("yaw"))
            return jsonify(cmd=c.tolist())

        @app.route("/heading")
        def heading():
            on = request.args.get("on", "1") not in ("0", "false", "")
            mgr.sim().set_heading_hold(on)
            return jsonify(heading_hold=on)

        @app.route("/arm")
        def arm():
            return jsonify(arm_pose=mgr.sim().set_overlay_arm_pose(
                request.args.get("pose", "carry")))

        @app.route("/spray")
        def spray():
            on = request.args.get("on")
            return jsonify(spraying=mgr.sim().set_spray(
                None if on is None else on not in ("0", "false", "")))

        @app.route("/reignite")
        def reignite():
            return jsonify(fire_health=mgr.sim().reignite())

        @app.route("/scenes")
        def scenes_list():
            return jsonify(scenes=mgr.catalog_list(), current=mgr.current_key(),
                           has_camera=mgr.has_camera())

        @app.route("/scene")
        def scene_switch():
            key = request.args.get("key") or request.args.get("name")
            if not key:
                return jsonify(error="missing key"), 400
            try:
                mgr.switch(key)
            except ValueError as e:
                return jsonify(error=str(e)), 400
            return jsonify(current=mgr.current_key(), has_camera=mgr.has_camera())

        @app.route("/nav")
        def nav_view():
            snap = mgr.sim().nav_snapshot()
            if snap is None:
                return jsonify(error="no procedural spec")
            return jsonify(snap)

        @app.route("/approach")
        def approach_toggle():
            on = request.args.get("on", "1") not in ("0", "false", "")
            active = mgr.sim().set_approach(on)
            phase = mgr.sim().get_state().get("approach_phase")
            return jsonify(approach=active, phase=phase)

        @app.route("/reset")
        def reset_pose():
            return jsonify(pos=mgr.sim().reset_to_start().tolist())

        @app.route("/target")
        def target_fire():
            idx = request.args.get("idx", type=int)
            if idx is None:
                return jsonify(error="missing idx"), 400
            return jsonify(targeted_fire=mgr.sim().set_targeted_fire(idx))

        @app.route("/camera")
        def camera():
            if not mgr.has_camera():
                return "", 404
            return mjpeg_response(mgr.cam_frames, RENDER_FPS)

    def state_fn():
        s = mgr.sim().get_state()
        return dict(pos=s["pos"].tolist(), cmd=s["cmd"].tolist(),
                    fell=bool(s["fell"]), sim_time=float(s["sim_time"]),
                    arm_pose=s["arm_pose"], spraying=bool(s["spraying"]),
                    hitting=bool(s["hitting"]), fire_health=float(s["fire_health"]),
                    fires=[{"pos": list(f["pos"]), "health": float(f["health"])}
                           for f in s["fires"]],
                    targeted_fire=int(s["targeted_fire"]),
                    blocked=bool(s["blocked"]),
                    approach_phase=s["approach_phase"])

    return create_app(page_html=_PAGE, frame_source=mgr.frames,
                      state_fn=state_fn, register_routes=register)


def start(block=True, serve_web=True, port=8088, render=True, initial_cmd=None,
          scene="flat", loop=False, overlay=True, spec=None):
    """Start the sim. With block=False, runs the loop on a background thread.

    Pass a ``SceneSpec`` via ``spec`` to run a procedural scene: the XMLs are
    (re)built from it and the robot spawns at ``spec.start``."""
    initial_key = spec.name if spec is not None else (
        scene if scene in SCENES else "flat")

    mgr = SimManager(initial_key, overlay=overlay, render=render,
                     loop=loop, initial_cmd=initial_cmd)

    if serve_web:
        serve(_make_app(mgr), port, label="G1 walker")

    if block:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            mgr.shutdown()
    return mgr.sim()


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default="flat", help="flat | obstacles | <path.xml>")
    p.add_argument("--spec", default=None,
                   help="path to a SceneSpec JSON (procedural scene; spawns at start)")
    p.add_argument("--idle", action="store_true",
                   help="stand still until commanded (default: walk forward)")
    p.add_argument("--vx", type=float, default=None, help="initial forward speed")
    p.add_argument("--port", type=int, default=8088)
    p.add_argument("--loop", dest="loop", action="store_true", default=None,
                   help="auto-reset & replay (default: on for obstacles)")
    p.add_argument("--no-loop", dest="loop", action="store_false")
    p.add_argument("--no-overlay", dest="overlay", action="store_false", default=True,
                   help="render the bare 12-DOF body (no arms)")
    args = p.parse_args()

    spec = from_json(args.spec) if args.spec else None

    # Procedural scenes have walls; stand still by default so the blind walker
    # doesn't march into one. Named demo scenes keep their walk-forward default.
    default_vx = 0.0 if spec is not None else (0.4 if args.scene == "obstacles" else 0.5)
    vx = 0.0 if args.idle else (args.vx if args.vx is not None else default_vx)
    loop = (args.scene == "obstacles") if args.loop is None else args.loop
    start(block=True, serve_web=True, port=args.port,
          initial_cmd=(vx, 0.0, 0.0), scene=args.scene, loop=loop,
          overlay=args.overlay, spec=spec)


if __name__ == "__main__":
    main()
