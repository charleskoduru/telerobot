#!/usr/bin/env python3

import argparse
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from telerobot.config import load_config, load_robot
from telerobot.controller.kinematics import build_kinematics


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("workspace_bounds_gui")


HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SO-101 Workspace Bounds Recorder</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #111;
      color: #eee;
      margin: 20px;
    }
    button {
      font-size: 18px;
      padding: 12px 18px;
      margin: 6px;
      border-radius: 8px;
      border: none;
      cursor: pointer;
    }
    #capture { background: #2e8b57; color: white; }
    #undo { background: #666; color: white; }
    #done { background: #1e90ff; color: white; }
    canvas {
      background: #222;
      border: 1px solid #555;
      margin: 10px;
    }
    pre {
      background: #000;
      color: #0f0;
      padding: 15px;
      border-radius: 8px;
      font-size: 16px;
      white-space: pre-wrap;
    }
    .row { display: flex; flex-wrap: wrap; }
    .card {
      background: #1b1b1b;
      padding: 14px;
      border-radius: 10px;
      margin-bottom: 14px;
    }
  </style>
</head>
<body>
  <h1>SO-101 Workspace Bounds Recorder</h1>

  <div class="card">
    <p><b>Instructions:</b></p>
    <p>Move the robot end effector to your 4 workspace corners: 2 bottom points and 2 top points.</p>
    <p>Click <b>Capture Point</b> at each corner. Then click <b>Done</b>.</p>

    <button id="capture" onclick="post('/capture')">Capture Point</button>
    <button id="undo" onclick="post('/undo')">Undo</button>
    <button id="done" onclick="post('/done')">Done</button>

    <br><br>

    <button onclick="post('/release_motors')" style="background:#cc7722;color:white;">
      Release Motors
    </button>

    <button onclick="post('/lock_motors')" style="background:#aa3333;color:white;">
      Lock Motors
    </button>

    <p><b>Warning:</b> Hold the robot arm before pressing Release Motors. It may sag.</p>
  </div>

  <div class="card">
    <h2>Live End Effector</h2>
    <div id="xyz">Waiting for robot...</div>
    <div id="count">Captured points: 0</div>
  </div>

  <div class="row">
    <div>
      <h3>Top View: X / Y</h3>
      <canvas id="top" width="500" height="500"></canvas>
    </div>
    <div>
      <h3>Side View: X / Z</h3>
      <canvas id="side" width="500" height="500"></canvas>
    </div>
  </div>

  <h2>Output</h2>
  <pre id="output">No bounds yet.</pre>

<script>
let state = null;

async function post(path) {
  const r = await fetch(path, {method: 'POST'});
  const txt = await r.text();
  console.log(txt);
  await update();
}

function drawCanvas(canvasId, aName, bName) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "#444";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 10; i++) {
    let p = i * canvas.width / 10;
    ctx.beginPath();
    ctx.moveTo(p, 0);
    ctx.lineTo(p, canvas.height);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, p);
    ctx.lineTo(canvas.width, p);
    ctx.stroke();
  }

  ctx.fillStyle = "#aaa";
  ctx.font = "14px Arial";
  ctx.fillText(aName, canvas.width - 30, canvas.height / 2 - 8);
  ctx.fillText(bName, canvas.width / 2 + 8, 18);

  if (!state) return;

  let pts = [];
  if (state.points) pts = pts.concat(state.points);
  if (state.current) pts.push(state.current);

  if (pts.length === 0) return;

  let ai = {x:0, y:1, z:2}[aName];
  let bi = {x:0, y:1, z:2}[bName];

  let valsA = pts.map(p => p[ai]);
  let valsB = pts.map(p => p[bi]);

  let minA = Math.min(...valsA) - 0.05;
  let maxA = Math.max(...valsA) + 0.05;
  let minB = Math.min(...valsB) - 0.05;
  let maxB = Math.max(...valsB) + 0.05;

  function project(p) {
    let a = p[ai];
    let b = p[bi];

    let x = (a - minA) / (maxA - minA) * canvas.width;
    let y = canvas.height - (b - minB) / (maxB - minB) * canvas.height;
    return [x, y];
  }

  // Draw captured points
  ctx.fillStyle = "#1e90ff";
  state.points.forEach((p, idx) => {
    let [x, y] = project(p);
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillText(String(idx + 1), x + 10, y - 10);
  });

  // Draw current point
  if (state.current) {
    let [x, y] = project(state.current);
    ctx.fillStyle = "#00ff66";
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw bounds rectangle if done
  if (state.bounds) {
    let mn = state.bounds.min;
    let mx = state.bounds.max;

    let corners = [
      [mn[ai], mn[bi]],
      [mx[ai], mn[bi]],
      [mx[ai], mx[bi]],
      [mn[ai], mx[bi]]
    ];

    function project2(a, b) {
      let x = (a - minA) / (maxA - minA) * canvas.width;
      let y = canvas.height - (b - minB) / (maxB - minB) * canvas.height;
      return [x, y];
    }

    ctx.strokeStyle = "#ffcc00";
    ctx.lineWidth = 3;
    ctx.beginPath();
    let c0 = project2(corners[0][0], corners[0][1]);
    ctx.moveTo(c0[0], c0[1]);
    for (let i = 1; i < corners.length; i++) {
      let c = project2(corners[i][0], corners[i][1]);
      ctx.lineTo(c[0], c[1]);
    }
    ctx.closePath();
    ctx.stroke();
  }
}

async function update() {
  const r = await fetch('/state');
  state = await r.json();

  if (state.current) {
    document.getElementById("xyz").innerText =
      `x=${state.current[0].toFixed(4)} m, y=${state.current[1].toFixed(4)} m, z=${state.current[2].toFixed(4)} m`;
  } else {
    document.getElementById("xyz").innerText = state.status || "Waiting...";
  }

  document.getElementById("count").innerText =
    `Captured points: ${state.points.length}`;

  if (state.output) {
    document.getElementById("output").innerText = state.output;
  }

  drawCanvas("top", "x", "y");
  drawCanvas("side", "x", "z");
}

setInterval(update, 100);
update();
</script>
</body>
</html>
"""


class SharedState:
    def __init__(self, margin_m: float):
        self.lock = threading.Lock()
        self.current = None
        self.points = []
        self.bounds = None
        self.output = ""
        self.status = "Starting..."
        self.margin_m = margin_m
        self.torque_requests = []
        self.motors_released = False

    def snapshot(self):
        with self.lock:
            return {
                "current": self.current,
                "points": self.points,
                "bounds": self.bounds,
                "output": self.output,
                "status": self.status,
            }


def format_bounds(points, margin_m):
    arr = np.array(points, dtype=float)
    mn = arr.min(axis=0) - margin_m
    mx = arr.max(axis=0) + margin_m

    mn = [round(float(v), 4) for v in mn]
    mx = [round(float(v), 4) for v in mx]

    output = (
        "end_effector_bounds:\n"
        f"  min: [{mn[0]}, {mn[1]}, {mn[2]}]\n"
        f"  max: [{mx[0]}, {mx[1]}, {mx[2]}]\n"
    )

    return {"min": mn, "max": mx}, output


def set_robot_torque(robot, enabled: bool):
    """Enable or disable SO-101 servo torque."""
    bus = robot.bus

    if enabled:
        print("Locking motors / enabling torque...")
        bus.enable_torque()
    else:
        print("Releasing motors / disabling torque...")
        bus.disable_torque()


def robot_reader(config_path, state: SharedState):
    cfg = load_config(config_path)

    loaded = load_robot(config_path)

    # Some Telerobot versions return robot directly.
    # Others return a tuple like (robot, cameras) or similar.
    if isinstance(loaded, tuple):
        robot = None
        for item in loaded:
            if hasattr(item, "connect") and hasattr(item, "get_observation"):
                robot = item
                break

        if robot is None:
            raise RuntimeError(
                f"Could not find robot object inside load_robot() tuple. "
                f"Tuple types: {[type(x).__name__ for x in loaded]}"
            )
    else:
        robot = loaded

    arm_name = next(iter(cfg.arms))
    arm_cfg = cfg.arms[arm_name]

    log.info("Connecting robot...")
    robot.connect()
    log.info("Robot connected.")

    motor_names = list(robot.bus.motors.keys())

    kinematics = build_kinematics(
        arm_type=arm_cfg.type,
        motor_names=motor_names,
        regularization=arm_cfg.regularization,
    )

    while True:
        try:
            torque_request = None
            with state.lock:
                if state.torque_requests:
                    torque_request = state.torque_requests.pop(0)

            if torque_request == "release":
                set_robot_torque(robot, enabled=False)
                with state.lock:
                    state.motors_released = True
                    state.status = "Motors released. Move arm by hand."
                    state.output = "Motors released. Move arm by hand, then Capture Point."
                time.sleep(0.2)

            elif torque_request == "lock":
                set_robot_torque(robot, enabled=True)
                with state.lock:
                    state.motors_released = False
                    state.status = "Motors locked."
                    state.output = "Motors locked."
                time.sleep(0.2)

            obs = robot.get_observation()

            q = np.array(
                [float(obs[f"{name}.pos"]) for name in motor_names],
                dtype=float,
            )

            t = kinematics.forward_kinematics(q)
            pos = t[:3, 3]

            with state.lock:
                state.current = [
                    round(float(pos[0]), 5),
                    round(float(pos[1]), 5),
                    round(float(pos[2]), 5),
                ]
                state.status = "Robot connected."

            time.sleep(0.03)

        except Exception as e:
            with state.lock:
                state.status = f"Robot read error: {e}"
            time.sleep(0.25)


def make_handler(state: SharedState, output_path: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            return

        def _send(self, body, code=200, content_type="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            if isinstance(body, str):
                body = body.encode("utf-8")
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self._send(HTML, content_type="text/html")
                return

            if self.path == "/state":
                self._send(json.dumps(state.snapshot()), content_type="application/json")
                return

            self._send("Not found", code=404)

        def do_POST(self):
            if self.path == "/release_motors":
                with state.lock:
                    state.torque_requests.append("release")
                    state.output = "Queued motor release..."
                self._send("release queued")
                return

            if self.path == "/lock_motors":
                with state.lock:
                    state.torque_requests.append("lock")
                    state.output = "Queued motor lock..."
                self._send("lock queued")
                return

            if self.path == "/capture":
                with state.lock:
                    if state.current is None:
                        self._send("No current robot pose yet.", code=400)
                        return

                    state.points.append(list(state.current))
                    state.output = (
                        f"Captured point {len(state.points)}:\n"
                        f"{state.current}\n\n"
                        "Need at least 4 points, then press Done."
                    )

                self._send("captured")
                return

            if self.path == "/undo":
                with state.lock:
                    if state.points:
                        removed = state.points.pop()
                        state.output = f"Removed point: {removed}"
                        state.bounds = None
                    else:
                        state.output = "No points to undo."
                self._send("undo")
                return

            if self.path == "/done":
                with state.lock:
                    if len(state.points) < 4:
                        state.output = (
                            f"Only {len(state.points)} points captured.\n"
                            "Capture at least 4 points: 2 bottom and 2 top."
                        )
                        self._send("Need at least 4 points.", code=400)
                        return

                    bounds, output = format_bounds(state.points, state.margin_m)
                    state.bounds = bounds
                    state.output = output

                    output_path.write_text(output)

                    print("\n========== COPY THIS INTO config.yaml ==========")
                    print(output)
                    print(f"Saved to: {output_path}")
                    print("===============================================\n")

                self._send("done")
                return

            self._send("Not found", code=404)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/workspace/telerobot/config.yaml",
        help="Path to Telerobot config.yaml",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8787,
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.02,
        help="Extra safety margin added to min/max bounds in meters.",
    )
    parser.add_argument(
        "--output",
        default="/workspace/telerobot/workspace_bounds.yaml",
    )
    args = parser.parse_args()

    state = SharedState(margin_m=args.margin)
    output_path = Path(args.output)

    t = threading.Thread(
        target=robot_reader,
        args=(args.config, state),
        daemon=True,
    )
    t.start()

    server = ThreadingHTTPServer(
        (args.host, args.port),
        make_handler(state, output_path),
    )

    print(f"\nOpen this on Ubuntu browser:")
    print(f"http://127.0.0.1:{args.port}")
    print("\nMove robot to 4 points, click Capture Point, then Done.\n")

    server.serve_forever()


if __name__ == "__main__":
    main()
