# demo_self_parking_sim.py — Self-parking simulator (MATLAB layers + SAT collisions) with IPC control
# Usage:
#   python demo_self_parking_sim.py                  # default: IPC 127.0.0.1:55555
#   python demo_self_parking_sim.py --host 127.0.0.1 --port 55556
#   python demo_self_parking_sim.py --mode wasd      # (debug) keyboard drive mode

from __future__ import annotations

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_HINT_IME_SHOW_UI", "0")  # suppress macOS IME popups

import argparse, errno, json, math, random, socket, copy, subprocess, sys
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pygame
from pygame import gfxdraw
from scipy.io import loadmat

# --------- 폰트 헬퍼 ---------
FONT_CANDIDATES = [
    ["Apple SD Gothic Neo", "AppleGothic", "NanumGothic"],
    ["Noto Sans CJK KR", "Noto Sans KR", "NotoSansKR", "Arial Unicode MS"],
    ["Malgun Gothic", "Gulim", "UnDotum"],
]


def load_font_with_fallback(size: int, bold: bool = False):
    """Search through multiple font families so localized text renders cleanly."""
    font_path = None
    for group in FONT_CANDIDATES:
        font_path = pygame.font.match_font(group, bold=bold)
        if font_path:
            break
    if not font_path:
        return pygame.font.Font(None, size)
    return pygame.font.Font(font_path, size)

# Viewport (screen sub-rectangle) for prettier HUD layout; configured in main().
vp_ox = vp_oy = vp_w = vp_h = None

BASE_MAP_CACHE = {}
RECENT_RESULTS = deque(maxlen=3)
REPLAY_DIR = "replays"

MARKER_COLORS = {
    "boundary": (200, 60, 60),
    "occupied_slot": (210, 150, 40),
    "stationary": (140, 60, 180),
    "unknown": (80, 80, 80),
    "line": (30, 30, 30),
}

PARKING_SUCCESS_IOU = 0.30  # minimum IoU threshold (30%) to consider a pass
# Minimum alignment cosine (≈ ±48°) to treat the orientation as correct
ORIENTATION_ALIGNMENT_THRESHOLD = math.cos(math.radians(48.0))

BASE_WINDOW_SIZE = (1480, 900)
MIN_WINDOW_SIZE = (1120, 720)
SIDEBAR_WIDTH_RANGE = (320, 460)
MAIN_VIEW_MIN_WIDTH = 560
MAP_CARD_GAP = 20
MAP_CARD_PADDING = 32
MAP_CARD_ASPECT = 16 / 9
ACCENT_COLOR = (65, 110, 220)
HIGHLIGHT_ALPHA = 80
STEER_FLIP_DEADZONE = math.radians(2.0)
MIN_TARGET_DISTANCE = 8.0  # meters; ensure target slot not adjacent to start
LEFT_EXCLUDE_MARGIN = 6.0  # meters; ignore slots hugging the left wall


def build_obs_payload(t: float, state, target_slot, params) -> Dict[str, Any]:
    """Convert the runtime state into an observation payload for agents/replays."""
    return {
        "t": t,
        "state": {
            "x": float(state.x),
            "y": float(state.y),
            "yaw": float(state.yaw),
            "v": float(state.v),
        },
        "target_slot": list(map(float, target_slot)),
        "limits": {
            "dt": float(params.dt),
            "L": float(params.L),
            "maxSteer": float(params.maxSteer),
            "maxAccel": float(params.maxAccel),
            "maxBrake": float(params.maxBrake),
            "steerRate": float(params.steerRate),
        },
    }


def enforce_min_window_size(width: int, height: int) -> tuple[int, int]:
    min_w, min_h = MIN_WINDOW_SIZE
    return max(width, min_w), max(height, min_h)


def compute_layout(sw: int, sh: int) -> tuple[pygame.Rect, pygame.Rect]:
    """Adjust main view and sidebar widths responsively to match the window."""
    base_width = BASE_WINDOW_SIZE[0]
    scale = clamp(sw / base_width, 0.65, 1.0)

    base_min_sidebar, base_max_sidebar = SIDEBAR_WIDTH_RANGE
    min_sidebar = max(260, int(base_min_sidebar * scale))
    max_sidebar = max(min_sidebar, int(base_max_sidebar * scale))

    min_main = max(500, int(MAIN_VIEW_MIN_WIDTH * scale))
    target_sidebar = int(sw * (0.24 + (1.0 - scale) * 0.08))
    sidebar_width = int(clamp(target_sidebar, min_sidebar, max_sidebar))
    main_width = sw - sidebar_width

    if main_width < min_main:
        main_width = max(min_main, sw - min_sidebar)
        sidebar_width = sw - main_width

    if sidebar_width < min_sidebar:
        sidebar_width = min_sidebar
        main_width = max(sw - sidebar_width, min_main)

    sidebar_width = max(0, min(sidebar_width, sw))
    main_width = max(0, min(sw - sidebar_width, sw))

    main_rect = pygame.Rect(0, 0, main_width, sh)
    sidebar_rect = pygame.Rect(main_rect.right, 0, sw - main_rect.width, sh)
    return main_rect, sidebar_rect


def determine_map_columns(window_width: int) -> int:
    if window_width >= 1400:
        return 3
    if window_width >= 980:
        return 2
    return 1


def ellipsize_text(font: pygame.font.Font, text: str, max_width: int) -> str:
    if font.size(text)[0] <= max_width:
        return text
    ellipsis = "…"
    ellipsis_width = font.size(ellipsis)[0]
    if ellipsis_width >= max_width:
        return ellipsis
    buf = []
    for ch in text:
        buf.append(ch)
        if font.size("".join(buf))[0] + ellipsis_width > max_width:
            buf.pop()
            break
    return "".join(buf).rstrip() + ellipsis


def draw_wrapped_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    rect: pygame.Rect,
    align: str = "left",
    line_height: float = 1.35,
    max_lines: int | None = None,
    ellipsis: bool = True,
) -> int:
    if not text or rect.width <= 0:
        return 0
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if font.size(candidate)[0] <= rect.width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if max_lines is not None and len(lines) > max_lines:
        visible = lines[:max_lines]
        if ellipsis and visible:
            visible[-1] = ellipsize_text(font, visible[-1], rect.width)
        lines = visible

    y = rect.top
    drawn = 0
    line_spacing = int(font.get_height() * line_height)
    for line in lines:
        if y + font.get_height() > rect.bottom:
            break
        text_surface = font.render(line, True, color)
        if align == "center":
            x = rect.left + (rect.width - text_surface.get_width()) // 2
        elif align == "right":
            x = rect.right - text_surface.get_width()
        else:
            x = rect.left
        surface.blit(text_surface, (x, y))
        y += line_spacing
        drawn += 1
    return drawn


def update_viewport(sw: int, sh: int) -> None:
    global vp_ox, vp_oy, vp_w, vp_h
    margin = 24
    hud_h = 64
    vp_ox = margin
    vp_oy = hud_h
    vp_w = max(1, sw - 2 * margin)
    vp_h = max(1, sh - hud_h - margin)


def load_replay_catalog(limit: int = 10) -> list[dict]:
    root = Path(REPLAY_DIR)
    if not root.exists():
        return []
    try:
        files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    except (OSError, PermissionError):
        return []
    entries: list[dict] = []
    for path in files[:limit]:
        entry = {"path": path, "meta": {}, "error": None}
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            meta = data.get("meta", {}) if isinstance(data, dict) else {}
            entry["meta"] = meta
        except Exception as exc:
            entry["error"] = str(exc)
        entries.append(entry)
    return entries


def discover_agent_entries(base_dir: Path) -> list[dict]:
    entries: list[dict] = []
    local_agent = base_dir / "student_algorithms.py"
    if local_agent.exists():
        entries.append({
            "label": "student_algorithms.py (local)",
            "path": local_agent,
            "cwd": base_dir,
            "type": "process",
        })
    sibling_dir = base_dir.parent / "self-parking-user-algorithms"
    sibling = (sibling_dir / "my_agent.py").resolve()
    if sibling.exists():
        entries.append({
            "label": "self-parking-user-algorithms/my_agent.py",
            "path": sibling,
            "cwd": sibling.parent,
            "type": "process",
        })
    entries.append({
        "label": "Built-in WASD control (this computer)",
        "path": None,
        "cwd": base_dir,
        "type": "local_manual",
    })
    return entries


class AgentManager:
    """Manage student algorithm subprocesses."""

    def __init__(self, entries: list[dict], python_executable: str | None = None):
        self.entries: list[dict] = []
        for entry in entries:
            self.entries.append({
                "label": entry.get("label", "Student Algorithm"),
                "path": Path(entry.get("path")) if entry.get("path") else None,
                "cwd": Path(entry.get("cwd", Path.cwd())),
                "type": entry.get("type", "process"),
            })
        self.python_executable = python_executable or sys.executable
        self.process: subprocess.Popen | None = None
        self.active_idx: int | None = None
        self.manual_active: bool = False

    def is_running(self) -> bool:
        if self.manual_active:
            return True
        return self.process is not None and self.process.poll() is None

    def poll(self) -> None:
        if self.manual_active:
            return
        if self.process and self.process.poll() is not None:
            self.process = None
            self.active_idx = None

    def start(self, idx: int, host: str, port: int) -> tuple[bool, str | None]:
        if idx < 0 or idx >= len(self.entries):
            return False, "No agent entry has been selected."
        entry = self.entries[idx]
        path_obj = entry["path"]
        if path_obj is not None and not path_obj.exists():
            return False, "Agent file could not be found."

        self.stop()
        entry_type = entry.get("type", "process")
        if entry_type == "local_manual":
            self.manual_active = True
            self.active_idx = idx
            return True, None

        if entry["path"] is None:
            return False, "No executable interpreter available for this agent."

        cmd = [self.python_executable, str(entry["path"]), "--host", host, "--port", str(port)]
        try:
            self.process = subprocess.Popen(cmd, cwd=str(entry["cwd"]))
            self.active_idx = idx
            return True, None
        except Exception as exc:
            self.process = None
            self.active_idx = None
            return False, str(exc)

    def stop(self) -> None:
        if self.manual_active:
            self.manual_active = False
            self.active_idx = None
            return
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            except Exception:
                pass
        self.process = None
        self.active_idx = None

    def is_manual_active(self) -> bool:
        return self.manual_active and self.active_idx is not None

# ----------------- 유틸 -----------------
def clamp(x,a,b): return max(a, min(b,x))
def wrap_to_pi(a): return (a+math.pi)%(2*math.pi)-math.pi
def move_toward(x,tgt,dx): return x + clamp(tgt-x, -dx, dx)

def rotate_and_translate(points, yaw, tx, ty):
    c,s = math.cos(yaw), math.sin(yaw)
    return [(c*x - s*y + tx, s*x + c*y + ty) for x,y in points]

def world_to_screen(x, y, world, sw, sh):
    global vp_ox, vp_oy, vp_w, vp_h
    xmin, xmax, ymin, ymax = world
    if vp_ox is None:
        ox, oy, iw, ih = 0, 0, sw, sh
    else:
        ox, oy, iw, ih = vp_ox, vp_oy, vp_w, vp_h
    sx = ox + (x - xmin) / (xmax - xmin) * iw
    sy = oy + (1.0 - (y - ymin) / (ymax - ymin)) * ih
    return int(sx), int(sy)

def draw_polygon(surface, poly, color, world, sw, sh, width=0):
    pts = [world_to_screen(x,y,world,sw,sh) for (x,y) in poly]
    if width == 0:
        gfxdraw.aapolygon(surface, pts, color)
        gfxdraw.filled_polygon(surface, pts, color)
    else:
        pygame.draw.polygon(surface, color, pts, width)

def draw_rect(surface, rect, color, world, sw, sh, width=0):
    xmin,xmax,ymin,ymax = rect
    poly = [(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
    draw_polygon(surface, poly, color, world, sw, sh, width=width)

def rect_contains_poly(rect, poly_xy):
    xmin,xmax,ymin,ymax = rect
    for (x,y) in poly_xy:
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            return False
    return True

# --------- Geometry helpers for robust collision (SAT) ---------
def poly_from_rect(rect):
    xmin, xmax, ymin, ymax = rect
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

def _project(points, axis):
    ax, ay = axis
    dots = [p[0]*ax + p[1]*ay for p in points]
    return min(dots), max(dots)

def polys_intersect_SAT(polyA, polyB):
    """Convex polygon intersection via Separating Axis Theorem (SAT)."""
    for poly in (polyA, polyB):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % n]
            axis = (-(y2 - y1), (x2 - x1))
            length = math.hypot(axis[0], axis[1])
            if length == 0: continue
            axis = (axis[0]/length, axis[1]/length)
            minA, maxA = _project(polyA, axis)
            minB, maxB = _project(polyB, axis)
            if maxA < minB or maxB < minA:  # separating axis exists
                return False
    return True

def poly_intersects_rect(poly, rect):
    return polys_intersect_SAT(poly, poly_from_rect(rect))

def polygon_area(poly: list[tuple[float, float]]) -> float:
    """Shoelace area for simple polygons."""
    if len(poly) < 3:
        return 0.0
    acc = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5

def clip_polygon_with_rect(poly: list[tuple[float, float]], rect: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    """Sutherland–Hodgman clipping of polygon with axis-aligned rectangle."""
    xmin, xmax, ymin, ymax = rect
    if len(poly) < 3:
        return []

    def clip_edge(points, keep_fn, intersect_fn):
        if not points:
            return []
        output = []
        s = points[-1]
        s_inside = keep_fn(s)
        for e in points:
            e_inside = keep_fn(e)
            if e_inside:
                if not s_inside:
                    output.append(intersect_fn(s, e))
                output.append(e)
            elif s_inside:
                output.append(intersect_fn(s, e))
            s, s_inside = e, e_inside
        return output

    def intersect_vertical(s, e, x_bound):
        sx, sy = s
        ex, ey = e
        if abs(ex - sx) < 1e-9:
            return x_bound, sy
        t = (x_bound - sx) / (ex - sx)
        return x_bound, sy + t * (ey - sy)

    def intersect_horizontal(s, e, y_bound):
        sx, sy = s
        ex, ey = e
        if abs(ey - sy) < 1e-9:
            return sx, y_bound
        t = (y_bound - sy) / (ey - sy)
        return sx + t * (ex - sx), y_bound

    points = poly
    points = clip_edge(points, lambda p: p[0] >= xmin, lambda s, e: intersect_vertical(s, e, xmin))
    points = clip_edge(points, lambda p: p[0] <= xmax, lambda s, e: intersect_vertical(s, e, xmax))
    points = clip_edge(points, lambda p: p[1] >= ymin, lambda s, e: intersect_horizontal(s, e, ymin))
    points = clip_edge(points, lambda p: p[1] <= ymax, lambda s, e: intersect_horizontal(s, e, ymax))
    return points

def compute_slot_iou(car_poly: list[tuple[float, float]], slot_rect: tuple[float, float, float, float]) -> float:
    """Compute IoU between the car polygon and an axis-aligned slot rectangle."""
    intersection_poly = clip_polygon_with_rect(car_poly, slot_rect)
    inter_area = polygon_area(intersection_poly)
    if inter_area <= 0.0:
        return 0.0
    car_area = polygon_area(car_poly)
    xmin, xmax, ymin, ymax = slot_rect
    slot_area = max(0.0, (xmax - xmin) * (ymax - ymin))
    union_area = max(car_area + slot_area - inter_area, 1e-9)
    return inter_area / union_area

def determine_parking_orientation(state: "State", slot_rect: tuple[float, float, float, float]) -> str:
    """Return 'front_in' or 'rear_in' depending on vehicle heading relative to slot."""
    width = slot_rect[1] - slot_rect[0]
    height = slot_rect[3] - slot_rect[2]
    forward_vec = (math.cos(state.yaw), math.sin(state.yaw))
    axis_value = forward_vec[1] if height >= width else forward_vec[0]
    if abs(axis_value) < ORIENTATION_ALIGNMENT_THRESHOLD:
        return "unknown"
    return "front_in" if axis_value >= 0 else "rear_in"

def perimeter_points(poly, spacing=0.10):
    """Sample points uniformly along polygon perimeter (meters)."""
    pts = []
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        seg_len = math.hypot(x2-x1, y2-y1)
        if seg_len < 1e-9: continue
        steps = max(1, int(seg_len/spacing))
        for k in range(steps+1):
            t = k/steps
            pts.append((x1 + t*(x2-x1), y1 + t*(y2-y1)))
    return pts


def draw_overlay(surface, title, lines, font, sw, sh):
    """Render a translucent overlay in the viewport to show round summary."""
    global vp_ox, vp_oy, vp_w, vp_h
    if vp_ox is None:
        ox, oy, iw, ih = 0, 0, sw, sh
    else:
        ox, oy, iw, ih = vp_ox, vp_oy, vp_w, vp_h

    overlay = pygame.Surface((iw, ih), pygame.SRCALPHA)
    overlay.fill((255, 255, 255, 220))
    surface.blit(overlay, (ox, oy))

    pygame.draw.rect(surface, (0, 0, 0), pygame.Rect(ox, oy, iw, ih), 3)

    y = oy + 30
    title_color = (20, 80, 20) if title == "SUCCESS" else (120, 30, 30)
    title_img = font.render(title, True, title_color)
    surface.blit(title_img, (ox + 20, y))
    y += 30

    for text in lines:
        img = font.render(text, True, (0, 0, 0))
        surface.blit(img, (ox + 20, y))
        y += 22

    tip = "R: restart  ·  M: change map  ·  ESC/Q: quit"
    tip_img = font.render(tip, True, (0, 0, 0))
    surface.blit(tip_img, (ox + 20, oy + ih - 32))

# ----------------- 파라미터/상태 -----------------
@dataclass
class Params:
    dt: float = 1.0/60.0             # 60 Hz
    L: float = 2.6                   # wheelbase
    maxSteer: float = math.radians(35)
    maxAccel: float = 3.0            # m/s^2
    maxBrake: float = 7.0            # m/s^2
    steerRate: float = math.radians(180)
    cmdRate: float = 3.0
    timeout: float = 180.0
    # 차량 외형 (후축 기준 시각화용)
    Lf: float = 1.6
    Lr: float = 1.4
    W: float  = 1.6
    selfCenterRate: float = math.radians(90)  # no input → steer toward 0
    coastDecel: float = 0.8                   # m/s^2

class State:
    def __init__(self, x=0.0,y=0.0,yaw=0.0,v=0.0):
        self.x=x; self.y=y; self.yaw=yaw; self.v=v

class InputCmd:
    def __init__(self):
        self.delta_tgt=0.0; self.accel=0.0; self.brake=0.0; self.gear='D'

@dataclass
class RoundStats:
    """Accumulates driving metrics over a single simulation round."""

    elapsed: float = 0.0          # elapsed driving time (seconds)
    distance: float = 0.0         # total travel distance (meters)
    gear_switches: int = 0        # number of gear changes
    avg_speed_accum: float = 0.0  # sum of speeds for average computation
    speed_samples: int = 0        # number of samples contributing to the average
    min_abs_steer: float = float("inf")  # minimum absolute steering angle observed
    direction_flips: int = 0            # count of direction changes (left/right)
    prev_gear: str = "D"                # previous gear to detect transitions
    prev_delta_sign: int = 0            # previous steer sign for flip detection
    final_iou: float = 0.0              # IoU with the parking slot at completion
    final_orientation: str = "unknown"  # final parking orientation label
    final_speed: float = 0.0            # last recorded speed for stop validation

# ----------------- 모델/차량 -----------------
def step_kinematic(state: State, delta, a, P: Params):
    state.v += a * P.dt
    state.x += state.v * math.cos(state.yaw) * P.dt
    state.y += state.v * math.sin(state.yaw) * P.dt
    state.yaw = wrap_to_pi(state.yaw + (state.v / P.L) * math.tan(delta) * P.dt)
    return state

def car_polygon(state: State, P: Params):
    pts = [( P.Lf,  P.W/2),
           ( P.Lf, -P.W/2),
           (-P.Lr, -P.W/2),
           (-P.Lr,  P.W/2)]
    return rotate_and_translate(pts, state.yaw, state.x, state.y)

def draw_car(surface, state: State, delta, P: Params, world, sw, sh):
    body_poly = car_polygon(state, P)
    draw_polygon(surface, body_poly, (150,200,255), world, sw, sh, width=0)
    draw_polygon(surface, body_poly, (30,60,120),   world, sw, sh, width=2)
    # 앞표식(노즈 삼각형)
    nose = (P.Lf + 0.10, 0.0)
    tri = [(P.Lf - 0.30, -0.30), (P.Lf - 0.30, 0.30), nose]
    tri_xy = rotate_and_translate(tri, state.yaw, state.x, state.y)
    draw_polygon(surface, tri_xy, (255,220,100), world, sw, sh, width=0)
    draw_polygon(surface, tri_xy, (120,90,20),   world, sw, sh, width=1)
    # 바퀴(간단)
    wheel_L = 0.50; wheel_W = 0.12
    fx = P.Lf - 0.25; fy =  P.W/2 - 0.20
    wheel_rect = [(-wheel_L/2, -wheel_W/2), (wheel_L/2, -wheel_W/2),
                  (wheel_L/2,  wheel_W/2), (-wheel_L/2,  wheel_W/2)]
    wl = rotate_and_translate(wheel_rect, delta, fx,  fy)
    wl = rotate_and_translate(wl, state.yaw, state.x, state.y)
    wr = rotate_and_translate(wheel_rect, delta, fx, -fy)
    wr = rotate_and_translate(wr, state.yaw, state.x, state.y)
    draw_polygon(surface, wl, (40,40,40), world, sw, sh, width=0)
    draw_polygon(surface, wr, (40,40,40), world, sw, sh, width=0)

# ----------------- Map loading -----------------
class MapAssets:
    """Bundle layered parking-lot data extracted from MATLAB."""

    def __init__(self, C, Cs, Cm, Cp, cellSize, extent, slots, occupied_idx, border, lines, FreeThr, OccThr, walls_rects):
        self.C = C
        self.Cs = Cs
        self.Cm = Cm
        self.Cp = Cp
        self.cellSize = cellSize
        self.extent = extent
        self.slots = slots
        self.occupied_idx = occupied_idx
        self.border = border
        self.lines = lines
        self.FreeThr = FreeThr
        self.OccThr = OccThr
        self.walls_rects = walls_rects

def load_parking_assets(mat_path="parking_assets_layers_75x50.mat") -> MapAssets:
    """Load layered assets from a MATLAB `.mat` file."""
    m = loadmat(mat_path)
    # Convert MATLAB doubles to float32 to reduce memory usage.
    C = m["C"].astype(np.float32)
    Cs = m["C_stationary"].astype(np.float32)
    Cm = m["C_markings"].astype(np.float32)
    Cp = m["C_parked"].astype(np.float32)
    # Squeeze scalars to floats.
    cellSize = float(np.array(m["cellSize"]).squeeze())
    extent = tuple(np.array(m["extent"]).squeeze().tolist())
    # Keep slots, lines, etc. as float arrays.
    slots = np.array(m["slots"]).astype(float)
    occupied_idx = np.array(m["occupied_idx"]).astype(bool).ravel()
    border = tuple(np.array(m["border"]).squeeze().tolist())
    lines = np.array(m["lines"]).astype(float)
    # Preserve occupancy thresholds as floats.
    FreeThr = float(np.array(m["FreeThreshold"]).squeeze())
    OccThr = float(np.array(m["OccupiedThreshold"]).squeeze())
    walls_rects = np.array(m["walls_rects"]).astype(float)
    return MapAssets(C, Cs, Cm, Cp, cellSize, extent, slots, occupied_idx,
                     border, lines, FreeThr, OccThr, walls_rects)

def world_to_rc(x, y, extent, cellSize, H):
    xmin,xmax,ymin,ymax = extent
    col = int(np.floor((x - xmin) / cellSize))
    row_from_bottom = int(np.floor((y - ymin) / cellSize))
    row = H - 1 - row_from_bottom  # compensate for MATLAB row ordering
    return row, col

def draw_walls_rects(surface, world, sw, sh, rects):
    for r in rects:
        xmin, xmax, ymin, ymax = float(r[0]), float(r[1]), float(r[2]), float(r[3])
        draw_rect(surface, (xmin, xmax, ymin, ymax), (0,0,0), world, sw, sh, width=0)

def grid_cell_rect(row: int, col: int, extent, cellSize, H):
    """Convert (row,col) to world-space rectangle."""
    xmin = extent[0] + col * cellSize
    xmax = xmin + cellSize
    ymax = extent[3] - row * cellSize
    ymin = ymax - cellSize
    return xmin, xmax, ymin, ymax

def get_base_map(filename: str) -> MapAssets:
    """Load and cache the base `.mat` dataset."""
    if filename not in BASE_MAP_CACHE:
        BASE_MAP_CACHE[filename] = load_parking_assets(filename)
    return BASE_MAP_CACHE[filename]

def generate_map_thumbnail(M: MapAssets, size=(280, 160), highlight_idx: int | None = None):
    """Render the map layers into a compact preview card."""
    width, height = size
    surface = pygame.Surface((width, height))
    surface.fill((248, 250, 255))

    xmin, xmax, ymin, ymax = M.extent
    w_span = max(xmax - xmin, 1e-6)
    h_span = max(ymax - ymin, 1e-6)

    def to_screen(x, y):
        sx = (x - xmin) / w_span * (width - 10) + 5
        sy = (1.0 - (y - ymin) / h_span) * (height - 10) + 5
        return int(sx), int(sy)

    pygame.draw.rect(surface, (90, 120, 200), pygame.Rect(2, 2, width - 4, height - 4), 2, border_radius=8)

    for idx, rect in enumerate(M.slots):
        pts = [to_screen(rect[0], rect[2]),
               to_screen(rect[1], rect[2]),
               to_screen(rect[1], rect[3]),
               to_screen(rect[0], rect[3])]
        if M.occupied_idx[idx]:
            pygame.draw.polygon(surface, (200, 210, 230), pts)
        pygame.draw.lines(surface, (110, 130, 180), True, pts, 1)

    if highlight_idx is not None and 0 <= highlight_idx < len(M.slots):
        rect = M.slots[highlight_idx]
        pts = [to_screen(rect[0], rect[2]),
               to_screen(rect[1], rect[2]),
               to_screen(rect[1], rect[3]),
               to_screen(rect[0], rect[3])]
        pygame.draw.polygon(surface, (180, 240, 190), pts)
        pygame.draw.lines(surface, (60, 140, 60), True, pts, 3)

    # draw walls/line markings
    for r in M.walls_rects:
        pts = [to_screen(r[0], r[2]),
               to_screen(r[1], r[2]),
               to_screen(r[1], r[3]),
               to_screen(r[0], r[3])]
        pygame.draw.polygon(surface, (150, 60, 60), pts, 0)

    for x1, y1, x2, y2 in M.lines:
        pygame.draw.line(surface, (80, 100, 150), to_screen(x1, y1), to_screen(x2, y2), 2)

    border_pts = [to_screen(M.border[0], M.border[2]),
                  to_screen(M.border[1], M.border[2]),
                  to_screen(M.border[1], M.border[3]),
                  to_screen(M.border[0], M.border[3])]
    pygame.draw.lines(surface, (50, 70, 120), True, border_pts, 3)

    return surface

def _ensure_at_least_one_free_slot(M: MapAssets, rng: random.Random) -> None:
    """Ensure at least one slot remains free by clearing one at random if needed."""
    if M.slots.size == 0:
        return
    if np.all(M.occupied_idx):
        free_idx = rng.randrange(len(M.occupied_idx))
        M.occupied_idx[free_idx] = False

CAR_LENGTH = 3.0  # 차량 길이(전후 오버행 포함)
CAR_WIDTH = 1.6   # 차량 폭
SLOT_LENGTH_MARGIN = 1.2  # 슬롯 길이 마진 (전후 총합)
SLOT_WIDTH_MARGIN = 0.6   # 슬롯 폭 마진 (좌우 총합)
ENABLE_STATIONARY_COLLISIONS = False
ENABLE_BOUNDARY_COLLISIONS = True
LINE_COLLISION_HALF_WIDTH = 0.25

def scale_map_geometry(M: MapAssets, factor: float) -> None:
    """Scale the map about its center, adjusting slots and boundaries."""
    if factor <= 0:
        return

    xmin, xmax, ymin, ymax = M.extent
    cx = (xmin + xmax) * 0.5
    cy = (ymin + ymax) * 0.5

    def scale_x(values):
        return cx + (values - cx) * factor

    def scale_y(values):
        return cy + (values - cy) * factor

    M.extent = (
        float(scale_x(xmin)),
        float(scale_x(xmax)),
        float(scale_y(ymin)),
        float(scale_y(ymax)),
    )

    bxmin, bxmax, bymin, bymax = M.border
    M.border = (
        float(scale_x(bxmin)),
        float(scale_x(bxmax)),
        float(scale_y(bymin)),
        float(scale_y(bymax)),
    )

    if M.slots.size > 0:
        M.slots[:, 0] = scale_x(M.slots[:, 0])
        M.slots[:, 1] = scale_x(M.slots[:, 1])
        M.slots[:, 2] = scale_y(M.slots[:, 2])
        M.slots[:, 3] = scale_y(M.slots[:, 3])

    if M.lines.size > 0:
        M.lines[:, 0] = scale_x(M.lines[:, 0])
        M.lines[:, 2] = scale_x(M.lines[:, 2])
        M.lines[:, 1] = scale_y(M.lines[:, 1])
        M.lines[:, 3] = scale_y(M.lines[:, 3])

    if M.walls_rects.size > 0:
        M.walls_rects[:, 0] = scale_x(M.walls_rects[:, 0])
        M.walls_rects[:, 1] = scale_x(M.walls_rects[:, 1])
        M.walls_rects[:, 2] = scale_y(M.walls_rects[:, 2])
        M.walls_rects[:, 3] = scale_y(M.walls_rects[:, 3])

    M.cellSize *= factor

def resize_slots_to_vehicle(M: MapAssets) -> None:
    """Resize slots around their center with vehicle-friendly margins."""
    if M.slots.size == 0:
        return

    half_width = 0.5 * (CAR_WIDTH + SLOT_WIDTH_MARGIN)
    half_length = 0.5 * (CAR_LENGTH + SLOT_LENGTH_MARGIN)
    xmin, xmax, ymin, ymax = M.extent

    centers_x = 0.5 * (M.slots[:, 0] + M.slots[:, 1])
    centers_y = 0.5 * (M.slots[:, 2] + M.slots[:, 3])

    M.slots[:, 0] = np.clip(centers_x - half_width, xmin, xmax)
    M.slots[:, 1] = np.clip(centers_x + half_width, xmin, xmax)
    M.slots[:, 2] = np.clip(centers_y - half_length, ymin, ymax)
    M.slots[:, 3] = np.clip(centers_y + half_length, ymin, ymax)


def open_top_parking_lane(M: MapAssets, tolerance: float = 0.25) -> None:
    """Remove the top boundary so the entry lane remains open."""
    top = float(M.extent[3])
    if M.lines.size > 0:
        keep_lines = []
        horiz_y = [
            float(line[1])
            for line in M.lines
            if np.isclose(line[1], line[3], atol=tolerance)
        ]
        top_slot_y = max(horiz_y) if horiz_y else None
        for line in M.lines:
            y1, y2 = float(line[1]), float(line[3])
            if np.isclose(y1, top, atol=tolerance) and np.isclose(y2, top, atol=tolerance):
                continue  # remove horizontal line hugging the very top
            if (
                top_slot_y is not None
                and np.isclose(y1, y2, atol=tolerance)
                and np.isclose(y1, top_slot_y, atol=max(tolerance, 0.6))
            ):
                continue  # remove the top-most slot boundary
            keep_lines.append(line)
        M.lines = np.array(keep_lines, dtype=float) if keep_lines else np.zeros((0, 4), dtype=float)

def compute_line_rects(M: MapAssets, half_width: float = LINE_COLLISION_HALF_WIDTH) -> np.ndarray:
    """Extrude lane lines into thin rectangles for collision testing."""
    if M.lines.size == 0:
        return np.zeros((0, 4), dtype=float)
    rects = []
    xmin, xmax, ymin, ymax = M.extent
    for x1, y1, x2, y2 in M.lines:
        if np.isclose(x1, x2, atol=1e-6):  # vertical segment
            x_min = min(x1, x2) - half_width
            x_max = max(x1, x2) + half_width
            y_min = min(y1, y2)
            y_max = max(y1, y2)
        elif np.isclose(y1, y2, atol=1e-6):  # horizontal segment
            x_min = min(x1, x2)
            x_max = max(x1, x2)
            y_min = min(y1, y2) - half_width
            y_max = max(y1, y2) + half_width
        else:
            x_min = min(x1, x2) - half_width
            x_max = max(x1, x2) + half_width
            y_min = min(y1, y2) - half_width
            y_max = max(y1, y2) + half_width
        x_min = max(x_min, xmin)
        x_max = min(x_max, xmax)
        y_min = max(y_min, ymin)
        y_max = min(y_max, ymax)
        if x_max <= x_min or y_max <= y_min:
            continue
        rects.append([x_min, x_max, y_min, y_max])
    return np.array(rects, dtype=float)


def apply_map_variant(base: MapAssets, variant: str, seed: int | None = None) -> MapAssets:
    """Apply variant tweaks on top of the base map to alter difficulty."""
    rng = random.Random(seed)
    M = copy.deepcopy(base)

    # 기본 점유 상태도 매번 시드를 통해 섞어 변화를 준다.
    total_slots = len(M.occupied_idx)
    if total_slots > 0:
        occupied_total = int(np.count_nonzero(M.occupied_idx))
        occupied_total = min(occupied_total, max(0, total_slots - 1))  # 최소 한 슬롯은 비워 둔다.
        shuffled = list(range(total_slots))
        rng.shuffle(shuffled)
        new_occ = np.zeros(total_slots, dtype=bool)
        if occupied_total > 0:
            new_occ[np.array(shuffled[:occupied_total])] = True
        M.occupied_idx = new_occ

    if variant == "dense_center":
        occupied = M.occupied_idx.copy()
        free_indices = np.where(~occupied)[0]
        rng.shuffle(free_indices)
        take = max(4, len(free_indices) // 2)
        occupied[free_indices[:take]] = True
        M.occupied_idx = occupied
    elif variant == "one_way_training":
        # 중앙 차선에 장애물을 추가해 통로를 제한한다.
        extras = []
        lane_y = (M.border[2] + M.border[3]) / 2.0
        width = 0.8
        spacing = 4.0
        x = M.border[0] + 6.0 + rng.uniform(-1.0, 1.0)
        while x + width < M.border[1] - 6.0:
            height = rng.uniform(1.0, 2.0)
            extras.append([x, x + width, lane_y - height, lane_y + height])
            x += spacing + rng.uniform(-1.0, 1.5)
        if extras:
            if M.walls_rects.size == 0:
                M.walls_rects = np.array(extras, dtype=float)
            else:
                M.walls_rects = np.vstack([M.walls_rects, np.array(extras, dtype=float)])
    elif variant == "single_free_slot":
        # 하나의 슬롯만 비워 놓고 나머지는 모두 점유 상태로 만든다.
        if total_slots > 0:
            occ = np.ones(total_slots, dtype=bool)
            free_idx = rng.randrange(total_slots)
            occ[free_idx] = False
            M.occupied_idx = occ
        # 3단계는 완전 만차 시나리오로 장애물을 제거하고 맵을 축소한다.
        scale_map_geometry(M, factor=0.8)
        if M.walls_rects.size > 0:
            xmin, xmax, ymin, ymax = M.extent
            mask = (
                np.isclose(M.walls_rects[:, 0], xmin, atol=1e-3) |
                np.isclose(M.walls_rects[:, 1], xmax, atol=1e-3) |
                np.isclose(M.walls_rects[:, 2], ymin, atol=1e-3) |
                np.isclose(M.walls_rects[:, 3], ymax, atol=1e-3)
            )
            M.walls_rects = M.walls_rects[mask] if mask.any() else np.zeros((0, 4), dtype=float)
        if M.Cs.size > 0:
            M.Cs.fill(0.0)
    elif variant == "open_training":
        # 거의 모든 점유 슬롯을 비워 넓은 연습장을 만든다.
        M.occupied_idx = np.zeros_like(M.occupied_idx, dtype=bool)
    _ensure_at_least_one_free_slot(M, rng)
    return M

AVAILABLE_MAPS = [
    {
        "key": "default_lot",
        "name": "Default Lot 75x50",
        "filename": "parking_assets_layers_75x50.mat",
        "summary": "Balanced baseline environment",
        "variant": "",
        "stage": 1,
        "expected_orientation": "front_in",
    },
    {
        "key": "dense_lot",
        "name": "Crowded Lot",
        "filename": "parking_assets_layers_75x50.mat",
        "summary": "Tight layout with occupied neighbors",
        "variant": "dense_center",
        "stage": 2,
        "expected_orientation": "front_in",
    },
    {
        "key": "training_course",
        "name": "Full House Lot",
        "filename": "parking_assets_layers_75x50.mat",
        "summary": "Only a single slot remains open",
        "variant": "single_free_slot",
        "stage": 3,
        "expected_orientation": "rear_in",
    },
]

# Define per-stage scoring thresholds and penalty weights.
# `weights` controls penalty importance and `expected_orientation` enforces the final parking heading.
STAGE_RULES = {
    1: {
        "label": "Stage 1",
        "time_target": 65.0,
        "distance_factor": 0.90,
        "speed_target": 2.0,
        "steer_flip_target": 4,
        "expected_orientation": "front_in",
        "weights": {
            "time": 8.0,
            "distance": 6.0,
            "speed": 5.0,
            "steer_flip": 4.0,
            "parking_iou": 9.0,
            "parking_orientation": 5.0,
            "parking_stop": 3.0,
        },
    },
    2: {
        "label": "Stage 2",
        "time_target": 80.0,
        "distance_factor": 0.95,
        "speed_target": 2.8,
        "steer_flip_target": 5,
        "expected_orientation": "front_in",
        "weights": {
            "time": 9.0,
            "distance": 6.0,
            "speed": 6.0,
            "steer_flip": 5.0,
            "parking_iou": 9.0,
            "parking_orientation": 6.0,
            "parking_stop": 3.0,
        },
    },
    3: {
        "label": "Stage 3",
        "time_target": 95.0,
        "distance_factor": 1.05,
        "speed_target": 3.2,
        "steer_flip_target": 6,
        "expected_orientation": "rear_in",
        "weights": {
            "time": 11.0,
            "distance": 7.0,
            "speed": 6.0,
            "steer_flip": 5.0,
            "parking_iou": 10.0,
            "parking_orientation": 7.0,
            "parking_stop": 3.0,
        },
    },
}

ORIENTATION_LABELS = {
    "front_in": "Front-in parking",
    "rear_in": "Rear-in parking",
}


ORIENTATION_LABELS_SHORT = {
    "front_in": "Front",
    "rear_in": "Rear",
}


def describe_orientation(expected: str | None) -> str:
    if not expected:
        return "Orientation flexible"
    return ORIENTATION_LABELS.get(expected, expected)


def compute_weighted_score(component_score: float, weight: float) -> float:
    return clamp(component_score, 0.0, 1.0) * weight

def orientation_alignment_score(observed: str, expected: str) -> float:
    if not expected:
        return 1.0
    if observed == expected:
        return 1.0
    if observed == "unknown":
        return 0.25
    # wrong orientation still gets small partial credit to avoid zeroing
    return 0.0

def parking_iou_score(final_iou: float) -> float:
    if final_iou <= 0.0:
        return 0.0
    if final_iou >= 1.0:
        return 1.0
    span = max(1e-6, 1.0 - PARKING_SUCCESS_IOU)
    return clamp((final_iou - PARKING_SUCCESS_IOU) / span + 0.5, 0.0, 1.0)

def smooth_ratio(actual: float, target: float) -> float:
    if target <= 1e-6:
        return 1.0
    if actual <= 0.0:
        return 1.0
    return clamp(target / actual, 0.0, 1.0)

def inverse_ratio(actual: float, target: float) -> float:
    if target <= 1e-6:
        return 1.0
    return clamp(target / max(actual, 1e-6), 0.0, 1.0)

def limited_ratio(actual: float, target: float) -> float:
    if target <= 1e-6:
        return 1.0
    return clamp(1.0 - (actual / (target * 2.0)), 0.0, 1.0)


def get_stage_profile(map_cfg: dict | None):
    if not map_cfg:
        return 1, STAGE_RULES[1]
    stage = int(map_cfg.get("stage", 1))
    profile = STAGE_RULES.get(stage, STAGE_RULES[1])
    return stage, profile

def compute_round_score(stats: RoundStats, stage_profile: dict, result_reason: str, world_extent) -> tuple[float, dict]:
    """Compute round scores based on stage rules and gather detailed metrics."""
    # Prepare base metrics shared with the HUD and replay summaries.
    details = {
        "elapsed": stats.elapsed,
        "distance": stats.distance,
        "gear_switches": stats.gear_switches,
        "avg_speed": (stats.avg_speed_accum / stats.speed_samples) if stats.speed_samples > 0 else 0.0,
        "reason": result_reason,
        "parking_iou": stats.final_iou,
        "parking_orientation": stats.final_orientation,
        "final_speed": stats.final_speed,
    }

    # 성공하지 못한 경우에는 즉시 0점을 부여하고 세부 정보만 반환합니다.
    if result_reason != "success":
        details["score"] = 0.0
        return 0.0, details

    # 맵 대각선 길이를 사용해 상대적인 주행 거리 목표를 계산합니다.
    xmin, xmax, ymin, ymax = world_extent
    diag = math.hypot(xmax - xmin, ymax - ymin)

    # 스테이지 설정에서 목표값과 가중치를 추출합니다.
    time_target = stage_profile.get("time_target", 1.0)
    distance_factor = stage_profile.get("distance_factor", 1.0)
    distance_target = max(1e-6, diag * distance_factor)
    speed_target = max(1e-6, float(stage_profile.get("speed_target", 1.0)))
    steer_flip_target = max(1, int(stage_profile.get("steer_flip_target", 1)))
    expected_orientation = stage_profile.get("expected_orientation")
    weights = stage_profile.get("weights", {})
    # 모든 스테이지에서 동일한 안전 점수를 부여해 기본 점수를 고정한다.
    safe_base = 50.0

    avg_speed = details["avg_speed"]
    # 항목별 기여도(패널티 반영 전의 가중 점수)를 기록합니다.
    component_breakdown = {}

    # 각 항목의 성취율을 0~1 범위로 계산하고 가중치를 곱합니다.
    time_score = smooth_ratio(stats.elapsed, time_target)
    component_breakdown["time"] = compute_weighted_score(time_score, weights.get("time", 0.0))

    distance_score = inverse_ratio(stats.distance, distance_target)
    component_breakdown["distance"] = compute_weighted_score(distance_score, weights.get("distance", 0.0))

    # 평균 속도 달성도를 시간 대비 0~1 범위로 정규화합니다.
    speed_score = clamp(avg_speed / speed_target, 0.0, 1.0)
    component_breakdown["speed"] = compute_weighted_score(speed_score, weights.get("speed", 0.0))

    steer_flip_score = limited_ratio(stats.direction_flips, steer_flip_target)
    component_breakdown["steer_flip"] = compute_weighted_score(steer_flip_score, weights.get("steer_flip", 0.0))

    # 슬롯 점유율과 정렬 방향을 통해 최종 주차 품질을 평가합니다.
    parking_iou_component = parking_iou_score(stats.final_iou)
    component_breakdown["parking_iou"] = compute_weighted_score(parking_iou_component, weights.get("parking_iou", 0.0))

    orientation_component = orientation_alignment_score(stats.final_orientation, expected_orientation)
    component_breakdown["parking_orientation"] = compute_weighted_score(orientation_component, weights.get("parking_orientation", 0.0))

    # 최종 속도가 낮을수록 완전 정차로 간주하여 추가 가점을 부여합니다.
    stop_component = clamp(1.0 - stats.final_speed / 0.3, 0.0, 1.0)
    component_breakdown["parking_stop"] = compute_weighted_score(stop_component, weights.get("parking_stop", 0.0))

    total_weight = sum(float(weights.get(key, 0.0)) for key in component_breakdown)
    # 가중 패널티를 모두 합산하여 성능 점수를 계산합니다.
    performance_component = sum(component_breakdown.values())
    score_cap = 100.0
    final_score = clamp(safe_base + performance_component, 0.0, score_cap)

    # 세부 내역을 details에 삽입하여 HUD와 리플레이에서 활용합니다.
    details.update({
        "component_scores": component_breakdown,
        "safe_base": safe_base,
        "performance_component": performance_component,
        "score": final_score,
        "expected_orientation": expected_orientation,
        "weight_total": total_weight,
        "score_cap": score_cap,
    })
    return final_score, details

def _slugify(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    slug = slug.strip("_")
    return slug or "session"

def save_replay_log(frames: list[dict], meta: dict) -> str | None:
    """Persist the IPC communication log as a replay file."""
    if not frames:
        return None
    try:
        os.makedirs(REPLAY_DIR, exist_ok=True)
    except Exception as exc:
        print(f"[replay] failed to create directory: {exc}")
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    map_label = _slugify(meta.get("map_key", meta.get("map_name", "session")))
    filename = f"{timestamp}_{map_label}.json"
    path = os.path.join(REPLAY_DIR, filename)
    payload = {
        "meta": meta,
        "frames": frames,
    }
    try:
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        print(f"[replay] saved: {path}")
        mirror_replay_to_student_dir(Path(path), payload)
        return path
    except Exception as exc:
        print(f"[replay] failed to write: {exc}")
        return None


def mirror_replay_to_student_dir(src_path: Path, payload: dict) -> None:
    """Copy the replay into the student-algorithm workspace when available."""

    student_root = Path(__file__).resolve().parent.parent / "self-parking-user-algorithms"
    if not student_root.exists():
        print(f"[replay] student workspace missing: {student_root}")
        return

    try:
        target_dir = student_root / "student_replays"
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[replay] failed to create student directory: {exc}")
        return

    target_path = target_dir / src_path.name
    try:
        with open(target_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        # 학생 쪽 경로는 조용히 동기화하되, 필요 시 디버그용 로그만 남긴다.
        print(f"[replay] copied for student: {target_path}")
    except Exception as exc:
        print(f"[replay] failed to copy for student: {exc}")

def load_replay_file(path: str) -> tuple[dict, list[dict]]:
    """Load a replay JSON file and return metadata plus frames."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Replay file not found: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Replay file corrupted: {exc}") from exc
    meta = data.get("meta", {})
    frames = data.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("Replay file has no frame data.")
    return meta, frames

def build_map_payload(M: MapAssets) -> dict:
    """Serialize map data so it can be delivered to the student client."""
    # Convert numpy arrays to lists/scalars so JSON encoders can handle them.
    return {
        "extent": [float(x) for x in M.extent],
        "cellSize": float(M.cellSize),
        "slots": np.array(M.slots).astype(float).tolist(),
        "occupied_idx": [int(bool(v)) for v in M.occupied_idx],
        "walls_rects": np.array(M.walls_rects).astype(float).tolist(),
        "lines": np.array(M.lines).astype(float).tolist(),
        "grid": {
            "stationary": M.Cs.astype(float).tolist(),
            "parked": M.Cp.astype(float).tolist(),
        },
    }

def ensure_map_loaded(map_cfg: dict, cache: dict, seed: int | None = None) -> dict:
    """Load a map configuration, apply any variants, and cache the result."""
    map_key = map_cfg.get("key", map_cfg["filename"])
    cache_key = (map_key, seed)
    if cache_key not in cache:
        # 기본 맵을 불러옵니다. (같은 파일을 반복 로드하는 것을 방지합니다.)
        base = get_base_map(map_cfg["filename"])
        variant = map_cfg.get("variant", "")
        if variant:
            # variant 문자열에 따라 슬롯/점유 정보를 수정합니다.
            assets = apply_map_variant(base, variant, seed)
        else:
            assets = copy.deepcopy(base)
        # 차량 크기에 맞게 슬롯 경계값을 리사이즈합니다.
        resize_slots_to_vehicle(assets)
        # 진입로를 열어 초기 진입이 막히지 않도록 조정합니다.
        open_top_parking_lane(assets)
        assets.line_rects = compute_line_rects(assets)
        seed_val = seed if seed is not None else 0
        target_rng = random.Random(seed_val ^ 0x5F3759DF)
        free_idx = np.where(~assets.occupied_idx)[0]
        # 비어 있는 슬롯 중 하나를 타깃으로 선정합니다. (없으면 None으로 둡니다.)
        target_idx = int(target_rng.choice(free_idx.tolist())) if free_idx.size > 0 else None
        # IPC 송신용 payload를 미리 구성하여 캐시에 함께 보관합니다.
        payload = build_map_payload(assets)
        payload["expected_orientation"] = map_cfg.get("expected_orientation")
        slots_total = int(len(assets.slots))
        occupied = int(np.count_nonzero(assets.occupied_idx))
        thumbnail = generate_map_thumbnail(assets, highlight_idx=target_idx)
        cache[cache_key] = {
            "assets": assets,
            "payload": payload,
            "target_idx": target_idx,
            "thumbnail": thumbnail,
            "meta": {
                "slots_total": slots_total,
                "occupied_total": occupied,
            },
            "seed": seed,
        }
    return cache[cache_key]

def detect_stationary_collision(car_poly, M: MapAssets, threshold=0.5):
    """Perform high-resolution collision checks between the stationary grid and vehicle polygon."""
    H, W = M.Cs.shape
    car_x = [p[0] for p in car_poly]
    car_y = [p[1] for p in car_poly]
    min_x = clamp(min(car_x), M.extent[0], M.extent[1])
    max_x = clamp(max(car_x), M.extent[0], M.extent[1])
    min_y = clamp(min(car_y), M.extent[2], M.extent[3])
    max_y = clamp(max(car_y), M.extent[2], M.extent[3])

    col_min = int(math.floor((min_x - M.extent[0]) / M.cellSize))
    col_max = int(math.floor((max_x - M.extent[0]) / M.cellSize))
    row_min = int(math.floor((M.extent[3] - max_y) / M.cellSize))
    row_max = int(math.floor((M.extent[3] - min_y) / M.cellSize))

    col_min = clamp(col_min, 0, W - 1)
    col_max = clamp(col_max, 0, W - 1)
    row_min = clamp(row_min, 0, H - 1)
    row_max = clamp(row_max, 0, H - 1)

    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            if M.Cs[row, col] <= threshold:
                continue
            rect = grid_cell_rect(row, col, M.extent, M.cellSize, H)
            if poly_intersects_rect(car_poly, rect):
                center = ((rect[0] + rect[1]) * 0.5, (rect[2] + rect[3]) * 0.5)
                return True, center
    return False, None

def draw_collision_markers(surface, markers, world, sw, sh):
    """Visualize collision impact points for debugging."""
    for marker in markers:
        mx, my = marker.get("pos", (None, None))
        if mx is None or my is None:
            continue
        reason = marker.get("reason", "unknown")
        color = MARKER_COLORS.get(reason, MARKER_COLORS["unknown"])
        sx, sy = world_to_screen(mx, my, world, sw, sh)
        pygame.draw.circle(surface, color, (sx, sy), 8, 2)
        pygame.draw.line(surface, color, (sx - 6, sy - 6), (sx + 6, sy + 6), 2)
        pygame.draw.line(surface, color, (sx - 6, sy + 6), (sx + 6, sy - 6), 2)

def draw_pause_overlay(
    screen,
    fonts,
    sw,
    sh,
    stats_lines,
    instructions,
):
    """Render the pause overlay together with the right-hand HUD panels."""
    dim = pygame.Surface((sw, sh), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 140))
    screen.blit(dim, (0, 0))

    panel_w = 320
    panel = pygame.Surface((panel_w, sh), pygame.SRCALPHA)
    panel.fill((252, 252, 255, 235))
    screen.blit(panel, (sw - panel_w, 0))

    title_img = fonts["title"].render("PAUSED", True, (40, 60, 140))
    tx = sw - panel_w + 24
    ty = 36
    screen.blit(title_img, (tx, ty))
    ty += title_img.get_height() + 12

    for line in stats_lines:
        text_img = fonts["regular"].render(line, True, (55, 65, 90))
        screen.blit(text_img, (tx, ty))
        ty += 24

    ty += 24
    for line in instructions:
        text_img = fonts["large"].render(line, True, (35, 80, 140))
        screen.blit(text_img, (tx, ty))
        ty += 34

def compute_map_card_rects(sw: int, sh: int, count: int, margin: int = 48, gap: int = 32):
    if count <= 0:
        return []
    usable_width = max(sw - 2 * margin, 200)
    card_width = usable_width // count - (gap * (count - 1) // max(count, 1))
    card_width = max(220, min(320, card_width))
    total_width = card_width * count + gap * (count - 1)
    start_x = max(margin, (sw - total_width) // 2)
    card_height = 320
    y = sh // 2 - card_height // 2 + 20
    rects = []
    for i in range(count):
        x = start_x + i * (card_width + gap)
        rects.append(pygame.Rect(x, y, card_width, card_height))
    return rects

def wrap_text_lines(font, text: str, max_width: int):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if not candidate:
            continue
        if font.size(candidate)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines
def render_map_selection(
    surface: pygame.Surface,
    sw: int,
    sh: int,
    fonts,
    maps_cfg,
    map_cache,
    map_states,
    selected_idx: int,
    hovered_idx: int | None,
    error_text: str | None = None,
    connection_text: str | None = None,
    ipc_connected: bool = False,
    focus_mode: str = "maps",
    agent_entries: list[dict] | None = None,
    agent_idx: int = 0,
    agent_active_idx: int | None = None,
    agent_message: str | None = None,
    replay_entries: list[dict] | None = None,
    replay_idx: int = 0,
    replay_window_start: int = 0,
) -> tuple[list[pygame.Rect], list[tuple[int, pygame.Rect]], list[tuple[int, pygame.Rect]], int]:
    agent_entries = agent_entries or []
    replay_entries = replay_entries or []

    surface.fill((243, 246, 252))
    main_rect, sidebar_rect = compute_layout(sw, sh)
    pygame.draw.rect(surface, (248, 250, 253), main_rect)

    title_surface = fonts["title"].render("Select Driving Map", True, (28, 34, 78))
    surface.blit(title_surface, (main_rect.x + (main_rect.width - title_surface.get_width()) // 2, 52))

    subtitle_surface = fonts["regular"].render("Choose a course before connecting the agent.", True, (82, 92, 115))
    surface.blit(subtitle_surface, (main_rect.x + (main_rect.width - subtitle_surface.get_width()) // 2, 112))

    if connection_text:
        status_font = fonts["large"]
        max_status_width = main_rect.width - MAP_CARD_PADDING * 2
        status_label = ellipsize_text(status_font, connection_text, max_status_width)
        status_color = (50, 140, 80) if ipc_connected else (200, 60, 60)
        status_surface = status_font.render(status_label, True, status_color)
        surface.blit(status_surface, (main_rect.x + (main_rect.width - status_surface.get_width()) // 2, 160))

    columns = max(1, min(determine_map_columns(sw), len(maps_cfg) or 1))
    available_width = max(220, main_rect.width - MAP_CARD_PADDING * 2)
    card_width = (available_width - (columns - 1) * MAP_CARD_GAP) / max(columns, 1)
    card_width = max(230, min(card_width, 380))
    thumbnail_height = card_width / MAP_CARD_ASPECT
    info_height = 180
    card_height = int(thumbnail_height + info_height)

    rows = (len(maps_cfg) + columns - 1) // columns if columns else 0
    total_height = rows * card_height + max(0, rows - 1) * MAP_CARD_GAP
    grid_top = max(main_rect.y + 200, main_rect.y + (main_rect.height - total_height) // 2)

    total_row_width = columns * card_width + max(0, columns - 1) * MAP_CARD_GAP
    grid_left = int(main_rect.x + (main_rect.width - total_row_width) // 2)

    card_rects: list[pygame.Rect] = []
    for idx, cfg in enumerate(maps_cfg):
        row = idx // columns if columns else 0
        col = idx % columns if columns else 0
        x = int(grid_left + col * (card_width + MAP_CARD_GAP))
        y = int(grid_top + row * (card_height + MAP_CARD_GAP))
        card_rects.append(pygame.Rect(x, y, int(card_width), int(card_height)))

    for idx, cfg in enumerate(maps_cfg):
        rect = card_rects[idx]
        selected = idx == selected_idx
        hovered = hovered_idx == idx and not selected
        base_color = (226, 232, 244)
        if selected:
            base_color = (210, 220, 245)
        elif hovered:
            base_color = (234, 238, 248)
        pygame.draw.rect(surface, base_color, rect, border_radius=18)
        border_color = ACCENT_COLOR if selected else (166, 180, 205)
        border_width = 2 if selected else 1
        pygame.draw.rect(surface, border_color, rect, border_width, border_radius=18)

        if selected and focus_mode == "maps":
            overlay = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            overlay.fill(ACCENT_COLOR + (HIGHLIGHT_ALPHA,))
            surface.blit(overlay, rect.topleft)

        name_rect = pygame.Rect(rect.x + 20, rect.y + 16, rect.width - 40, 44)
        draw_wrapped_text(surface, cfg.get("name", "Map"), fonts["large"], (24, 32, 60), name_rect, max_lines=1, ellipsis=True)

        state = map_states.setdefault(cfg["key"], {"seed": random.randrange(1 << 30)})
        seed = state["seed"]
        preview = None
        preview_error = None
        bundle = None
        try:
            bundle = ensure_map_loaded(cfg, map_cache, seed=seed)
            preview = bundle.get("thumbnail")
        except FileNotFoundError:
            preview_error = "Map file not found."
        except Exception as exc:
            preview_error = f"Preview failed: {exc}"

        canvas_rect = pygame.Rect(rect.x + 16, rect.y + 70, rect.width - 32, int(thumbnail_height))
        pygame.draw.rect(surface, (232, 236, 248), canvas_rect, border_radius=12)
        pygame.draw.rect(surface, (152, 170, 210), canvas_rect, 1, border_radius=12)
        if preview:
            scaled = pygame.transform.smoothscale(preview, (canvas_rect.width - 12, canvas_rect.height - 12))
            surface.blit(scaled, (canvas_rect.x + 6, canvas_rect.y + 6))
        elif preview_error:
            draw_wrapped_text(
                surface,
                preview_error,
                fonts["regular"],
                (180, 70, 70),
                canvas_rect.inflate(-12, -12),
                align="center",
                line_height=1.25,
                max_lines=3,
            )

        summary_rect = pygame.Rect(rect.x + 20, canvas_rect.bottom + 16, rect.width - 40, 72)
        draw_wrapped_text(
            surface,
            cfg.get("summary", ""),
            fonts["regular"],
            (58, 68, 92),
            summary_rect,
            align="center",
            line_height=1.28,
            max_lines=3,
        )

        orientation_value = cfg.get("expected_orientation")
        orientation_label = describe_orientation(orientation_value)
        requirement_prefix = "Required orientation"
        requirement_text = f"{requirement_prefix}: {orientation_label}"
        requirement_font = fonts["large"] if orientation_value else fonts["regular"]
        requirement_color = (44, 64, 128) if orientation_value else (96, 104, 132)
        max_req_width = rect.width - 40
        requirement_display = ellipsize_text(requirement_font, requirement_text, max_req_width)
        requirement_surface = requirement_font.render(requirement_display, True, requirement_color)
        orientation_margin_top = 96 if orientation_value else 88
        requirement_y = summary_rect.bottom + 8
        requirement_y = max(requirement_y, rect.y + orientation_margin_top)
        requirement_y = min(requirement_y, rect.bottom - 84)
        requirement_pos = (
            rect.x + (rect.width - requirement_surface.get_width()) // 2,
            requirement_y,
        )
        surface.blit(requirement_surface, requirement_pos)

        stats_top = requirement_pos[1] + requirement_surface.get_height() + 8
        stats_top = min(stats_top, rect.bottom - 58)

        if bundle:
            meta = bundle.get("meta", {})
            stats_text = f"Slots {meta.get('slots_total', '?')} · Occupied {meta.get('occupied_total', '?')}"
            stats_rect = pygame.Rect(rect.x + 20, stats_top, rect.width - 40, 24)
            draw_wrapped_text(surface, stats_text, fonts["regular"], (100, 110, 140), stats_rect, align="center", max_lines=1)

        footer_surface = fonts["regular"].render(f"{idx + 1}/{len(maps_cfg)}", True, (120, 130, 150))
        surface.blit(footer_surface, (rect.right - footer_surface.get_width() - 14, rect.bottom - footer_surface.get_height() - 10))

        if selected and focus_mode == "maps":
            tag_surface = fonts["regular"].render("Selected", True, (255, 255, 255))
            tag_rect = pygame.Rect(rect.x + 18, rect.y + rect.height - 42, tag_surface.get_width() + 24, 26)
            pygame.draw.rect(surface, ACCENT_COLOR, tag_rect, border_radius=12)
            surface.blit(tag_surface, (tag_rect.x + 12, tag_rect.y + 4))

    if not maps_cfg:
        placeholder = fonts["regular"].render("No maps registered.", True, (110, 120, 140))
        surface.blit(placeholder, (main_rect.x + (main_rect.width - placeholder.get_width()) // 2, main_rect.y + main_rect.height // 2))

    instructions = "← → or A/D: change map  ·  Enter/Space: select  ·  TAB: focus  ·  ESC/Q: quit  ·  R: refresh replays"
    instr_surface = fonts["regular"].render(instructions, True, (72, 82, 95))
    surface.blit(instr_surface, (main_rect.x + (main_rect.width - instr_surface.get_width()) // 2, main_rect.bottom - 48))

    if error_text:
        err_surface = fonts["regular"].render(error_text, True, (192, 60, 60))
        surface.blit(err_surface, (main_rect.x + (main_rect.width - err_surface.get_width()) // 2, main_rect.bottom - 80))

    agent_rects: list[tuple[int, pygame.Rect]] = []
    replay_rects: list[tuple[int, pygame.Rect]] = []
    if sidebar_rect.width > 0:
        agent_rects, replay_rects, replay_window_start = render_selection_side_panel(
            surface,
            fonts,
            sidebar_rect,
            focus_mode,
            agent_entries,
            agent_idx,
            agent_active_idx,
            agent_message,
            replay_entries,
            replay_idx,
            replay_window_start,
        )

    return card_rects, agent_rects, replay_rects, replay_window_start


def render_selection_side_panel(
    surface: pygame.Surface,
    fonts,
    sidebar_rect: pygame.Rect,
    focus_mode: str,
    agent_entries: list[dict],
    agent_idx: int,
    agent_active_idx: int | None,
    agent_message: str | None,
    replay_entries: list[dict],
    replay_idx: int,
    replay_window_start: int,
) -> tuple[list[tuple[int, pygame.Rect]], list[tuple[int, pygame.Rect]], int]:
    agent_rects: list[tuple[int, pygame.Rect]] = []
    replay_rects: list[tuple[int, pygame.Rect]] = []

    if sidebar_rect.width <= 0:
        return agent_rects, replay_rects, replay_window_start

    font = fonts["regular"]
    font_large = fonts["large"]

    pygame.draw.rect(surface, (234, 237, 246), sidebar_rect)
    pygame.draw.line(surface, (186, 194, 208), (sidebar_rect.x, sidebar_rect.y), (sidebar_rect.x, sidebar_rect.bottom), 2)

    section_x = sidebar_rect.x + 20
    section_width = sidebar_rect.width - 40
    y = sidebar_rect.y + 36

    def draw_header(text: str, top: int) -> int:
        header = font_large.render(text, True, (45, 55, 95))
        surface.blit(header, (section_x, top))
        divider_y = top + header.get_height() + 6
        pygame.draw.line(surface, (192, 200, 214), (section_x, divider_y), (section_x + section_width, divider_y), 1)
        return divider_y + 12

    y = draw_header("Student Algorithms", y)
    agent_item_height = 66
    if not agent_entries:
        empty_rect = pygame.Rect(section_x, y, section_width, 70)
        draw_wrapped_text(surface, "No algorithms detected.", font, (145, 155, 175), empty_rect, max_lines=3)
        y = empty_rect.bottom + 16
    else:
        for idx, entry in enumerate(agent_entries):
            item_rect = pygame.Rect(section_x, y, section_width, agent_item_height)
            running = agent_active_idx == idx and entry.get("running")
            available = entry.get("available", True) and entry.get("exists", True)
            focused = focus_mode == "agents" and idx == agent_idx

            base_color = (239, 242, 251)
            if running:
                base_color = (206, 243, 214)
            elif not available:
                base_color = (245, 223, 223)
            if focused:
                base_color = (210, 222, 247)
            pygame.draw.rect(surface, base_color, item_rect, border_radius=14)
            border_color = ACCENT_COLOR if focused else (178, 188, 204)
            pygame.draw.rect(surface, border_color, item_rect, 2, border_radius=14)

            status_color = (44, 136, 72) if running else ((185, 68, 68) if not available else (110, 120, 150))
            status_text = "Running" if running else ("Missing file" if not available else "Idle")

            pygame.draw.circle(surface, status_color, (item_rect.x + 18, item_rect.y + item_rect.height // 2), 8)
            label_rect = pygame.Rect(item_rect.x + 36, item_rect.y + 10, item_rect.width - 44, 24)
            draw_wrapped_text(surface, entry.get("label", "Student Algorithm"), font, (35, 45, 82), label_rect, max_lines=1, ellipsis=True)
            status_rect = pygame.Rect(item_rect.x + 36, item_rect.y + 36, item_rect.width - 44, 22)
            draw_wrapped_text(surface, status_text, font, status_color, status_rect, max_lines=1, ellipsis=True)

            agent_rects.append((idx, item_rect))
            y += agent_item_height + 12

    if agent_message:
        message_rect = pygame.Rect(section_x, y, section_width, 70)
        draw_wrapped_text(surface, agent_message, font, (125, 70, 70), message_rect, max_lines=3)
        y = message_rect.bottom + 18

    y = max(y, sidebar_rect.y + int(sidebar_rect.height * 0.48))
    y = draw_header("Replays", y)

    if not replay_entries:
        empty_rect = pygame.Rect(section_x, y, section_width, 80)
        draw_wrapped_text(surface, "No saved replays.", font, (145, 155, 175), empty_rect, max_lines=3)
        y = empty_rect.bottom + 18
    else:
        max_visible = 7 if sidebar_rect.height > 880 else 5
        total = len(replay_entries)
        replay_window_start = max(0, min(replay_window_start, max(0, total - max_visible)))
        if replay_idx < replay_window_start:
            replay_window_start = replay_idx
        elif replay_idx >= replay_window_start + max_visible:
            replay_window_start = replay_idx - max_visible + 1
        visible_entries = replay_entries[replay_window_start: replay_window_start + max_visible]
        replay_item_height = 74

        for offset, entry in enumerate(visible_entries):
            idx = replay_window_start + offset
            item_rect = pygame.Rect(section_x, y, section_width, replay_item_height)
            focused = focus_mode == "replays" and idx == replay_idx
            base_color = (240, 243, 250)
            if focused:
                base_color = (212, 224, 244)
            pygame.draw.rect(surface, base_color, item_rect, border_radius=14)
            border_color = ACCENT_COLOR if focused else (178, 188, 204)
            pygame.draw.rect(surface, border_color, item_rect, 2, border_radius=14)

            meta = entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {}
            timestamp = meta.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    timestamp = dt.strftime("%m-%d %H:%M") if sidebar_rect.width < 420 else dt.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    pass
            map_name = meta.get("map_name") or meta.get("map_key") or "Map"
            score = meta.get("score")
            result = meta.get("result", "")
            status_color = (190, 60, 60) if result == "collision" else (102, 112, 142)

            headline_rect = pygame.Rect(item_rect.x + 16, item_rect.y + 10, item_rect.width - 32, 26)
            headline = f"{timestamp} · {map_name}" if timestamp else map_name
            draw_wrapped_text(surface, headline, font, (42, 52, 88), headline_rect, max_lines=1, ellipsis=True)

            details_rect = pygame.Rect(item_rect.x + 16, item_rect.y + 38, item_rect.width - 32, 24)
            detail_parts = []
            if isinstance(score, (int, float)):
                detail_parts.append(f"{score:.1f} pts")
            if result:
                detail_parts.append(result)
            draw_wrapped_text(surface, " · ".join(detail_parts), font, status_color, details_rect, max_lines=1, ellipsis=True)

            if entry.get("error"):
                error_rect = pygame.Rect(item_rect.x + 16, item_rect.bottom - 24, item_rect.width - 32, 18)
                draw_wrapped_text(surface, "Failed to load", font, (190, 70, 70), error_rect, max_lines=1, ellipsis=True)

            replay_rects.append((idx, item_rect))
            y += replay_item_height + 12

    return agent_rects, replay_rects, replay_window_start


def map_selection_loop(
    screen_state: dict,
    clock,
    fonts,
    maps_cfg,
    map_cache,
    map_states,
    ipc=None,
    host="127.0.0.1",
    port=55556,
    initial_idx=0,
    error_text=None,
    agent_manager: AgentManager | None = None,
    replay_loader=None,
):
    selected_idx = min(initial_idx, max(len(maps_cfg) - 1, 0))
    focus_mode = "maps"
    agent_idx = 0
    replay_idx = 0
    hovered_idx = None
    agent_message: str | None = None
    agent_message_until = 0
    last_replay_click = 0
    replay_window_start = 0
    prev_agent_active: int | None = None

    replay_entries = replay_loader() if replay_loader else []
    agent_rects: list[tuple[int, pygame.Rect]] = []
    replay_rects: list[tuple[int, pygame.Rect]] = []
    card_rects: list[pygame.Rect] = []

    running = True
    while running:
        screen = screen_state["surface"]
        sw, sh = screen_state["size"]

        if ipc:
            try:
                ipc.poll_accept()
            except Exception:
                pass

        now = pygame.time.get_ticks()
        if agent_message and now > agent_message_until:
            agent_message = None

        agent_entries_info: list[dict] = []
        if agent_manager:
            agent_manager.poll()
            for idx, entry in enumerate(agent_manager.entries):
                path_obj = entry["path"]
                exists = True if path_obj is None else path_obj.exists()
                entry_type = entry.get("type", "process")
                if entry_type == "local_manual":
                    running_proc = agent_manager.is_manual_active() and agent_manager.active_idx == idx
                    available = True
                else:
                    running_proc = agent_manager.is_running() and agent_manager.active_idx == idx
                    available = exists
                agent_entries_info.append({
                    "label": entry.get("label", "Student Algorithm"),
                    "exists": exists,
                    "available": available,
                    "running": running_proc,
                    "type": entry_type,
                })
        current_agent_active = agent_manager.active_idx if agent_manager and agent_manager.is_running() else None
        if prev_agent_active is not None and current_agent_active is None and agent_message is None:
            agent_message = "Student algorithm terminated."
            agent_message_until = pygame.time.get_ticks() + 2500
        prev_agent_active = current_agent_active

        if agent_entries_info:
            agent_idx = max(0, min(agent_idx, len(agent_entries_info) - 1))
        else:
            agent_idx = 0

        if replay_entries:
            replay_idx = max(0, min(replay_idx, len(replay_entries) - 1))
        else:
            replay_idx = 0

        focus_sections = ["maps"]
        if agent_entries_info:
            focus_sections.append("agents")
        focus_sections.append("replays")
        if focus_mode not in focus_sections:
            focus_mode = focus_sections[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.VIDEORESIZE:
                new_size = enforce_min_window_size(event.w, event.h)
                screen_state["surface"] = pygame.display.set_mode(new_size, pygame.RESIZABLE)
                screen_state["size"] = new_size
                update_viewport(*new_size)
                continue
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:
                    current_idx = focus_sections.index(focus_mode)
                    if event.mod & pygame.KMOD_SHIFT:
                        current_idx = (current_idx - 1) % len(focus_sections)
                    else:
                        current_idx = (current_idx + 1) % len(focus_sections)
                    focus_mode = focus_sections[current_idx]
                    continue

                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return None

                if focus_mode == "maps":
                    if event.key in (pygame.K_RIGHT, pygame.K_d) and maps_cfg:
                        selected_idx = (selected_idx + 1) % len(maps_cfg)
                    elif event.key in (pygame.K_LEFT, pygame.K_a) and maps_cfg:
                        selected_idx = (selected_idx - 1) % len(maps_cfg)
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE) and maps_cfg:
                        return ("map", selected_idx)
                    continue

                if focus_mode == "agents":
                    if event.key in (pygame.K_UP, pygame.K_w) and agent_entries_info:
                        agent_idx = (agent_idx - 1) % len(agent_entries_info)
                        continue
                    if event.key in (pygame.K_DOWN, pygame.K_s) and agent_entries_info:
                        agent_idx = (agent_idx + 1) % len(agent_entries_info)
                        continue
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE) and agent_manager and agent_entries_info:
                        if agent_manager.is_running() and agent_manager.active_idx == agent_idx:
                            agent_manager.stop()
                            agent_message = "Stopped the student algorithm."
                            agent_message_until = pygame.time.get_ticks() + 2500
                        else:
                            success, err = agent_manager.start(agent_idx, host, port)
                            if success:
                                agent_message = "Launched the student algorithm."
                            else:
                                agent_message = f"Launch failed: {err}"
                            agent_message_until = pygame.time.get_ticks() + 3500
                        continue
                    if event.key in (pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_x):
                        if agent_manager and agent_manager.is_running():
                            agent_manager.stop()
                            agent_message = "Stopped the student algorithm."
                        else:
                            agent_message = "No algorithm is currently running."
                        agent_message_until = pygame.time.get_ticks() + 2500
                        continue
                    continue

                if focus_mode == "replays":
                    if event.key in (pygame.K_UP, pygame.K_w) and replay_entries:
                        replay_idx = (replay_idx - 1) % len(replay_entries)
                        continue
                    if event.key in (pygame.K_DOWN, pygame.K_s) and replay_entries:
                        replay_idx = (replay_idx + 1) % len(replay_entries)
                        continue
                    if event.key in (pygame.K_RETURN, pygame.K_SPACE) and replay_entries:
                        return ("replay", str(replay_entries[replay_idx]["path"]))
                    if event.key in (pygame.K_r, pygame.K_F5) and replay_loader:
                        replay_entries = replay_loader() or []
                        replay_idx = min(replay_idx, len(replay_entries) - 1) if replay_entries else 0
                        replay_window_start = 0
                    continue

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handled = False
                for idx, rect in agent_rects:
                    if rect.collidepoint(event.pos):
                        focus_mode = "agents"
                        agent_idx = idx
                        handled = True
                        break
                if handled:
                    continue

                for idx, rect in replay_rects:
                    if rect.collidepoint(event.pos):
                        focus_mode = "replays"
                        replay_idx = idx
                        ticks = pygame.time.get_ticks()
                        if ticks - last_replay_click < 350 and replay_entries:
                            return ("replay", str(replay_entries[replay_idx]["path"]))
                        last_replay_click = ticks
                        handled = True
                        break
                if handled:
                    continue

                for idx, rect in enumerate(card_rects):
                    if rect.collidepoint(event.pos):
                        focus_mode = "maps"
                        if idx == selected_idx:
                            return ("map", selected_idx)
                        selected_idx = idx
                        break

            if event.type == pygame.MOUSEWHEEL and replay_entries:
                if event.y > 0:
                    replay_idx = max(0, replay_idx - 1)
                elif event.y < 0:
                    replay_idx = min(len(replay_entries) - 1, replay_idx + 1)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 4 and replay_entries:
                replay_idx = max(0, replay_idx - 1)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 5 and replay_entries:
                replay_idx = min(len(replay_entries) - 1, replay_idx + 1)

        if ipc and ipc.is_connected:
            peer = f"{ipc.peer[0]}:{ipc.peer[1]}" if ipc.peer else f"{host}:{port}"
            connection_text = f"Student algorithm connected — {peer}"
            ipc_connected = True
        else:
            connection_text = f"Student algorithm waiting — {host}:{port}" if ipc else None
            ipc_connected = False

        card_rects, agent_rects, replay_rects, replay_window_start = render_map_selection(
            screen,
            sw,
            sh,
            fonts,
            maps_cfg,
            map_cache,
            map_states,
            selected_idx,
            hovered_idx,
            error_text=error_text,
            connection_text=connection_text,
            ipc_connected=ipc_connected,
            focus_mode=focus_mode,
            agent_entries=agent_entries_info,
            agent_idx=agent_idx,
            agent_active_idx=current_agent_active,
            agent_message=agent_message,
            replay_entries=replay_entries,
            replay_idx=replay_idx,
            replay_window_start=replay_window_start,
        )

        mouse_pos = pygame.mouse.get_pos()
        hovered_idx = None
        for idx, rect in enumerate(card_rects):
            if rect.collidepoint(mouse_pos):
                hovered_idx = idx
                break

        pygame.display.flip()
        clock.tick(60)

def reseed_map_states(map_states: dict, maps_cfg):
    for cfg in maps_cfg:
        map_states[cfg["key"]] = {"seed": random.randrange(1 << 30)}

# ----------------- IPC Controller -----------------
class IPCController:
    """Single-client TCP server that hosts JSONL IPC for the student algorithm."""

    def __init__(self, host="127.0.0.1", port=55556, timeout=0.15):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listen_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.listen_sock.bind((self.host, self.port))
        except OSError as exc:
            self.listen_sock.close()
            raise RuntimeError(
                f"[IPC] failed to bind {self.host}:{self.port} - port already in use"
            ) from exc

        self.listen_sock.listen(1)
        self.listen_sock.setblocking(False)
        print(f"[IPC] listening on {self.host}:{self.port}")

        self.sock = None
        self.buf = b""
        self.peer = None

    def poll_accept(self) -> bool:
        """Accept client connections without blocking."""
        if self.sock is not None:
            return True

        try:
            conn, addr = self.listen_sock.accept()
        except BlockingIOError:
            return False
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return False
            raise

        conn.settimeout(self.timeout)
        self.sock = conn
        self.buf = b""
        self.peer = addr
        print(f"[IPC] student connected from {addr[0]}:{addr[1]}")
        return True

    def close_connection(self):
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None
        self.buf = b""
        self.peer = None

    def shutdown(self):
        self.close_connection()
        if self.listen_sock:
            try:
                self.listen_sock.close()
            except Exception:
                pass
            self.listen_sock = None

    @property
    def is_connected(self) -> bool:
        return self.sock is not None

    def send_obs(self, obs: dict):
        """Send one observation tick as a JSON Lines record."""
        if not self.sock:
            raise ConnectionError("IPC send attempted without active connection")
        line = (json.dumps(obs, ensure_ascii=False) + "\n").encode()
        self.sock.sendall(line)

    def send_map(self, map_payload: dict):
        """Immediately send the map payload as `{"map": ...}` after connecting."""
        if not self.sock:
            raise ConnectionError("IPC send attempted without active connection")
        message = {"map": map_payload}
        # 모든 메시지는 JSON 문자열 뒤에 개행을 붙여 JSON Lines 형식으로 전송합니다.
        line = (json.dumps(message, ensure_ascii=False) + "\n").encode()
        self.sock.sendall(line)

    def recv_cmd(self) -> dict:
        """Receive one command line from the student algorithm and normalize expected keys."""
        if not self.sock:
            raise ConnectionError("IPC recv attempted without active connection")
        while b"\n" not in self.buf:
            # 줄바꿈이 수신될 때까지 버퍼에 누적합니다.
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("IPC closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        msg = json.loads(line.decode())
        # 기본값을 채워 넣어 planner가 누락된 키로 인해 실패하지 않도록 합니다.
        return {
            "steer": float(msg.get("steer", 0.0)),
            "accel": float(msg.get("accel", 0.0)),
            "brake": float(msg.get("brake", 0.0)),
            "gear": str(msg.get("gear", "D")),
        }

def run_replay_mode(screen, clock, sw, sh, fonts, replay_path: str, meta: dict, frames: list[dict]):
    """Play back a previously saved replay file."""
    map_key = meta.get("map_key")
    map_cfg = next((cfg for cfg in AVAILABLE_MAPS if cfg.get("key") == map_key), AVAILABLE_MAPS[0])
    map_seed = meta.get("map_seed")
    bundle = ensure_map_loaded(map_cfg, {}, seed=map_seed)
    M = bundle["assets"]
    world = M.extent

    target_idx = meta.get("target_idx")
    target_slot = None
    if target_idx is not None and isinstance(target_idx, int) and 0 <= target_idx < len(M.slots):
        target_slot = tuple(M.slots[target_idx].tolist())
    if target_slot is None:
        first_obs = frames[0].get("obs", {}) if frames else {}
        slot_payload = first_obs.get("target_slot")
        if slot_payload and len(slot_payload) == 4:
            target_slot = tuple(float(v) for v in slot_payload)
    if target_slot is None and len(M.slots) > 0:
        fallback_idx = bundle.get("target_idx", 0)
        fallback_idx = fallback_idx if 0 <= fallback_idx < len(M.slots) else 0
        target_slot = tuple(M.slots[fallback_idx].tolist())

    P = Params()
    state = State()
    delta = 0.0
    traj = []

    total_frames = len(frames)
    frame_idx = 0
    playing = True
    replay_done = False
    current_cmd = {}
    current_time = 0.0
    reason = meta.get("result", "replay")
    score_value = meta.get("score", 0.0)
    stats_meta = meta.get("stats", {}) if isinstance(meta.get("stats"), dict) else {}
    stage_label = meta.get("stage_label") or get_stage_profile(map_cfg)[1].get("label", "")

    def reset_playback():
        nonlocal frame_idx, playing, replay_done, delta, current_cmd, current_time, traj
        frame_idx = 0
        playing = True
        replay_done = False
        delta = 0.0
        current_cmd = {}
        current_time = 0.0
        traj.clear()

    def apply_frame(idx: int):
        nonlocal delta, current_cmd, current_time
        if idx < 0 or idx >= total_frames:
            return
        frame = frames[idx]
        obs = frame.get("obs", {})
        obs_state = obs.get("state", {})
        state.x = float(obs_state.get("x", state.x))
        state.y = float(obs_state.get("y", state.y))
        state.yaw = float(obs_state.get("yaw", state.yaw))
        state.v = float(obs_state.get("v", state.v))
        current_time = float(obs.get("t", current_time))
        traj.append((state.x, state.y))
        if len(traj) > 2000:
            traj.pop(0)
        current_cmd = frame.get("cmd") or {}
        delta = float(current_cmd.get("steer", delta))

    reset_playback()

    reason_map = {
        "success": "Success",
        "collision": "Collision",
        "timeout": "Timeout",
        "quit": "Quit",
        "replay": "Replay",
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    playing = not playing
                elif event.key == pygame.K_r:
                    reset_playback()

        if playing and frame_idx < total_frames:
            apply_frame(frame_idx)
            frame_idx += 1
            if frame_idx >= total_frames:
                playing = False
                replay_done = True

        clock.tick(int(1.0 / P.dt))

        screen.fill((245, 245, 245))
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(vp_ox, vp_oy, vp_w, vp_h), 3)
        draw_rect(screen, M.border, (0, 0, 0), world, sw, sh, width=4)
        draw_walls_rects(screen, world, sw, sh, M.walls_rects)
        for x1, y1, x2, y2 in M.lines:
            p1 = world_to_screen(x1, y1, world, sw, sh)
            p2 = world_to_screen(x2, y2, world, sw, sh)
            pygame.draw.line(screen, (0, 0, 0), p1, p2, 3)
        for i, rect in enumerate(M.slots):
            if M.occupied_idx[i]:
                draw_rect(screen, tuple(rect), (0, 0, 0), world, sw, sh, width=0)
        for rect in M.slots:
            draw_rect(screen, tuple(rect), (0, 0, 0), world, sw, sh, width=2)
        if target_slot:
            draw_rect(screen, target_slot, (180, 255, 180), world, sw, sh, width=0)
            draw_rect(screen, target_slot, (50, 140, 50), world, sw, sh, width=2)
        if len(traj) >= 2:
            pts = [world_to_screen(x, y, world, sw, sh) for (x, y) in traj]
            pygame.draw.lines(screen, (50, 50, 200), False, pts, 2)
        draw_car(screen, state, delta, P, world, sw, sh)

        hud1 = f"t={current_time:6.2f}s  frame={min(frame_idx, total_frames)}/{total_frames}"
        steer_deg = math.degrees(delta)
        hud2 = f"steer={steer_deg:6.2f}°  accel={float(current_cmd.get('accel', 0.0))*100:5.1f}%  brake={float(current_cmd.get('brake', 0.0))*100:5.1f}%  gear={current_cmd.get('gear', 'D')}"
        play_state = "Play" if playing else "Paused"
        hud3 = f"Map: {map_cfg['name']}  |  Mode: Replay ({play_state})"
        hint = "SPACE: play/pause  ·  R: restart  ·  ESC/Q: quit"
        screen.blit(fonts["regular"].render(hud1, True, (0, 0, 0)), (12, 8))
        screen.blit(fonts["regular"].render(hud2, True, (0, 0, 0)), (12, 26))
        hud3_img = fonts["regular"].render(hud3, True, (0, 0, 0))
        screen.blit(hud3_img, (12, 44))
        hint_img = fonts["regular"].render(hint, True, (70, 70, 70))
        screen.blit(hint_img, (sw - hint_img.get_width() - 16, 44))

        if replay_done:
            elapsed = stats_meta.get("elapsed", current_time)
            distance = stats_meta.get("distance", 0.0)
            gear_sw = stats_meta.get("gear_switches", 0)
            avg_speed = stats_meta.get("avg_speed", 0.0)
            reason_text = reason_map.get(reason, reason)
            info_lines = [
                f"Score: {score_value:.1f}",
                f"Result: {reason_text}",
                f"Stage: {stage_label}",
                f"Elapsed: {elapsed:.1f} s",
                f"Distance: {distance:.1f} m",
                f"Gear switches: {gear_sw}",
                f"Avg speed: {avg_speed:.2f} m/s",
                f"Frames: {total_frames}",
                f"Replay file: {os.path.basename(replay_path)}",
            ]
            draw_overlay(screen, "REPLAY", info_lines, fonts["regular"], sw, sh)

        pygame.display.flip()

# ----------------- 메인 -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ipc","wasd","replay"], default="ipc")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=55556)
    ap.add_argument("--replay", help="Play back a saved replay file (JSON).")
    return ap.parse_args()

def main():
    args = parse_args()
    replay_meta = None
    replay_frames = None
    if args.replay:
        args.mode = "replay"
        try:
            replay_meta, replay_frames = load_replay_file(args.replay)
        except Exception as exc:
            print(f"[replay] failed to load: {exc}")
            return

    # 1) pygame
    pygame.init()
    pygame.event.set_allowed([
        pygame.QUIT,
        pygame.KEYDOWN,
        pygame.KEYUP,
        pygame.MOUSEBUTTONDOWN,
        pygame.MOUSEBUTTONUP,
        pygame.MOUSEWHEEL,
        pygame.VIDEORESIZE,
    ])
    screen = pygame.display.set_mode(BASE_WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("Self-Parking — Layered Map from MATLAB (IPC)")
    screen_state = {"surface": screen, "size": BASE_WINDOW_SIZE}
    sw, sh = screen_state["size"]
    clock = pygame.time.Clock()
    font = load_font_with_fallback(18)
    font_large = load_font_with_fallback(28, bold=True)
    font_title = load_font_with_fallback(54, bold=True)
    font_requirement = load_font_with_fallback(36, bold=True)
    font_small = load_font_with_fallback(16, bold=True)
    fonts = {
        "regular": font,
        "large": font_large,
        "title": font_title,
        "requirement": font_requirement,
        "small": font_small,
    }

    # Viewport margins so HUD stays outside of the lot rendering area.
    update_viewport(*screen_state["size"])

    if args.mode == "replay":
        run_replay_mode(screen, clock, sw, sh, fonts, args.replay, replay_meta, replay_frames)
        pygame.quit()
        return

    # ----- 재시작 가능한 라운드 루프 -----
    ipc = IPCController(args.host, args.port) if args.mode == "ipc" else None
    base_dir = Path(__file__).resolve().parent
    agent_manager = AgentManager(discover_agent_entries(base_dir)) if args.mode == "ipc" else None
    replay_loader = load_replay_catalog
    map_cache = {}
    map_states = {}
    map_seeds_dirty = True
    selected_map_idx = 0
    selection_error = None
    ui_mode = "map_select"
    M = None
    map_payload = None
    world = None
    H = W = 0
    current_target_idx = None
    active_map_cfg = None
    banner_show_until = 0
    student_connected = ipc.is_connected if ipc else False
    map_sent = False

    running=True
    while running:
        if ui_mode == "map_select":
            manual_override_active = agent_manager.is_manual_active() if agent_manager else False
            if map_seeds_dirty:
                reseed_map_states(map_states, AVAILABLE_MAPS)
                map_cache.clear()
                map_seeds_dirty = False
                current_target_idx = None
                active_map_cfg = None
            choice = map_selection_loop(
                screen_state,
                clock,
                fonts,
                AVAILABLE_MAPS,
                map_cache,
                map_states,
                ipc=ipc if (args.mode == "ipc" and not manual_override_active) else None,
                host=args.host,
                port=args.port,
                initial_idx=selected_map_idx,
                error_text=selection_error,
                agent_manager=agent_manager,
                replay_loader=replay_loader,
            )
            if choice is None:
                break
            choice_kind, payload = choice
            if choice_kind == "replay":
                try:
                    replay_meta, replay_frames = load_replay_file(payload)
                except Exception as exc:
                    selection_error = f"Failed to load replay: {exc}"
                    continue
            
                screen = screen_state["surface"]
                sw, sh = screen_state["size"]
                update_viewport(sw, sh)
                run_replay_mode(screen, clock, sw, sh, fonts, payload, replay_meta, replay_frames)
                selection_error = None
                map_seeds_dirty = True
                continue
            if choice_kind != "map":
                selection_error = "Unknown selection item."
                continue

            selected_map_idx = payload
            selection_error = None
            selected_cfg = AVAILABLE_MAPS[selected_map_idx]
            selected_seed = map_states[selected_cfg["key"]]["seed"]
            try:
                bundle = ensure_map_loaded(selected_cfg, map_cache, seed=selected_seed)
            except FileNotFoundError:
                missing = selected_cfg["filename"]
                selection_error = f"Map file not found: {missing}"
                continue
            except Exception as e:
                selection_error = f"Failed to load map: {e}"
                continue

            M = bundle["assets"]
            map_payload = bundle["payload"]
            current_target_idx = bundle.get("target_idx")
            map_states[selected_cfg["key"]]["target_idx"] = current_target_idx
            active_map_cfg = selected_cfg
            screen = screen_state["surface"]
            sw, sh = screen_state["size"]
            update_viewport(sw, sh)
            xmin, xmax, ymin, ymax = M.extent
            world = (xmin, xmax, ymin, ymax)
            H, W = M.C.shape
            ui_mode = "sim_round"
            banner_show_until = 0
            continue

        if M is None:
            selection_error = "No map selected."
            ui_mode = "map_select"
            map_seeds_dirty = True
            continue

        manual_override_active = agent_manager.is_manual_active() if agent_manager else False
        control_mode = "local_wasd" if (args.mode == "wasd" or manual_override_active) else "ipc"
        if control_mode == "ipc":
            student_connected = ipc.is_connected if ipc else False
            map_sent = False
        else:
            student_connected = False
            map_sent = True

        # 2) 시뮬레이션 라운드 초기화
        # 고정 시작 위치/자세 (좌하단 슬롯 근처, 위쪽을 바라보도록 설정)
        start_x = xmin + 4.0
        start_y = ymin + 6.0
        start_yaw = math.radians(90.0)

        P = Params()
        free_slot_indices = [i for i, occ in enumerate(M.occupied_idx) if not occ]
        if not free_slot_indices:
            raise RuntimeError("No available parking slots in the map.")

        def slot_center(idx: int) -> tuple[float, float]:
            rect = M.slots[idx]
            cx = (rect[0] + rect[1]) * 0.5
            cy = (rect[2] + rect[3]) * 0.5
            return cx, cy

        def is_far_enough(idx: int) -> bool:
            cx, cy = slot_center(idx)
            return math.hypot(cx - start_x, cy - start_y) >= MIN_TARGET_DISTANCE

        left_limit = xmin + LEFT_EXCLUDE_MARGIN

        def is_left_excluded(idx: int) -> bool:
            cx, _ = slot_center(idx)
            return cx <= left_limit

        candidate_indices = [i for i in free_slot_indices if not is_left_excluded(i) and is_far_enough(i)]
        allowed_indices = [i for i in free_slot_indices if not is_left_excluded(i)]

        if (
            current_target_idx in free_slot_indices
            and not is_left_excluded(current_target_idx)
            and is_far_enough(current_target_idx)
        ):
            target_idx = current_target_idx
        else:
            if candidate_indices:
                pick_pool = candidate_indices
            elif allowed_indices:
                pick_pool = allowed_indices
            else:
                pick_pool = free_slot_indices
            target_idx = random.choice(pick_pool)
            current_target_idx = target_idx
            if active_map_cfg:
                map_states[active_map_cfg["key"]]["target_idx"] = current_target_idx
        target_slot = tuple(M.slots[target_idx].tolist())

        state = State(start_x, start_y, start_yaw, 0.0)
        u = InputCmd()
        delta = 0.0
        traj = []
        t=0.0; why="running"
        move_dist = 0.0
        prev_x, prev_y = state.x, state.y
        collision_count = 0
        collision_markers = []
        abort_to_menu = False
        paused = False
        round_stats = RoundStats()
        round_stats.prev_gear = u.gear
        round_stats.min_abs_steer = abs(delta)
        round_stats.prev_delta_sign = 0
        replay_frames = []
        waiting_for_ipc = False

        # ---- 메인 주행 루프 ----
        while why == "running":
            waiting_overlay_active = (control_mode == "ipc") and bool(ipc and not ipc.is_connected)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    why="quit"; break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
                        if waiting_overlay_active:
                            abort_to_menu = True
                            why = "to_menu"
                            break
                        paused = not paused
                        continue
                    if paused:
                        if event.key == pygame.K_m:
                            abort_to_menu = True
                            why = "to_menu"
                            break
                        if event.key in (pygame.K_ESCAPE, pygame.K_q):
                            why = "quit"
                            running = False
                            break
                if why!="running": break
            if why != "running":
                break

            waiting_for_ipc = False

            if control_mode == "local_wasd":
                # 키 입력 (디버그/예비용)
                keys = pygame.key.get_pressed()
                if not paused:
                    step_cmd = P.cmdRate * P.dt
                    u.accel = clamp(u.accel + (step_cmd if keys[pygame.K_w] else -step_cmd), 0, 1)
                    u.brake = clamp(u.brake + (step_cmd if keys[pygame.K_s] else -step_cmd), 0, 1)
                    steer_step = math.radians(120) * P.dt
                    if keys[pygame.K_a]:
                        u.delta_tgt = clamp(u.delta_tgt + steer_step, -P.maxSteer, P.maxSteer)
                    if keys[pygame.K_d]:
                        u.delta_tgt = clamp(u.delta_tgt - steer_step, -P.maxSteer, P.maxSteer)
                    if not keys[pygame.K_a] and not keys[pygame.K_d]:
                        u.delta_tgt = move_toward(u.delta_tgt, 0.0, P.selfCenterRate * P.dt)
                    if keys[pygame.K_r]:
                        u.gear = 'R' if u.gear == 'D' else 'D'
                    if keys[pygame.K_SPACE]:
                        state.v = 0.0
                        u.accel = 0.0
                        u.brake = 1.0
                        u.delta_tgt = 0.0
                    obs = build_obs_payload(t, state, target_slot, P)
                    cmd_snapshot = {
                        "steer": float(u.delta_tgt),
                        "accel": float(u.accel),
                        "brake": float(u.brake),
                        "gear": u.gear,
                    }
                    replay_frames.append({"t": t, "obs": obs, "cmd": cmd_snapshot})
                else:
                    u.delta_tgt = move_toward(u.delta_tgt, 0.0, P.selfCenterRate * P.dt)
                    u.accel = 0.0
                    u.brake = 0.0
            else:
                # IPC로 명령 수신 (시뮬레이터가 서버 역할 수행)
                if not ipc.is_connected:
                    if student_connected:
                        student_connected = False
                        banner_show_until = 0
                        map_sent = False
                    if ipc.poll_accept():
                        student_connected = True
                        banner_show_until = pygame.time.get_ticks() + 1500
                        map_sent = False
                        try:
                            ipc.send_map(map_payload)
                            map_sent = True
                            print("[IPC] sent static map payload to student")
                        except Exception as e:
                            print(f"[IPC] failed to send map payload: {e}")
                            ipc.close_connection()
                            student_connected = False
                            banner_show_until = 0

                if ipc.is_connected:
                    if not map_sent:
                        try:
                            ipc.send_map(map_payload)
                            map_sent = True
                            print("[IPC] re-sent static map payload to student")
                        except Exception as e:
                            print(f"[IPC] map payload resend failed: {e}")
                            ipc.close_connection()
                            student_connected = False
                            banner_show_until = 0
                            waiting_for_ipc = True
                            continue

                    if not paused:
                        obs = build_obs_payload(t, state, target_slot, P)
                        replay_entry = {"t": t, "obs": obs, "cmd": None}
                        try:
                            ipc.send_obs(obs)
                            cmd = ipc.recv_cmd()
                            u.delta_tgt = clamp(float(cmd.get("steer", 0.0)), -P.maxSteer, P.maxSteer)
                            u.accel = clamp(float(cmd.get("accel", 0.0)), 0.0, 1.0)
                            u.brake = clamp(float(cmd.get("brake", 0.0)), 0.0, 1.0)
                            u.gear = 'R' if str(cmd.get("gear", "D")).upper().startswith('R') else 'D'
                            replay_entry["cmd"] = cmd
                            replay_frames.append(replay_entry)
                        except Exception as e:
                            replay_frames.append(replay_entry)
                            print(f"[IPC] comm fail: {e} -> connection will be reset")
                            ipc.close_connection()
                            student_connected = False
                            banner_show_until = 0
                            map_sent = False
                            waiting_for_ipc = True
                else:
                    waiting_for_ipc = True

                if waiting_for_ipc:
                    # 연결 대기 중에는 차량을 정지 상태로 유지한다.
                    u.delta_tgt = move_toward(u.delta_tgt, 0.0, P.selfCenterRate * P.dt)
                    u.accel = 0.0
                    u.brake = 0.0
                    u.gear = 'D'

            # 조향 레이트 제한
            delta = move_toward(delta, u.delta_tgt, P.steerRate * P.dt)

            if not waiting_for_ipc and not paused:
                if u.gear != round_stats.prev_gear:
                    round_stats.gear_switches += 1
                    round_stats.prev_gear = u.gear
                round_stats.min_abs_steer = min(round_stats.min_abs_steer, abs(delta))
                steer_sign = 0
                if abs(delta) >= STEER_FLIP_DEADZONE:
                    steer_sign = 1 if delta > 0 else -1
                if steer_sign != 0:
                    prev_sign = round_stats.prev_delta_sign
                    if prev_sign != 0 and steer_sign != prev_sign:
                        round_stats.direction_flips += 1
                    round_stats.prev_delta_sign = steer_sign

                # 가감속 합성(+ coast)
                sgn = +1.0 if u.gear == 'D' else -1.0
                a_accel = sgn * P.maxAccel * u.accel
                vsgn = math.copysign(1.0, state.v) if abs(state.v) > 1e-6 else 0.0
                a_brake = -vsgn * P.maxBrake * u.brake
                a_coast = 0.0
                if u.accel < 1e-3 and u.brake < 1e-3 and abs(state.v) > 1e-3:
                    a_coast = -vsgn * P.coastDecel
                a = a_accel + a_brake + a_coast

                # 동역학
                state = step_kinematic(state, delta, a, P)

                # 궤적 기록 (최근 N개만 유지)
                traj.append((state.x, state.y))
                if len(traj) > 2000:
                    traj.pop(0)

                # 이동거리
                move_dist += math.hypot(state.x - prev_x, state.y - prev_y)
                prev_x, prev_y = state.x, state.y
                round_stats.avg_speed_accum += abs(state.v)
                round_stats.speed_samples += 1

                # 차량 폴리곤(충돌용)
                car_poly = car_polygon(state, P)
                # 차량이 슬롯을 얼마나 채웠는지(IoU)와 진입 방향(전면/후면)을 계산한다.
                slot_iou = compute_slot_iou(car_poly, target_slot)
                slot_orientation = determine_parking_orientation(state, target_slot)
                if slot_iou > round_stats.final_iou:
                    round_stats.final_iou = slot_iou
                    round_stats.final_orientation = slot_orientation
                    round_stats.final_speed = abs(state.v)

                # -------- 충돌 판정 --------
                collided = False
                collision_reason = None
                collision_marker = None

                # (A) 경계 충돌은 필요 시에만 사용.
                if ENABLE_BOUNDARY_COLLISIONS:
                    for (vx, vy) in car_poly:
                        if not (xmin <= vx <= xmax and ymin <= vy <= ymax):
                            collided = True
                            collision_reason = "boundary"
                            collision_marker = (clamp(vx, xmin, xmax), clamp(vy, ymin, ymax))
                            break

                # (B) 점유된 슬롯과 SAT 충돌
                if not collided:
                    for i, rect in enumerate(M.slots):
                        if not M.occupied_idx[i]:
                            continue
                        if poly_intersects_rect(car_poly, tuple(rect)):
                            collided = True
                            collision_reason = "occupied_slot"
                            collision_marker = ((rect[0] + rect[1]) * 0.5, (rect[2] + rect[3]) * 0.5)
                            break

                # (C) 주차 라인과 충돌
                if not collided:
                    line_rects = getattr(M, "line_rects", None)
                    if line_rects is None:
                        line_rects = compute_line_rects(M)
                        M.line_rects = line_rects
                    for rect in line_rects:
                        if poly_intersects_rect(car_poly, tuple(rect)):
                            collided = True
                            collision_reason = "line"
                            collision_marker = ((rect[0] + rect[1]) * 0.5, (rect[2] + rect[3]) * 0.5)
                            break

                # (D) 정지물 레이어와 고정밀 교차 검사
                if not collided and ENABLE_STATIONARY_COLLISIONS:
                    collided, hit_point = detect_stationary_collision(car_poly, M, threshold=M.FreeThr)
                    if collided:
                        collision_reason = "stationary"
                        collision_marker = hit_point

                # (E) 타깃 슬롯 내부는 허용
                if rect_contains_poly(target_slot, car_poly):
                    collided = False
                    collision_reason = None
                    collision_marker = None

                # 성공 판정(간단 버전): 슬롯 완전 포함 + 저속
                # IoU 50% 이상 + 속도 저속 + 정해진 방향(전/후면)으로 정렬된 경우에만 성공 처리
                reached = (
                    slot_iou >= PARKING_SUCCESS_IOU
                    and abs(state.v) <= 0.2
                    and slot_orientation != "unknown"
                )

                if collided:
                    collision_markers.append({
                        "pos": collision_marker if collision_marker else (state.x, state.y),
                        "reason": collision_reason or "unknown",
                    })
                    collision_count += 1
                    why = "collision"
                    break
                if reached:
                    round_stats.final_iou = slot_iou
                    round_stats.final_orientation = slot_orientation
                    round_stats.final_speed = abs(state.v)
                    why = "success"
                    break

        # ---- 렌더 ----
            if control_mode == "ipc" and waiting_for_ipc:
                screen.fill((255, 255, 255))
                title = font_title.render("PARKING - SIM", True, (20, 20, 20))
                title_pos = ((sw - title.get_width()) // 2, sh // 3 - title.get_height() // 2)
                screen.blit(title, title_pos)

                blink = (pygame.time.get_ticks() // 400) % 2 == 0
                status_color = (200, 40, 40) if blink else (255, 160, 160)
                status_text = "Waiting for student algorithm"
                status = font_large.render(status_text, True, status_color)
                status_pos = ((sw - status.get_width()) // 2, title_pos[1] + title.get_height() + 30)
                screen.blit(status, status_pos)

                info_text = f"Listening: {args.host}:{args.port}"
                info = font.render(info_text, True, (90, 90, 90))
                info_pos = ((sw - info.get_width()) // 2, status_pos[1] + status.get_height() + 18)
                screen.blit(info, info_pos)
                map_info = font.render(f"Selected map: {AVAILABLE_MAPS[selected_map_idx]['name']}", True, (70, 70, 70))
                map_pos = ((sw - map_info.get_width()) // 2, info_pos[1] + info.get_height() + 16)
                screen.blit(map_info, map_pos)
                hint = font.render("P: main screen", True, (80, 80, 80))
                screen.blit(hint, ((sw - hint.get_width()) // 2, map_pos[1] + map_info.get_height() + 16))
            else:
                screen.fill((245, 245, 245))

                orientation_value = active_map_cfg.get("expected_orientation") if active_map_cfg else None
                orientation_label = describe_orientation(orientation_value)
                orientation_short = ORIENTATION_LABELS_SHORT.get(orientation_value, orientation_label)
                requirement_color = (182, 52, 52) if orientation_value else (90, 100, 130)

                # Helpful axes (fade grey) and viewport border.
                x0s, y0s = world_to_screen(0, 0, world, sw, sh)
                pygame.draw.line(screen, (220, 220, 220), (0, y0s), (sw, y0s), 1)
                pygame.draw.line(screen, (220, 220, 220), (x0s, 0), (x0s, sh), 1)
                pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(vp_ox, vp_oy, vp_w, vp_h), 3)

                # 외곽 및 환경 요소
                draw_rect(screen, M.border, (0, 0, 0), world, sw, sh, width=4)
                draw_walls_rects(screen, world, sw, sh, M.walls_rects)

                for x1, y1, x2, y2 in M.lines:
                    p1 = world_to_screen(x1, y1, world, sw, sh)
                    p2 = world_to_screen(x2, y2, world, sw, sh)
                    pygame.draw.line(screen, (0, 0, 0), p1, p2, 3)

                for i, rect in enumerate(M.slots):
                    if M.occupied_idx[i]:
                        draw_rect(screen, tuple(rect), (0, 0, 0), world, sw, sh, width=0)

                for rect in M.slots:
                    draw_rect(screen, tuple(rect), (0, 0, 0), world, sw, sh, width=2)

                draw_rect(screen, target_slot, (180, 255, 180), world, sw, sh, width=0)
                draw_rect(screen, target_slot, (50, 140, 50), world, sw, sh, width=2)

                if orientation_value:
                    slot_cx = (target_slot[0] + target_slot[1]) * 0.5
                    slot_cy = (target_slot[2] + target_slot[3]) * 0.5
                    slot_screen = world_to_screen(slot_cx, slot_cy, world, sw, sh)
                    slot_font = fonts["small"]
                    slot_surface = slot_font.render(orientation_short, True, requirement_color)
                    slot_rect = slot_surface.get_rect()
                    slot_rect.midbottom = (slot_screen[0], slot_screen[1] - 6)
                    slot_bg_rect = slot_rect.inflate(12, 8)
                    slot_bg_surface = pygame.Surface(slot_bg_rect.size, pygame.SRCALPHA)
                    slot_bg_surface.fill((255, 255, 255, 220))
                    screen.blit(slot_bg_surface, slot_bg_rect.topleft)
                    screen.blit(slot_surface, slot_rect.topleft)

                if len(traj) >= 2:
                    pts = [world_to_screen(x, y, world, sw, sh) for (x, y) in traj]
                    pygame.draw.lines(screen, (50, 50, 200), False, pts, 2)

                draw_car(screen, state, delta, P, world, sw, sh)
                if collision_markers:
                    draw_collision_markers(screen, collision_markers, world, sw, sh)

                # HUD (세 줄 구성)
                hud1 = f"t={t:5.1f}s  gear={u.gear}  v={state.v:5.2f} m/s  steer={math.degrees(delta):6.1f} deg"
                hud2 = f"accel={u.accel*100:5.1f}%  brake={u.brake*100:5.1f}%  collisions={collision_count}"
                if control_mode == "ipc":
                    if ipc.is_connected:
                        peer = f"{ipc.peer[0]}:{ipc.peer[1]}" if ipc.peer else f"{args.host}:{args.port}"
                        hud3 = f"Map: {AVAILABLE_MAPS[selected_map_idx]['name']}  |  IPC connected: {peer}"
                    else:
                        hud3 = f"Map: {AVAILABLE_MAPS[selected_map_idx]['name']}  |  IPC waiting: {args.host}:{args.port}"
                else:
                    if args.mode == "wasd":
                        hud3 = f"Map: {AVAILABLE_MAPS[selected_map_idx]['name']}  |  Mode: WASD (W/S throttle, A/D steer, R gear, SPACE brake)"
                    else:
                        hud3 = f"Map: {AVAILABLE_MAPS[selected_map_idx]['name']}  |  Mode: built-in WASD (no agent)"
                hud_hint = "P: pause"

                screen.blit(font.render(hud1, True, (0, 0, 0)), (12, 8))
                screen.blit(font.render(hud2, True, (0, 0, 0)), (12, 26))
                hud3_img = font.render(hud3, True, (0, 0, 0))
                screen.blit(hud3_img, (12, 44))
                hint_img = font.render(hud_hint, True, (70, 70, 70))
                screen.blit(hint_img, (sw - hint_img.get_width() - 16, 44))

                requirement_text = f"Requirement: {orientation_label}"
                requirement_surface = fonts["requirement"].render(requirement_text, True, requirement_color)
                requirement_y = max(12, vp_oy - requirement_surface.get_height() - 12)
                requirement_pos = (
                    vp_ox + (vp_w - requirement_surface.get_width()) // 2,
                    requirement_y,
                )
                screen.blit(requirement_surface, requirement_pos)

                if paused:
                    stats_lines = [
                        f"Time: {t:0.1f} s",
                        f"Speed: {state.v:0.2f} m/s",
                        f"Steer: {math.degrees(delta):0.1f} deg",
                        f"Collisions: {collision_count}",
                        f"Map: {AVAILABLE_MAPS[selected_map_idx]['name']}",
                    ]
                    instruction_lines = [
                        "P: continue",
                        "M: select map",
                        "R: restart (after finish)",
                        "ESC/Q: quit",
                    ]
                    draw_pause_overlay(screen, fonts, sw, sh, stats_lines, instruction_lines)

                if control_mode == "ipc" and pygame.time.get_ticks() < banner_show_until:
                    banner = font_large.render("Student algorithm connected!", True, (30, 110, 40))
                    banner_pos = (vp_ox + (vp_w - banner.get_width()) // 2, vp_oy + 20)
                    screen.blit(banner, banner_pos)

            pygame.display.flip()
            clock.tick(int(1.0 / P.dt))
            if not waiting_for_ipc and not paused:
                t += P.dt
                if t >= P.timeout:
                    why = "timeout"
                    break
            round_stats.elapsed = t
            round_stats.distance = move_dist
            round_stats.final_speed = max(round_stats.final_speed, abs(state.v))

        if why == "to_menu" or abort_to_menu:
            ui_mode = "map_select"
            banner_show_until = 0
            map_sent = False
            map_seeds_dirty = True
            current_target_idx = None
            active_map_cfg = None
            continue

        round_stats.elapsed = t
        round_stats.distance = move_dist
        round_stats.final_speed = max(round_stats.final_speed, abs(state.v))
        stage_idx, stage_profile = get_stage_profile(active_map_cfg)
        score_value, score_details = compute_round_score(round_stats, stage_profile, why, M.extent)
        RECENT_RESULTS.append({
            "map": active_map_cfg["name"] if active_map_cfg else "Unknown",
            "stage": stage_idx,
            "stage_label": stage_profile.get("label", f"Stage {stage_idx}"),
            "score": score_value,
            "details": score_details,
            "reason": why,
        })
        map_seed = None
        if active_map_cfg:
            state_entry = map_states.get(active_map_cfg["key"], {})
            map_seed = state_entry.get("seed")

        map_key = active_map_cfg.get("key") if active_map_cfg else None
        map_name = active_map_cfg["name"] if active_map_cfg else None

        replay_mode_label = "manual_wasd" if control_mode == "local_wasd" else args.mode
        replay_meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "map_key": map_key,
            "map_name": map_name,
            "map_seed": map_seed,
            "target_idx": current_target_idx,
            "stage": stage_idx,
            "stage_label": stage_profile.get("label", f"Stage {stage_idx}"),
            "result": why,
            "score": score_value,
            "mode": replay_mode_label,
            "stats": {
                "elapsed": round_stats.elapsed,
                "distance": round_stats.distance,
                "gear_switches": round_stats.gear_switches,
                "avg_speed": score_details.get("avg_speed"),
                "parking_iou": round_stats.final_iou,
                "parking_orientation": round_stats.final_orientation,
                "parking_iou_threshold": PARKING_SUCCESS_IOU,
                "expected_orientation": stage_profile.get("expected_orientation"),
            },
            "frame_count": len(replay_frames),
        }
        replay_path = save_replay_log(replay_frames, replay_meta)

        # 종료 오버레이
        screen.fill((245, 245, 245))
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(vp_ox, vp_oy, vp_w, vp_h), 3)
        draw_rect(screen, M.border, (0, 0, 0), world, sw, sh, width=4)
        draw_walls_rects(screen, world, sw, sh, M.walls_rects)
        for x1, y1, x2, y2 in M.lines:
            p1 = world_to_screen(x1, y1, world, sw, sh)
            p2 = world_to_screen(x2, y2, world, sw, sh)
            pygame.draw.line(screen, (0, 0, 0), p1, p2, 3)
        for i, rect in enumerate(M.slots):
            if M.occupied_idx[i]:
                draw_rect(screen, tuple(rect), (0, 0, 0), world, sw, sh, width=0)
        for rect in M.slots:
            draw_rect(screen, tuple(rect), (0, 0, 0), world, sw, sh, width=2)
        draw_rect(screen, target_slot, (180, 255, 180), world, sw, sh, width=0)
        draw_rect(screen, target_slot, (50, 140, 50), world, sw, sh, width=2)
        draw_car(screen, state, delta, P, world, sw, sh)
        if collision_markers:
            draw_collision_markers(screen, collision_markers, world, sw, sh)

        title = "SUCCESS" if why == "success" else ("TIMEOUT" if why == "timeout" else "COLLISION" if why == "collision" else "QUIT")
        score_cap = score_details.get("score_cap", 100.0)
        info_lines = []
        info_lines.append(f"Final score {score_value:.1f} / {score_cap:.0f}")
        perf = score_details.get("performance_component", 0.0)
        safe_base = score_details.get("safe_base", 50.0)
        info_lines.append(f"Breakdown: safety {safe_base:.0f} + performance {perf:.1f}")

        iou_value = score_details.get("parking_iou")
        if isinstance(iou_value, (int, float)):
            info_lines.append(f"IoU {iou_value * 100:.1f}%")

        component_scores = score_details.get("component_scores", {}) or {}
        if component_scores:
            info_lines.append("")
            info_lines.append("Detailed metrics")
            component_labels = {
                "time": "Time",
                "distance": "Distance",
                "speed": "Average speed",
                "steer_flip": "Steering reversals",
                "parking_iou": "Slot IoU",
                "parking_orientation": "Orientation match",
                "parking_stop": "Stopped in slot",
            }
            for key, value in component_scores.items():
                label = component_labels.get(key, key)
                weight = stage_profile.get("weights", {}).get(key, 0.0)
                info_lines.append(f"- {label}: {value:.1f} / {weight:.0f}")

        if why == "success":
            xmin, xmax, ymin, ymax = M.extent
            diag = math.hypot(xmax - xmin, ymax - ymin)
            time_target = stage_profile.get("time_target", 0.0)
            distance_target = diag * stage_profile.get("distance_factor", 1.0)
            speed_target = stage_profile.get("speed_target", 0.0)
            steer_flip_target = stage_profile.get("steer_flip_target", 0)
            info_lines.append("")
            info_lines.append("Vs targets")
            if time_target > 0:
                info_lines.append(f"- Time {round_stats.elapsed:.1f}s / {time_target:.1f}s")
            info_lines.append(f"- Distance {round_stats.distance:.1f}m / {distance_target:.1f}m")
            info_lines.append(f"- Average speed {score_details.get('avg_speed', 0.0):.2f}m/s / {speed_target:.2f}m/s")
            info_lines.append(f"- Steering reversals {round_stats.direction_flips} / {steer_flip_target}")

        draw_overlay(screen, title, info_lines, font, sw, sh)
        pygame.display.flip()

        # 대기 루프
        waiting = True
        return_to_selection = False
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
                running = False
                waiting = False
            if keys[pygame.K_r]:
                waiting = False
            if keys[pygame.K_m]:
                ui_mode = "map_select"
                return_to_selection = True
                waiting = False
            if keys[pygame.K_p]:
                ui_mode = "map_select"
                return_to_selection = True
                waiting = False
            clock.tick(60)

        if return_to_selection:
            banner_show_until = 0
            map_sent = False
            map_seeds_dirty = True
            continue


    if ipc:
        ipc.shutdown()
    if agent_manager:
        agent_manager.stop()

    pygame.quit()

if __name__ == "__main__":
    main()
