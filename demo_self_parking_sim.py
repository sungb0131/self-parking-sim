# demo_self_parking_sim.py — Self-Parking simulator (MATLAB 레이어 + SAT 충돌) + IPC 제어
# 사용법:
#   python demo_self_parking_sim.py                  # 기본: IPC 127.0.0.1:55555
#   python demo_self_parking_sim.py --host 127.0.0.1 --port 55556
#   python demo_self_parking_sim.py --mode wasd      # (디버그용) 키보드 제어

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_HINT_IME_SHOW_UI", "0")  # macOS IME 노이즈 억제

import argparse, errno, json, math, random, socket, copy, subprocess, sys
from datetime import datetime
from collections import deque
from dataclasses import dataclass
from pathlib import Path

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
    """한국어를 포함한 텍스트가 깨지지 않도록 다단계로 폰트를 탐색."""
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
}

BASE_WINDOW_SIZE = (1600, 1000)
MIN_WINDOW_SIZE = (1280, 800)
SIDEBAR_WIDTH_RANGE = (360, 480)
MAP_CARD_GAP = 20
MAP_CARD_PADDING = 32
MAP_CARD_ASPECT = 16 / 9
ACCENT_COLOR = (65, 110, 220)
HIGHLIGHT_ALPHA = 80


def enforce_min_window_size(width: int, height: int) -> tuple[int, int]:
    min_w, min_h = MIN_WINDOW_SIZE
    return max(width, min_w), max(height, min_h)


def compute_layout(sw: int, sh: int) -> tuple[pygame.Rect, pygame.Rect]:
    min_sidebar, max_sidebar = SIDEBAR_WIDTH_RANGE
    sidebar_width = int(sw * 0.28)
    sidebar_width = max(min_sidebar, min(sidebar_width, max_sidebar))
    main_width = sw - sidebar_width
    if main_width < 720:
        sidebar_width = max(min_sidebar, sw - 720)
        main_width = sw - sidebar_width
    sidebar_width = max(min_sidebar, min(sidebar_width, sw - 480))
    main_rect = pygame.Rect(0, 0, max(main_width, 480), sh)
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
            "label": "student_algorithms.py (로컬)",
            "path": local_agent,
            "cwd": base_dir,
        })
    sibling = (base_dir.parent / "self-parking-user-algorithms" / "my_agent.py").resolve()
    if sibling.exists():
        entries.append({
            "label": "self-parking-user-algorithms/my_agent.py",
            "path": sibling,
            "cwd": sibling.parent,
        })
    return entries


class AgentManager:
    """학생 알고리즘 프로세스를 관리한다."""

    def __init__(self, entries: list[dict], python_executable: str | None = None):
        self.entries: list[dict] = []
        for entry in entries:
            self.entries.append({
                "label": entry.get("label", "학생 알고리즘"),
                "path": Path(entry.get("path")),
                "cwd": Path(entry.get("cwd", Path.cwd())),
            })
        self.python_executable = python_executable or sys.executable
        self.process: subprocess.Popen | None = None
        self.active_idx: int | None = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def poll(self) -> None:
        if self.process and self.process.poll() is not None:
            self.process = None
            self.active_idx = None

    def start(self, idx: int, host: str, port: int) -> tuple[bool, str | None]:
        if idx < 0 or idx >= len(self.entries):
            return False, "선택된 알고리즘이 없습니다."
        entry = self.entries[idx]
        if not entry["path"].exists():
            return False, "파일을 찾을 수 없습니다."

        self.stop()
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

    tip = "R: 재시작  ·  M: 맵 변경  ·  ESC/Q: 종료"
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
    elapsed: float = 0.0
    distance: float = 0.0
    gear_switches: int = 0
    avg_speed_accum: float = 0.0
    speed_samples: int = 0
    min_abs_steer: float = float("inf")
    prev_gear: str = "D"

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

# ----------------- 맵 로드 -----------------
class MapAssets:
    def __init__(self, C, Cs, Cm, Cp, cellSize, extent, slots, occupied_idx, border, lines, FreeThr, OccThr, walls_rects):
        self.C=C; self.Cs=Cs; self.Cm=Cm; self.Cp=Cp
        self.cellSize=cellSize; self.extent=extent
        self.slots=slots; self.occupied_idx=occupied_idx
        self.border=border; self.lines=lines
        self.FreeThr=FreeThr; self.OccThr=OccThr
        self.walls_rects=walls_rects

def load_parking_assets(mat_path="parking_assets_layers_75x50.mat") -> MapAssets:
    m = loadmat(mat_path)
    C  = m["C"].astype(np.float32)                      # combined cost map
    Cs = m["C_stationary"].astype(np.float32)           # stationary
    Cm = m["C_markings"].astype(np.float32)             # road markings
    Cp = m["C_parked"].astype(np.float32)               # parked cars
    cellSize = float(np.array(m["cellSize"]).squeeze())
    extent = tuple(np.array(m["extent"]).squeeze().tolist())  # (xmin,xmax,ymin,ymax)
    slots = np.array(m["slots"]).astype(float)                 # Nx4 [xmin xmax ymin ymax]
    occupied_idx = np.array(m["occupied_idx"]).astype(bool).ravel()
    border = tuple(np.array(m["border"]).squeeze().tolist())   # (xmin,xmax,ymin,ymax)
    lines  = np.array(m["lines"]).astype(float)                # Mx4 [x1 y1 x2 y2]
    FreeThr = float(np.array(m["FreeThreshold"]).squeeze())
    OccThr  = float(np.array(m["OccupiedThreshold"]).squeeze())
    walls_rects = np.array(m["walls_rects"]).astype(float)
    return MapAssets(C, Cs, Cm, Cp, cellSize, extent, slots, occupied_idx,
                     border, lines, FreeThr, OccThr, walls_rects)

def world_to_rc(x, y, extent, cellSize, H):
    xmin,xmax,ymin,ymax = extent
    col = int(np.floor((x - xmin) / cellSize))
    row_from_bottom = int(np.floor((y - ymin) / cellSize))
    row = H - 1 - row_from_bottom  # MATLAB행렬과 스크린 좌표 뒤집힘 보정
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
    if filename not in BASE_MAP_CACHE:
        BASE_MAP_CACHE[filename] = load_parking_assets(filename)
    return BASE_MAP_CACHE[filename]

def generate_map_thumbnail(M: MapAssets, size=(280, 160), highlight_idx: int | None = None):
    """맵 레이어를 소형 카드 형태로 미리 렌더링."""
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

    # 벽체/라인 표현
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

def apply_map_variant(base: MapAssets, variant: str, seed: int | None = None) -> MapAssets:
    """기본 맵 자산에 변형을 적용해 다른 난이도/환경을 구성."""
    if not variant:
        return base
    rng = random.Random(seed)
    M = copy.deepcopy(base)

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
    elif variant == "open_training":
        # 거의 모든 점유 슬롯을 비워 넓은 연습장을 만든다.
        M.occupied_idx = np.zeros_like(M.occupied_idx, dtype=bool)
    return M

AVAILABLE_MAPS = [
    {
        "key": "default_lot",
        "name": "기본 주차장 75x50",
        "filename": "parking_assets_layers_75x50.mat",
        "summary": "균일한 배치의 표준 테스트 환경",
        "variant": "",
        "stage": 1,
    },
    {
        "key": "dense_lot",
        "name": "혼잡 주차장",
        "filename": "parking_assets_layers_75x50.mat",
        "summary": "주변 슬롯이 가득 찬 협소 환경",
        "variant": "dense_center",
        "stage": 2,
    },
    {
        "key": "training_course",
        "name": "장애물 훈련장",
        "filename": "parking_assets_layers_75x50.mat",
        "summary": "중앙 통로에 임시 장애물을 배치한 연습 코스",
        "variant": "one_way_training",
        "stage": 3,
    },
]

STAGE_RULES = {
    1: {
        "label": "1단계",
        "base_score": 100.0,
        "time_target": 60.0,
        "distance_factor": 0.85,
        "turn_target": 2,
        "speed_target": 2.5,
        "weights": {
            "time": 35.0,
            "distance": 35.0,
            "turn": 20.0,
            "speed": 10.0,
        },
    },
    2: {
        "label": "2단계",
        "base_score": 100.0,
        "time_target": 75.0,
        "distance_factor": 0.95,
        "turn_target": 3,
        "speed_target": 3.0,
        "weights": {
            "time": 30.0,
            "distance": 30.0,
            "turn": 25.0,
            "speed": 15.0,
        },
    },
    3: {
        "label": "3단계",
        "base_score": 100.0,
        "time_target": 90.0,
        "distance_factor": 1.05,
        "turn_target": 4,
        "speed_target": 3.5,
        "weights": {
            "time": 28.0,
            "distance": 32.0,
            "turn": 25.0,
            "speed": 15.0,
        },
    },
}

def get_stage_profile(map_cfg: dict | None):
    if not map_cfg:
        return 1, STAGE_RULES[1]
    stage = int(map_cfg.get("stage", 1))
    profile = STAGE_RULES.get(stage, STAGE_RULES[1])
    return stage, profile

def compute_round_score(stats: RoundStats, stage_profile: dict, result_reason: str, world_extent) -> tuple[float, dict]:
    """라운드 종료 시 점수와 지표 상세를 계산한다."""
    details = {
        "elapsed": stats.elapsed,
        "distance": stats.distance,
        "gear_switches": stats.gear_switches,
        "avg_speed": (stats.avg_speed_accum / stats.speed_samples) if stats.speed_samples > 0 else 0.0,
        "reason": result_reason,
    }

    if result_reason != "success":
        return 0.0, details

    xmin, xmax, ymin, ymax = world_extent
    diag = math.hypot(xmax - xmin, ymax - ymin)

    time_target = stage_profile.get("time_target", 1.0)
    distance_factor = stage_profile.get("distance_factor", 1.0)
    distance_target = max(1e-6, diag * distance_factor)
    turn_target = max(1, int(stage_profile.get("turn_target", 1)))
    speed_target = max(1e-6, float(stage_profile.get("speed_target", 1.0)))
    weights = stage_profile.get("weights", {})

    time_ratio = stats.elapsed / time_target if time_target > 0 else 0.0
    dist_ratio = stats.distance / distance_target
    turn_ratio = stats.gear_switches / turn_target
    avg_speed = details["avg_speed"]
    speed_ratio = avg_speed / speed_target

    penalties = (
        weights.get("time", 0.0) * clamp(time_ratio, 0.0, 2.0) +
        weights.get("distance", 0.0) * clamp(dist_ratio, 0.0, 2.0) +
        weights.get("turn", 0.0) * clamp(turn_ratio, 0.0, 2.0) +
        weights.get("speed", 0.0) * clamp(speed_ratio, 0.0, 2.0)
    )

    base_score = stage_profile.get("base_score", 100.0)
    score = clamp(base_score - penalties, 0.0, base_score)

    details.update({
        "time_ratio": time_ratio,
        "dist_ratio": dist_ratio,
        "turn_ratio": turn_ratio,
        "speed_ratio": speed_ratio,
        "score": score,
    })
    return score, details

def _slugify(text: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    slug = slug.strip("_")
    return slug or "session"

def save_replay_log(frames: list[dict], meta: dict) -> str | None:
    """IPC 통신 로그를 리플레이 파일로 저장한다."""
    if not frames:
        return None
    try:
        os.makedirs(REPLAY_DIR, exist_ok=True)
    except Exception as exc:
        print(f"[replay] 디렉터리 생성 실패: {exc}")
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
        print(f"[replay] 저장 완료: {path}")
        return path
    except Exception as exc:
        print(f"[replay] 저장 실패: {exc}")
        return None

def load_replay_file(path: str) -> tuple[dict, list[dict]]:
    """리플레이 JSON 파일을 읽어 메타와 프레임 리스트를 반환한다."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"리플레이 파일을 찾을 수 없습니다: {path}")
    with open(path, "r", encoding="utf-8") as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError as exc:
            raise ValueError(f"리플레이 파일이 손상되었습니다: {exc}") from exc
    meta = data.get("meta", {})
    frames = data.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("리플레이 파일에 프레임 데이터가 없습니다.")
    return meta, frames

def build_map_payload(M: MapAssets) -> dict:
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
    """Load map variant (with optional random seed) and cache the result."""
    map_key = map_cfg.get("key", map_cfg["filename"])
    cache_key = (map_key, seed)
    if cache_key not in cache:
        base = get_base_map(map_cfg["filename"])
        variant = map_cfg.get("variant", "")
        if variant:
            assets = apply_map_variant(base, variant, seed)
        else:
            assets = copy.deepcopy(base)
        seed_val = seed if seed is not None else 0
        target_rng = random.Random(seed_val ^ 0x5F3759DF)
        free_idx = np.where(~assets.occupied_idx)[0]
        target_idx = int(target_rng.choice(free_idx.tolist())) if free_idx.size > 0 else None
        payload = build_map_payload(assets)
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
    """정지물 레이어와 차량 폴리곤 충돌을 고해상도로 판정한다. [UPDATE] 기존 둘레 샘플링 대신 셀 교차 검사로 정확도를 높였다."""
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
    """충돌 지점을 시각화한다. [UPDATE] 충돌 위치 확인을 위한 마커 추가."""
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
    """일시정지 오버레이와 우측 패널을 렌더링."""
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

    title_surface = fonts["title"].render("주행 맵 선택", True, (28, 34, 78))
    surface.blit(title_surface, (main_rect.x + (main_rect.width - title_surface.get_width()) // 2, 52))

    subtitle_surface = fonts["regular"].render("연결 전에 사용할 주행 환경을 선택하세요.", True, (82, 92, 115))
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
    info_height = 160
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
        draw_wrapped_text(surface, cfg.get("name", "맵"), fonts["large"], (24, 32, 60), name_rect, max_lines=1, ellipsis=True)

        state = map_states.setdefault(cfg["key"], {"seed": random.randrange(1 << 30)})
        seed = state["seed"]
        preview = None
        preview_error = None
        bundle = None
        try:
            bundle = ensure_map_loaded(cfg, map_cache, seed=seed)
            preview = bundle.get("thumbnail")
        except FileNotFoundError:
            preview_error = "맵 파일을 찾을 수 없습니다."
        except Exception as exc:
            preview_error = f"미리보기 실패: {exc}"

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

        if bundle:
            meta = bundle.get("meta", {})
            stats_text = f"전체 슬롯 {meta.get('slots_total', '?')}개 · 점유 {meta.get('occupied_total', '?')}개"
            stats_rect = pygame.Rect(rect.x + 20, rect.bottom - 56, rect.width - 40, 24)
            draw_wrapped_text(surface, stats_text, fonts["regular"], (100, 110, 140), stats_rect, align="center", max_lines=1)

        footer_surface = fonts["regular"].render(f"{idx + 1}/{len(maps_cfg)}", True, (120, 130, 150))
        surface.blit(footer_surface, (rect.right - footer_surface.get_width() - 14, rect.bottom - footer_surface.get_height() - 10))

        if selected and focus_mode == "maps":
            tag_surface = fonts["regular"].render("선택됨", True, (255, 255, 255))
            tag_rect = pygame.Rect(rect.x + 18, rect.y + rect.height - 42, tag_surface.get_width() + 24, 26)
            pygame.draw.rect(surface, ACCENT_COLOR, tag_rect, border_radius=12)
            surface.blit(tag_surface, (tag_rect.x + 12, tag_rect.y + 4))

    if not maps_cfg:
        placeholder = fonts["regular"].render("등록된 맵이 없습니다.", True, (110, 120, 140))
        surface.blit(placeholder, (main_rect.x + (main_rect.width - placeholder.get_width()) // 2, main_rect.y + main_rect.height // 2))

    instructions = "← → 또는 A/D: 맵 변경  ·  Enter/Space: 선택  ·  TAB: 영역 이동  ·  ESC/Q: 종료"
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

    y = draw_header("학생 알고리즘", y)
    agent_item_height = 66
    if not agent_entries:
        empty_rect = pygame.Rect(section_x, y, section_width, 70)
        draw_wrapped_text(surface, "감지된 알고리즘이 없습니다.", font, (145, 155, 175), empty_rect, max_lines=3)
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
            status_text = "실행 중" if running else ("파일 없음" if not available else "대기")

            pygame.draw.circle(surface, status_color, (item_rect.x + 18, item_rect.y + item_rect.height // 2), 8)
            label_rect = pygame.Rect(item_rect.x + 36, item_rect.y + 10, item_rect.width - 44, 24)
            draw_wrapped_text(surface, entry.get("label", "학생 알고리즘"), font, (35, 45, 82), label_rect, max_lines=1, ellipsis=True)
            status_rect = pygame.Rect(item_rect.x + 36, item_rect.y + 36, item_rect.width - 44, 22)
            draw_wrapped_text(surface, status_text, font, status_color, status_rect, max_lines=1, ellipsis=True)

            agent_rects.append((idx, item_rect))
            y += agent_item_height + 12

    if agent_message:
        message_rect = pygame.Rect(section_x, y, section_width, 70)
        draw_wrapped_text(surface, agent_message, font, (125, 70, 70), message_rect, max_lines=3)
        y = message_rect.bottom + 18

    y = max(y, sidebar_rect.y + int(sidebar_rect.height * 0.48))
    y = draw_header("리플레이", y)

    if not replay_entries:
        empty_rect = pygame.Rect(section_x, y, section_width, 80)
        draw_wrapped_text(surface, "저장된 리플레이가 없습니다.", font, (145, 155, 175), empty_rect, max_lines=3)
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
            map_name = meta.get("map_name") or meta.get("map_key") or "맵"
            score = meta.get("score")
            result = meta.get("result", "")
            status_color = (190, 60, 60) if result == "collision" else (102, 112, 142)

            headline_rect = pygame.Rect(item_rect.x + 16, item_rect.y + 10, item_rect.width - 32, 26)
            headline = f"{timestamp} · {map_name}" if timestamp else map_name
            draw_wrapped_text(surface, headline, font, (42, 52, 88), headline_rect, max_lines=1, ellipsis=True)

            details_rect = pygame.Rect(item_rect.x + 16, item_rect.y + 38, item_rect.width - 32, 24)
            detail_parts = []
            if isinstance(score, (int, float)):
                detail_parts.append(f"{score:.1f}점")
            if result:
                detail_parts.append(result)
            draw_wrapped_text(surface, " · ".join(detail_parts), font, status_color, details_rect, max_lines=1, ellipsis=True)

            if entry.get("error"):
                error_rect = pygame.Rect(item_rect.x + 16, item_rect.bottom - 24, item_rect.width - 32, 18)
                draw_wrapped_text(surface, "불러오기 실패", font, (190, 70, 70), error_rect, max_lines=1, ellipsis=True)

            replay_rects.append((idx, item_rect))
            y += replay_item_height + 12

    instructions = [
        "Enter: 실행 / 선택",
        "Backspace: 알고리즘 종료",
        "R: 리플레이 새로고침",
        "TAB: 영역 이동",
    ]
    instr_y = sidebar_rect.bottom - len(instructions) * 24 - 28
    for line in instructions:
        instr_surface = font.render(line, True, (96, 105, 135))
        surface.blit(instr_surface, (section_x, instr_y))
        instr_y += 24

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
                exists = entry["path"].exists()
                running_proc = agent_manager.is_running() and agent_manager.active_idx == idx
                agent_entries_info.append({
                    "label": entry.get("label", "학생 알고리즘"),
                    "exists": exists,
                    "available": exists,
                    "running": running_proc,
                })
        current_agent_active = agent_manager.active_idx if agent_manager and agent_manager.is_running() else None
        if prev_agent_active is not None and current_agent_active is None and agent_message is None:
            agent_message = "학생 알고리즘이 종료되었습니다."
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
                            agent_message = "학생 알고리즘을 종료했습니다."
                            agent_message_until = pygame.time.get_ticks() + 2500
                        else:
                            success, err = agent_manager.start(agent_idx, host, port)
                            if success:
                                agent_message = "학생 알고리즘을 실행했습니다."
                            else:
                                agent_message = f"실행 실패: {err}"
                            agent_message_until = pygame.time.get_ticks() + 3500
                        continue
                    if event.key in (pygame.K_BACKSPACE, pygame.K_DELETE, pygame.K_x):
                        if agent_manager and agent_manager.is_running():
                            agent_manager.stop()
                            agent_message = "학생 알고리즘을 종료했습니다."
                        else:
                            agent_message = "실행 중인 알고리즘이 없습니다."
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
            connection_text = f"학생 알고리즘 연결됨 — {peer}"
            ipc_connected = True
        else:
            connection_text = f"학생 알고리즘 연결 대기 중 — {host}:{port}" if ipc else None
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
    """학생 알고리즘과의 JSONL IPC를 호스팅하는 단일 클라이언트 TCP 서버."""

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
                f"[IPC] 포트 {self.host}:{self.port} 바인딩 실패 - 다른 프로세스가 사용 중입니다"
            ) from exc

        self.listen_sock.listen(1)
        self.listen_sock.setblocking(False)
        print(f"[IPC] listening on {self.host}:{self.port}")

        self.sock = None
        self.buf = b""
        self.peer = None

    def poll_accept(self) -> bool:
        """알고리즘 클라이언트 연결을 논블로킹으로 수락."""
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
        """관측값 한 틱을 JSON 한 줄로 직렬화하여 송신."""
        if not self.sock:
            raise ConnectionError("IPC send attempted without active connection")
        line = (json.dumps(obs, ensure_ascii=False) + "\n").encode()
        self.sock.sendall(line)

    def send_map(self, map_payload: dict):
        if not self.sock:
            raise ConnectionError("IPC send attempted without active connection")
        message = {"map": map_payload}
        line = (json.dumps(message, ensure_ascii=False) + "\n").encode()
        self.sock.sendall(line)

    def recv_cmd(self) -> dict:
        """학생 알고리즘이 보낸 명령 한 줄을 수신하고 표준화."""
        if not self.sock:
            raise ConnectionError("IPC recv attempted without active connection")
        while b"\n" not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("IPC closed")
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        msg = json.loads(line.decode())
        return {
            "steer": float(msg.get("steer", 0.0)),
            "accel": float(msg.get("accel", 0.0)),
            "brake": float(msg.get("brake", 0.0)),
            "gear": str(msg.get("gear", "D")),
        }

def run_replay_mode(screen, clock, sw, sh, fonts, replay_path: str, meta: dict, frames: list[dict]):
    """저장된 리플레이 파일을 재생한다."""
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
        "success": "성공",
        "collision": "충돌",
        "timeout": "시간초과",
        "quit": "종료",
        "replay": "리플레이",
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
        play_state = "재생" if playing else "일시정지"
        hud3 = f"맵: {map_cfg['name']}  |  모드: Replay ({play_state})"
        hint = "SPACE: 재생/일시정지  ·  R: 처음부터  ·  ESC/Q: 종료"
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
    ap.add_argument("--replay", help="저장된 리플레이 파일(JSON)을 재생합니다.")
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
            print(f"[replay] 로드 실패: {exc}")
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
    fonts = {"regular": font, "large": font_large, "title": font_title}

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
                ipc=ipc if args.mode == "ipc" else None,
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
                    selection_error = f"리플레이 로드 실패: {exc}"
                    continue
            
                screen = screen_state["surface"]
                sw, sh = screen_state["size"]
                update_viewport(sw, sh)
                run_replay_mode(screen, clock, sw, sh, fonts, payload, replay_meta, replay_frames)
                selection_error = None
                map_seeds_dirty = True
                continue
            if choice_kind != "map":
                selection_error = "알 수 없는 선택 항목입니다."
                continue

            selected_map_idx = payload
            selection_error = None
            selected_cfg = AVAILABLE_MAPS[selected_map_idx]
            selected_seed = map_states[selected_cfg["key"]]["seed"]
            try:
                bundle = ensure_map_loaded(selected_cfg, map_cache, seed=selected_seed)
            except FileNotFoundError:
                missing = selected_cfg["filename"]
                selection_error = f"맵 파일을 찾을 수 없습니다: {missing}"
                continue
            except Exception as e:
                selection_error = f"맵 로드 실패: {e}"
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
            student_connected = ipc.is_connected if ipc else False
            map_sent = False
            continue

        if M is None:
            selection_error = "맵이 선택되지 않았습니다."
            ui_mode = "map_select"
            map_seeds_dirty = True
            continue

        # 2) 시뮬레이션 라운드 초기화
        P = Params()
        free_slot_indices = [i for i, occ in enumerate(M.occupied_idx) if not occ]
        if not free_slot_indices:
            raise RuntimeError("맵에 사용 가능한 주차 슬롯이 없습니다.")
        if current_target_idx in free_slot_indices:
            target_idx = current_target_idx
        else:
            target_idx = random.choice(free_slot_indices)
            current_target_idx = target_idx
            if active_map_cfg:
                map_states[active_map_cfg["key"]]["target_idx"] = current_target_idx
        target_slot = tuple(M.slots[target_idx].tolist())

        # 고정 시작 위치/자세 (좌하단 슬롯 근처, 위쪽을 바라보도록 설정)
        start_x = xmin + 4.0
        start_y = ymin + 6.0
        start_yaw = math.radians(90.0)

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
        replay_frames = []

        # ---- 메인 주행 루프 ----
        while why == "running":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    why="quit"; break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:
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

            waiting_for_ipc = False

            if args.mode == "wasd":
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
                        obs = {
                            "t": t,
                            "state": {"x": state.x, "y": state.y, "yaw": state.yaw, "v": state.v},
                            "target_slot": list(target_slot),
                            "limits": {
                                "dt": P.dt,
                                "L": P.L,
                                "maxSteer": P.maxSteer,
                                "maxAccel": P.maxAccel,
                                "maxBrake": P.maxBrake,
                                "steerRate": P.steerRate,
                            },
                        }
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

                # -------- 충돌 판정 --------
                collided = False
                collision_reason = None
                collision_marker = None

                # (A) 경계 밖은 즉시 충돌로 유지. [UPDATE] 감지 지점 기록 추가.
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

                # (C) 정지물 레이어와 고정밀 교차 검사
                if not collided:
                    collided, hit_point = detect_stationary_collision(car_poly, M, threshold=M.FreeThr)
                    if collided:
                        collision_reason = "stationary"
                        collision_marker = hit_point

                # (D) 타깃 슬롯 내부는 허용
                if rect_contains_poly(target_slot, car_poly):
                    collided = False
                    collision_reason = None
                    collision_marker = None

                # 성공 판정(간단 버전): 슬롯 완전 포함 + 저속
                reached = rect_contains_poly(target_slot, car_poly) and abs(state.v) <= 0.2

                if collided:
                    collision_markers.append({
                        "pos": collision_marker if collision_marker else (state.x, state.y),
                        "reason": collision_reason or "unknown",
                    })
                    collision_count += 1
                    why = "collision"
                    break
                if reached:
                    why = "success"
                    break

        # ---- 렌더 ----
            if args.mode == "ipc" and waiting_for_ipc:
                screen.fill((255, 255, 255))
                title = font_title.render("PARKING - SIM", True, (20, 20, 20))
                title_pos = ((sw - title.get_width()) // 2, sh // 3 - title.get_height() // 2)
                screen.blit(title, title_pos)

                blink = (pygame.time.get_ticks() // 400) % 2 == 0
                status_color = (200, 40, 40) if blink else (255, 160, 160)
                status_text = "학생 알고리즘 연결 대기 중"
                status = font_large.render(status_text, True, status_color)
                status_pos = ((sw - status.get_width()) // 2, title_pos[1] + title.get_height() + 30)
                screen.blit(status, status_pos)

                info_text = f"수신 대기: {args.host}:{args.port}"
                info = font.render(info_text, True, (90, 90, 90))
                info_pos = ((sw - info.get_width()) // 2, status_pos[1] + status.get_height() + 18)
                screen.blit(info, info_pos)
                map_info = font.render(f"선택된 맵: {AVAILABLE_MAPS[selected_map_idx]['name']}", True, (70, 70, 70))
                map_pos = ((sw - map_info.get_width()) // 2, info_pos[1] + info.get_height() + 16)
                screen.blit(map_info, map_pos)
                hint = font.render("P: 대기 화면", True, (80, 80, 80))
                screen.blit(hint, ((sw - hint.get_width()) // 2, map_pos[1] + map_info.get_height() + 16))
            else:
                screen.fill((245, 245, 245))

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

                if len(traj) >= 2:
                    pts = [world_to_screen(x, y, world, sw, sh) for (x, y) in traj]
                    pygame.draw.lines(screen, (50, 50, 200), False, pts, 2)

                draw_car(screen, state, delta, P, world, sw, sh)
                if collision_markers:
                    draw_collision_markers(screen, collision_markers, world, sw, sh)

                # HUD (세 줄 구성)
                hud1 = f"t={t:5.1f}s  gear={u.gear}  v={state.v:5.2f} m/s  steer={math.degrees(delta):6.1f} deg"
                hud2 = f"accel={u.accel*100:5.1f}%  brake={u.brake*100:5.1f}%  collisions={collision_count}"
                if args.mode == "ipc":
                    if ipc.is_connected:
                        peer = f"{ipc.peer[0]}:{ipc.peer[1]}" if ipc.peer else f"{args.host}:{args.port}"
                        hud3 = f"맵: {AVAILABLE_MAPS[selected_map_idx]['name']}  |  IPC 연결: {peer}"
                    else:
                        hud3 = f"맵: {AVAILABLE_MAPS[selected_map_idx]['name']}  |  IPC 대기: {args.host}:{args.port}"
                else:
                    hud3 = f"맵: {AVAILABLE_MAPS[selected_map_idx]['name']}  |  Mode: WASD (W/S throttle, A/D steer, R gear, SPACE brake)"
                hud_hint = "P: 대기 화면"

                screen.blit(font.render(hud1, True, (0, 0, 0)), (12, 8))
                screen.blit(font.render(hud2, True, (0, 0, 0)), (12, 26))
                hud3_img = font.render(hud3, True, (0, 0, 0))
                screen.blit(hud3_img, (12, 44))
                hint_img = font.render(hud_hint, True, (70, 70, 70))
                screen.blit(hint_img, (sw - hint_img.get_width() - 16, 44))

                if paused:
                    stats_lines = [
                        f"시간: {t:0.1f} s",
                        f"속도: {state.v:0.2f} m/s",
                        f"조향: {math.degrees(delta):0.1f} deg",
                        f"충돌 횟수: {collision_count}",
                        f"맵: {AVAILABLE_MAPS[selected_map_idx]['name']}",
                    ]
                    instruction_lines = [
                        "P: 계속",
                        "M: 맵 선택",
                        "R: 재시작 (라운드 종료 후)",
                        "ESC/Q: 종료",
                    ]
                    draw_pause_overlay(screen, fonts, sw, sh, stats_lines, instruction_lines)

                if args.mode == "ipc" and pygame.time.get_ticks() < banner_show_until:
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
        stage_idx, stage_profile = get_stage_profile(active_map_cfg)
        score_value, score_details = compute_round_score(round_stats, stage_profile, why, M.extent)
        RECENT_RESULTS.append({
            "map": active_map_cfg["name"] if active_map_cfg else "알 수 없음",
            "stage": stage_idx,
            "stage_label": stage_profile.get("label", f"{stage_idx}단계"),
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

        replay_meta = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "map_key": map_key,
            "map_name": map_name,
            "map_seed": map_seed,
            "target_idx": current_target_idx,
            "stage": stage_idx,
            "stage_label": stage_profile.get("label", f"{stage_idx}단계"),
            "result": why,
            "score": score_value,
            "mode": args.mode,
            "stats": {
                "elapsed": round_stats.elapsed,
                "distance": round_stats.distance,
                "gear_switches": round_stats.gear_switches,
                "avg_speed": score_details.get("avg_speed"),
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
        stage_label = stage_profile.get("label", f"{stage_idx}단계")
        avg_speed = score_details.get("avg_speed", 0.0)
        info_lines = [
            f"Score: {score_value:.1f}",
            f"Stage: {stage_label}",
            f"Elapsed: {round_stats.elapsed:.1f} s",
            f"Distance: {round_stats.distance:.1f} m",
            f"Gear switches: {round_stats.gear_switches}",
            f"Avg speed: {avg_speed:.2f} m/s",
            f"Collisions: {collision_count}",
        ]
        if why == "success":
            info_lines.extend([
                f"시간 비율: {score_details.get('time_ratio', 0.0):.2f}",
                f"거리 비율: {score_details.get('dist_ratio', 0.0):.2f}",
                f"방향전환 비율: {score_details.get('turn_ratio', 0.0):.2f}",
                f"속도 비율: {score_details.get('speed_ratio', 0.0):.2f}",
            ])
        info_lines.append(f"맵: {AVAILABLE_MAPS[selected_map_idx]['name']}")
        if args.mode == "ipc":
            if ipc and ipc.is_connected:
                peer = f"{ipc.peer[0]}:{ipc.peer[1]}" if ipc.peer else f"{args.host}:{args.port}"
                info_lines.append(f"IPC connected: {peer}")
            else:
                info_lines.append(f"IPC waiting on {args.host}:{args.port}")
        if replay_path:
            info_lines.append(f"Replay: {replay_path}")

        if RECENT_RESULTS:
            info_lines.append("최근 3회 기록:")
            reason_map = {
                "success": "성공",
                "collision": "충돌",
                "timeout": "시간초과",
                "quit": "종료",
            }
            for entry in reversed(RECENT_RESULTS):
                label = entry.get("stage_label", "")
                reason_text = reason_map.get(entry.get("reason"), entry.get("reason", ""))
                info_lines.append(f"- {label} {entry.get('map', '')}: {entry.get('score', 0.0):.1f}점 ({reason_text})")

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
