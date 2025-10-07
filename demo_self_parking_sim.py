# demo_self_parking_sim.py — Self-Parking simulator (MATLAB 레이어 + SAT 충돌) + IPC 제어
# 사용법:
#   python demo_self_parking_sim.py                  # 기본: IPC 127.0.0.1:55555
#   python demo_self_parking_sim.py --host 127.0.0.1 --port 55556
#   python demo_self_parking_sim.py --mode wasd      # (디버그용) 키보드 제어

import os
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_HINT_IME_SHOW_UI", "0")  # macOS IME 노이즈 억제

import argparse, errno, json, math, random, socket
from dataclasses import dataclass

import numpy as np
import pygame
from pygame import gfxdraw
from scipy.io import loadmat

# Viewport (screen sub-rectangle) for prettier HUD layout; configured in main().
vp_ox = vp_oy = vp_w = vp_h = None

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

    tip = "Press R to retry, ESC/Q to quit"
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

# ----------------- 메인 -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ipc","wasd"], default="ipc")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=55556)
    return ap.parse_args()

def main():
    args = parse_args()

    # 0) 맵 로드
    M = load_parking_assets("parking_assets_layers_75x50.mat")
    xmin, xmax, ymin, ymax = M.extent
    world = (xmin, xmax, ymin, ymax)
    H, W = M.C.shape

    map_payload = {
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

    # 1) pygame
    pygame.init()
    pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
    sw, sh = 1100, 700
    screen = pygame.display.set_mode((sw, sh))
    pygame.display.set_caption("Self-Parking — Layered Map from MATLAB (IPC)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    font_large = pygame.font.SysFont("consolas", 28)
    font_title = pygame.font.SysFont("consolas", 48, bold=True)

    # Viewport margins so HUD stays outside of the lot rendering area.
    hud_h = 64
    margin = 24
    global vp_ox, vp_oy, vp_w, vp_h
    vp_ox, vp_oy = margin, hud_h
    vp_w, vp_h = sw - 2 * margin, sh - hud_h - margin

    # ----- 재시작 가능한 라운드 루프 -----
    ipc = IPCController(args.host, args.port) if args.mode == "ipc" else None
    student_connected = ipc.is_connected if ipc else False
    banner_show_until = 0
    map_sent = False

    running=True
    while running:
        # 2) 시뮬레이션 라운드 초기화
        P = Params()
        free_slot_indices = [i for i, occ in enumerate(M.occupied_idx) if not occ]
        target_slot = tuple(M.slots[random.choice(free_slot_indices)].tolist())

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

        # ---- 메인 주행 루프 ----
        while why == "running":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    why="quit"; break
            if why!="running": break

            waiting_for_ipc = False

            if args.mode == "wasd":
                # 키 입력 (디버그/예비용)
                keys = pygame.key.get_pressed()
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
                    try:
                        ipc.send_obs(obs)
                        cmd = ipc.recv_cmd()
                        u.delta_tgt = clamp(float(cmd.get("steer", 0.0)), -P.maxSteer, P.maxSteer)
                        u.accel = clamp(float(cmd.get("accel", 0.0)), 0.0, 1.0)
                        u.brake = clamp(float(cmd.get("brake", 0.0)), 0.0, 1.0)
                        u.gear = 'R' if str(cmd.get("gear", "D")).upper().startswith('R') else 'D'
                    except Exception as e:
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

            if not waiting_for_ipc:
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

                # 차량 폴리곤(충돌용)
                car_poly = car_polygon(state, P)

                # -------- 충돌 판정 --------
                collided = False
                # (A) 경계 밖
                for (x, y) in car_poly:
                    if not (xmin <= x <= xmax and ymin <= y <= ymax):
                        collided = True
                        break
                # (B) 점유된 슬롯 사각형과 SAT 충돌
                if not collided:
                    for i, rect in enumerate(M.slots):
                        if not M.occupied_idx[i]:
                            continue
                        if poly_intersects_rect(car_poly, tuple(rect)):
                            collided = True
                            break
                # (C) 정지물(래스터): 둘레 샘플 → stationary 레이어
                if not collided:
                    samples = perimeter_points(car_poly, spacing=0.08)
                    for (x, y) in samples:
                        r, c = world_to_rc(x, y, M.extent, M.cellSize, H)
                        if 0 <= r < H and 0 <= c < W and M.Cs[r, c] > 0.5:
                            collided = True
                            break
                # (D) 타깃 슬롯 내부면 충돌 무시
                if rect_contains_poly(target_slot, car_poly):
                    collided = False

                # 성공 판정(간단 버전): 슬롯 완전 포함 + 저속
                reached = rect_contains_poly(target_slot, car_poly) and abs(state.v) <= 0.2

                if collided:
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
                status_text = "Disconnected - waiting for student algorithm"
                status = font_large.render(status_text, True, status_color)
                status_pos = ((sw - status.get_width()) // 2, title_pos[1] + title.get_height() + 30)
                screen.blit(status, status_pos)

                info_text = f"Listening on {args.host}:{args.port}"
                info = font.render(info_text, True, (90, 90, 90))
                info_pos = ((sw - info.get_width()) // 2, status_pos[1] + status.get_height() + 18)
                screen.blit(info, info_pos)
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

                # HUD (세 줄 구성)
                hud1 = f"t={t:5.1f}s  gear={u.gear}  v={state.v:5.2f} m/s  steer={math.degrees(delta):6.1f} deg"
                hud2 = f"accel={u.accel*100:5.1f}%  brake={u.brake*100:5.1f}%  collisions={collision_count}"
                if args.mode == "ipc":
                    if ipc.is_connected:
                        peer = f"{ipc.peer[0]}:{ipc.peer[1]}" if ipc.peer else f"{args.host}:{args.port}"
                        hud3 = f"IPC: connected ({peer})"
                    else:
                        hud3 = f"IPC: waiting on {args.host}:{args.port}"
                else:
                    hud3 = "Mode: WASD (W/S throttle, A/D steer, R gear toggle, SPACE brake)"

                screen.blit(font.render(hud1, True, (0, 0, 0)), (12, 8))
                screen.blit(font.render(hud2, True, (0, 0, 0)), (12, 26))
                screen.blit(font.render(hud3, True, (0, 0, 0)), (12, 44))

                if args.mode == "ipc" and pygame.time.get_ticks() < banner_show_until:
                    banner = font_large.render("Student algorithm connected!", True, (30, 110, 40))
                    banner_pos = (vp_ox + (vp_w - banner.get_width()) // 2, vp_oy + 20)
                    screen.blit(banner, banner_pos)

            pygame.display.flip()
            clock.tick(int(1.0 / P.dt))
            if not waiting_for_ipc:
                t += P.dt
                if t >= P.timeout:
                    why = "timeout"
                    break

        # 종료 오버레이
        screen.fill((245, 245, 245))
        pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(vp_ox, vp_oy, vp_w, vp_h), 3)
        draw_rect(screen, M.border, (0, 0, 0), world, sw, sh, width=4)

        title = "SUCCESS" if why == "success" else ("TIMEOUT" if why == "timeout" else "COLLISION" if why == "collision" else "QUIT")
        info_lines = [
            f"Elapsed: {t:.1f} s",
            f"Distance: {move_dist:.1f} m",
            f"Collisions: {collision_count}",
        ]
        if args.mode == "ipc":
            if ipc and ipc.is_connected:
                peer = f"{ipc.peer[0]}:{ipc.peer[1]}" if ipc.peer else f"{args.host}:{args.port}"
                info_lines.append(f"IPC connected: {peer}")
            else:
                info_lines.append(f"IPC waiting on {args.host}:{args.port}")

        draw_overlay(screen, title, info_lines, font, sw, sh)
        pygame.display.flip()

        # 대기 루프
        waiting = True
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
            clock.tick(60)


    if ipc:
        ipc.shutdown()

    pygame.quit()

if __name__ == "__main__":
    main()
