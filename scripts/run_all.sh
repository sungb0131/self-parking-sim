#!/bin/bash
# run_all.sh — venv 활성화 후 시뮬레이터(서버) → 학생 알고리즘(클라이언트) 순으로 실행
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  echo "[run_all] .venv 디렉터리가 없습니다. 먼저 가상환경을 생성하세요." >&2
  exit 1
fi

source .venv/bin/activate

SIM_HOST="127.0.0.1"
SIM_PORT="55556"

python3 demo_self_parking_sim.py --mode ipc --host "$SIM_HOST" --port "$SIM_PORT" &
SIM_PID=$!
echo "[run_all] simulator started (pid=$SIM_PID). Waiting before launching student client..."

cleanup() {
  echo "[run_all] cleaning up"
  kill "${SIM_PID}" 2>/dev/null || true
  if [[ -n "${CLIENT_PID:-}" ]]; then
    kill "${CLIENT_PID}" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

sleep 2

python3 student_algo_server.py --host "$SIM_HOST" --port "$SIM_PORT" &
CLIENT_PID=$!
echo "[run_all] student algorithm started (pid=$CLIENT_PID)."

wait "$SIM_PID"
