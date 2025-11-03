# Parking Workspace

이 저장소는 시뮬레이터와 학생 에이전트 구현을 한 번에 다루기 위한 상위 워크스페이스입니다.

- `parking-sim/` — 기존 [`self-parking-sim`](https://github.com/sungb0131/self-parking-sim) 프로젝트를 그대로 옮겨 놓은 디렉터리입니다. 시뮬레이터 실행, 맵 자산, 스크립트가 모두 여기에 포함됩니다.
- `parking-agent/` — [`self-parking-user-algorithms`](https://github.com/sungb0131/self-parking-user-algorithms) 저장소를 클론해 사용하는 디렉터리입니다.

## 초기 설정

```bash
# 에이전트 저장소 클론 (이미 존재한다면 생략)
git clone https://github.com/sungb0131/self-parking-user-algorithms.git parking-agent
```

시뮬레이터는 `parking-sim/` 디렉터리에서 기존과 동일하게 실행합니다.

```bash
cd parking-sim
python3 -m pip install -r requirements.txt
python3 demo_self_parking_sim.py
```

에이전트는 `parking-agent/` 디렉터리에서 개발·실행하면 됩니다.
