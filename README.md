# Parking Workspace

이 저장소는 [`self-parking-sim`](https://github.com/sungb0131/self-parking-sim) 시뮬레이터와 학생 에이전트 구현을 한 번에 다루기 위한 상위 워크스페이스입니다. 기존에 `parking-sim/` 폴더 안에 있던 시뮬레이터 파일을 모두 최상위 디렉터리로 옮겨 두었습니다.

## 구성

- `demo_self_parking_sim.py` — 시뮬레이터 데모 실행 진입점
- `student_algorithms.py` — 학생용 에이전트 베이스 코드
- `parking_assets_layers_75x50.mat` — 데모에서 사용하는 주차장 자산
- `requirements.txt` — 시뮬레이터 실행에 필요한 파이썬 패키지 목록
- (선택) `parking-agent/` — [`self-parking-user-algorithms`](https://github.com/sungb0131/self-parking-user-algorithms) 저장소를 클론해 사용하는 위치

## 초기 설정

```bash
# 에이전트 저장소 클론 (이미 존재한다면 생략)
git clone https://github.com/sungb0131/self-parking-user-algorithms.git parking-agent
```

## 시뮬레이터 실행

이 프로젝트는 Python 3.10 이상에서 테스트되었습니다. 시스템에 3.10 해석기가 없다면 `pyenv`나 `asdf`로 설치한 뒤 아래 명령을 실행하세요.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python demo_self_parking_sim.py
```

에이전트는 `parking-agent/` 디렉터리(또는 별도 저장소)에서 개발하면 됩니다.
