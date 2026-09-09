# Electric Circuit Analogy

[English](README.md) | **한국어**

미세유체 채널 네트워크를 전기회로 등가 모델로 해석하고, 목표 outlet 유량과
농도에 맞도록 채널 길이를 최적화하는 Python 프로그램입니다. 기본 회로
해석과 함께 NSGA-II, PSO, multi-start SLSQP 최적화 방법을 지원합니다.

## 주요 기능

- incidence matrix 기반 채널 네트워크 해석
- Kirchhoff 법칙을 이용한 채널별 유량 및 압력 계산
- 다성분 유체의 outlet 농도 계산
- NSGA-II, PSO 또는 multi-start SLSQP를 이용한 채널 길이 최적화
- system1~system4 예제 토폴로지 및 입력 데이터 제공
- 시스템과 optimizer별 결과 디렉터리 자동 분리

## 요구 환경

- Python 3.10 이상
- NumPy 1.24 이상, 3 미만
- pandas 2 이상, 4 미만
- SciPy 1.10 이상, 2 미만
- Matplotlib 3.7 이상, 4 미만
- pymoo 0.6 이상, 0.7 미만

## 설치

사용하려는 Python 환경에서 저장소 루트로 이동한 후 의존성을 설치합니다.
시스템 Python, conda 환경 또는 다른 환경 관리 도구 중 어느 것을 사용해도
됩니다.

```bash
python -m pip install -r requirements.txt
```

시스템에 따라 Python 실행 명령이 `python3`인 경우에는
`python3 -m pip install -r requirements.txt`를 사용하십시오. conda 등 별도
환경을 사용한다면 원하는 환경을 먼저 선택한 후 위 명령을 실행하면 됩니다.

설치와 CLI 인식을 확인하려면 다음 명령을 실행합니다.

```bash
python electric_analogy_programming.py --help
```

## 빠른 시작

처음 실행할 때는 최적화 없이 system1의 원 설계를 해석하는 다음 명령을
권장합니다.

```bash
python electric_analogy_programming.py --system system1 --solve-only
```

기본 결과는 `results/system1/nsga2/reviewed_additive/`에 저장됩니다.
`--solve-only`에서는 `--optimizer`가 최적화에 사용되지는 않지만 결과 경로를
구분하는 이름으로 사용됩니다.

길이를 최적화하려면 원하는 optimizer를 선택합니다.

```bash
# NSGA-II
python electric_analogy_programming.py --system system1 --optimizer nsga2

# Particle Swarm Optimization
python electric_analogy_programming.py --system system2 --optimizer pso

# Multi-start SLSQP
python electric_analogy_programming.py --system system3 --optimizer slsqp
```

인자 없이 실행하면 system1에 대해 NSGA-II 길이 최적화를 수행합니다.
기본 설정은 계산량이 클 수 있으므로 기능 확인이나 시험 실행에는 작은
population과 반복 횟수를 지정하는 것이 좋습니다.

```bash
python electric_analogy_programming.py --system system1 --optimizer nsga2 --pop-size 20 --n-gen 10
python electric_analogy_programming.py --system system1 --optimizer pso --pop-size 10 --n-gen 10
python electric_analogy_programming.py --system system1 --optimizer slsqp --n-starts 2 --maxiter 20
```

## 명령행 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--system` | `system1`, `system2`, `system3`, `system4` 중 실행할 시스템 | `system1` |
| `--optimizer` | `nsga2`, `pso`, `slsqp` 중 최적화 방법 | `nsga2` |
| `--seed` | PSO/SLSQP 난수 seed | `1` |
| `--pop-size` | NSGA-II population 또는 PSO swarm 크기 | NSGA-II `600`, PSO `50` |
| `--n-gen` | NSGA-II/PSO 최대 generation 수 | NSGA-II `600`; PSO는 평가 예산으로 계산 |
| `--n-starts` | SLSQP 독립 재시작 횟수 | `20` |
| `--maxiter` | 각 SLSQP 실행의 최대 iteration 수 | `1000` |
| `--edge-mode` | `reviewed`: CAD 검토 edge 집합, `all`: 전체 논리 edge | `reviewed` |
| `--bound-mode` | `additive`: 초기값 −5.5/+2.0 mm, `relative`: 초기값 0.5배/2.0배 | `additive` |
| `--output-dir` | 사용자가 지정하는 결과 디렉터리 | `results/<system>/<optimizer>/<edge_mode>_<bound_mode>/` |
| `--solve-only` | 길이 최적화를 생략하고 입력 설계만 해석 | 사용하지 않음 |

재현 가능한 PSO/SLSQP 실행에는 동일한 `--seed`와 나머지 optimizer 옵션을
함께 기록하십시오.

## 입력 데이터

각 시스템 디렉터리에는 다음 세 입력 CSV가 있어야 합니다.

```text
system1/
├── incidence_mat.csv
├── length_mat.csv
└── concentration_mat.csv
```

- `incidence_mat.csv`: edge와 node의 연결 관계를 나타내는 incidence matrix
- `length_mat.csv`: 각 edge의 채널 길이
- `concentration_mat.csv`: 각 outlet에서 원하는 성분별 목표 농도

시스템별 유체 물성, 채널 단면, inlet/outlet 인덱스, inlet 유량 및 최적화할
edge 목록은 `codes/system_configs.py`에 정의되어 있습니다. 입력 CSV의 edge와
node 순서는 해당 설정의 인덱스와 일치해야 합니다.

edge 집합과 bound 방식은 하나의 코드에서 서로 독립적으로 선택합니다.

| 조합 | 의미 | CAD 배포 상태 |
|---|---|---|
| `reviewed + additive` | 검토 edge, `초기값 - 5.5 mm`~`초기값 + 2.0 mm` | 생산 기본값; 기존 CAD workflow 적용 가능 |
| `reviewed + relative` | 검토 edge, `초기값 × 0.5`~`초기값 × 2.0` | 탐색용; CAD 재교정 필요 |
| `all + relative` | 전체 edge, `초기값 × 0.5`~`초기값 × 2.0` | 탐색용; 일부 edge는 CAD controller가 없을 수 있음 |
| `all + additive` | 전체 edge, `초기값 - 5.5 mm`~`초기값 + 2.0 mm` | 하한이 0 이하인 edge가 있으면 즉시 중단 |

additive mode는 잘못된 하한을 임의로 clipping하지 않습니다. 예를 들어 짧은
fixed edge까지 `--edge-mode all`로 포함해 `초기값 - 5.5 mm <= 0`이 되면,
실행을 중단하고 위험한 논리 edge를 모두 표시합니다. 두 bound profile과
검토된 changing-edge 목록은 `codes/system_configs.py`에 정의되어 있습니다.

| System | Topology | 최적화 가능한 논리 edge |
|---|---|---|
| `system1` | system1 | E01-E05, E07 |
| `system2` | `b` | E01-E05, E08, E11, E13, E15 |
| `system3` | `c` | E01-E05, E12-E15, E20-E22 |
| `system4` | `3_4` | E01-E04, E11-E15, E22-E25 |

각 system에서 위 목록에 포함되지 않은 edge는 고정됩니다.

실행 예시는 다음과 같습니다.

```bash
# 생산 기본 profile
python electric_analogy_programming.py --system system3 --optimizer nsga2 \
  --edge-mode reviewed --bound-mode additive

# 전체 edge 탐색 profile
python electric_analogy_programming.py --system system3 --optimizer pso \
  --edge-mode all --bound-mode relative
```

모든 최적화 실행은 `optimization_run_config.json`을 생성합니다. 이 파일에는
선택한 mode, changing-edge index와 E 번호, edge별 초기/하한/상한 길이,
요청한 delta/scale, `cad_release_eligible`가 기록되며 optimizer에 실제 전달된
profile의 기준 증적이 됩니다.

입력 디렉터리는 원본 데이터 보관용으로 취급됩니다. 프로그램은 입력 CSV를
결과 디렉터리로 복사한 후 그 복사본을 사용하며, 시스템 입력 디렉터리 또는
그 하위 경로를 `--output-dir`로 지정하는 것을 차단합니다.

농도 행렬에 대한 자세한 설명은
[`conc_matrix_explanation.md`](conc_matrix_explanation.md)를 참고하십시오.

## 결과 확인

기본 디렉터리 구조는 다음과 같습니다.

```text
results/
└── system1/
    ├── nsga2/
    │   ├── reviewed_additive/
    │   └── all_relative/
    ├── pso/
    └── slsqp/
```

실행이 끝나면 콘솔에 실제 결과 경로가 `Results written to: ...` 형식으로
표시됩니다. 실행 방법에 따라 다음 파일들이 생성될 수 있습니다.

- `new_length_mat.csv`: 최적화된 채널 길이
- `optimization_run_config.json`: 확정 edge/bound profile과 CAD 적용 가능 여부
- `current_vec_total.csv`, `voltage_vec_total.csv`: 원 설계의 유량과 압력
- `current_vec_revised.csv`, `voltage_vec_revised.csv`: 최적화 설계의 유량과 압력
- `outlet_*.png`: outlet 유량·농도 비교 그래프
- `pareto_front.png`, `pareto_optimal_*.csv`: NSGA-II 결과
- `pso_optimization_summary.csv`: PSO 실행 요약
- `slsqp_restart_summary.csv`: SLSQP 재시작별 결과

사용자 지정 경로에 결과를 저장하려면 다음과 같이 실행합니다.

```bash
python electric_analogy_programming.py --system system4 --optimizer pso --output-dir my_results/system4_pso
```

## Python에서 실행

CLI 대신 다른 Python 코드에서 공통 드라이버를 호출할 수도 있습니다.

```python
from codes.electric_analogy_programming import run

flow, concentration, result_dir = run(
    system="system1",
    optimizer="pso",
    seed=42,
    pop_size=20,
    n_gen=50,
)
```

최적화 없이 해석하려면 `optimize=False`를 지정합니다.

```python
flow, concentration, result_dir = run(
    system="system1",
    optimizer="nsga2",
    optimize=False,
)
```

저수준 회로 해석 API가 필요한 경우 `codes.electric_analogy`를 import합니다.
저장소 루트의 `electric_analogy.py`는 기존 `import electric_analogy` 사용자를
위한 호환 facade입니다.

## 프로젝트 구조

```text
electric_circuit_analogy/
├── electric_analogy.py                 # 루트 호환 facade
├── electric_analogy_programming.py     # 기본 CLI launcher
├── requirements.txt
├── codes/
│   ├── electric_analogy.py             # 공통 solver와 NSGA-II
│   ├── electric_analogy_pso.py         # PSO backend
│   ├── electric_analogy_slsqp.py       # multi-start SLSQP backend
│   ├── electric_analogy_programming.py # 공통 CLI/driver
│   ├── optimization_profiles.py        # 공통 edge/bound mode 해석기
│   ├── system_configs.py               # system1~system4 설정
│   └── resistance.py                   # 채널 저항식
├── system1/                            # 입력 데이터와 호환 launcher
├── system2/
├── system3/
├── system4/
└── results/                            # 실행 시 생성; Git에서 제외
```

## 기존 시스템별 경로로 실행

시스템 폴더에 있는 launcher도 사용할 수 있습니다. 저장소 루트에서 실행하는
것이 권장됩니다.

```bash
python system1/electric_analogy_programming.py --optimizer pso
python system4/electric_analogy_programming.py --optimizer slsqp
```

각 launcher는 해당 시스템을 기본값으로 선택하며, 실제 계산은 동일한 공통
드라이버와 solver를 사용합니다.

## 문제 해결

- `ModuleNotFoundError`가 발생하면 현재 실행 중인 Python 환경에 의존성이
  설치되었는지 확인하고 `python -m pip install -r requirements.txt`를 다시
  실행하십시오.
- 입력 파일 오류가 발생하면 선택한 system 디렉터리에 세 CSV 파일이 모두
  있는지 확인하십시오.
- 출력 경로 관련 `ValueError`가 발생하면 `system1`~`system4` 밖의 별도
  디렉터리를 지정하십시오.
- 최적화가 오래 걸리면 시험 실행에서 `--pop-size`, `--n-gen`,
  `--n-starts`, `--maxiter`를 줄이십시오.
- 정확한 현재 옵션은 `python electric_analogy_programming.py --help`로
  확인할 수 있습니다.
