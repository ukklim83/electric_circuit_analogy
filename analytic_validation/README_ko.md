# Analytic validation 자료

[English](README.md) | **한국어**

이 디렉터리는 기존 `analytic/` 작업 폴더에서 재현에 필요한 파일만 선별한
배포용 구성입니다. 생성된 그림·전개된 기호식 TXT·optimizer 결과·출력이
포함된 notebook·중복 solver·debug 파일은 포함하지 않았습니다.

## 검증 범위

현재 자료는 8개 resistance를 갖는 2-inlet/2-outlet prototype의 수학적 구조를
분석합니다. 정확한 행렬 방정식을 기반으로 목적함수의 joint convexity에 대한
반례를 찾고 수치 정밀도를 바꿔 재확인합니다.

다음 항목을 직접 검증하는 도구는 아닙니다.

- `system1`~`system4` 전체의 수치 정확도
- NSGA-II·PSO·SLSQP의 global optimum 수렴
- `reviewed`/`all`, `additive`/`relative` mode
- CAD 형상, clearance, 제조 가능성 및 controller calibration

이 항목에는 별도의 solver 회귀 검사와 CAD acceptance test가 필요합니다.

## 포함 파일

| 파일 | 역할 |
|---|---|
| `prototype_reduced_objective.py` | 상태 변수를 수치적으로 제거한 정확한 8-resistance 목적함수 |
| `prototype_joint_convexity_analysis.py` | Sobol sampling, Hessian, eigenvalue 및 Jensen 검사 |
| `prototype_directional_confirmation.py` | 선택된 음의 곡률 방향의 arbitrary-precision 재확인 |
| `plot_prototype_presentation_figures.py` | 결과 보고용 그림 생성 |
| `analytic_validation_script.py` | 기존 대규모 symbolic derivation; legacy reference |
| `CV_NCV_map.xlsx` | legacy script가 사용하는 수동 convex/non-convex 분류표 |

## 권장 실행 순서

저장소 루트에서 의존성을 설치합니다.

```bash
python -m pip install -r requirements.txt
```

그다음 이 디렉터리에서 실행합니다.

```bash
python prototype_joint_convexity_analysis.py --samples 128 --replicates 1
python prototype_directional_confirmation.py
python plot_prototype_presentation_figures.py
```

확장 검사는 다음과 같습니다.

```bash
python prototype_joint_convexity_analysis.py --samples 256 --replicates 4 \
  --output-dir outputs/prototype_joint_convexity_n256_b4
python prototype_directional_confirmation.py \
  --input-dir outputs/prototype_joint_convexity_n256_b4
python plot_prototype_presentation_figures.py \
  --input-dir outputs/prototype_joint_convexity_n256_b4
```

생성 파일은 모두 Git에서 제외되는 `outputs/` 아래에 기록됩니다.

검증 흐름은 정확한 유량·농도 행렬 구성, 내부 Sobol point의 full Hessian 계산,
step-stable 음의 eigenvalue 탐색, arbitrary-precision 대칭 차분, Jensen gap
확인 순서입니다. 안정적인 음의 방향 곡률은 이 prototype이 joint convex가
아님을 보이지만, 다수의 local optimum이 존재한다는 증명은 아닙니다.

기존 symbolic script는 다음처럼 실행할 수 있습니다.

```bash
python analytic_validation_script.py
```

이 스크립트는 local LaTeX 환경이 필요하고 계산량과 그림 크기가 매우 큽니다.
출력은 `outputs/legacy_symbolic/`에 저장됩니다.
