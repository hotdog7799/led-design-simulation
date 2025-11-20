새로운 채팅에서 프로젝트를 설명하고 이어서 진행할 수 있도록, 현재까지 구축한 \*\*"렌즈리스 이미징을 위한 LED 조명계 최적화 시뮬레이터"\*\*에 대한 종합적인 **README 및 가이드라인**을 작성해 드립니다.

이 내용을 그대로 복사해서 새로운 채팅의 첫 메시지로 사용하시면, AI가 프로젝트의 맥락을 즉시 파악할 수 있을 것입니다.

-----

# 🦷 LED Illumination Optimization Simulator for Lensless Imaging

이 프로젝트는 렌즈리스 구강 스캐너를 위한 **최적의 LED 조명계 배치**를 찾기 위해 개발된 Python 기반 시뮬레이터입니다. 405nm UV LED(형광 여기용)와 White LED(컬러 이미징용)를 제한된 공간(Camera Cavity 주변)에 배치하여, 목표 ROI에 균일하고 충분한 광량을 전달하는 것을 목표로 합니다.

-----

## 📂 1. 프로젝트 파일 구조

이 시뮬레이터는 유지보수와 확장성을 위해 3개의 파일로 모듈화되어 있습니다.

### **1) `config.yaml` (설정 파일)**

시뮬레이션의 모든 파라미터를 관리하는 컨트롤 타워입니다. 코드를 수정하지 않고 실험 조건을 변경할 수 있습니다.

  * **`optics`**: 광학계 기구 정보 (Working Distance, ROI 크기 등).
  * **`scenario`**: 시나리오별(A: 평면형, B: 매입형) 기구적 제약 조건 (LED 두께, Cavity 크기 등).
  * **`led_specs`**: LED 데이터시트 정보 (방사각 프로파일, 물리적 크기) 및 보정된 파워 값.
  * **`optimization_ranges`**: 탐색할 LED 좌표 범위 (a, b).
  * **`loss_weights`**: 최적화 평가 지표(Loss Function)의 가중치 (Power, Uniformity, Crosstalk 등).

### **2) `led_simulator_utils.py` (라이브러리)**

핵심 물리 모델과 유틸리티 함수들이 구현된 모듈입니다.

  * **물리 모델:**
      * 데이터시트 기반 **Beam Profile Interpolation**.
      * **Inverse Square Law** (거리 감쇠) 및 **Lambert's Cosine Law** (입사각 효과) 적용.
      * **Area Source Model:** LED를 점이 아닌 실제 크기(WxH)를 가진 면 광원으로 모델링 (Sub-sampling 지원).
      * **Passivation 효과 근사:** 레진 도포로 인한 빔 퍼짐 현상 반영.
  * **배치 생성기:**
      * `create_4_symmetric_uv_layout`: Cavity와 최소 간격을 고려한 4개 UV LED 대칭 배치 (±a, ±b).
      * `place_white_leds_fixed_xaxis`: Crosstalk(간섭)를 회피하며 X축 상에 White LED 배치.
  * **분석 및 시각화:**
      * ROI 내 **Total Power(mW)**, **Irradiance(W/m²)**, **Uniformity**, **CV** 계산.
      * 2D Irradiance Map 및 정규화된 1D Line Profile (X, Y, 대각선) 시각화.

### **3) `run_optimization.py` (실행 스크립트)**

시뮬레이션을 실행하는 메인 진입점입니다. 두 가지 모드를 지원합니다.

  * **최적화 모드 (`RUN_OPTIMIZATION = True`):**
      * `config.yaml`의 탐색 범위 내 모든 (a, b) 조합에 대해 시뮬레이션을 수행합니다.
      * 각 배치의 Loss를 계산하고, 최상위 5개 결과를 CSV로 저장합니다.
      * 가장 우수한 배치의 결과를 자동으로 시각화합니다.
  * **플로팅 모드 (`RUN_OPTIMIZATION = False`):**
      * 이전 최적화 결과(CSV)를 불러와, 특정 설정(Row Index)에 대한 상세 시뮬레이션 및 시각화만 빠르게 수행합니다.

-----

## ⚙️ 2. 핵심 기능 및 알고리즘

### **Optimization Logic (최적화 로직)**

다음 조건들을 만족하는 파라미터 `(a, b)`를 Grid Search 방식으로 탐색합니다.

1.  **배치 조건:** UV LED 4개는 `(±a, ±b)`에, White LED 2개는 Cavity 바깥쪽 X축에 배치.
2.  **Hard Constraints (필수 제약):**
      * 어떤 LED도 Camera Cavity 영역을 침범하지 않을 것.
      * LED 간 물리적 간격이 `min_led_separation_mm` 이상일 것.
      * White LED와 UV LED 사이 거리가 `crosstalk_min_distance_mm` (5mm) 이상일 것.
3.  **Soft Constraints (Loss Function):**
      * **Power Penalty:** ROI 내 총 파워가 목표치(34mW) 미달 시 페널티.
      * **Uniformity Penalty:** ROI 내 조도 균일도가 낮을수록 페널티.
      * **Center Boost:** 중앙부 조도가 전체 평균보다 낮을수록 페널티.

### **Calibration (파워 보정)**

실제 실험 데이터(4개 LED, WD=6mm에서 34mW 측정)를 기준으로 시뮬레이션의 광원 파워(`SINGLE_LED_TOTAL_POWER_MW`)를 역산하여 보정함으로써, 시뮬레이션 결과의 절대적 수치(mW)에 대한 신뢰도를 확보했습니다.

-----

## 🚀 3. 사용 가이드 (How to Use)

### **환경 설정**

필요한 라이브러리 설치:

```bash
pip install numpy matplotlib scipy pandas pyyaml
```

### **실행 방법**

1.  **파라미터 수정:** `config.yaml`에서 시나리오(A/B), 탐색 범위, 가중치 등을 설정합니다.
2.  **최적화 실행:** `run_optimization.py`에서 `RUN_OPTIMIZATION = True`로 설정하고 실행합니다.
    ```bash
    python run_optimization.py
    ```
      * 결과는 `plot_results/YYMMDD_HHMMSS_optimization_top5_results.csv`에 저장됩니다.
      * 최적 배치의 2D 맵과 라인 플롯 이미지가 저장되고 팝업으로 표시됩니다.
3.  **결과 재확인 (플로팅 모드):**
      * `run_optimization.py`에서 `RUN_OPTIMIZATION = False`로 변경합니다.
      * `PLOT_CSV_FILE`에 확인하고 싶은 결과 CSV 파일 경로를 입력합니다.
      * `PLOT_ROW_INDEX`를 지정(0=1등, 1=2등...)하여 실행하면 해당 배치를 다시 시뮬레이션하여 시각화합니다.

-----

이 가이드를 새로운 채팅에 붙여넣으시면, AI가 즉시 프로젝트의 구조와 목표를 이해하고 이어서 도움을 줄 수 있습니다.