# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import yaml
import itertools
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

output_dir = "./plot_result"
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

# 현재 스크립트의 디렉토리에서 유틸리티 함수 임포트
# (run_optimization.py와 led_simulator_utils.py가 같은 폴더에 있어야 함)
try:
    import led_simulator_utils as led_sim
except ImportError:
    print("Error: 'led_simulator_utils.py' not found.")
    print("Please make sure both scripts are in the same directory.")
    sys.exit(1)


# ======================================================================================
# 1. 설정 파일 로드 및 파라미터 재구성
# ======================================================================================
def load_config(config_file="config.yaml"):
    """YAML 설정 파일을 로드합니다."""
    if not os.path.exists(config_file):
        print(f"Error: '{config_file}' not found.")
        sys.exit(1)

    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def setup_parameters(config):
    """설정 딕셔너리를 시뮬레이션에 필요한 파라미터로 변환합니다."""
    # 시나리오 설정
    sc = config["scenario"]
    actual_dist = sc["actual_led_object_distance_mm"]
    cavity_w = sc["cavity_width_mm"]
    cavity_h = sc["cavity_height_mm"]

    # LED 특성
    led_spec = config["led_specs"]
    angles_orig = np.array(led_spec["angles_original"])
    intens_orig = np.array(led_spec["intensities_original"])
    spread_factor = led_spec["passivation_spread_factor"]
    angles_final = np.clip(angles_orig * spread_factor, 0, 95)
    angles_final[0] = 0
    beam_func = led_sim.get_beam_profile_func(angles_final, intens_orig)

    # 최적화 범위
    opt_range = config["optimization_ranges"]
    a_range = np.arange(
        opt_range["a_range"]["start"],
        opt_range["a_range"]["stop"],
        opt_range["a_range"]["step"],
    )
    b_range = np.arange(
        opt_range["b_range"]["start"],
        opt_range["b_range"]["stop"],
        opt_range["b_range"]["step"],
    )
    param_combinations = list(itertools.product(a_range, b_range))

    # 나머지 파라미터 래핑
    params = {
        "actual_dist": actual_dist,
        "cavity_w": cavity_w,
        "cavity_h": cavity_h,
        "beam_func": beam_func,
        "config": config,
        "param_combinations": param_combinations,
    }
    return params


# ======================================================================================
# 2. 메인 최적화 실행
# ======================================================================================
def plot_pareto_frontier(results_df, output_dir, timestamp):
    """
    Power vs Uniformity 관계를 산점도로 시각화하여 Trade-off를 분석합니다.
    Color는 Total Loss를 나타냅니다.
    """
    if results_df.empty:
        return

    plt.figure(figsize=(10, 7))

    # X축: Power, Y축: Overall Uniformity, Color: Total Loss (낮을수록 좋음)
    # cmap='viridis_r' : 노란색(낮은 Loss, 좋음) -> 보라색(높은 Loss, 나쁨)
    scatter = plt.scatter(
        results_df["Pwr_UV_mW"],
        results_df["Uni_UV_All"],
        c=results_df["Total_Loss"],
        cmap="viridis_r",
        alpha=0.7,
        edgecolors="k",
        s=50,
    )

    plt.colorbar(scatter, label="Total Loss (Lower(Yellow) is Better)")
    plt.xlabel("UV ROI Power (mW)")
    plt.ylabel("UV Overall Uniformity")
    plt.title("Pareto Frontier Analysis: Power vs Uniformity")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)

    # 최적점(Loss 최소) 표시
    best_idx = results_df["Total_Loss"].idxmin()
    best_row = results_df.loc[best_idx]
    plt.scatter(
        best_row["Pwr_UV_mW"],
        best_row["Uni_UV_All"],
        c="red",
        s=150,
        # marker="*",ㄴ
        label="Best Config",
    )

    # 텍스트로 최적점 정보 표시
    plt.text(
        best_row["Pwr_UV_mW"],
        best_row["Uni_UV_All"],
        f"  Best\n  (a={best_row['a_mm']}, b={best_row['b_mm']})",
        color="red",
        fontweight="bold",
    )

    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{timestamp}_pareto_frontier.png")
    plt.savefig(save_path)
    print(f"Pareto Frontier plot saved to: {save_path}")
    plt.show()


def run_optimization_sweep(params):
    config = params["config"]
    beam_func = params["beam_func"]
    actual_dist = params["actual_dist"]
    cavity_w = params["cavity_w"]
    cavity_h = params["cavity_h"]
    param_combinations = params["param_combinations"]

    roi_w = config["optics"]["roi_width_mm"]
    roi_h = config["optics"]["roi_height_mm"]
    center_ratio = config["optics"]["center_roi_ratio"]

    num_uv = config["led_specs"]["num_uv_leds"]
    num_white = config["white_led"]["num_to_place"]
    min_sep = config["led_specs"]["min_led_separation_mm"]
    white_x_offset = config["white_led"]["x_offset_mm"]
    min_crosstalk_dist = config["white_led"]["crosstalk_min_distance_mm"]
    single_power_mw = config["led_specs"]["single_led_total_power_mw"]
    loss_weights = config["loss_weights"]

    grid_size = config["simulation"]["grid_size_mm"]
    resolution = config["simulation"]["resolution"]

    # 면 광원 파라미터 (UV & White 공통)
    led_geom = {
        "led_width_mm": config["led_specs"]["led_width_mm"],
        "led_height_mm": config["led_specs"]["led_height_mm"],
        "subsample_x": config["led_specs"]["subsample_x"],
        "subsample_y": config["led_specs"]["subsample_y"],
    }

    # ==========================================================================
    # [NEW] White LED Reference Simulation (Pre-calculation)
    # White LED 위치는 (±x_offset, 0)으로 고정되어 있으므로 루프 밖에서 한 번만 계산
    # ==========================================================================
    print("--- Pre-calculating White LED Illumination (Area Source Mode) ---")
    fixed_white_positions = [
        np.array([white_x_offset, 0, 0]),
        np.array([-white_x_offset, 0, 0]),
    ]

    # White LED도 동일한 면 광원 로직(led_geom)을 사용하여 시뮬레이션
    _, _, white_illum_map = led_sim.simulate_illumination(
        fixed_white_positions,
        beam_func,
        actual_dist,
        grid_size,
        resolution,
        led_geom_params=led_geom,
    )

    # White LED의 Uniformity 점수 미리 계산
    _, white_uni_all, white_uni_center, _ = led_sim.analyze_roi(
        _, _, white_illum_map, roi_w, roi_h, single_power_mw, num_white, center_ratio
    )
    print(f"White LED Reference Uniformity: {white_uni_all:.3f}")

    results_list = []
    total_combinations = len(param_combinations)
    print(
        f"--- Starting UV Optimization Search ({total_combinations} combinations) ---"
    )

    for idx, (a, b) in enumerate(param_combinations):
        # 1. UV LED 배치
        uv_led_positions = led_sim.create_4_symmetric_uv_layout(
            a, b, cavity_w, cavity_h, min_sep
        )
        if not uv_led_positions:
            continue

        # 2. White LED 배치 검사 (Hard Constraint)
        white_led_positions, crosstalk_penalty = led_sim.place_white_leds_fixed_xaxis(
            uv_led_positions,
            white_x_offset,
            num_white,
            min_crosstalk_dist,
            0,  # Penalty 값은 이제 안 씀
        )

        # [UPDATE] Crosstalk 발생 시 아예 탈락시킴 (Hard Constraint)
        if len(white_led_positions) < num_white:
            continue  # Skip this configuration

        # 3. 시뮬레이션 수행
        X_mm, Y_mm, uv_illum_map = led_sim.simulate_illumination(
            uv_led_positions,
            beam_func,
            actual_dist,
            grid_size,
            resolution,
            led_geom_params=led_geom,
        )
        power_uv, uni_uv_all, uni_uv_cen, _ = led_sim.analyze_roi(
            X_mm,
            Y_mm,
            uv_illum_map,
            roi_w,
            roi_h,
            single_power_mw,
            num_uv,
            center_ratio,
        )

        # 4. Loss 계산 (정규화된 방식)
        # calculate_loss 함수 인자에서 crosstalk_penalty 제거됨
        total_loss, p_pen, u_all_pen, u_cen_pen = led_sim.calculate_loss(
            power_uv, uni_uv_all, uni_uv_cen, white_uni_all, loss_weights
        )

        results_list.append(
            {
                "a_mm": round(a, 2),
                "b_mm": round(b, 2),
                "Pwr_UV_mW": round(power_uv, 2),
                "Uni_UV_All": round(uni_uv_all, 3),
                "Uni_UV_Cen": round(uni_uv_cen, 3),
                "Total_Loss": round(
                    total_loss, 4
                ),  # Loss가 작아질 수 있으므로 소수점 늘림
                "Valid": True,
            }
        )

        # 진행 상황 출력 (100개마다)
        if idx % 100 == 0:
            print(
                f"({idx}/{total_combinations}) a={a:.2f}, b={b:.2f} -> Loss={total_loss:.4f}"
            )

    return results_list, white_illum_map  # 시각화를 위해 화이트 맵도 반환


# ======================================================================================
# 3. 결과 분석 및 시각화
# ======================================================================================
def analyze_and_plot_results(results_list, white_illum_map_ref, params):
    config = params["config"]
    if not results_list:
        print("No valid results.")
        return

    results_df = pd.DataFrame(results_list)
    results_df_sorted = results_df.sort_values(by="Total_Loss").reset_index(drop=True)

    # 결과 저장
    results_df_sorted.to_csv(config["logging"]["results_file"], index=False)

    best_config = results_df_sorted.iloc[0]
    print("\n--- Best Configuration ---")
    print(best_config)

    plot_pareto_frontier(results_df_sorted, output_dir, timestamp)

    best_a = best_config["a_mm"]
    best_b = best_config["b_mm"]

    # 재구성 및 시각화 준비
    uv_leds = led_sim.create_4_symmetric_uv_layout(
        best_a,
        best_b,
        params["cavity_w"],
        params["cavity_h"],
        config["led_specs"]["min_led_separation_mm"],
    )
    white_leds, _ = led_sim.place_white_leds_fixed_xaxis(
        uv_leds,
        config["white_led"]["x_offset_mm"],
        config["white_led"]["num_to_place"],
        config["white_led"]["crosstalk_min_distance_mm"],
        0,
    )

    led_geom = {
        "led_width_mm": config["led_specs"]["led_width_mm"],
        "led_height_mm": config["led_specs"]["led_height_mm"],
        "subsample_x": config["led_specs"]["subsample_x"],
        "subsample_y": config["led_specs"]["subsample_y"],
    }

    # UV Map 재생성
    X, Y, uv_map = led_sim.simulate_illumination(
        uv_leds,
        params["beam_func"],
        params["actual_dist"],
        config["simulation"]["grid_size_mm"],
        config["simulation"]["resolution"],
        led_geom_params=led_geom,
    )

    # 통계 다시 계산
    p_uv, u_uv_all, u_uv_cen, scale = led_sim.analyze_roi(
        X,
        Y,
        uv_map,
        config["optics"]["roi_width_mm"],
        config["optics"]["roi_height_mm"],
        config["led_specs"]["single_led_total_power_mw"],
        len(uv_leds),
        config["optics"]["center_roi_ratio"],
    )

    # White 통계
    p_wh, u_wh_all, u_wh_cen, _ = led_sim.analyze_roi(
        X,
        Y,
        white_illum_map_ref,
        config["optics"]["roi_width_mm"],
        config["optics"]["roi_height_mm"],
        config["led_specs"]["single_led_total_power_mw"],
        len(white_leds),
        config["optics"]["center_roi_ratio"],
    )

    # [PLOT] 1x3 서브플롯 (UV / White / Combined)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    plot_params = {
        "roi_w": config["optics"]["roi_width_mm"],
        "roi_h": config["optics"]["roi_height_mm"],
        "cavity_w": params["cavity_w"],
        "cavity_h": params["cavity_h"],
        "center_ratio": config["optics"]["center_roi_ratio"],
    }
    extent = [
        -config["simulation"]["grid_size_mm"] / 2,
        config["simulation"]["grid_size_mm"] / 2,
        -config["simulation"]["grid_size_mm"] / 2,
        config["simulation"]["grid_size_mm"] / 2,
    ]

    # 1. UV Plot
    led_sim.plot_irradiance_map(
        axes[0],
        uv_map,
        extent,
        plot_params,
        uv_leds,
        [],
        {"power": p_uv, "uni_all": u_uv_all, "uni_cen": u_uv_cen},
        "UV Light Only",
        led_geom_params=led_geom,
    )

    # 2. White Plot
    led_sim.plot_irradiance_map(
        axes[1],
        white_illum_map_ref,
        extent,
        plot_params,
        [],
        white_leds,
        {"power": p_wh, "uni_all": u_wh_all, "uni_cen": u_wh_cen},
        "White Light Only",
        led_geom_params=led_geom,
    )

    # 3. Combined Plot
    # Don't need this one because we don't use the combined illumination simulatenously, moreover, it could be a littlebit helpful if the colormap of the radiance field was different, but nevermind. -from me
    # combined_map = uv_map + white_illum_map_ref
    # # Combined 통계
    # p_comb, u_comb_all, u_comb_cen, _ = led_sim.analyze_roi(
    #     X,
    #     Y,
    #     combined_map,
    #     config["optics"]["roi_width_mm"],
    #     config["optics"]["roi_height_mm"],
    #     config["led_specs"]["single_led_total_power_mw"],
    #     len(uv_leds) + len(white_leds),
    #     config["optics"]["center_roi_ratio"],
    # )

    # led_sim.plot_irradiance_map(
    #     axes[2],
    #     combined_map,
    #     extent,
    #     plot_params,
    #     uv_leds,
    #     white_leds,
    #     {"power": p_comb, "uni_all": u_comb_all, "uni_cen": u_comb_cen},
    #     "Combined (UV + White)",
    #     led_geom_params=led_geom,
    # )

    # plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{timestamp}_result_maps.png"))
    plt.show()


# ======================================================================================
# 5. 스크립트 실행
# ======================================================================================
if __name__ == "__main__":
    config_data = load_config("config.yaml")
    sim_params = setup_parameters(config_data)
    results, white_map_ref = run_optimization_sweep(sim_params)
    analyze_and_plot_results(results, white_map_ref, sim_params)
