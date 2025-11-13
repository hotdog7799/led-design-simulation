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
def run_optimization_sweep(params):
    """
    (a, b) 좌표의 모든 조합을 탐색하여 Loss를 계산하고 결과를 로깅합니다.
    """
    # 파라미터 풀기
    config = params["config"]
    beam_func = params["beam_func"]
    actual_dist = params["actual_dist"]
    cavity_w = params["cavity_w"]
    cavity_h = params["cavity_h"]
    param_combinations = params["param_combinations"]

    # 설정값
    roi_w = config["optics"]["roi_width_mm"]
    roi_h = config["optics"]["roi_height_mm"]
    num_uv = config["led_specs"]["num_uv_leds"]
    num_white = config["white_led"]["num_to_place"]
    min_sep = config["led_specs"]["min_led_separation_mm"]
    white_x_offset = config["white_led"]["x_offset_mm"]
    min_crosstalk_dist = config["white_led"]["crosstalk_min_distance_mm"]
    single_power_mw = config["led_specs"]["single_led_total_power_mw"]
    loss_weights = config["loss_weights"]
    grid_size = config["simulation"]["grid_size_mm"]
    resolution = config["simulation"]["resolution"]

    results_list = []
    total_combinations = len(param_combinations)
    print(f"--- Starting Optimization Search ({total_combinations} combinations) ---")
    start_time_total = datetime.now()

    for idx, (a, b) in enumerate(param_combinations):
        uv_led_positions = led_sim.create_4_symmetric_uv_layout(
            a, b, cavity_w, cavity_h, min_sep
        )
        if not uv_led_positions:
            # print(f"({idx+1}/{total_combinations}) a={a:.2f}, b={b:.2f} -> Skipped (Invalid UV Layout)")
            continue

        white_led_positions, crosstalk_penalty = led_sim.place_white_leds_fixed_xaxis(
            uv_led_positions,
            white_x_offset,
            num_white,
            min_crosstalk_dist,
            loss_weights["W_CROSSTALK"],
        )
        crosstalk_occurred = crosstalk_penalty > 0

        if crosstalk_occurred:
            power_on_roi, uniformity_roi = 0.0, 0.0
            print_status = "No!"
            total_loss, power_pen, unif_pen = led_sim.calculate_loss(
                power_on_roi, uniformity_roi, crosstalk_penalty, loss_weights
            )
        else:
            X_mm, Y_mm, illum_map_arb = led_sim.simulate_illumination(
                uv_led_positions, beam_func, actual_dist, grid_size, resolution
            )
            power_on_roi, uniformity_roi, scale_factor = led_sim.analyze_roi(
                X_mm, Y_mm, illum_map_arb, roi_w, roi_h, single_power_mw, num_uv
            )
            total_loss, power_pen, unif_pen = led_sim.calculate_loss(
                power_on_roi, uniformity_roi, crosstalk_penalty, loss_weights
            )
            print_status = "Yes"

        results_list.append(
            {
                "a_mm": round(a, 2),
                "b_mm": round(b, 2),
                "Num_UV_LEDs": num_uv,
                "Num_White_LEDs": num_white,
                "Power_ROI_mW": round(power_on_roi, 2),
                "Uniformity_ROI": round(uniformity_roi, 3),
                "White_LED_Placeable": not crosstalk_occurred,
                "Total_Loss": round(total_loss, 2),
                "Power_Penalty": round(power_pen, 2),
                "Uniformity_Penalty": round(unif_pen, 3),
                "Crosstalk_Penalty": int(crosstalk_penalty),
            }
        )
        print(
            f"({idx+1}/{total_combinations}) a={a:.2f}, b={b:.2f} -> Pwr={power_on_roi:.2f}, Uni={uniformity_roi:.3f}, Wht OK?={print_status}, Loss={total_loss:.2f}"
        )

    end_time_total = datetime.now()
    print("-" * 60)
    print(
        f"Optimization Search Finished. Total time: {end_time_total - start_time_total}"
    )

    return results_list


# ======================================================================================
# 3. 결과 분석 및 시각화
# ======================================================================================
def analyze_and_plot_results(results_list, params):
    """결과를 로깅하고 최적의 배치를 시각화합니다."""

    config = params["config"]

    if not results_list:
        print("\nNo valid configurations found in the specified range.")
        return

    results_df = pd.DataFrame(results_list)
    results_df_sorted = results_df.sort_values(by="Total_Loss").reset_index(drop=True)

    log_file = config["logging"]["results_file"]
    results_df_sorted.to_csv(log_file, index=False)
    print(f"\n✅ Optimization results logged to '{log_file}'")

    valid_configs = results_df_sorted[
        (results_df_sorted["White_LED_Placeable"] == True)
        & (results_df_sorted["Power_Penalty"] == 0.0)
    ]

    if valid_configs.empty:
        print(
            "\n--- No configurations met all constraints. Showing best overall results: ---"
        )
        print(results_df_sorted.head(5).to_string())
        best_config = results_df_sorted.iloc[0]
    else:
        print("\n--- Top 5 Valid Configurations ---")
        print(valid_configs.head(5).to_string())
        best_config = valid_configs.iloc[0]

    print(
        f"\n--- Visualizing Selected Best Configuration (Loss={best_config['Total_Loss']:.2f}) ---"
    )
    best_a = best_config["a_mm"]
    best_b = best_config["b_mm"]

    # 시각화를 위해 파라미터 다시 준비
    best_uv_leds = led_sim.create_4_symmetric_uv_layout(
        best_a,
        best_b,
        params["cavity_w"],
        params["cavity_h"],
        config["led_specs"]["min_led_separation_mm"],
    )
    best_white_leds, _ = led_sim.place_white_leds_fixed_xaxis(
        best_uv_leds,
        config["white_led"]["x_offset_mm"],
        config["white_led"]["num_to_place"],
        config["white_led"]["crosstalk_min_distance_mm"],
        0,  # Loss는 0으로 전달
    )

    if not best_uv_leds:
        print("Error: Could not generate layout for the best configuration.")
        return

    # 최적의 결과에 대해 시뮬레이션 다시 실행 (정확한 맵 데이터 확보)
    X_best, Y_best, illum_best_arb = led_sim.simulate_illumination(
        best_uv_leds,
        params["beam_func"],
        params["actual_dist"],
        config["simulation"]["grid_size_mm"],
        config["simulation"]["resolution"],
    )
    power_best, uniformity_best, scale_factor_best = led_sim.analyze_roi(
        X_best,
        Y_best,
        illum_best_arb,
        config["optics"]["roi_width_mm"],
        config["optics"]["roi_height_mm"],
        config["led_specs"]["single_led_total_power_mw"],
        len(best_uv_leds),
    )
    irradiance_map_best_W_m2 = (
        illum_best_arb
        * scale_factor_best
        * (config["simulation"]["mm_per_m"] ** 2 / 1000.0)
    )

    # 1. 2D 조도 맵 표시
    fig_2d, ax_2d = plt.subplots(figsize=(10, 8))
    plot_params = {
        "roi_w": config["optics"]["roi_width_mm"],
        "roi_h": config["optics"]["roi_height_mm"],
        "cavity_w": params["cavity_w"],
        "cavity_h": params["cavity_h"],
    }
    plot_stats = {
        "power": power_best,
        "uniformity": uniformity_best,
        "a": best_a,
        "b": best_b,
    }
    im = led_sim.plot_irradiance_map(
        ax_2d,
        irradiance_map_best_W_m2,
        [
            -config["simulation"]["grid_size_mm"] / 2,
            config["simulation"]["grid_size_mm"] / 2,
            -config["simulation"]["grid_size_mm"] / 2,
            config["simulation"]["grid_size_mm"] / 2,
        ],
        plot_params,
        best_uv_leds,
        best_white_leds,
        plot_stats,
    )
    cbar = fig_2d.colorbar(im, ax=ax_2d, fraction=0.046, pad=0.04)
    cbar.set_label("Irradiance (W/m^2)")
    # plt.savefig(config["logging"]["best_config_plot_file"])
    config_plots_file = os.path.join(
        output_dir,
        f"{timestamp}_{os.path.basename(config['logging']['best_config_plot_file'])}",
    )
    plt.savefig(config_plots_file)
    plt.show()

    # 2. 1D 라인 플롯 (X=0, Y=0, 대각선)
    fig_lines, (ax_x, ax_y, ax_diag) = plt.subplots(1, 3, figsize=(24, 7))

    # 데이터 준비
    resolution = config["simulation"]["resolution"]
    grid_size = config["simulation"]["grid_size_mm"]
    roi_w = config["optics"]["roi_width_mm"]
    roi_h = config["optics"]["roi_height_mm"]

    # Y=0 데이터
    y0_data_arb = illum_best_arb[resolution // 2, :]
    x0_coords = X_best[resolution // 2, :]
    x0_mask = np.abs(x0_coords) <= roi_w / 2
    x0_roi_coords = x0_coords[x0_mask]
    y0_roi_data = y0_data_arb[x0_mask]
    if len(y0_roi_data) > 0:
        y0_roi_norm = y0_roi_data / np.nanmax(y0_roi_data)
    else:
        y0_roi_norm = []

    # X=0 데이터
    x0_data_arb = illum_best_arb[:, resolution // 2]
    y0_coords = Y_best[:, resolution // 2]
    y0_mask = np.abs(y0_coords) <= roi_h / 2
    y0_roi_coords = y0_coords[y0_mask]
    x0_roi_data = x0_data_arb[y0_mask]
    if len(x0_roi_data) > 0:
        x0_roi_norm = x0_roi_data / np.nanmax(x0_roi_data)
    else:
        x0_roi_norm = []

    # 대각선 데이터
    diag_data_arb = illum_best_arb.diagonal()
    x_diag = np.linspace(-grid_size / 2, grid_size / 2, resolution)
    diag_mask = np.abs(x_diag) <= roi_w / 2
    x_diag_roi = x_diag[diag_mask]
    diag_data_roi = diag_data_arb[diag_mask]
    if len(diag_data_roi) > 0:
        diag_data_roi_norm = diag_data_roi / np.nanmax(diag_data_roi)
    else:
        diag_data_roi_norm = []

    # 플로팅
    led_sim.plot_line_profile(
        ax_x,
        x0_roi_coords,
        y0_roi_norm,
        f"Normalized Y=0 (a={best_a:.2f}, b={best_b:.2f})",
        "X Position (mm)",
        "Norm. Irradiance",
        [-roi_w / 2, roi_w / 2],
    )
    led_sim.plot_line_profile(
        ax_y,
        y0_roi_coords,
        x0_roi_norm,
        f"Normalized X=0 (a={best_a:.2f}, b={best_b:.2f})",
        "Y Position (mm)",
        "Norm. Irradiance",
        [-roi_h / 2, roi_h / 2],
    )
    led_sim.plot_line_profile(
        ax_diag,
        x_diag_roi,
        diag_data_roi_norm,
        f"Normalized Diagonal (a={best_a:.2f}, b={best_b:.2f})",
        "Approx. X Position (mm)",
        "Norm. Irradiance",
        [-roi_w / 2, roi_w / 2],
    )

    plt.tight_layout()
    line_plots_file = os.path.join(
        output_dir,
        f"{timestamp}_{os.path.basename(config['logging']['line_plots_file'])}",
    )
    # plt.savefig(config["logging"]["line_plots_file"])
    plt.savefig(line_plots_file)
    plt.show()


# ======================================================================================
# 5. 스크립트 실행
# ======================================================================================
if __name__ == "__main__":
    # 1. 설정 로드 및 파라미터 준비
    config_data = load_config("config.yaml")
    simulation_params = setup_parameters(config_data)

    # 2. 최적화 탐색 실행
    all_results = run_optimization_sweep(simulation_params)

    # 3. 결과 분석 및 최종 시각화
    analyze_and_plot_results(all_results, simulation_params)
