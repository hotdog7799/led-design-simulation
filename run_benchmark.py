# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import run_optimization as opt_main # 기존 설정 로드 함수 재사용
import led_simulator_utils as led_sim

# ==============================================================================
# 사용자 정의 프리셋 (Benchmarks)
# ==============================================================================
presets = {
    "Original_Design": {
        "uv": [np.array([1.53, 5.01, 0]), np.array([-1.53, 5.01, 0]), 
               np.array([-1.53, -5.01, 0]), np.array([1.53, -5.01, 0])],
        "white": [np.array([6.50, 0, 0]), np.array([-6.50, 0, 0])]
    },
    "Another_Design_1": {
        "uv": [np.array([4.43, 1.25, 0]), np.array([-1.55, 5.05, 0]), 
               np.array([-4.43, -1.25, 0]), np.array([1.55, -5.05, 0])],
        "white": [np.array([1.55, 5.05, 0]), np.array([-4.43, 1.25, 0]), 
                  np.array([-1.55, -5.05, 0]), np.array([4.43, -1.25, 0])]
    }
}

def evaluate_presets():
    print("=== Starting Benchmark Evaluation ===")
    
    # 1. 설정 로드 (run_optimization.py의 함수 재사용)
    config = opt_main.load_config("config.yaml")
    params = opt_main.setup_parameters(config)
    
    beam_func = params["beam_func"]
    actual_dist = params["actual_dist"]
    grid_size = config["simulation"]["grid_size_mm"]
    resolution = config["simulation"]["resolution"]
    roi_w = config["optics"]["roi_width_mm"]
    roi_h = config["optics"]["roi_height_mm"]
    center_ratio = config["optics"]["center_roi_ratio"]
    single_power_mw = config["led_specs"]["single_led_total_power_mw"]
    loss_weights = config["loss_weights"]
    
    led_geom = {
        "led_width_mm": config["led_specs"]["led_width_mm"],
        "led_height_mm": config["led_specs"]["led_height_mm"],
        "subsample_x": config["led_specs"]["subsample_x"],
        "subsample_y": config["led_specs"]["subsample_y"],
    }
    
    results = []
    output_dir = "./plot_result/benchmark"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")

    # 2. 프리셋 순회 및 평가
    for name, leds in presets.items():
        print(f"\nEvaluating: {name} ...")
        uv_pos = leds["uv"]
        white_pos = leds["white"]
        
        # 시뮬레이션
        X, Y, uv_map = led_sim.simulate_illumination(uv_pos, beam_func, actual_dist, grid_size, resolution, led_geom_params=led_geom)
        X, Y, white_map = led_sim.simulate_illumination(white_pos, beam_func, actual_dist, grid_size, resolution, led_geom_params=led_geom)
        
        # 분석
        p_uv, cp_uv, u_uv, uc_uv, _ = led_sim.analyze_roi(X, Y, uv_map, roi_w, roi_h, single_power_mw, len(uv_pos), center_ratio)
        p_wh, cp_wh, u_wh, uc_wh, _ = led_sim.analyze_roi(X, Y, white_map, roi_w, roi_h, single_power_mw, len(white_pos), center_ratio)
        
        # Loss 계산
        total_loss, p_loss, cp_loss, u_loss = led_sim.calculate_loss(p_uv, cp_uv, u_uv, uc_uv, u_wh, loss_weights, center_ratio)
        
        print(f"  -> Total Loss: {total_loss:.4f}")
        print(f"  -> UV Power: {p_uv:.2f}mW (Center: {cp_uv:.2f}mW), Uniformity: {u_uv:.3f}")
        
        results.append({
            "Name": name,
            "Total_Loss": total_loss,
            "Pwr_UV": p_uv, "Pwr_UV_Cen": cp_uv, "Uni_UV": u_uv,
            "Uni_White": u_wh
        })
        
        # 플로팅 (결과 저장)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plot_params = {
            "roi_w": roi_w, "roi_h": roi_h,
            "cavity_w": params["cavity_w"], "cavity_h": params["cavity_h"],
            "center_ratio": center_ratio
        }
        extent = [-grid_size/2, grid_size/2, -grid_size/2, grid_size/2]
        
        led_sim.plot_irradiance_map(axes[0], uv_map, extent, plot_params, uv_pos, [], 
                                    {"power": p_uv, "uni_all": u_uv, "uni_cen": uc_uv}, f"{name} - UV", led_geom_params=led_geom)
        led_sim.plot_irradiance_map(axes[1], white_map, extent, plot_params, [], white_pos,
                                    {"power": p_wh, "uni_all": u_wh, "uni_cen": uc_wh}, f"{name} - White", led_geom_params=led_geom)
        
        combined_map = uv_map + white_map
        p_comb, _, u_comb, uc_comb, _ = led_sim.analyze_roi(X, Y, combined_map, roi_w, roi_h, single_power_mw, len(uv_pos)+len(white_pos), center_ratio)
        
        led_sim.plot_irradiance_map(axes[2],
         combined_map,
          extent,
          plot_params,
           uv_pos,
            white_pos,
                                    {"power": p_comb, "uni_all": u_comb, "uni_cen": uc_comb}, f"{name} - Combined", led_geom_params=led_geom)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{timestamp}_{name}.png"))
        plt.close()

    # 결과 CSV 저장
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{timestamp}_benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark completed. Results saved to {csv_path}")
    print(df)

if __name__ == "__main__":
    evaluate_presets()