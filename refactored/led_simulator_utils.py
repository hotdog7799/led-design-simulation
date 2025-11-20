# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d

# ======================================================================================
# 1. 핵심 물리 및 배치 함수
# ======================================================================================


def get_beam_profile_func(angles, intensities):
    """
    데이터시트의 방사각 프로파일을 기반으로 1D 보간 함수를 생성합니다.
    """
    f = interp1d(angles, intensities, kind="linear", fill_value=0.0, bounds_error=False)
    return f


def create_4_symmetric_uv_layout(a, b, cavity_w, cavity_h, min_separation):
    """
    (±a, ±b) 대칭 위치에 4개의 UV LED를 배치하고 유효성을 검사합니다.
    (Cavity 침범, 최소 간격)
    """
    positions = [
        np.array([a, b, 0]),
        np.array([-a, b, 0]),
        np.array([-a, -b, 0]),
        np.array([a, -b, 0]),
    ]
    for x, y, z in positions:
        if abs(x) < cavity_w / 2.0 and abs(y) < cavity_h / 2.0:
            return []  # Cavity 침범 시 무효
    dist1 = np.linalg.norm(positions[0] - positions[1])  # 2a
    dist2 = np.linalg.norm(positions[0] - positions[3])  # 2b
    if dist1 < min_separation or dist2 < min_separation:
        return []  # 최소 간격 위반 시 무효
    return positions


def place_white_leds_fixed_xaxis(
    uv_led_positions, x_white, num_white_target, min_dist, w_crosstalk
):
    """
    X축 고정 위치에 White LED를 배치하고 Crosstalk 페널티를 계산합니다.
    """
    candidate_slots = [np.array([x_white, 0, 0]), np.array([-x_white, 0, 0])]
    selected_white_positions = []
    placement_successful = True

    for slot in candidate_slots:
        slot_ok = True
        for uv_pos in uv_led_positions:
            if np.linalg.norm(slot - uv_pos) < min_dist:
                slot_ok = False
                break
        if slot_ok:
            selected_white_positions.append(slot)

    if len(selected_white_positions) < num_white_target:
        placement_successful = False
        selected_white_positions = []

    selected_white_positions = selected_white_positions[:num_white_target]
    crosstalk_penalty = 0.0 if placement_successful else w_crosstalk
    return selected_white_positions, crosstalk_penalty


def simulate_illumination(
    led_positions, beam_profile_func, target_z, grid_size, resolution
):
    """
    (a,b) 좌표 기반 4개 UV LED 배치 및 유효성 검사 함수
    """
    x_coords = np.linspace(-grid_size / 2, grid_size / 2, resolution)
    y_coords = np.linspace(-grid_size / 2, grid_size / 2, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)
    total_illum_map = np.zeros_like(X)

    for led_pos in led_positions:
        vectors_to_led = led_pos - np.stack([X, Y, np.full_like(X, target_z)], axis=-1)
        distances = np.linalg.norm(vectors_to_led, axis=-1)
        distances = np.where(distances == 0, 1e-9, distances)
        cos_theta = (target_z - led_pos[2]) / distances
        angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
        beam_intensity = beam_profile_func(angle_degrees)
        intensity_contribution = (beam_intensity * cos_theta) / (distances**2)
        total_illum_map += intensity_contribution
    return X, Y, total_illum_map


def analyze_roi(X, Y, illum_map, roi_w, roi_h, effective_single_led_power_mw, num_leds):
    """
    ROI 영역 분석: 총 파워(mW), 균일도, 스케일 팩터 계산
    """
    roi_mask = (np.abs(X) <= roi_w / 2) & (np.abs(Y) <= roi_h / 2)
    roi_illum = np.where(roi_mask, illum_map, np.nan)
    if np.all(np.isnan(roi_illum)):
        return 0, 0, 0.0

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    pixel_area_mm2 = dx * dy
    sim_total_integral = np.sum(illum_map[~np.isnan(illum_map)]) * pixel_area_mm2
    total_led_power_mw_reference = effective_single_led_power_mw * num_leds
    scale_factor = (
        (total_led_power_mw_reference / sim_total_integral)
        if sim_total_integral > 0
        else 0.0
    )
    roi_power_mw = np.nansum(roi_illum) * pixel_area_mm2 * scale_factor
    uniformity = (
        np.nanmin(roi_illum) / np.nanmax(roi_illum) if np.nanmax(roi_illum) > 0 else 0
    )
    return roi_power_mw, uniformity, scale_factor


def calculate_loss(power_roi, uniformity_roi, crosstalk_penalty, loss_weights):
    """
    정의된 가중치를 사용하여 Loss 값을 계산합니다.
    """
    power_penalty = max(0, loss_weights["TARGET_POWER_MW"] - power_roi)
    uniformity_penalty = 1.0 - uniformity_roi
    loss = (
        loss_weights["W_POWER"] * power_penalty
        + loss_weights["W_UNIFORMITY"] * uniformity_penalty
        + crosstalk_penalty
    )  # W_CROSSTALK는 place_white_leds 함수에서 이미 적용됨
    return loss, power_penalty, uniformity_penalty


# ======================================================================================
# 2. 재사용 가능한 플로팅 함수
# ======================================================================================


def plot_irradiance_map(
    ax, map_data, extent_mm, config_params, led_positions_uv, led_positions_white, stats
):
    """
    주어진 Matplotlib 축(ax)에 2D 조도 맵과 오버레이를 그립니다.
    """
    im = ax.imshow(map_data, extent=extent_mm, cmap="inferno", origin="lower")

    # ROI 영역
    roi_rect = patches.Rectangle(
        (-config_params["roi_w"] / 2, -config_params["roi_h"] / 2),
        config_params["roi_w"],
        config_params["roi_h"],
        linewidth=2,
        edgecolor="cyan",
        facecolor="none",
        linestyle="--",
        label=f'ROI (Pwr={stats["power"]:.2f}mW, Uni={stats["uniformity"]:.3f})',
    )
    ax.add_patch(roi_rect)

    # Cavity 영역
    cavity_rect = patches.Rectangle(
        (-config_params["cavity_w"] / 2, -config_params["cavity_h"] / 2),
        config_params["cavity_w"],
        config_params["cavity_h"],
        linewidth=1.5,
        edgecolor="gray",
        facecolor="none",
        linestyle=":",
        label=f"Camera Cavity: {config_params['cavity_h']}x{config_params['cavity_w']}",
    )
    ax.add_patch(cavity_rect)

    # LED 위치
    if led_positions_uv:
        uv_x = [p[0] for p in led_positions_uv]
        uv_y = [p[1] for p in led_positions_uv]
        ax.scatter(
            uv_x,
            uv_y,
            c="w",
            marker="o",
            s=40,
            label=f"UV LEDs ({len(led_positions_uv)})",
        )
    if led_positions_white:
        wh_x = [p[0] for p in led_positions_white]
        wh_y = [p[1] for p in led_positions_white]
        ax.scatter(
            wh_x,
            wh_y,
            c="lime",
            marker="s",
            s=40,
            label=f"White LEDs ({len(led_positions_white)})",
        )

    ax.set_title(f"Irradiance Map (a={stats['a']:.2f}, b={stats['b']:.2f})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.legend()
    return im


def plot_line_profile(
    ax, x_data, y_data_normalized, title, xlabel, ylabel, roi_boundaries_mm
):
    """
    주어진 Matplotlib 축(ax)에 정규화된 1D 라인 플롯을 그립니다.
    """
    if len(x_data) > 0 and len(y_data_normalized) > 0:
        ax.plot(x_data, y_data_normalized, marker=".", linestyle="-")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.1)
        ax.grid(True)
        ax.axvline(
            roi_boundaries_mm[0], color="c", linestyle="--", label="ROI Boundary"
        )
        ax.axvline(roi_boundaries_mm[1], color="c", linestyle="--")
        # 극솟값 계산 및 점선 추가
        min_y = min(y_data_normalized)
        ax.axhline(min_y, color="m", linestyle="--", label=f"Min Value: {min_y:.3f}")
        ax.text(
            x_data[0],  # 그래프 왼쪽에 표시
            min_y,
            f"{min_y:.3f}",
            color="m",
            va="bottom",
            ha="left",
            fontsize=9,
        )

        ax.legend()
    else:
        ax.set_title(f"{title} - No Data in ROI")
