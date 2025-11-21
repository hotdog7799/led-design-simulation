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
        selected_white_positions = []  # 하나라도 안되면 배치 실패 처리

    selected_white_positions = selected_white_positions[:num_white_target]
    crosstalk_penalty = 0.0 if placement_successful else w_crosstalk
    return selected_white_positions, crosstalk_penalty


def generate_sub_points(center_pos, w, h, nx, ny):
    """
    LED 중심 좌표를 기준으로 면 광원을 모사하는 sub-point들의 좌표 리스트를 생성합니다.
    (면 광원 로직 핵심 함수)
    """
    if nx <= 1 and ny <= 1:
        return [center_pos]  # 점 광원

    # LED 표면 내 격자 생성
    xs = np.linspace(-w / 2.0, w / 2.0, nx)
    ys = np.linspace(-h / 2.0, h / 2.0, ny)

    sub_points = []
    for dx in xs:
        for dy in ys:
            offset = np.array([dx, dy, 0.0])
            sub_points.append(center_pos + offset)

    return sub_points


def simulate_illumination(
    led_positions,
    beam_profile_func,
    target_z,
    grid_size,
    resolution,
    led_geom_params=None,  # [중요] 이 인자가 있어야 에러가 나지 않습니다.
):
    """
    (a,b) 좌표 기반 LED 배치에 따른 조도 분포를 시뮬레이션합니다.
    led_geom_params가 제공되면 면 광원(Area Source) 모델을 사용합니다.
    """
    x_coords = np.linspace(-grid_size / 2, grid_size / 2, resolution)
    y_coords = np.linspace(-grid_size / 2, grid_size / 2, resolution)
    X, Y = np.meshgrid(x_coords, y_coords)

    target_points = np.stack([X, Y, np.full_like(X, target_z)], axis=-1)
    total_illum_map = np.zeros_like(X)

    # 면 광원 파라미터 추출
    if led_geom_params:
        w = led_geom_params.get("led_width_mm", 0)
        h = led_geom_params.get("led_height_mm", 0)
        nx = led_geom_params.get("subsample_x", 1)
        ny = led_geom_params.get("subsample_y", 1)
        num_sub_points = nx * ny
    else:
        w, h, nx, ny = 0, 0, 1, 1
        num_sub_points = 1

    for led_center_pos in led_positions:
        # 이 LED를 구성하는 점 광원들(Sub-LEDs) 생성
        sub_points = generate_sub_points(led_center_pos, w, h, nx, ny)

        # 단일 LED에 대한 조도 누적 (나중에 평균 냄)
        single_led_map = np.zeros_like(X)

        for sp in sub_points:
            vectors_to_led = sp - target_points  # (H, W, 3)
            dist_sq = np.sum(vectors_to_led**2, axis=-1)
            distances = np.sqrt(dist_sq)
            distances = np.where(distances == 0, 1e-9, distances)

            dz = target_z - sp[2]
            cos_theta = dz / distances

            angle_degrees = np.degrees(np.arccos(np.clip(cos_theta, 0, 1)))
            beam_intensity = beam_profile_func(angle_degrees)

            intensity_contribution = (beam_intensity * cos_theta) / dist_sq
            single_led_map += intensity_contribution

        # 면 광원 효과: N개의 점으로 나눴으니 총 파워 보존을 위해 N으로 나눔
        total_illum_map += single_led_map / num_sub_points

    return X, Y, total_illum_map


def analyze_roi(
    X,
    Y,
    illum_map,
    roi_w,
    roi_h,
    effective_single_led_power_mw,
    num_leds,
    center_ratio=0.5,
):
    """
    ROI 전체 및 중앙부(Center) 통계를 계산합니다.
    """
    roi_mask = (np.abs(X) <= roi_w / 2) & (np.abs(Y) <= roi_h / 2)
    roi_vals = illum_map[roi_mask]

    if roi_vals.size == 0 or np.all(np.isnan(roi_vals)):
        return 0, 0.0, 0.0, 0.0

    dx = X[0, 1] - X[0, 0]
    dy = Y[1, 0] - Y[0, 0]
    pixel_area_mm2 = dx * dy

    sim_total_integral = np.sum(illum_map) * pixel_area_mm2
    total_led_power_mw_reference = effective_single_led_power_mw * num_leds
    scale_factor = (
        (total_led_power_mw_reference / sim_total_integral)
        if sim_total_integral > 0
        else 0.0
    )

    roi_power_mw = np.sum(roi_vals) * pixel_area_mm2 * scale_factor

    max_val = np.max(roi_vals)
    min_val = np.min(roi_vals)
    uni_overall = (min_val / max_val) if max_val > 0 else 0.0

    # 중앙부 ROI
    center_w = roi_w * center_ratio
    center_h = roi_h * center_ratio
    center_mask = (np.abs(X) <= center_w / 2) & (np.abs(Y) <= center_h / 2)
    center_vals = illum_map[center_mask]

    if center_vals.size > 0:
        c_max = np.max(center_vals)
        c_min = np.min(center_vals)
        uni_center = (c_min / c_max) if c_max > 0 else 0.0
    else:
        uni_center = 0.0

    return roi_power_mw, uni_overall, uni_center, scale_factor


def calculate_loss(
    power_roi, uni_overall, uni_center, uni_white, crosstalk_penalty, weights
):
    """
    전체, 중앙, 화이트 균일도를 모두 고려하여 Loss를 계산합니다.
    """
    power_penalty = max(0, weights["TARGET_POWER_MW"] - power_roi)

    pen_uni_overall = 1.0 - uni_overall
    pen_uni_center = 1.0 - uni_center
    pen_uni_white = 1.0 - uni_white

    loss = (
        weights["W_POWER"] * power_penalty
        + weights["W_UNIFORMITY_OVERALL"] * pen_uni_overall
        + weights["W_UNIFORMITY_CENTER"] * pen_uni_center
        + weights["W_UNIFORMITY_WHITE"] * pen_uni_white
        + crosstalk_penalty
    )

    return loss, power_penalty, pen_uni_overall, pen_uni_center


# ======================================================================================
# 2. 재사용 가능한 플로팅 함수
# ======================================================================================


def plot_irradiance_map(
    ax,
    map_data,
    extent_mm,
    config_params,
    led_positions_uv,
    led_positions_white,
    stats,
    title_prefix="",
):
    """
    주어진 Matplotlib 축(ax)에 2D 조도 맵과 오버레이를 그립니다.
    """
    im = ax.imshow(map_data, extent=extent_mm, cmap="inferno", origin="lower")

    # 전체 ROI
    roi_rect = patches.Rectangle(
        (-config_params["roi_w"] / 2, -config_params["roi_h"] / 2),
        config_params["roi_w"],
        config_params["roi_h"],
        linewidth=2,
        edgecolor="cyan",
        facecolor="none",
        linestyle="--",
        label="Full ROI",
    )
    ax.add_patch(roi_rect)

    # 중앙부 ROI
    center_w = config_params["roi_w"] * config_params["center_ratio"]
    center_h = config_params["roi_h"] * config_params["center_ratio"]
    center_rect = patches.Rectangle(
        (-center_w / 2, -center_h / 2),
        center_w,
        center_h,
        linewidth=1.5,
        edgecolor="yellow",
        facecolor="none",
        linestyle="-.",
        label="Center ROI",
    )
    ax.add_patch(center_rect)

    # Cavity
    cavity_rect = patches.Rectangle(
        (-config_params["cavity_w"] / 2, -config_params["cavity_h"] / 2),
        config_params["cavity_w"],
        config_params["cavity_h"],
        linewidth=1.0,
        edgecolor="gray",
        facecolor="none",
        linestyle=":",
    )
    ax.add_patch(cavity_rect)

    # LED Positions
    if led_positions_uv:
        uv_x = [p[0] for p in led_positions_uv]
        uv_y = [p[1] for p in led_positions_uv]
        ax.scatter(uv_x, uv_y, c="w", marker="o", s=30, label="UV")

    if led_positions_white:
        wh_x = [p[0] for p in led_positions_white]
        wh_y = [p[1] for p in led_positions_white]
        ax.scatter(wh_x, wh_y, c="lime", marker="s", s=30, label="White")

    if stats:
        title = f"{title_prefix}\nPwr={stats.get('power',0):.1f}mW, Uni(All)={stats.get('uni_all',0):.2f}, Uni(Cen)={stats.get('uni_cen',0):.2f}"
    else:
        title = title_prefix

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (mm)")
    ax.legend(loc="upper right", fontsize=8)
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
        min_y = min(y_data_normalized)
        ax.axhline(min_y, color="m", linestyle="--", label=f"Min Value: {min_y:.3f}")
        ax.text(
            x_data[0],
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
