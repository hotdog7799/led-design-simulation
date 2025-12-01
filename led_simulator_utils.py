# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import interp1d

# ======================================================================================
# 1. 핵심 물리 및 배치 함수
# ======================================================================================
# [NEW] White LED 동적 배치 알고리즘
def optimize_white_leds_radial(
    uv_led_positions, cavity_w, cavity_h, num_white_target, min_crosstalk_dist, w_crosstalk
):
    """
    Cavity 주변을 360도 스캔하여 UV LED와 간섭이 없는 '명당'을 찾아 White LED를 배치합니다.
    """
    # 탐색 궤도: Cavity보다 1mm 정도 여유를 둔 타원형/원형 궤도
    margin = 1.0 
    rx = (cavity_w / 2.0) + margin
    ry = (cavity_h / 2.0) + margin
    
    # 1도 단위로 후보 위치 스캔
    angles = np.deg2rad(np.linspace(0, 360, 360, endpoint=False))
    valid_positions = []
    
    for theta in angles:
        # 타원 궤도상의 좌표 계산
        x = rx * np.cos(theta)
        y = ry * np.sin(theta)
        pos = np.array([x, y, 0])
        
        # Crosstalk 거리 검사
        distances = [np.linalg.norm(pos - uv) for uv in uv_led_positions]
        min_dist = min(distances) if distances else 100.0
        
        if min_dist >= min_crosstalk_dist:
            # (위치, UV와의 거리) 저장 -> 멀리 떨어질수록 안전하지만, 너무 멀면 광량 손해.
            # 여기서는 '안전하기만 하면' 일단 후보로 등록
            valid_positions.append(pos)
    
    # 배치 로직: 최대한 서로 멀리 떨어지게 배치 (Greedy 방식)
    selected_positions = []
    if not valid_positions:
        return [], w_crosstalk # 배치 실패
        
    # 첫 번째는 임의로(또는 UV 사이 가장 넓은 공간에) 배치할 수 있으나,
    # 간단히 0도에 가장 가까운 유효 위치부터 시작
    # (더 정교한 알고리즘 가능하지만 시뮬레이션 속도 고려)
    # 여기서는 단순히 리스트에서 간격을 두며 뽑습니다.
    
    # valid_positions 리스트에서 등간격으로 뽑아내기 시도
    if len(valid_positions) < num_white_target:
        # 후보가 목표보다 적으면 있는 거라도 다 넣음
        selected_positions = valid_positions
    else:
        # K-Means나 복잡한 로직 대신, 각도 기준으로 4분면 등을 고려해 뽑는게 좋음.
        # 가장 간단하고 효과적인 방법: 
        # 후보군 중 하나 뽑고 -> 그거랑 먼거 뽑고 -> 반복
        import random
        # 시뮬레이션 재현성을 위해 시드 고정 혹은 결정론적 방법 사용 권장
        # 여기서는 가장 단순하게: 
        # 4개 목표라면 0, 90, 180, 270도 근처의 유효 좌표를 우선 탐색
        target_angles = np.linspace(0, 2*np.pi, num_white_target, endpoint=False)
        
        for target_th in target_angles:
            # 타겟 각도와 가장 가까운 유효 후보 찾기
            best_candidate = None
            min_ang_diff = 100.0
            
            target_pos = np.array([rx * np.cos(target_th), ry * np.sin(target_th), 0])
            
            for vp in valid_positions:
                # 이미 선택된 애들과 너무 가까우면(겹치면) 패스
                if any(np.linalg.norm(vp - sp) < 2.0 for sp in selected_positions): 
                    continue
                
                dist = np.linalg.norm(vp - target_pos)
                if dist < min_ang_diff: # 좌표 거리로 근사
                    min_ang_diff = dist
                    best_candidate = vp
            
            if best_candidate is not None:
                selected_positions.append(best_candidate)

    # 개수 부족 시 페널티
    penalty = 0.0
    if len(selected_positions) < num_white_target:
        # 목표 개수보다 적으면 개당 페널티 부과 (강력하게)
        penalty = w_crosstalk * (num_white_target - len(selected_positions))

    return selected_positions, penalty

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
        # [핵심] 중앙부 절대 파워 (mW)
        center_power_mw = np.sum(center_vals) * pixel_area_mm2 * scale_factor
        
        c_max = np.max(center_vals)
        c_min = np.min(center_vals)
        uni_center = (c_min / c_max) if c_max > 0 else 0.0
    else:
        center_power_mw = 0.0
        uni_center = 0.0

    return roi_power_mw,center_power_mw, uni_overall, uni_center, scale_factor


def calculate_loss(power_roi, center_power_mw, uni_overall, uni_center, uni_white, weights, center_ratio=0.5):
    target_power = weights["TARGET_POWER_MW"]
    
    # 1. Total Power Loss (Normalized)
    if power_roi < target_power:
        p_loss = weights["W_POWER"] * ((target_power - power_roi)/target_power)**2
    else:
        p_loss = 0.0
        
    # 2. [NEW] Center Power Loss (핵심)
    # 목표: 전체 파워 목표의 (면적비 * 0.8) 정도는 중앙에 있어야 한다고 가정
    # 예: 면적비가 0.25라면(0.5*0.5), 전체 파워의 20%는 중앙에 집중되어야 함.
    target_center = target_power * (center_ratio**2) * 0.8 
    
    if center_power_mw < target_center:
        # 중앙이 어두우면 강력한 페널티!
        cp_loss = weights["W_POWER_CENTER"] * ((target_center - center_power_mw)/target_center)**2
    else:
        cp_loss = 0.0

    # 3. Uniformity Loss
    u_all_loss = weights["W_UNIFORMITY_OVERALL"] * (1 - uni_overall)**2
    u_cen_loss = weights["W_UNIFORMITY_CENTER"] * (1 - uni_center)**2
    u_wht_loss = weights["W_UNIFORMITY_WHITE"] * (1 - uni_white)**2

    total_loss = p_loss + cp_loss + u_all_loss + u_cen_loss + u_wht_loss
    return total_loss, p_loss, cp_loss, u_all_loss


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
    led_geom_params=None,
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

    # 1. 면 광원 파라미터 준비
    if led_geom_params:
        w = led_geom_params.get("led_width_mm", 0)
        h = led_geom_params.get("led_height_mm", 0)
        nx = led_geom_params.get("subsample_x", 1)
        ny = led_geom_params.get("subsample_y", 1)
        marker_size = 5  # 점이 많아지므로 사이즈를 좀 줄임
    else:
        w, h, nx, ny = 0, 0, 1, 1
        marker_size = 30  # 기존 사이즈

    # 2. UV LED 그리기
    if led_positions_uv:
        uv_points_x = []
        uv_points_y = []

        for pos in led_positions_uv:
            # 중심점 하나를 N개의 점으로 쪼개서 리스트에 추가
            sub_pts = generate_sub_points(pos, w, h, nx, ny)
            for sp in sub_pts:
                uv_points_x.append(sp[0])
                uv_points_y.append(sp[1])

        ax.scatter(
            uv_points_x,
            uv_points_y,
            c="w",
            marker="o",
            s=marker_size,
            label=f"UV ({len(uv_points_x)} pts)",
        )

    # 3. White LED 그리기
    if led_positions_white:
        wh_points_x = []
        wh_points_y = []

        for pos in led_positions_white:
            sub_pts = generate_sub_points(pos, w, h, nx, ny)
            for sp in sub_pts:
                wh_points_x.append(sp[0])
                wh_points_y.append(sp[1])

        ax.scatter(
            wh_points_x,
            wh_points_y,
            c="lime",
            marker="s",
            s=marker_size,
            label=f"White ({len(wh_points_x)} pts)",
        )

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
