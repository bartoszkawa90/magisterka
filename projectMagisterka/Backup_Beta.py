import math
import matplotlib.pyplot as plt
import numpy as np

# --- Reward functions ---
def R1(v, vmin, vtarget, vmax, d_norm, a_rew, infraction):
    if infraction:
        return -10.0
    if v < vmin:
        vel_term = (v / vmin) if vmin != 0 else 0.0
    elif v < vtarget:
        vel_term = 1.0
    else:
        if vmax > vtarget:
            vel_term = max(0.0, 1.0 - (v - vtarget) / (vmax - vtarget))
        else:
            vel_term = 0.0
    return vel_term * (1.0 - d_norm) * a_rew

def R2(collision, lane_change, goal_reached, car_in_lane, v_t, phi_t, d_t):
    if collision or lane_change:
        return -200.0
    if goal_reached:
        return 100.0
    if car_in_lane:
        return abs(v_t * math.cos(phi_t)) - abs(v_t * math.sin(phi_t)) - abs(d_t * v_t)
    return 0.0

def R3(state, k_v=None, v_ego=None, t_out=None):
    if state == "standard_driving":
        return k_v * v_ego
    if state == "crossing_intersection":
        return 1.0
    if state == "collision":
        return -2.0
    if state == "penalty":
        return 0.2
    if state == "duration_penalty":
        return -0.2 / t_out if t_out and t_out != 0 else 0.0
    return 0.0

def R4(dt, dt_prev, vt, vt_prev, ct, ct_prev, st, st_prev, ot, ot_prev):
    reward = 1000.0 * (dt_prev - dt)
    reward += 0.05   * (vt - vt_prev)
    reward -= 0.00002 * (ct - ct_prev)
    reward -= 2.0    * (st - st_prev)
    reward -= 2.0    * (ot - ot_prev)
    return reward

def R5(r_speed, r_travel, p_deviation, c_steer, alpha_tr, alpha_dev, alpha_st):
    return (r_speed +
            alpha_tr * r_travel +
            alpha_dev * p_deviation +
            alpha_st * c_steer)

def R6(collisions, f_i, d, v, d_obs, a, R):
    if collisions > 0 or abs(f_i) > 100 or abs(d) > 3:
        return -200
    elif abs(d) > 2:
        return R - 10
    else:
        return R

# --- Piecewise Normalization function ---
# This function normalizes the data such that:
#   - if values are all positive: they are scaled to [0, 1],
#   - if values are all negative: they are scaled to [0, -1],
#   - if mixed: positives are scaled to [0, 1] and negatives to [0, -1],
# leaving zeros unchanged.
def normalize_piecewise(arr):
    arr = np.array(arr)
    res = np.zeros_like(arr, dtype=float)
    pos_mask = arr > 0
    neg_mask = arr < 0
    if np.any(pos_mask):
        max_val = np.max(arr[pos_mask])
        if max_val != 0:
            res[pos_mask] = arr[pos_mask] / max_val
    if np.any(neg_mask):
        min_val = np.min(arr[neg_mask])  # negative value
        if min_val != 0:
            res[neg_mask] = arr[neg_mask] / abs(min_val)
    return res

phi_t = 0.0

# ------------------------------------------------------------------
# 1) Reward Functions vs. Velocity-like Input
# ------------------------------------------------------------------
x = np.linspace(0, 30, 200)
r1_x = [R1(v, 5.0, 15.0, 25.0, 0.2, 1.0, False) for v in x]
r2_x = [R2(False, False, False, True, v, phi_t, 5.0) for v in x]
r3_x = [R3("standard_driving", k_v=1.0, v_ego=v, t_out=None) for v in x]
r4_x = [R4(10.0, 10.0, v, 10.0, 0, 0, 0, 0, 0, 0) for v in x]
r5_x = [R5(v, 0.5, 1.0, 0.1, 0.2, -0.5, -0.1) for v in x]
r6_x = [R6(0, 0.0, 0.5, v, 5.0, 0.0, 50.0) for v in x]

# Normalize piecewise for velocity input
norm_r1 = normalize_piecewise(r1_x)
norm_r2 = normalize_piecewise(r2_x)
norm_r3 = normalize_piecewise(r3_x)
norm_r4 = normalize_piecewise(r4_x)
norm_r5 = normalize_piecewise(r5_x)
norm_r6 = normalize_piecewise(r6_x)

# Create a figure with two subplots: left = raw, right = normalized
fig1, (ax1_raw, ax1_norm) = plt.subplots(1, 2, figsize=(14, 6))

# Raw subplot
ax1_raw.plot(x, r1_x, label='R1(v)', linewidth=2, color='tab:blue')
ax1_raw.plot(x, r2_x, label='R2(v_t)', linewidth=2, marker='o', markersize=4, color='tab:orange')
ax1_raw.plot(x, r3_x, label='R3(v_ego)', linewidth=2, linestyle='--', color='tab:green')
ax1_raw.plot(x, r4_x, label='R4(vt)', linewidth=2, color='tab:red')
ax1_raw.plot(x, r5_x, label='R5(r_speed)', linewidth=2, marker='^', color='tab:purple')
ax1_raw.plot(x, r6_x, label='R6(v)', linewidth=2, linestyle='-.', color='tab:brown')
ax1_raw.set_xlabel('Velocity-like input')
ax1_raw.set_ylabel('Reward (Raw)')
ax1_raw.set_title('Raw Reward Functions vs. Velocity-like Input')
ax1_raw.legend()
ax1_raw.grid(True)

# Normalized subplot
ax1_norm.plot(x, norm_r1, label='R1(v)', linewidth=2, color='tab:blue')
ax1_norm.plot(x, norm_r2, label='R2(v_t)', linewidth=2, marker='o', markersize=4, color='tab:orange')
ax1_norm.plot(x, norm_r3, label='R3(v_ego)', linewidth=2, linestyle='--', color='tab:green')
ax1_norm.plot(x, norm_r4, label='R4(vt)', linewidth=2, color='tab:red')
ax1_norm.plot(x, norm_r5, label='R5(r_speed)', linewidth=2, marker='^', color='tab:purple')
ax1_norm.plot(x, norm_r6, label='R6(v)', linewidth=2, linestyle='-.', color='tab:brown')
ax1_norm.set_xlabel('Velocity-like input')
ax1_norm.set_ylabel('Normalized Reward (piecewise)')
ax1_norm.set_title('Normalized Reward Functions vs. Velocity-like Input')
ax1_norm.legend()
ax1_norm.grid(True)

fig1.tight_layout()


# ------------------------------------------------------------------
# 2) Reward Functions vs. Collision/Infraction Flags (0 and 1)
# ------------------------------------------------------------------
flags = [0, 1]
r1_flag = [R1(10.0, 5.0, 15.0, 25.0, 0.2, 1.0, f > 0) for f in flags]
r2_flag = [R2(f > 0, False, False, True, 10.0, phi_t, 5.0) for f in flags]
r3_states = [R3("standard_driving", 1.0, 10.0, None),
             R3("collision", 1.0, 10.0, None)]
r3_flag = [r3_states[f] for f in flags]
const_R4 = R4(10.0, 10.0, 10.0, 10.0, 0, 0, 0, 0, 0, 0)
const_R5 = R5(1.0, 0.5, 1.0, 0.1, 0.2, -0.5, -0.1)
r4_flag = [const_R4] * 2
r5_flag = [const_R5] * 2
r6_flag = [R6(int(f), 0.0, 0.5, 10.0, 5.0, 0.0, 50.0) for f in flags]

# Normalize piecewise for flags
norm_r1_flag = normalize_piecewise(r1_flag)
norm_r2_flag = normalize_piecewise(r2_flag)
norm_r3_flag = normalize_piecewise(r3_flag)
norm_r4_flag = normalize_piecewise(r4_flag)
norm_r5_flag = normalize_piecewise(r5_flag)
norm_r6_flag = normalize_piecewise(r6_flag)

fig2, (ax2_raw, ax2_norm) = plt.subplots(1, 2, figsize=(10, 5))

# Raw subplot for flags
ax2_raw.plot(flags, r1_flag, marker='o', label='R1(infraction)', linestyle='-', linewidth=2, color='tab:blue')
ax2_raw.plot(flags, r2_flag, marker='s', label='R2(collision)', linestyle='-', linewidth=2, color='tab:orange')
ax2_raw.plot(flags, r3_flag, marker='^', label='R3(state)', linestyle='--', linewidth=2, color='tab:green')
ax2_raw.plot(flags, r4_flag, marker='d', label='R4(constant)', linestyle='-.', linewidth=2, color='tab:red')
ax2_raw.plot(flags, r5_flag, marker='x', label='R5(constant)', linestyle=':', linewidth=2, color='tab:purple')
ax2_raw.plot(flags, r6_flag, marker='v', label='R6(collisions)', linestyle='-', linewidth=2, color='tab:brown')
ax2_raw.set_xticks(flags)
ax2_raw.set_xticklabels(['0', '1'])
ax2_raw.set_xlabel('Collision/Infraction Flag (0 or 1)')
ax2_raw.set_ylabel('Reward (Raw)')
ax2_raw.set_title('Raw R-functions vs. 0/1 Collision/Infraction Flags')
ax2_raw.legend()
ax2_raw.grid(True)

# Normalized subplot for flags
ax2_norm.plot(flags, norm_r1_flag, marker='o', label='R1(infraction)', linestyle='-', linewidth=2, color='tab:blue')
ax2_norm.plot(flags, norm_r2_flag, marker='s', label='R2(collision)', linestyle='-', linewidth=2, color='tab:orange')
ax2_norm.plot(flags, norm_r3_flag, marker='^', label='R3(state)', linestyle='--', linewidth=2, color='tab:green')
ax2_norm.plot(flags, norm_r4_flag, marker='d', label='R4(constant)', linestyle='-.', linewidth=2, color='tab:red')
ax2_norm.plot(flags, norm_r5_flag, marker='x', label='R5(constant)', linestyle=':', linewidth=2, color='tab:purple')
ax2_norm.plot(flags, norm_r6_flag, marker='v', label='R6(collisions)', linestyle='-', linewidth=2, color='tab:brown')
ax2_norm.set_xticks(flags)
ax2_norm.set_xticklabels(['0', '1'])
ax2_norm.set_xlabel('Collision/Infraction Flag (0 or 1)')
ax2_norm.set_ylabel('Normalized Reward (piecewise)')
ax2_norm.set_title('Normalized R-functions vs. 0/1 Collision/Infraction Flags')
ax2_norm.legend()
ax2_norm.grid(True)

fig2.tight_layout()


# ------------------------------------------------------------------
# 3) Reward Functions vs. Distance-like Input
# ------------------------------------------------------------------
dist = np.linspace(0, 10, 200)
r1_dist = [R1(10.0, 5.0, 15.0, 25.0, 0.2, 1.0, False) for _ in dist]
r2_dist = [R2(False, False, False, True, 10.0, phi_t, d) for d in dist]
r3_dist = [R3("standard_driving", 1.0, 10.0, None) for _ in dist]
r4_dist = [R4(d, d, 10.0, 10.0, 0, 0, 0, 0, 0, 0) for d in dist]
r5_dist = [R5(1.0, 0.5, d, 0.1, 0.2, -0.5, -0.1) for d in dist]
r6_dist = [R6(0, 0.0, d, 10.0, 5.0, 0.0, 50.0) for d in dist]

# Normalize piecewise for distance input
norm_r1_dist = normalize_piecewise(r1_dist)
norm_r2_dist = normalize_piecewise(r2_dist)
norm_r3_dist = normalize_piecewise(r3_dist)
norm_r4_dist = normalize_piecewise(r4_dist)
norm_r5_dist = normalize_piecewise(r5_dist)
norm_r6_dist = normalize_piecewise(r6_dist)

fig3, (ax3_raw, ax3_norm) = plt.subplots(1, 2, figsize=(14, 6))

# Raw subplot for distance
ax3_raw.plot(dist, r1_dist, label='R1(constant)', linewidth=2, color='tab:blue')
ax3_raw.plot(dist, r2_dist, label='R2(d_t)', linewidth=2, marker='o', markersize=4, color='tab:orange')
ax3_raw.plot(dist, r3_dist, label='R3(constant)', linewidth=2, linestyle='--', color='tab:green')
ax3_raw.plot(dist, r4_dist, label='R4(constant)', linewidth=2, color='tab:red')
ax3_raw.plot(dist, r5_dist, label='R5(p_deviation)', linewidth=2, marker='^', color='tab:purple')
ax3_raw.plot(dist, r6_dist, label='R6(d)', linewidth=2, linestyle='-.', color='tab:brown')
ax3_raw.set_xlabel('Distance-like input')
ax3_raw.set_ylabel('Reward (Raw)')
ax3_raw.set_title('Raw R-functions vs. Distance-like Input')
ax3_raw.legend()
ax3_raw.grid(True)

# Normalized subplot for distance
ax3_norm.plot(dist, norm_r1_dist, label='R1(constant)', linewidth=2, color='tab:blue')
ax3_norm.plot(dist, norm_r2_dist, label='R2(d_t)', linewidth=2, marker='o', markersize=4, color='tab:orange')
ax3_norm.plot(dist, norm_r3_dist, label='R3(constant)', linewidth=2, linestyle='--', color='tab:green')
ax3_norm.plot(dist, norm_r4_dist, label='R4(constant)', linewidth=2, color='tab:red')
ax3_norm.plot(dist, norm_r5_dist, label='R5(p_deviation)', linewidth=2, marker='^', color='tab:purple')
ax3_norm.plot(dist, norm_r6_dist, label='R6(d)', linewidth=2, linestyle='-.', color='tab:brown')
ax3_norm.set_xlabel('Distance-like input')
ax3_norm.set_ylabel('Normalized Reward (piecewise)')
ax3_norm.set_title('Normalized R-functions vs. Distance-like Input')
ax3_norm.legend()
ax3_norm.grid(True)

fig3.tight_layout()

plt.show()
