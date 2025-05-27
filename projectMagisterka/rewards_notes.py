import math
import matplotlib.pyplot as plt
import numpy as np

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
    reward += 0.05 * (vt - vt_prev)
    reward -= 0.00002 * (ct - ct_prev)
    reward -= 2.0 * (st - st_prev)
    reward -= 2.0 * (ot - ot_prev)
    return reward

def R5(r_speed, r_travel, p_deviation, c_steer, alpha_tr, alpha_dev, alpha_st):
    return (r_speed +
            alpha_tr * r_travel +
            alpha_dev * p_deviation +
            alpha_st * c_steer)

def R_break(collisions: int, v: float, d_obs: float, a: int) -> float:
    R = 0.0
    # term 1
    if v < 10*d_obs + 10:
        R += 3 * int(a == 0) - int(a == 1)
    # term 2
    if v > 10*d_obs + 10:
        R += 2 * (2 * int(a == 1) - 1)
    # term 3
    if v < 1 and d_obs > 100:
        R -= 10
    # term 4
    if v == 0 and d_obs < 150:
        R += 200
    # term 5
    if collisions > 0:
        R -= 200
    return R


def R6_drive(collisions, f_i, d, v, d_obs, a, R):
    if collisions > 0 or abs(f_i) > 100 or abs(d) > 3:
        return -200
    elif abs(d) > 2:
        return R - 10
    else:
        return R

# def R_total(r_pi, collisions, f_i, d, v, d_obs, a):
#     if collisions > 0 or abs(f_i) > 100 or abs(d) > 3:
#         r_drive = -200
#     elif abs(d) <= 2:
#         r_drive = r_pi
#     else:
#         # here 2 < |d| <= 3
#         r_drive = r_pi - 10
#     r_break = 0
#     # collision penalty
#     if collisions > 0:
#         r_break -= 200
#     # speed vs obstacle
#     if v < d_obs + 10:
#         # if action = 0, +3; if action = 1, –1
#         r_break += 3 if a == 0 else -1
#     elif v > d_obs + 10:
#         # if action = 1, +2; if action = 0, –1
#         r_break += 2 if a == 1 else -1
#     # too slow & far from obstacle
#     if v < 1 and d_obs > 100:
#         r_break -= 10
#     # stopped & near obstacle
#     if v == 0 and d_obs < 150:
#         r_break += 200
#     return r_drive + r_break


def normalize_preserve_shape(arr):
    arr = np.array(arr)
    fmin = arr.min()
    fmax = arr.max()
    if fmin >= 0:  # all nonnegative
        return arr / (fmax if fmax != 0 else 1)
    elif fmax <= 0:  # all nonpositive
        return arr / (abs(fmin) if fmin != 0 else 1)
    else:
        # Mixed: Linearly transform so that fmin -> -1 and fmax -> 1.
        return 2 * (arr - fmin) / (fmax - fmin) - 1

# --- Prepare data ---
phi_t = 0.0
x    = np.linspace(0, 30, 200)
dist = np.linspace(0, 10, 200)
flags = [0, 1]

# Velocity‐based rewards
r1_x       = [R1(v, 5.0, 20.0, 25.0, 0.333, 0.6, False)    for v in x]
r2_x       = [R2(False, False, False, True, v, phi_t, 2.0) for v in x]
r3_x       = [R3("standard_driving", 1.0, v, None)        for v in x]
r4_x       = [R4(1.0, 1.05, x[i], 10.0, 0,0,0,0,0,0) if i>0 else
              R4(1.0, 1.05, x[i], 0.0, 0,0,0,0,0,0)        for i in range(len(x))]
r5_x       = [R5(v, 0.5, 1.0, 0.1, 0.2, -0.5, -0.1)         for v in x]
r6_x       = [R6_drive(0, 0.0, 0.5, v, 5.0, 0.0, 20.0)      for v in x]
rbreak_x   = [R_break(0, v, d_obs=5.0, a=1)                 for v in x]

# Collision‐flag‐based rewards
r1_flag       = [R1(10.0, 5.0, 20.0, 25.0, 0.333, 0.6, f>0)        for f in flags]
r2_flag       = [R2(f>0, False, False, True, 10.0, phi_t, 2.0)    for f in flags]
r3_states     = [R3("standard_driving",1.0,10.0,None),
                 R3("collision",         1.0,10.0,None)]
r3_flag       = [r3_states[f]                                            for f in flags]
const_R4      = R4(1.0,1.05,10.0,10.0,0,0,0,0,0,0)
r4_flag       = [const_R4]*2
const_R5      = R5(1.0,0.5,1.0,0.1,0.2,-0.5,-0.1)
r5_flag       = [const_R5]*2
r6_flag       = [R6_drive(int(f),0.0,0.5,10.0,5.0,0.0,50.0)          for f in flags]
rbreak_flag   = [R_break(int(f), 10.0, d_obs=5.0, a=1)               for f in flags]

# Distance‐based rewards
r1_dist       = [R1(10.0,5.0,20.0,25.0,0.333,0.6,False) for _ in dist]
r2_dist       = [R2(False,False,False,True,10.0,phi_t,d) for d in dist]
r3_dist       = [R3("standard_driving",1.0,10.0,None) for _ in dist]
r4_dist       = [R4(1.0,1.05,10.0,10.0,0,0,0,0,0,0)  for _ in dist]
r5_dist       = [R5(1.0,0.5,d,0.1,0.2,-0.5,-0.1)       for d in dist]
r6_dist       = [R6_drive(0,0.0,d,10.0,5.0,0.0,20.0)    for d in dist]
rbreak_dist   = [R_break(0,20.0,d, a=1)                for d in dist]

# Normalize all
norm_r1       = normalize_preserve_shape(r1_x)
norm_r2       = normalize_preserve_shape(r2_x)
norm_r3       = normalize_preserve_shape(r3_x)
norm_r4       = normalize_preserve_shape(r4_x)
norm_r5       = normalize_preserve_shape(r5_x)
norm_r6       = normalize_preserve_shape(r6_x)
norm_rbreak_x = normalize_preserve_shape(rbreak_x)

norm_r1_flag       = normalize_preserve_shape(r1_flag)
norm_r2_flag       = normalize_preserve_shape(r2_flag)
norm_r3_flag       = normalize_preserve_shape(r3_flag)
norm_r4_flag       = normalize_preserve_shape(r4_flag)
norm_r5_flag       = normalize_preserve_shape(r5_flag)
norm_r6_flag       = normalize_preserve_shape(r6_flag)
norm_rbreak_flag   = normalize_preserve_shape(rbreak_flag)

norm_r1_dist       = normalize_preserve_shape(r1_dist)
norm_r2_dist       = normalize_preserve_shape(r2_dist)
norm_r3_dist       = normalize_preserve_shape(r3_dist)
norm_r4_dist       = normalize_preserve_shape(r4_dist)
norm_r5_dist       = normalize_preserve_shape(r5_dist)
norm_r6_dist       = normalize_preserve_shape(r6_dist)
norm_rbreak_dist   = normalize_preserve_shape(rbreak_dist)

# --- Plotting ---

# Fig 1: velocity‐based
fig1, (ax1_raw, ax1_norm) = plt.subplots(1,2,figsize=(14,6))
ax1_raw.plot(x, r1_x,        label='R1(v)',      linewidth=2,   color='tab:blue')
ax1_raw.plot(x, r2_x,        label='R2(v_t)',    linewidth=2,   marker='o', markersize=4, color='tab:orange')
ax1_raw.plot(x, r3_x,        label='R3(v_ego)',  linewidth=2,   linestyle='--', color='tab:green')
ax1_raw.plot(x, r4_x,        label='R4(vt)',     linewidth=2.5, marker='^', color='tab:red')
ax1_raw.plot(x, r5_x,        label='R5(r_speed)',linewidth=2,   color='tab:purple')
ax1_raw.plot(x, r6_x,        label='R6_drive(v)',linewidth=2,   linestyle='-.', color='tab:brown')
ax1_raw.plot(x, rbreak_x,    label='R_break(v)', linewidth=2.5, linestyle=':',  color='tab:gray')
ax1_raw.set(xlabel='Velocity', ylabel='Reward (Raw)', title='Reward vs Velocity')
ax1_raw.grid(True)
ax1_raw.legend()

ax1_norm.plot(x, norm_r1,        label='R1(v)',      linewidth=2, color='tab:blue')
ax1_norm.plot(x, norm_r2,        label='R2(v_t)',    linewidth=2, marker='o', markersize=4, color='tab:orange')
ax1_norm.plot(x, norm_r3,        label='R3(v_ego)',  linewidth=2, linestyle='--', color='tab:green')
ax1_norm.plot(x, norm_r4,        label='R4(vt)',     linewidth=2.5, marker='^', color='tab:red')
ax1_norm.plot(x, norm_r5,        label='R5(r_speed)',linewidth=2, color='tab:purple')
ax1_norm.plot(x, norm_r6,        label='R6_drive(v)',linewidth=2, linestyle='-.', color='tab:brown')
ax1_norm.plot(x, norm_rbreak_x,  label='R_break(v)', linewidth=2.5, linestyle=':',  color='tab:gray')
ax1_norm.set(xlabel='Velocity', ylabel='Normalized Reward', title='Reward vs Velocity (Normalized)')
ax1_norm.grid(True)
ax1_norm.legend()

fig1.tight_layout()


# Fig 2: collision‐flag‐based
fig2, (ax2_raw, ax2_norm) = plt.subplots(1,2,figsize=(10,5))
ax2_raw.plot(flags, r1_flag,       marker='o', label='R1(infraction)', linestyle='-', linewidth=2, color='tab:blue')
ax2_raw.plot(flags, r2_flag,       marker='s', label='R2(collision)',  linestyle='-', linewidth=2, color='tab:orange')
ax2_raw.plot(flags, r3_flag,       marker='^', label='R3(state)',      linestyle='--',linewidth=2, color='tab:green')
ax2_raw.plot(flags, r4_flag,       marker='d', label='R4(constant)',   linestyle='-.',linewidth=2, color='tab:red')
ax2_raw.plot(flags, r5_flag,       marker='x', label='R5(constant)',   linestyle=':', linewidth=2, color='tab:purple')
ax2_raw.plot(flags, r6_flag,       marker='v', label='R6_drive(collision)', linestyle='-', linewidth=2, color='tab:brown')
ax2_raw.plot(flags, rbreak_flag,   marker='*', label='R_break(collision)',  linestyle=':', linewidth=2.5, color='tab:gray')
ax2_raw.set_xticks(flags); ax2_raw.set_xticklabels(['0','1'])
ax2_raw.set(xlabel='Collision Flag', ylabel='Reward (Raw)', title='Reward vs Collision Flag')
ax2_raw.grid(True)
ax2_raw.legend()

ax2_norm.plot(flags, norm_r1_flag,     marker='o', label='R1(infraction)', linestyle='-', linewidth=2, color='tab:blue')
ax2_norm.plot(flags, norm_r2_flag,     marker='s', label='R2(collision)',  linestyle='-', linewidth=2, color='tab:orange')
ax2_norm.plot(flags, norm_r3_flag,     marker='^', label='R3(state)',      linestyle='--',linewidth=2, color='tab:green')
ax2_norm.plot(flags, norm_r4_flag,     marker='d', label='R4(constant)',   linestyle='-.',linewidth=2, color='tab:red')
ax2_norm.plot(flags, norm_r5_flag,     marker='x', label='R5(constant)',   linestyle=':', linewidth=2, color='tab:purple')
ax2_norm.plot(flags, norm_r6_flag,     marker='v', label='R6_drive(collision)', linestyle='-', linewidth=2, color='tab:brown')
ax2_norm.plot(flags, norm_rbreak_flag, marker='*', label='R_break(collision)',  linestyle=':', linewidth=2.5, color='tab:gray')
ax2_norm.set_xticks(flags); ax2_norm.set_xticklabels(['0','1'])
ax2_norm.set(xlabel='Collision Flag', ylabel='Normalized Reward', title='Reward vs Collision Flag (Norm)')
ax2_norm.grid(True)
ax2_norm.legend()

fig2.tight_layout()


# Fig 3: distance‐based
fig3, (ax3_raw, ax3_norm) = plt.subplots(1,2,figsize=(14,6))
ax3_raw.plot(dist, r1_dist,     label='R1(constant)',    linewidth=2, color='tab:blue')
ax3_raw.plot(dist, r2_dist,     label='R2(d_t)',          linewidth=2, marker='o', markersize=4, color='tab:orange')
ax3_raw.plot(dist, r3_dist,     label='R3(constant)',    linewidth=2, linestyle='--', color='tab:green')
ax3_raw.plot(dist, r4_dist,     label='R4(constant)',    linewidth=2, color='tab:red')
ax3_raw.plot(dist, r5_dist,     label='R5(p_dev)',       linewidth=2, color='tab:purple')
ax3_raw.plot(dist, r6_dist,     label='R6_drive(d)',     linewidth=2, linestyle='-.', color='tab:brown')
ax3_raw.plot(dist, rbreak_dist, label='R_break(d)',      linewidth=2.5, linestyle=':',  color='tab:gray')
ax3_raw.set(xlabel="Distance from lane's center", ylabel='Reward (Raw)', title="Reward vs Distance")
ax3_raw.grid(True)
ax3_raw.legend()

ax3_norm.plot(dist, norm_r1_dist,     label='R1(constant)',    linewidth=2, color='tab:blue')
ax3_norm.plot(dist, norm_r2_dist,     label='R2(d_t)',          linewidth=2, marker='o', markersize=4, color='tab:orange')
ax3_norm.plot(dist, norm_r3_dist,     label='R3(constant)',    linewidth=2, linestyle='--', color='tab:green')
ax3_norm.plot(dist, norm_r4_dist,     label='R4(constant)',    linewidth=2, color='tab:red')
ax3_norm.plot(dist, norm_r5_dist,     label='R5(p_dev)',       linewidth=2, color='tab:purple')
ax3_norm.plot(dist, norm_r6_dist,     label='R6_drive(d)',     linewidth=2, linestyle='-.', color='tab:brown')
ax3_norm.plot(dist, norm_rbreak_dist, label='R_break(d)',      linewidth=2.5, linestyle=':',  color='tab:gray')
ax3_norm.set(xlabel="Distance from lane's center", ylabel='Normalized Reward', title="Reward vs Distance (Norm)")
ax3_norm.grid(True)
ax3_norm.legend()

fig3.tight_layout()
plt.show()
