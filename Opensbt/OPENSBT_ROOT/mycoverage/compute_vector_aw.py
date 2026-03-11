import json
import numpy as np

# -------------------------
# Interaction Metrics (Timestamp-aware)
# -------------------------

def min_TTC(simout, ego_id='ego', radius=1e-3):
    print("[DEBUG] Computing min_TTC...")
    ego_vel = np.array(simout['velocity'][ego_id])
    ego_pos = np.array([loc[:2] for loc in simout['location'][ego_id]])
    min_ttc = np.inf

    for key, pos_list in simout['location'].items():
        if key == ego_id:
            continue
        npc_pos = np.array([loc[:2] for loc in pos_list])
        npc_vel = np.array(simout['velocity'][key])
        
        for t in range(len(ego_pos)):
            d = npc_pos[t] - ego_pos[t]
            v_rel = npc_vel[t] - ego_vel[t]
            v_rel_norm_sq = np.dot(v_rel, v_rel)
            if v_rel_norm_sq < radius:
                continue
            ttc_val = -np.dot(d, v_rel) / v_rel_norm_sq
            if np.isscalar(ttc_val):
                if 0 < ttc_val < min_ttc:
                    min_ttc = ttc_val
            else:
                valid = ttc_val[np.array(ttc_val) > 0]
                if len(valid) > 0:
                    min_ttc = min(min_ttc, valid.min())
    print(f"[DEBUG] min_TTC = {min_ttc}")
    return min_ttc

def min_dist(simout, ego_id='ego'):
    print("[DEBUG] Computing min_dist...")
    ego_pos = np.array([loc[:2] for loc in simout['location'][ego_id]])
    min_d = np.inf

    for key, pos_list in simout['location'].items():
        if key == ego_id:
            continue
        npc_pos = np.array([loc[:2] for loc in pos_list])
        dists = np.linalg.norm(ego_pos - npc_pos, axis=1)
        if np.any(dists < min_d):
            min_d = dists.min()
    print(f"[DEBUG] min_dist = {min_d}")
    return min_d

def top2_min_dist(simout, ego_id='ego'):
    print("[DEBUG] Computing top2_min_dist...")
    min_list = []
    for key, pos_list in simout['location'].items():
        if key == ego_id:
            continue
        ego_pos = np.array([loc[:2] for loc in simout['location'][ego_id]])
        npc_pos = np.array([loc[:2] for loc in pos_list])
        dists = np.linalg.norm(ego_pos - npc_pos, axis=1)
        min_list.append(dists.min())
    result = sorted(min_list)[1] if len(min_list) >= 2 else np.inf
    print(f"[DEBUG] top2_min_dist = {result}")
    return result

def num_neighbors_within_R(simout, R=5.0, ego_id='ego'):
    print("[DEBUG] Computing num_neighbors_within_R (timestamp-weighted)...")
    ego_pos = np.array([loc[:2] for loc in simout['location'][ego_id]])
    times = np.array(simout['timestamps'][ego_id])
    time_deltas = np.diff(times, prepend=times[0])
    counts = []

    for t in range(len(ego_pos)):
        cnt = 0
        for key, pos_list in simout['location'].items():
            if key == ego_id:
                continue
            npc_pos = np.array(pos_list[t][:2])
            if np.linalg.norm(npc_pos - ego_pos[t]) <= R:
                cnt += 1
        counts.append(cnt * time_deltas[t])

    result = np.sum(counts) / (times[-1] - times[0])
    print(f"[DEBUG] num_neighbors_within_R = {result}")
    return result

def time_with_neighbors(simout, R=10.0, ego_id='ego'):
    print("[DEBUG] Computing time_with_neighbors (timestamp-weighted)...")
    ego_pos = np.array([loc[:2] for loc in simout['location'][ego_id]])
    times = np.array(simout['timestamps'][ego_id])
    time_deltas = np.diff(times, prepend=times[0])
    time_sum = 0.0

    for t in range(len(ego_pos)):
        for key, pos_list in simout['location'].items():
            if key == ego_id:
                continue
            npc_pos = np.array(pos_list[t][:2])
            if np.linalg.norm(npc_pos - ego_pos[t]) <= R:
                time_sum += time_deltas[t]
                break

    print(f"[DEBUG] time_with_neighbors = {time_sum}")
    return time_sum

def min_lateral_gap(simout, ego_id='ego'):
    print("[DEBUG] Computing min_lateral_gap...")
    ego_pos = np.array([loc[:2] for loc in simout['location'][ego_id]])
    ego_yaw = np.array(simout['yaw'][ego_id])
    min_gap = np.inf

    for key, pos_list in simout['location'].items():
        if key == ego_id:
            continue
        npc_pos = np.array([loc[:2] for loc in pos_list])
        for t in range(len(ego_pos)):
            dx, dy = npc_pos[t] - ego_pos[t]
            lateral = -dx * np.sin(ego_yaw[t]) + dy * np.cos(ego_yaw[t])
            if abs(lateral) < min_gap:
                min_gap = abs(lateral)
    print(f"[DEBUG] min_lateral_gap = {min_gap}")
    return min_gap

def min_longitudinal_gap(simout, ego_id='ego'):
    print("[DEBUG] Computing min_longitudinal_gap...")
    ego_pos = np.array([loc[:2] for loc in simout['location'][ego_id]])
    ego_yaw = np.array(simout['yaw'][ego_id])
    min_gap = np.inf

    for key, pos_list in simout['location'].items():
        if key == ego_id:
            continue
        npc_pos = np.array([loc[:2] for loc in pos_list])
        for t in range(len(ego_pos)):
            dx, dy = npc_pos[t] - ego_pos[t]
            longitudinal = dx * np.cos(ego_yaw[t]) + dy * np.sin(ego_yaw[t])
            if abs(longitudinal) < min_gap:
                min_gap = abs(longitudinal)
    print(f"[DEBUG] min_longitudinal_gap = {min_gap}")
    return min_gap

def avg_speed_ego(simout, ego_id='ego'):
    print("[DEBUG] Computing avg_speed_ego (timestamp-weighted)...")
    speeds = np.array(simout['speed'][ego_id])
    times = np.array(simout['timestamps'][ego_id])
    time_deltas = np.diff(times, prepend=times[0])
    avg_speed = np.sum(speeds * time_deltas) / (times[-1] - times[0])
    print(f"[DEBUG] avg_speed_ego = {avg_speed}")
    return avg_speed

def max_speed_ego(simout, ego_id='ego'):
    print("[DEBUG] Computing max_speed_ego...")
    speeds = np.array(simout['speed'][ego_id])
    max_speed = np.max(speeds)
    print(f"[DEBUG] max_speed_ego = {max_speed}")
    return max_speed

# -------------------------
# Compute Full Interaction Vector
# -------------------------

def compute_interaction_vector(simout):
    print("[DEBUG] Computing full interaction vector...")
    vector = [
        min_TTC(simout),
        min_dist(simout),
        top2_min_dist(simout),
        num_neighbors_within_R(simout, R=10.0),
        time_with_neighbors(simout, R=10.0),
        min_lateral_gap(simout),
        min_longitudinal_gap(simout),
        avg_speed_ego(simout),
        max_speed_ego(simout)
    ]
    print("[DEBUG] Full interaction vector computed.")
    return vector

# -------------------------
# Example Usage
# -------------------------

if __name__ == "__main__":
    path = rf"/home/user/testing/topic1-opensbt-aw/results_wandb/LoFi_GA_pop10_t06_00_00_seed13/jxhix0h9/rerun_hifi/simout/simout_S151.432725_0.459841_11.348637_23.177858_1.090066_2.525470_10.010484_16.557151_70.612127_0.705612_0.283797.json"  # replace with your path
    print("[DEBUG] Loading simout JSON file...")
    with open(path, "r") as f:
        simout = json.load(f)

    vector = compute_interaction_vector(simout)
    print("\n[RESULT] Interaction-space vector:")
    print(vector)