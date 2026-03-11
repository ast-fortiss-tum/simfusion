import json
import numpy as np

# -------------------------
# Helpers
# -------------------------

def _align_vec_dims(a, b, prefer_dims=2):
    """
    Make vectors a and b have the same length.
    - If both have at least prefer_dims components, truncate both to prefer_dims (default 2D).
    - Else truncate both to the minimum available dimension.
    Returns (a_aligned, b_aligned).
    """
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)

    if a.size >= prefer_dims and b.size >= prefer_dims:
        k = prefer_dims
    else:
        k = min(a.size, b.size)

    return a[:k], b[:k]


def _get_time_vector(simout, ego_id="ego"):
    """
    Return a single global time vector to use for timestamp-weighting.

    Priority:
      1) simout["timestamps"][ego_id] if exists and non-empty
      2) simout["timestamps"] if it's a list/array and non-empty (global)
      3) simout["times"] if it's a list/array and non-empty (global)
      4) simout["time"] if it's a list/array and non-empty (global)

    Raises ValueError if none found.
    """
    ts = simout.get("timestamps", None)

    # Case 1: timestamps is a dict keyed by agent
    if isinstance(ts, dict):
        v = ts.get(ego_id, None)
        if v is not None and len(v) > 0:
            return np.asarray(v, dtype=float)

    # Case 2: timestamps is already a global vector
    if isinstance(ts, (list, tuple, np.ndarray)) and len(ts) > 0:
        return np.asarray(ts, dtype=float)

    # Fallbacks: times/time (global)
    for k in ("times", "time"):
        v = simout.get(k, None)
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
            return np.asarray(v, dtype=float)

    raise ValueError(
        "No valid time vector found. Expected non-empty simout['timestamps'] "
        "(dict keyed by ego_id or global list) or simout['times']/simout['time']."
    )


def _time_deltas(times):
    """
    Compute per-sample weights.
    Uses dt[0]=0 so the first sample doesn't get an artificial full interval.
    """
    times = np.asarray(times, dtype=float)
    if times.size == 0:
        return times
    dt = np.diff(times, prepend=times[0])
    dt[0] = 0.0
    # guard against weird non-monotonic timestamps
    dt = np.clip(dt, 0.0, None)
    return dt


# -------------------------
# Interaction Metrics
# -------------------------

def min_TTC(simout, ego_id="ego", radius=1e-3, prefer_dims=2):
    """
    NOTE: This is actually "time to closest approach" (CPA time) under constant velocity,
    not a finite-radius collision TTC. Kept as-is for compatibility.
    """
    print("[DEBUG] Computing min_TTC...")

    ego_vel = np.asarray(simout["velocity"][ego_id])
    ego_loc = np.asarray(simout["location"][ego_id])

    min_ttc = np.inf

    for key, npc_loc_list in simout["location"].items():
        if key == ego_id:
            continue

        npc_vel = np.asarray(simout["velocity"][key])
        npc_loc = np.asarray(npc_loc_list)

        T = min(len(ego_loc), len(npc_loc), len(ego_vel), len(npc_vel))
        for t in range(T):
            d_raw = npc_loc[t] - ego_loc[t]
            vrel_raw = npc_vel[t] - ego_vel[t]

            d, v_rel = _align_vec_dims(d_raw, vrel_raw, prefer_dims=prefer_dims)
            if d.size == 0 or v_rel.size == 0:
                continue

            v_rel_norm_sq = float(np.dot(v_rel, v_rel))
            if v_rel_norm_sq < radius:
                continue

            ttc_val = -float(np.dot(d, v_rel)) / v_rel_norm_sq
            if 0 < ttc_val < min_ttc:
                min_ttc = ttc_val

    print(f"[DEBUG] min_TTC = {min_ttc}")
    return min_ttc


def min_dist(simout, ego_id="ego"):
    print("[DEBUG] Computing min_dist...")
    ego_pos = np.asarray([loc[:2] for loc in simout["location"][ego_id]], dtype=float)
    min_d = np.inf

    for key, pos_list in simout["location"].items():
        if key == ego_id:
            continue
        npc_pos = np.asarray([loc[:2] for loc in pos_list], dtype=float)
        n = min(len(ego_pos), len(npc_pos))
        if n == 0:
            continue
        dists = np.linalg.norm(ego_pos[:n] - npc_pos[:n], axis=1)
        if dists.size:
            min_d = min(min_d, float(dists.min()))

    print(f"[DEBUG] min_dist = {min_d}")
    return min_d


def top2_min_dist(simout, ego_id="ego"):
    """
    Second-smallest of {min distance to each other agent}.
    Useful only if you have >=2 non-ego agents.
    """
    print("[DEBUG] Computing top2_min_dist...")
    ego_pos = np.asarray([loc[:2] for loc in simout["location"][ego_id]], dtype=float)

    min_list = []
    for key, pos_list in simout["location"].items():
        if key == ego_id:
            continue
        npc_pos = np.asarray([loc[:2] for loc in pos_list], dtype=float)
        n = min(len(ego_pos), len(npc_pos))
        if n == 0:
            continue
        dists = np.linalg.norm(ego_pos[:n] - npc_pos[:n], axis=1)
        if dists.size:
            min_list.append(float(dists.min()))

    result = sorted(min_list)[1] if len(min_list) >= 2 else np.inf
    print(f"[DEBUG] top2_min_dist = {result}")
    return result


def num_neighbors_within_R(simout, R=5.0, ego_id="ego"):
    print("[DEBUG] Computing num_neighbors_within_R (timestamp-weighted)...")
    ego_pos = np.asarray([loc[:2] for loc in simout["location"][ego_id]], dtype=float)

    times = _get_time_vector(simout, ego_id=ego_id)
    dt = _time_deltas(times)

    T = min(len(ego_pos), len(times), len(dt))
    if T == 0:
        print("[DEBUG] num_neighbors_within_R = inf (no samples)")
        return np.inf

    counts_weighted = []
    for t in range(T):
        cnt = 0
        for key, pos_list in simout["location"].items():
            if key == ego_id:
                continue
            if t >= len(pos_list):
                continue
            npc_pos = np.asarray(pos_list[t][:2], dtype=float)
            if np.linalg.norm(npc_pos - ego_pos[t]) <= R:
                cnt += 1
        counts_weighted.append(cnt * float(dt[t]))

    denom = float(times[T - 1] - times[0])
    denom = denom if denom != 0 else 1.0

    result = float(np.sum(counts_weighted) / denom)
    print(f"[DEBUG] num_neighbors_within_R = {result}")
    return result


def time_with_neighbors(simout, R=10.0, ego_id="ego"):
    print("[DEBUG] Computing time_with_neighbors (timestamp-weighted)...")
    ego_pos = np.asarray([loc[:2] for loc in simout["location"][ego_id]], dtype=float)

    times = _get_time_vector(simout, ego_id=ego_id)
    dt = _time_deltas(times)

    T = min(len(ego_pos), len(times), len(dt))
    if T == 0:
        print("[DEBUG] time_with_neighbors = 0.0 (no samples)")
        return 0.0

    time_sum = 0.0
    for t in range(T):
        for key, pos_list in simout["location"].items():
            if key == ego_id:
                continue
            if t >= len(pos_list):
                continue
            npc_pos = np.asarray(pos_list[t][:2], dtype=float)
            if np.linalg.norm(npc_pos - ego_pos[t]) <= R:
                time_sum += float(dt[t])
                break

    print(f"[DEBUG] time_with_neighbors = {time_sum}")
    return time_sum


def min_lateral_gap(simout, ego_id="ego"):
    print("[DEBUG] Computing min_lateral_gap...")
    ego_pos = np.asarray([loc[:2] for loc in simout["location"][ego_id]], dtype=float)
    ego_yaw = np.asarray(simout["yaw"][ego_id], dtype=float)
    min_gap = np.inf

    for key, pos_list in simout["location"].items():
        if key == ego_id:
            continue
        npc_pos = np.asarray([loc[:2] for loc in pos_list], dtype=float)
        n = min(len(ego_pos), len(npc_pos), len(ego_yaw))
        for t in range(n):
            dx, dy = npc_pos[t] - ego_pos[t]
            lateral = -dx * np.sin(ego_yaw[t]) + dy * np.cos(ego_yaw[t])
            min_gap = min(min_gap, float(abs(lateral)))

    print(f"[DEBUG] min_lateral_gap = {min_gap}")
    return min_gap


def min_longitudinal_gap(simout, ego_id="ego"):
    print("[DEBUG] Computing min_longitudinal_gap...")
    ego_pos = np.asarray([loc[:2] for loc in simout["location"][ego_id]], dtype=float)
    ego_yaw = np.asarray(simout["yaw"][ego_id], dtype=float)
    min_gap = np.inf

    for key, pos_list in simout["location"].items():
        if key == ego_id:
            continue
        npc_pos = np.asarray([loc[:2] for loc in pos_list], dtype=float)
        n = min(len(ego_pos), len(npc_pos), len(ego_yaw))
        for t in range(n):
            dx, dy = npc_pos[t] - ego_pos[t]
            longitudinal = dx * np.cos(ego_yaw[t]) + dy * np.sin(ego_yaw[t])
            min_gap = min(min_gap, float(abs(longitudinal)))

    print(f"[DEBUG] min_longitudinal_gap = {min_gap}")
    return min_gap


def avg_speed_ego(simout, ego_id="ego"):
    print("[DEBUG] Computing avg_speed_ego (timestamp-weighted)...")
    speeds = np.asarray(simout["speed"][ego_id], dtype=float)

    times = _get_time_vector(simout, ego_id=ego_id)
    dt = _time_deltas(times)

    T = min(len(speeds), len(times), len(dt))
    if T == 0:
        print("[DEBUG] avg_speed_ego = nan (no samples)")
        return np.nan

    denom = float(times[T - 1] - times[0])
    denom = denom if denom != 0 else 1.0

    avg_speed = float(np.sum(speeds[:T] * dt[:T]) / denom)
    print(f"[DEBUG] avg_speed_ego = {avg_speed}")
    return avg_speed


def max_speed_ego(simout, ego_id="ego"):
    print("[DEBUG] Computing max_speed_ego...")
    speeds = np.asarray(simout["speed"][ego_id], dtype=float)
    max_speed = float(np.max(speeds)) if speeds.size else np.nan
    print(f"[DEBUG] max_speed_ego = {max_speed}")
    return max_speed


def max_deceleration_ego(simout, ego_id="ego"):
    """
    Peak deceleration magnitude (positive number, m/s^2).
    Uses simout['acceleration'][ego_id] if present, else estimates from speed/time.
    """
    print("[DEBUG] Computing max_deceleration_ego...")

    acc = simout.get("acceleration", None)
    if isinstance(acc, dict) and ego_id in acc and len(acc[ego_id]) > 0:
        a = np.asarray(acc[ego_id], dtype=float)
        # If accel is vector-valued, use x-component as a "longitudinal" proxy
        if a.ndim == 2 and a.shape[1] >= 1:
            a_long = a[:, 0]
        else:
            a_long = a.reshape(-1)

        if a_long.size == 0:
            print("[DEBUG] max_deceleration_ego = nan (empty acceleration)")
            return np.nan

        most_negative = float(np.min(a_long))
        max_decel = max(0.0, -most_negative)
        print(f"[DEBUG] max_deceleration_ego = {max_decel}")
        return max_decel

    # Fallback: estimate from speed derivative
    speeds = np.asarray(simout.get("speed", {}).get(ego_id, []), dtype=float)
    if speeds.size < 2:
        print("[DEBUG] max_deceleration_ego = nan (no speed samples)")
        return np.nan

    times = _get_time_vector(simout, ego_id=ego_id)
    T = min(len(speeds), len(times))
    speeds = speeds[:T]
    times = times[:T]
    if T < 2:
        print("[DEBUG] max_deceleration_ego = nan (insufficient aligned samples)")
        return np.nan

    dt = np.diff(times)
    dv = np.diff(speeds)

    mask = dt > 1e-9
    if not np.any(mask):
        print("[DEBUG] max_deceleration_ego = nan (non-positive dt)")
        return np.nan

    a_est = dv[mask] / dt[mask]  # m/s^2
    most_negative = float(np.min(a_est)) if a_est.size else 0.0
    max_decel = max(0.0, -most_negative)

    print(f"[DEBUG] max_deceleration_ego = {max_decel}")
    return max_decel


# -------------------------
# Offsets ego<->other agent (signed, at min-abs)
# -------------------------

def min_signed_longitudinal_offset_car_lengths(
    simout,
    ego_id="ego",
    ego_length_m=4.5,
    at_min_dist=False,
):
    """
    Signed longitudinal offset (ego forward axis), expressed in car lengths.

    Default behavior (at_min_dist=False):
      - returns the signed longitudinal value at the time when |longitudinal| is minimal.

    If at_min_dist=True:
      - returns the signed longitudinal value at the time when Euclidean distance
        between ego and the other agent is minimal.
    """
    print("[DEBUG] Computing min_signed_longitudinal_offset_car_lengths...")
    ego_pos = np.asarray([loc[:2] for loc in simout["location"][ego_id]], dtype=float)
    ego_yaw = np.asarray(simout["yaw"][ego_id], dtype=float)

    best_score = np.inf
    best_signed = np.nan

    for key, pos_list in simout["location"].items():
        if key == ego_id:
            continue
        npc_pos = np.asarray([loc[:2] for loc in pos_list], dtype=float)
        n = min(len(ego_pos), len(npc_pos), len(ego_yaw))
        if n == 0:
            continue

        if at_min_dist:
            # Choose the timestep where Euclidean distance is minimal for this agent
            dists = np.linalg.norm(npc_pos[:n] - ego_pos[:n], axis=1)
            t_indices = [int(np.argmin(dists))]
        else:
            # Scan all timesteps and choose where |longitudinal| is minimal (original behavior)
            t_indices = range(n)

        for t in t_indices:
            dx, dy = npc_pos[t] - ego_pos[t]
            longitudinal = dx * np.cos(ego_yaw[t]) + dy * np.sin(ego_yaw[t])  # signed

            score = float(np.hypot(dx, dy)) if at_min_dist else abs(longitudinal)
            if score < best_score:
                best_score = score
                best_signed = float(longitudinal)

    if ego_length_m is not None:
        result = best_signed / float(ego_length_m) if np.isfinite(best_score) else np.nan
    else:
        result = best_signed

    print(f"[DEBUG] min_signed_longitudinal_offset_car_lengths = {result}")
    return result


def min_signed_lateral_offset(
    simout,
    ego_id="ego",
    at_min_dist=False,
):
    """
    Signed lateral offset (ego left axis), meters.

    Default behavior (at_min_dist=False):
      - returns the signed lateral value at the time when |lateral| is minimal.

    If at_min_dist=True:
      - returns the signed lateral value at the time when Euclidean distance
        between ego and the other agent is minimal.
    """
    print("[DEBUG] Computing min_signed_lateral_offset...")
    ego_pos = np.asarray([loc[:2] for loc in simout["location"][ego_id]], dtype=float)
    ego_yaw = np.asarray(simout["yaw"][ego_id], dtype=float)

    best_score = np.inf
    best_signed = np.nan

    for key, pos_list in simout["location"].items():
        if key == ego_id:
            continue
        npc_pos = np.asarray([loc[:2] for loc in pos_list], dtype=float)
        n = min(len(ego_pos), len(npc_pos), len(ego_yaw))
        if n == 0:
            continue

        if at_min_dist:
            dists = np.linalg.norm(npc_pos[:n] - ego_pos[:n], axis=1)
            t_indices = [int(np.argmin(dists))]
        else:
            t_indices = range(n)

        for t in t_indices:
            dx, dy = npc_pos[t] - ego_pos[t]
            lateral = -dx * np.sin(ego_yaw[t]) + dy * np.cos(ego_yaw[t])  # signed

            score = float(np.hypot(dx, dy)) if at_min_dist else abs(lateral)
            if score < best_score:
                best_score = score
                best_signed = float(lateral)

    print(f"[DEBUG] min_signed_lateral_offset = {best_signed}")
    return best_signed

def speed_at_min_dist(simout, ego_id="ego"):
    ego_pos = np.asarray([loc[:2] for loc in simout["location"][ego_id]], dtype=float)
    ego_speed = np.asarray(simout["speed"][ego_id], dtype=float)

    best_d = np.inf
    best_t = None

    for key, pos_list in simout["location"].items():
        if key == ego_id:
            continue
        npc_pos = np.asarray([loc[:2] for loc in pos_list], dtype=float)
        T = min(len(ego_pos), len(npc_pos), len(ego_speed))
        if T == 0:
            continue

        d = np.linalg.norm(ego_pos[:T] - npc_pos[:T], axis=1)
        t = int(np.argmin(d))
        if float(d[t]) < best_d:
            best_d = float(d[t])
            best_t = t

    if best_t is None:
        return np.nan
    return float(ego_speed[best_t])

# -------------------------
# Compute Full Interaction Vector
# -------------------------

def compute_interaction_vector(simout, project_name="planer_final"):
    if project_name == "planer_final":
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
            max_speed_ego(simout),
            # max_deceleration_ego(simout),
            # min_signed_longitudinal_offset_car_lengths(simout, ego_length_m=None),
            # min_signed_lateral_offset(simout),
        ]
        print("[DEBUG] Full interaction vector computed.")
        return vector

    elif project_name == "autoware_final":
        print("[DEBUG] Computing autoware_final interaction vector...")
        vector = [
            # min_TTC(simout),
            min_dist(simout),
            speed_at_min_dist(simout),
            # avg_speed_ego(simout),
            max_deceleration_ego(simout),
            min_signed_longitudinal_offset_car_lengths(simout, ego_length_m=None, at_min_dist=True),
            min_signed_lateral_offset(simout, at_min_dist=True),
        ]
        print("[DEBUG] autoware_final interaction vector computed.")
        return vector

    raise ValueError(f"Unknown project_name: {project_name}")


# -------------------------
# Example Usage
# -------------------------

if __name__ == "__main__":
    # path = r"/home/user/testing/topic1-opensbt-aw/results_wandb/LoFi_GA_pop10_t06_00_00_seed13/jxhix0h9/rerun_hifi/simout/simout_S151.432725_0.459841_11.348637_23.177858_1.090066_2.525470_10.010484_16.557151_70.612127_0.705612_0.283797.json"
    # print("[DEBUG] Loading simout JSON file...")
    # with open(path, "r") as f:
    #     simout = json.load(f)

    # vector = compute_interaction_vector(simout, project_name="autoware_final")
    # print("\n[RESULT] Interaction-space vector:")
    # print(vector)
    
    
    path = r"/home/user/testing/topic1-opensbt-aw/results_aw_wandb/Hifi_GA_pop20_gen30_t03_00_00_seed4/xe42gjpn/simout/simout_S0.309816_7.142730_17.111465.json"
    print("[DEBUG] Loading simout JSON file...")
    with open(path, "r") as f:
        simout = json.load(f)

    vector = compute_interaction_vector(simout, project_name="autoware_final")
    print("\n[RESULT] Interaction-space vector:")
    print(vector)