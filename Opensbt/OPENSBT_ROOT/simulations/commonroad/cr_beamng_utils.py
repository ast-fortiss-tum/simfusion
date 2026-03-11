"""
Small, single-purpose helpers for CommonRoad&BeamNG and OpenSCENARIO workflows.
"""

import random
import os, re
import math
from copy import deepcopy
from pathlib import Path
from matplotlib.axes import Axes
import numpy as np
from typing import Any, Dict, List, Tuple, Sequence
from matplotlib.patches import Polygon
import xml.etree.ElementTree as ET
import json, hashlib
from pathlib import Path


__all__ = [
    "parse_to_array",
    "pad_array",
    "pad_trajectory_3d",
    "create_scenario_instance",
    "draw_goal_from_goalstate"
]

TIME_MIN, TIME_MAX = 1, 400
X_MAX_DEFAULT = 220.0

# ---------- Arrays / parsing ----------
def parse_to_array(s: object) -> np.ndarray:
    """Convert text with numbers into a 1D numpy float array."""
    s = str(s).strip()
    if "," in s:
        numbers = re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+", s)
        return np.array([float(n) for n in numbers])
    else:
        try:
            return np.array([float(s)])
        except:
            return np.array([])


def pad_array(arr: Sequence[float], target_len: int) -> List[float]:
    """
    Truncate longer arrays to match the maximum length.
    Pad shorter arrays to match the maximum length by repeating the last value.
    """
    if len(arr) > target_len:
        # Truncate from the end
        return arr[:target_len]

    if len(arr) < target_len:
        # Repeat the last value to fill the gap
        last_value = arr[-1]
        padding = [last_value] * (target_len - len(arr))
        return arr + padding

    return arr


def pad_trajectory_3d(
    traj_list: Sequence[Tuple[float, float, float]],
    target_len: int,
) -> List[Tuple[float, float, float]]:
    """Pad 3D trajectory by repeating the last position."""
    if len(traj_list) > target_len:
        return traj_list[:target_len]

    if len(traj_list) < target_len:
        # Repeat the last position to fill the gap
        last_pos = traj_list[-1]
        padding = [last_pos] * (target_len - len(traj_list))
        return traj_list + padding

    return traj_list


# ---------- Visualization ----------
def _rot_rect(cx: float, cy: float, L: float, W: float, theta: float) -> np.ndarray:
    """
    Compute corners of a rectangle centered at (cx, cy) with size (L, W)
    rotated by `theta` radians. Returns an array of shape (4, 2).
    """
    dx, dy = L / 2.0, W / 2.0
    P = np.array([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (P @ R.T) + np.array([cx, cy])

def draw_goal_from_goalstate(
    pp_set,
    ax: Axes,
    fc: str = "#00FFFF",
    ec: str = "#00CCCC",
    alpha: float = 0.35,
    lw: float = 2.0,
    z: int = 60,
) -> None:
    """
    Draw the first rectangular goal region from a PlanningProblemSet onto `ax`.
    """
    pps = list(getattr(pp_set, "planning_problem_dict", {}).values())
    for pp in pps:
        for st in getattr(pp.goal, "state_list", []) or []:
            pos = getattr(st, "position", None)
            shp = getattr(pos, "shape", None)
            if shp and all(hasattr(shp, a) for a in ("center", "length", "width")):
                cx, cy = float(shp.center.x), float(shp.center.y)
                L, W = float(shp.length), float(shp.width)
                theta = float(getattr(shp, "orientation", 0.0))
                poly = _rot_rect(cx, cy, L, W, theta)
                ax.add_patch(
                    Polygon(
                        poly,
                        True,
                        facecolor=fc,
                        edgecolor=ec,
                        alpha=alpha,
                        linewidth=lw,
                        zorder=z,
                    )
                )
                ax.text(
                    cx,
                    cy,
                    "GOAL",
                    ha="center",
                    va="center",
                    fontsize=11,
                    weight="bold",
                    zorder=z + 1,
                )
                return


# ---------- XML helpers ----------
def _find_planning_problem(root: ET.Element, pp_id: int) -> ET.Element:
    """
    Find planningProblem by id. Checks numeric and if needed string format in XML.
    Raises ValueError if not found.
    """
    pp = root.find(f".//planningProblem[@id='{pp_id}']")
    if pp is None:
        for cand in root.findall(".//planningProblem"):
            if cand.get("id") == str(pp_id):
                return cand
        raise ValueError(f"planningProblem id='{pp_id}' not found.")
    return pp

def _get_or_create_velocity_interval(
    goal_el: ET.Element,
) -> Tuple[ET.Element, ET.Element]:
    """
    Ensure goal_el/velocity has intervalStart and intervalEnd children.
    Returns (vmin_el, vmax_el). Clears existing children if needed.
    """
    vel = goal_el.find("velocity")
    if vel is None:
        vel = ET.SubElement(goal_el, "velocity")
    else:
        for ch in list(vel):
            vel.remove(ch)
    vmin_el = ET.SubElement(vel, "intervalStart")
    vmax_el = ET.SubElement(vel, "intervalEnd")
    return vmin_el, vmax_el

def _set_text(el: ET.Element, value: object) -> None:
    """Set element text to str(value)."""
    el.text = str(value)


def create_scenario_instance(filename: str, values_dict: dict, outfolder=None) -> str:
    """
    Updates planningProblem -> goalState, velocity interval(ego) and generates lanelets & NPCs.
    """
    # --- Load XML ---
    tree = ET.parse(filename)
    root = tree.getroot()

    # find planningProblem id
    pp_id = int(values_dict.get("planning_problem_id", 3001))
    # find planningProblem element
    pp = _find_planning_problem(root, pp_id)
    # --- GoalState updates ---
    goal = pp.find("goalState")
    if goal is None:
        raise ValueError("goalState node not found.")

    # --- Goal center update ---
    if ("goal_center_x" in values_dict) or ("goal_center_y" in values_dict):
        # find rectangle center(ET.Element)
        rect_center_x = goal.find("position/rectangle/center/x")
        rect_center_y = goal.find("position/rectangle/center/y")

        if rect_center_x is not None and rect_center_y is not None:
            if "goal_center_x" in values_dict:
                _set_text(rect_center_x, values_dict["goal_center_x"])
            if "goal_center_y" in values_dict:
                _set_text(rect_center_y, values_dict["goal_center_y"])
        else:
            raise ValueError(
                "Goal center cannot be found; goal position could not be updated."
            )

    # --- Goal velocity interval (min/max) ---
    if ("goal_velocity_min" in values_dict) or ("goal_velocity_max" in values_dict):
        vmin_el, vmax_el = _get_or_create_velocity_interval(goal)
        vmin = values_dict.get("goal_velocity_min", vmin_el.text or 0.0)
        vmax = values_dict.get("goal_velocity_max", vmax_el.text or vmin)
        vmin, vmax = _ensure_min_le_max(vmin, vmax)
        _set_text(vmin_el, vmin)
        _set_text(vmax_el, vmax)

    # check if x_max is provided otherwise use default
    x_max = float(_default(values_dict, "x_max", X_MAX_DEFAULT))
    # check if lane width, lane dx, lane y0 provided otherwise use defaults
    lw      = float(_default(values_dict, "lane_width", 3.5))
    lane_dx = float(_default(values_dict, "lane_dx", 1.0))
    lane_y0 = float(_default(values_dict, "lane_y0", 0.0))
    # check if lane_count is provided otherwise set None 
    want_n  = values_dict.get("lane_count", None)

    # Read existing lanelets and their geometric centers
    existing_ids, lane_centers_y, next_lane_id = _read_existing_lanelets(root)

    # Ego lane center y (planning problem initialState) - exclude from NPC placement
    ego_y_text = pp.findtext("initialState/position/point/y", default="0.0")
    ego_y = float(ego_y_text)
    ego_x_text = pp.findtext("initialState/position/point/x", default="0.0")
    ego_x = float(ego_x_text)

    # Find obstacle 44 y (exclude lane)
    obs44 = root.find(".//dynamicObstacle[@id='44']/initialState/position/point/y")
    obs44_y = float(obs44.text) if obs44 is not None else None

    obs44_lane_y = None
    if obs44_y is not None and lane_centers_y:
        obs44_lane_y = min(lane_centers_y, key=lambda yc: abs(yc - obs44_y))
    
    # if lane_count provided: add that many new lanelets
    if want_n is not None:
        n_add = int(want_n)
        # reference height: just above current top (or start from lane_y0 - lw so first added is at lane_y0)
        top_base = (max(lane_centers_y) if lane_centers_y else (lane_y0 - lw))
        under_id = _find_lane_id_by_center(root, top_base, tol=1e-6) if lane_centers_y else None

        for i in range(n_add):
            # new lane center y
            yc = top_base + (i + 1) * lw
            new_id = next_lane_id + i

            # adjacency: no lane on the left (upper side). Right neighbor = the lane just below.
            if i == 0:
                right_ref = under_id  # last existing top lane (may be None if none exist)
            else:
                right_ref = new_id - 1

            _add_lanelet(root, new_id, yc, lw, x_max, lane_dx, adj_left=None, adj_right=right_ref)
            lane_centers_y.append(yc)

        next_lane_id += n_add

    # ----- NPC synthesis -----
    num_npc = int(_default(values_dict, "num_npc", 0))
    if num_npc > 0:
        # Random generator for NPC placement/trajectories
        seed_raw = _default(values_dict, "npc_seed", 42)
        seed = int(round(float(seed_raw)))
        rng = random.Random(seed)

        # Initial npc position
        x0_base = float(_default(values_dict, "npc_x0", 40.0))
        # NPC velocity/acceleration
        v_max = float(_default(values_dict, "npc_v_max", 8.0))
        a_nom = float(_default(values_dict, "npc_acc", 1.0)) 
        # NPC size and heading
        npc_L = float(_default(values_dict, "npc_length", 4.7))
        npc_W = float(_default(values_dict, "npc_width", 1.8))
        heading = float(_default(values_dict, "npc_heading", 0.0))
        # npc time step(same as scenario)
        dt = 0.1
        # npc trajectory horizon
        horizon = float(_default(values_dict, "npc_horizon", 40.0))
        # npc buffer distance for placement
        npc_buffer = float(_default(values_dict, "npc_buffer_min", 8.0))

        # Trajectory configuration
        #traj_mode = str(_default(values_dict, "npc_traj", "straight")).lower()
        traj_mode = str(_default(values_dict, "npc_traj", "cutout")).lower()
        # npc cut-in time constant
        cutin_tx = float(_default(values_dict, "cutin_tx", 0.1))
        # cut-in delay
        cutin_delay = float(_default(values_dict, "cutin_delay", 0.1))
        # cut-in profile
        cutin_profile = str(_default(values_dict, "cutin_profile", "sigmoid")).lower()
        # cut-in profile steepness
        cutin_k = float(_default(values_dict, "cutin_k", 8.0))
        # cut-in target end velocity
        v_end_opt = values_dict.get("cutin_v_end", None)
        v_end_opt = float(v_end_opt) if v_end_opt is not None else None

        # Randomize per-NPC mode: straight vs cut-in
        choices = _default(values_dict, "npc_traj_choices", ["cutin", "cutout", "straight"])
        weights = _default(values_dict, "npc_traj_probs",  [0.7, 0.0, 0.3])

        next_id = _next_free_dyn_id(root)

        # Track already placed NPC positions to avoid overlaps
        placed_xy = []
        obs = []
        for dob in root.findall("./dynamicObstacle"):
            try:
                obs_id = int(dob.get("id"))
            except:
                obs_id = 10**9
            obs.append((obs_id, dob))

        for _, dob in sorted(obs, key=lambda t: t[0]):
            x_el = dob.find("./initialState/position/point/x")
            y_el = dob.find("./initialState/position/point/y")
            if x_el is not None and y_el is not None:
                try:
                    placed_xy.append((float(x_el.text), float(y_el.text)))
                except:
                    pass

        placed_x_all: List[float] = [float(x) for (x, y) in placed_xy]

        # add ego position to placed positions to avoid npc-ego collision
        placed_xy.append((ego_x, ego_y))
        placed_x_all.append(ego_x)

        npc_long_gap = float(_default(values_dict, "npc_long_gap", 8.0))

        # possible lane centers for NPC placement (exclude ego and obs44 lanes(to prevent onstacle collision))
        # potential bug: if only 2 lane exist and one is ego lane, other is obs44 lane -> no lane left
        lane_centers_non_ego = sorted([
            yc for yc in lane_centers_y
            if abs(yc - ego_y) > 1.5
            and (obs44_lane_y is None or abs(yc - obs44_lane_y) > 1e-6)
        ])

        lane_pool = lane_centers_non_ego.copy()
        rng.shuffle(lane_pool)

        # Generate NPCs
        for i in range(num_npc):

            # select trajectory mode
            traj_mode_i = rng.choices(choices, weights=weights, k=1)[0]
            traj_mode_i = str(traj_mode_i).lower()

            # # select initial lane based on mode
            # if traj_mode_i in ("cutout", "straight"):
            #     # must start in ego lane
            #     y0 = float(ego_y)
            # else:
            #     # cutin starts in non-ego lane
            #     y0 = float(rng.choice(lane_centers_non_ego))

            # # select initial lane based on mode
            # if traj_mode_i in ("cutin", "straight"):
            #     # cutin starts in non-ego lane
            #     y0 = float(rng.choice(lane_centers_non_ego))
            # else:
            #     # must start in ego lane
            #     y0 = float(ego_y)

            if traj_mode_i == "straight":
                y0 = float(ego_y)
            else:  # cutin/cutout
                y0 = float(rng.choice(lane_centers_non_ego))

            t0 = 0.0

            # choose x0_init
            if traj_mode_i == "straight":
                x0_init = float(x0_base)
            elif traj_mode_i == "cutout":
                # cutout: also typically after ego, otherwise it may start behind and never matter
                #x0_init = max(float(x0_base), float(ego_x) + npc_buffer)
                x0_init = float(x0_base)
            else:
                # cutin: can be anywhere
                x0_init = float(x0_base)

            # place with gap+buffer (now using correct y0)
            x0 = _place_with_longitudinal_gap_and_buffer(
                x0_init=x0_init,
                y0=y0,
                placed_xy=placed_xy,
                placed_x_all=placed_x_all,
                x_gap=npc_long_gap,
                buf=npc_buffer,
                x_max=x_max,
            )

            placed_xy.append((x0, y0))
            placed_x_all.append(x0)

            # generate trajectory
            if traj_mode_i == "straight":
                traj = _make_straight_traj(
                    t0=t0, x0=x0, y0=y0, v0=0.0, T=t0 + horizon, dt=dt,
                    x_max=x_max, a=0.25, v_max=v_max
                )

            else:
                # choose target lane y_target (must differ when lane change expected)
                if traj_mode_i == "cutin":
                    y_tgt = float(ego_y)
                elif traj_mode_i == "cutout":
                    #y_tgt = float(rng.choice(lane_centers_non_ego))
                    # allow cutout target to be ANY lane (including ego lane), but not the current lane
                    all_lanes = [float(yc) for yc in lane_centers_y]

                    # optional: exclude obs44 lane from targets unless explicitly allowed
                    allow_obs44_target = bool(_default(values_dict, "allow_obs44_target", False))
                    if (not allow_obs44_target) and (obs44_lane_y is not None):
                        all_lanes = [yc for yc in all_lanes if abs(yc - float(obs44_lane_y)) > 1e-6]

                    # remove current lane to ensure an actual lane change
                    candidates = [yc for yc in all_lanes if abs(yc - float(y0)) > 1e-6]

                    # fallback: if only one lane remains, just keep the same (no change possible)
                    y_tgt = float(rng.choice(candidates)) if candidates else float(y0)

                # generate trajectory
                traj = _make_cutin_traj(
                    t0=t0, x0=x0, y0=y0, v0=0.0, T=t0 + horizon, dt=dt,
                    y_target=y_tgt, tx=cutin_tx, delay=cutin_delay,
                    profile=cutin_profile, k=cutin_k, v_end=v_end_opt,
                    x_max=x_max, a=a_nom, v_max=v_max,
                    hold_after_cutin=False, v_hold=None)

            dob = _make_dynamic_obstacle(
                new_id=next_id + i,
                length=npc_L, width=npc_W,
                x0=x0, y0=y0, v0=0.0, heading=heading,
                traj=traj
            )

            root.append(dob)          
    
    _sync_all_adjacency(root)

    # --- Output path and suffix ---
    if outfolder is not None:
        Path(outfolder).mkdir(parents=True, exist_ok=True)
        filename = os.path.join(outfolder, os.path.basename(filename))

    base, ext = os.path.splitext(filename)
    suffix = ""
    for _, v in values_dict.items():
        suffix += f"_{v}"
    new_file = f"{base}{suffix}{ext}"

    tree.write(new_file, encoding="utf-8", xml_declaration=True)
    _write_pretty(tree, new_file)
    return new_file

# ---------- Trajectory generation for NPCs ----------
def _ensure_min_le_max(vmin, vmax) -> Tuple[float, float]:
    """Ensure vmin <= vmax, swap if needed. Returns (vmin, vmax) as floats."""
    vmin, vmax = float(vmin), float(vmax)
    return (vmin, vmax) if vmin <= vmax else (vmax, vmin)

def _next_free_dyn_id(root: ET.Element) -> int:
    """Find the next free dynamicObstacle id (max existing + 1)."""
    ids = []
    for dob in root.findall("./dynamicObstacle"):
        try:
            ids.append(int(dob.get("id")))
        except:
            pass
    return (max(ids) + 1) if ids else 40

def _cr_point(parent: ET.Element, x: float, y: float) -> ET.Element:
    """
    Create CommonRoad point element with given x,y under parent.
    """
    pos = ET.SubElement(parent, "position")
    pt = ET.SubElement(pos, "point")
    _set_text(ET.SubElement(pt, "x"), x)
    _set_text(ET.SubElement(pt, "y"), y)
    return pos

def _cr_time(parent: ET.Element, t_int: int) -> ET.Element:
    """Create CommonRoad time element with exact integer time under parent."""
    tim = ET.SubElement(parent, "time")
    ET.SubElement(tim, "exact").text = str(int(t_int))
    return tim

def _cr_exact(parent: ET.Element, tag: str, val: float) -> ET.Element:
    """Create CommonRoad exact element with given tag and value under parent."""
    el = ET.SubElement(parent, tag)
    _set_text(ET.SubElement(el, "exact"), val)
    return el

def _make_dynamic_obstacle(
    new_id: int,
    length: float,
    width: float,
    x0: float,
    y0: float,
    v0: float,
    heading: float,
    traj: List[Tuple[float, float, float, float]],
) -> ET.Element:
    """Create dynamicObstacle element with given parameters and trajectory."""
    dob = ET.Element("dynamicObstacle", id=str(new_id))
    _set_text(ET.SubElement(dob, "type"), "car")

    shp = ET.SubElement(dob, "shape")
    rect = ET.SubElement(shp, "rectangle")
    _set_text(ET.SubElement(rect, "length"), length)
    _set_text(ET.SubElement(rect, "width"), width)

    init = ET.SubElement(dob, "initialState")
    _cr_point(init, x0, y0)
    _cr_exact(init, "orientation", heading)
    _cr_time(init, 0)
    _cr_exact(init, "velocity", v0)
    _cr_exact(init, "yawRate", 0.0)
    _cr_exact(init, "slipAngle", 0.0)

    tr = ET.SubElement(dob, "trajectory")
    k = 1
    for (t, x, y, v) in traj:
        st = ET.SubElement(tr, "state")
        _cr_point(st, x, y)
        _cr_exact(st, "orientation", heading)
        _cr_time(st, t)
        _cr_exact(st, "velocity", v)
        k += 1
    return dob


def _default(values: Dict[str, Any], key: str, fallback):
    """Get value from dict or return fallback if not found or None."""
    return values[key] if key in values and values[key] is not None else fallback

def _profile(alpha: float, kind: str = "sigmoid", k: float = 10.0) -> float:
    """alpha∈[0,1] -> [0,1]. kind: 'linear' | 'sigmoid'."""
    a = max(0.0, min(1.0, alpha))
    if kind == "linear":
        return a

    s = 1.0 / (1.0 + math.exp(-k * (a - 0.5)))
    s0 = 1.0 / (1.0 + math.exp(-k * (0.0 - 0.5)))
    s1 = 1.0 / (1.0 + math.exp(-k * (1.0 - 0.5)))
    return (s - s0) / (s1 - s0 + 1e-12)

def _make_straight_traj(t0, x0, y0, v0, T, dt, x_max=X_MAX_DEFAULT, a=2.0, v_max=8.0):
    """
    Straight trajectory from (x0,y0) with initial velocity v0 over time T.
    Velocity increases with acceleration a up to v_max.
    Returns list of (t, x, y, v).
    """
    n = int(math.ceil((T - t0) / dt))
    out, t_start = [], 1
    t_float, x, v = t0, x0, v0

    for i in range(n):
        t_float += dt
        y = y0

        v = min(float(v), float(v_max))
        v = min(v + a * dt, v_max)

        x = x + v * dt
        t_tick = min(t_start + i, TIME_MAX)
        if x >= x_max:
            out.append((t_tick, x_max, y, v))
            break
        out.append((t_tick, x, y, v))

    return out

def _make_cutin_traj(
    t0, x0, y0, v0, T, dt, y_target,
    tx=3.0, delay=0.0,
    profile="sigmoid", k=10.0, v_end=None, x_max=X_MAX_DEFAULT,
    a=2.0, v_max=8.0, hold_after_cutin=True, v_hold=None,
):
    # number of dt steps (float time horizon)
    n = int(math.ceil((T - t0) / dt))
    out, t_start = [], 1

    # velocity target
    v_target = float(v_end) if v_end is not None else float(v_max)
    v_target = min(v_target, float(v_max))

    start_step = int(math.ceil(max(0.0, delay) / dt))
    dur_steps  = max(1, int(math.ceil(max(0.0, tx) / dt)))
    end_step   = start_step + dur_steps

    x = float(x0)
    v = float(v0)

    for i in range(n):
        # integer tick used in XML
        t_tick = min(t_start + i, TIME_MAX)

        # ---- y policy ----
        if i < start_step:
            y = float(y0)
        elif i < end_step:
            # progress in [0,1] over dur_steps
            alpha = (i - start_step) / float(dur_steps)
            y = float(y0) + _profile(alpha, profile, k) * (float(y_target) - float(y0))
        else:
            y = float(y_target)

        # ---- v policy ----
        if hold_after_cutin and (i >= end_step):
            v_hold_min = 0.8 * v_target
            if v_hold is None:
                if v < v_hold_min:
                    v = min(v + a * dt, v_target, v_max)
                else:
                    v_hold = v

            if v_hold is not None:
                v = float(v_hold)
        else:
            # accelerate/decelerate towards v_target with clamp to v_max
            if v < v_target:
                v = min(v + a * dt, v_target, v_max)
            else:
                v = max(v - a * dt, v_target)
            v = min(v, float(v_max))

        # ---- x integration ----
        x = x + v * dt

        if x >= x_max:
            out.append((t_tick, float(x_max), y, v))
            break
        out.append((t_tick, x, y, v))

    return out



def _write_pretty(tree: ET.ElementTree, out_path: str) -> None:
    try:
        ET.indent(tree, space="  ", level=0)
        tree.write(out_path, encoding="utf-8", xml_declaration=True, short_empty_elements=False)
    except AttributeError:
        from xml.dom import minidom
        xml_bytes = ET.tostring(tree.getroot(), encoding="utf-8")
        pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
        pretty = "\n".join(line for line in pretty.splitlines() if line.strip())
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(pretty)

def _read_existing_lanelets(root: ET.Element):
    """
    Reads existing lanelets in the CommonRoad XML tree and returns their IDs and center y-coordinates.
    Also returns the next available lanelet ID.
    """
    ids = []
    centers = []
    for ln in root.findall("./lanelet"):
        try:
            ids.append(int(ln.get("id")))
        except:
            continue
        lb0 = ln.find("./leftBound/point[1]/y")
        rb0 = ln.find("./rightBound/point[1]/y")
        if lb0 is not None and rb0 is not None:
            yL = float(lb0.text); yR = float(rb0.text)
            centers.append(0.5*(yL+yR))
    next_id = (max(ids)+1) if ids else 1
    return ids, centers, next_id

def _lane_polyline(y_const: float, x_max: float, dx: float):
    """Generate lane polyline points at constant y over x from 0 to x_max with step dx."""
    xs = [i*dx for i in range(int(x_max/dx)+1)]
    return [(x, y_const) for x in xs]

def _add_lanelet(root: ET.Element, lane_id: int, y_center: float, w: float,
                 x_max: float, dx: float, adj_left: int|None, adj_right: int|None):
    """Add a lanelet element to the XML tree."""
    ln = ET.SubElement(root, "lanelet", id=str(lane_id))
    yL, yR = y_center + 0.5*w, y_center - 0.5*w

    left = ET.SubElement(ln, "leftBound")
    for x,y in _lane_polyline(yL, x_max, dx):
        pt = ET.SubElement(left, "point")
        ET.SubElement(pt,"x").text=str(x); ET.SubElement(pt,"y").text=str(y)
    ET.SubElement(left, "lineMarking").text = "dashed"

    right = ET.SubElement(ln, "rightBound")
    for x,y in _lane_polyline(yR, x_max, dx):
        pt = ET.SubElement(right, "point")
        ET.SubElement(pt,"x").text=str(x); ET.SubElement(pt,"y").text=str(y)
    ET.SubElement(right, "lineMarking").text = "dashed"

    if adj_left is not None:
        al = ET.SubElement(ln, "adjacentLeft"); al.set("ref", str(adj_left)); al.set("drivingDir","same")
    if adj_right is not None:
        ar = ET.SubElement(ln, "adjacentRight"); ar.set("ref", str(adj_right)); ar.set("drivingDir","same")
    ET.SubElement(ln, "laneletType").text = "highway"

def _find_lane_id_by_center(root: ET.Element, y_center: float, tol=1e-6):
    """Find lanelet ID by center y-coordinate within a tolerance."""
    for ln in root.findall("./lanelet"):
        lb0 = ln.find("./leftBound/point[1]/y")
        rb0 = ln.find("./rightBound/point[1]/y")
        if lb0 is None or rb0 is None: 
            continue
        yc = 0.5*(float(lb0.text)+float(rb0.text))
        if abs(yc - y_center) <= tol:
            return int(ln.get("id"))
    return None


def _lanelet_by_id(root: ET.Element, lane_id: int) -> ET.Element | None:
    """Find lanelet element by its ID."""
    return root.find(f"./lanelet[@id='{lane_id}']")

def _get_adj_ref(ln: ET.Element, side: str) -> int | None:
    tag = "adjacentLeft" if side == "left" else "adjacentRight"
    el = ln.find(tag)
    if el is not None and el.get("ref"):
        try:
            return int(el.get("ref"))
        except:
            return None
    return None

def _set_adjacent(ln: ET.Element, side: str, ref_id: int | None) -> None:
    """Set or remove adjacent lanelet reference on given side."""
    tag = "adjacentLeft" if side == "left" else "adjacentRight"
    el = ln.find(tag)
    if ref_id is None:
        if el is not None:
            ln.remove(el)
        return
    if el is None:
        el = ET.SubElement(ln, tag)
    el.set("ref", str(ref_id))
    el.set("drivingDir", "same")

def _ensure_pair(root: ET.Element, a_id: int, side: str, b_id: int) -> None:
    """Ensure that if lanelet a has b as adjacent on side, then b has a as adjacent on opposite side."""
    a = _lanelet_by_id(root, a_id)
    b = _lanelet_by_id(root, b_id)
    if a is None or b is None:
        return
    if side == "right":
        _set_adjacent(b, "left", a_id)
    else:  # side == "left"
        _set_adjacent(b, "right", a_id)

def _sync_all_adjacency(root: ET.Element) -> None:
    """Check all lanelets and ensure adjacency is mutual."""
    for ln in root.findall("./lanelet"):
        try:
            my_id = int(ln.get("id"))
        except:
            continue
        r = _get_adj_ref(ln, "right")
        l = _get_adj_ref(ln, "left")
        if r is not None:
            _ensure_pair(root, my_id, "right", r)
        if l is not None:
            _ensure_pair(root, my_id, "left", l)


def _place_with_buffer(x0_init: float, y0: float, placed: List[Tuple[float,float]],
                       buf: float, x_max: float, max_tries: int = 200) -> float:
    """
    Place x0_init at y0 avoiding collisions with placed positions within buf distance.
    If no valid position found within max_tries, returns the farthest valid position.
    """
    x0 = x0_init
    tries = 0
    # x max
    hard_cap = x_max - buf
    while True:
        # check x max
        if x0 > hard_cap:
            x0 = hard_cap
            break
        # check collisions
        ok = True
        for (px, py) in placed:
            if math.hypot(x0 - px, y0 - py) < buf:
                ok = False
                break
        if ok:
            break
        x0 += buf
        tries += 1
        if tries > max_tries:
            # give up and return current x0
            break
    return x0

def _place_with_longitudinal_gap(
    x0_init: float,
    placed_x: List[float],
    gap: float,
    x_max: float,
    max_tries: int = 200,
) -> float:
    """
    Shift x0 forward in steps of gap until it is at least gap
    away from every x in placed_x.

    Returns x0 clamped to x_max - gap (hard cap).
    """
    gap = float(gap)
    if gap <= 0.0:
        return float(min(x0_init, x_max))

    x0 = float(x0_init)
    hard_cap = float(x_max) - gap
    tries = 0

    while True:
        if x0 > hard_cap:
            return hard_cap
        ok = True
        for px in placed_x:
            if abs(x0 - float(px)) < gap:
                ok = False
                break
        if ok:
            return x0
        x0 += gap
        tries += 1
        if tries > max_tries:
            return min(x0, hard_cap)
        

def _place_with_longitudinal_gap_and_buffer(
    x0_init: float,
    y0: float,
    placed_xy: List[Tuple[float, float]],
    placed_x_all: List[float],
    x_gap: float,
    buf: float,
    x_max: float,
    max_tries: int = 200,
) -> float:
    """
    Two-stage placement:
      1) enforce 1D longitudinal x-gap against all previously placed x positions
      2) enforce 2D Euclidean buffer against placed (x,y) positions
    """
    x0 = _place_with_longitudinal_gap(x0_init, placed_x_all, x_gap, x_max, max_tries=max_tries)
    x0 = _place_with_buffer(x0, y0, placed_xy, buf, x_max, max_tries=max_tries)
    return x0


def hash_from_ordered_params(norm_params, n=20):
    payload = json.dumps(norm_params, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]

def normalize_by_order(values_dict, keys):
    INT_KEYS = {"num_npc", "lane_count", "npc_seed"}
    FLOAT_PREC = 6

    out = {}
    for k in keys:
        v = values_dict.get(k, None)
        if v is None:
            out[k] = None
        elif isinstance(v, (int, float)):
            out[k] = int(v) if k in INT_KEYS else round(float(v), FLOAT_PREC)
        elif isinstance(v, str):
            try:
                fv = float(v)
                out[k] = int(fv) if k in INT_KEYS else round(fv, FLOAT_PREC)
            except:
                out[k] = v
        else:
            out[k] = str(v)
    return out
