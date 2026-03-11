"""
Senate chamber seating visualization.

Launch with:
    python chamber_map.py [map_seed]

Keybindings:
    Left / Right arrow  - cycle color mode
    1-9                 - jump to color mode
    C                   - toggle coalition overlay
    L                   - toggle seat labels
    S                   - sort by current color mode / return to arc
"""

from __future__ import annotations

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

from senate_model.model import SenateModel
from senate_model.agents import Senator
from senate_viz import RYG, compute_session_net_impact, build_tooltip, run_viz_loop

_COLOR_MODES = [
    {"key": "approval", "label": "Constituent Approval", "norm": (0, 1), "cmap": RYG,
     "val": lambda s: s.constituent_profile.approval},
    {"key": "reputation", "label": "Senator Reputation", "norm": (0, 1), "cmap": RYG, "val": lambda s: s.reputation},
    {"key": "net_impact", "label": "Net Legislative Impact", "norm": (-0.5, 0.5), "cmap": RYG, "val": None,
     "dynamic_norm": True},
    {"key": "vote_bloc", "label": "Voting Bloc (PCA on vote history)", "norm": (0, 1), "cmap": plt.cm.coolwarm,
     "val": None},
    {"key": "economic_district", "label": "District Econ Ideology (Interventionist <- -> Market)", "norm": (-1, 1),
     "cmap": plt.cm.PRGn, "val": lambda s: s.constituent_profile.ideology.economic},
    {"key": "social_district", "label": "District Social Ideology (Progressive <- -> Traditional)", "norm": (-1, 1),
     "cmap": plt.cm.RdBu_r, "val": lambda s: s.constituent_profile.ideology.social},
    {"key": "economic_personal", "label": "Personal Econ Ideology (Interventionist <- -> Market)", "norm": (-1, 1),
     "cmap": plt.cm.PRGn, "val": lambda s: s.personal_ideology.economic},
    {"key": "social_personal", "label": "Personal Social Ideology (Progressive <- -> Traditional)", "norm": (-1, 1),
     "cmap": plt.cm.RdBu_r, "val": lambda s: s.personal_ideology.social},
    {"key": "time_until_election", "label": "Sessions Until Election", "norm": (0, 6), "cmap": plt.cm.Reds_r,
     "val": lambda s: s.time_until_election},
]

_COALITION_COLORS = [
    "#4e9af1", "#f1a94e", "#a8e04e", "#e04e9a", "#9a4ef1",
    "#4ef1a8", "#f14e4e", "#4ef1e0", "#f1e04e", "#e0a84e",
    "#a84ef1", "#4ea8f1",
]


def _seat_order(senators: list[Senator]) -> list[Senator]:
    return sorted(senators, key=lambda s: s.personal_ideology.economic)


def _distribute_rows(n: int, n_rows: int) -> list[int]:
    weights = [i + 1 for i in range(n_rows)]
    total_w = sum(weights)
    counts = [max(1, round(w / total_w * n)) for w in weights]
    diff = n - sum(counts)
    for i in range(abs(diff)):
        counts[-(i % n_rows) - 1] += 1 if diff > 0 else -1
    return counts


def chamber_positions(n: int) -> list[tuple[float, float]]:
    n_rows = max(3, round(math.sqrt(n) * 0.65))
    seats_per_row = _distribute_rows(n, n_rows)
    pitch = 1.0
    aspect = 0.6
    rx_inner = (seats_per_row[0] * pitch) / (math.pi * (1 + aspect))
    rx_inner = max(rx_inner, pitch * 0.8)

    positions = []
    for row_idx, count in enumerate(seats_per_row):
        rx = rx_inner + row_idx * pitch
        ry = rx * aspect
        angles = np.linspace(math.radians(4), math.radians(176), count)
        for a in angles:
            positions.append((rx * math.cos(math.pi - a), ry * math.sin(a)))
    return positions


def sorted_positions(arc_positions: list[tuple[float, float]], values: list[float]) -> list[tuple[float, float]]:
    slot_order = sorted(range(len(arc_positions)), key=lambda i: arc_positions[i][0])
    ranked = sorted(range(len(values)), key=lambda i: values[i])
    result = [None] * len(values)
    for slot_rank, senator_idx in enumerate(ranked):
        result[senator_idx] = arc_positions[slot_order[slot_rank]]
    return result


def compute_vote_bloc_axis(model: SenateModel) -> dict[int, float] | None:
    senators = list(model.agents_by_type[Senator])
    uid_list = [s.unique_id for s in senators]
    uid_pos = {uid: i for i, uid in enumerate(uid_list)}
    entries = [e for e in getattr(model, "session_history", []) if e.get("votes")]
    if len(entries) < 2:
        return None

    matrix = np.full((len(uid_list), len(entries)), -1.0)
    for col, entry in enumerate(entries):
        for uid, vote in entry["votes"].items():
            if uid in uid_pos:
                matrix[uid_pos[uid], col] = 1.0 if vote == "yes" else 0.0

    for col in range(matrix.shape[1]):
        col_data = matrix[:, col]
        present = col_data[col_data >= 0]
        matrix[col_data < 0, col] = present.mean() if len(present) > 0 else 0.5

    matrix -= matrix.mean(axis=0)
    _, _, Vt = np.linalg.svd(matrix, full_matrices=False)
    pc1 = matrix @ Vt[0]
    lo, hi = pc1.min(), pc1.max()
    if hi - lo < 1e-9:
        return None
    normalised = (pc1 - lo) / (hi - lo)
    return {uid: float(normalised[i]) for i, uid in enumerate(uid_list)}


def compute_active_coalitions(model: SenateModel) -> list[set[int]]:
    if hasattr(model, "active_coalitions"):
        return list(model.active_coalitions)
    coalitions = []
    for entry in getattr(model, "session_history", []):
        coalition = entry.get("coalition")
        if coalition and isinstance(coalition, (set, list)):
            coalitions.append(set(coalition))
    return coalitions


def draw_chamber(model: SenateModel, map_seed: int = 0) -> None:
    senators = _seat_order(list(model.agents_by_type[Senator]))
    n = len(senators)
    uid_index = {s.unique_id: i for i, s in enumerate(senators)}
    arc_pos = chamber_positions(n)

    impact_by_id = compute_session_net_impact(model)
    vote_bloc_by_id = compute_vote_bloc_axis(model)
    for mode in _COLOR_MODES:
        if mode["key"] == "net_impact":
            mode["val"] = lambda s, _imp=impact_by_id: _imp.get(s.unique_id, 0.0)
        elif mode["key"] == "vote_bloc":
            mode["val"] = (lambda s, _b=vote_bloc_by_id: _b.get(s.unique_id, 0.5)
            if vote_bloc_by_id is not None else lambda s: 0.5)

    plt.rcParams["font.family"] = "monospace"
    plt.rcParams["font.monospace"] = ["Consolas", "Courier New", "monospace"]
    for key, binding in [("keymap.yscale", "l"), ("keymap.save", "s")]:
        try:
            plt.rcParams[key].remove(binding)
        except ValueError:
            pass

    fig = plt.figure(figsize=(16, 9), facecolor="#1a1a1a")
    ax = fig.add_axes([0.04, 0.06, 0.82, 0.88], facecolor="#1a1a1a")
    cbar_ax = fig.add_axes([0.88, 0.15, 0.018, 0.65])
    ax.set_aspect("equal")
    ax.axis("off")

    theta = np.linspace(0, math.pi, 100)
    ax.plot(0.03 * np.cos(theta), 0.03 * np.sin(theta), color="#555555", lw=1.5, zorder=1)
    ax.plot([-0.03, 0.03], [0, 0], color="#555555", lw=1.5, zorder=1)

    xs = [p[0] for p in arc_pos]
    ys = [p[1] for p in arc_pos]
    seat_w = 0.42

    patches_list = []
    centers = list(arc_pos)
    current_positions = list(arc_pos)
    tooltips = []

    for senator, (px, py) in zip(senators, arc_pos):
        rect = mpatches.FancyBboxPatch(
            (px - seat_w / 2, py - seat_w / 2), seat_w, seat_w,
            boxstyle="round,pad=0.005", linewidth=0.4,
            edgecolor="#333333", facecolor="gray", zorder=2,
        )
        ax.add_patch(rect)
        patches_list.append(rect)
        tooltips.append(build_tooltip(senator, impact_by_id.get(senator.unique_id)))

    pad = seat_w * 3
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(-pad, max(ys) + pad)
    ax.set_autoscale_on(False)

    sm = plt.cm.ScalarMappable();
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    title = ax.set_title("", fontsize=11, color="white", pad=10)
    ax.text(0.5, -0.015,
            "Left/Right: cycle color  |  1-9: jump  |  S: sort/unsort  |  C: coalitions  |  L: labels",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5, color="#666666")

    seat_labels = [
        ax.text(cx, cy, str(senator.seat_id), ha="center", va="center",
                fontsize=max(3, seat_w * 18), color="black", zorder=3, visible=False)
        for senator, (cx, cy) in zip(senators, arc_pos)
    ]
    show_labels = [False]
    mode_index = [0]
    active_timer = [None]
    current_norm = [None]
    overlay_mode = [None]
    overlay_artists: list = []

    def _clear_overlay():
        for a in overlay_artists:
            a.remove()
        overlay_artists.clear()

    def _draw_coalition_overlay():
        _clear_overlay()
        coalitions = compute_active_coalitions(model)
        if not coalitions:
            t = ax.text(0, max(ys) * 0.5, "No active coalitions found.",
                        color="#888888", ha="center", fontsize=8, zorder=5)
            overlay_artists.append(t)
            return
        legend_handles = []
        for ci, coalition in enumerate(coalitions):
            color = _COALITION_COLORS[ci % len(_COALITION_COLORS)]
            members = [uid_index[uid] for uid in coalition if uid in uid_index]
            for idx in members:
                cx, cy = centers[idx]
                h = mpatches.FancyBboxPatch(
                    (cx - seat_w * 0.6, cy - seat_w * 0.6), seat_w * 1.2, seat_w * 1.2,
                    boxstyle="round,pad=0.005", linewidth=1.2,
                    edgecolor=color, facecolor="none", zorder=4, alpha=0.85,
                )
                ax.add_patch(h)
                overlay_artists.append(h)
            legend_handles.append(mpatches.Patch(color=color, label=f"Coalition {ci + 1} ({len(members)} senators)"))
        leg = ax.legend(handles=legend_handles, loc="lower left", fontsize=7,
                        facecolor="#1e1e1e", edgecolor="#555555", labelcolor="white")
        overlay_artists.append(leg)

    def _refresh_overlay():
        if overlay_mode[0] == "coalition":
            _draw_coalition_overlay()
        else:
            _clear_overlay()
        fig.canvas.draw_idle()

    def _update_title():
        mode = _COLOR_MODES[mode_index[0]]
        ov_str = f"  |  overlay: {overlay_mode[0]}" if overlay_mode[0] else ""
        title.set_text(
            f"Senate Chamber — {n} senators  |  Session {model.current_session}"
            f"  |  [{mode_index[0] + 1}/{len(_COLOR_MODES)}] {mode['label']}{ov_str}"
        )

    def _update_cbar_and_title():
        mode = _COLOR_MODES[mode_index[0]]
        norm = current_norm[0] if current_norm[0] is not None else Normalize(*mode["norm"])
        sm.set_cmap(mode["cmap"]);
        sm.set_norm(norm)
        cbar.update_normal(sm)
        cbar.set_label(mode["label"])
        cbar.ax.yaxis.label.set_color("white")
        _update_title()
        fig.canvas.draw_idle()

    def _lerp_positions(start: list, end: list, steps: int = 18):
        if active_timer[0] is not None:
            active_timer[0].stop();
            active_timer[0] = None

        def tick(frame=[0]):
            frame[0] += 1
            t = frame[0] / steps
            t = t * t * (3 - 2 * t)
            for i, (patch, lbl) in enumerate(zip(patches_list, seat_labels)):
                sx, sy = start[i]
                ex, ey = end[i]
                nx, ny = sx + (ex - sx) * t, sy + (ey - sy) * t
                patch.set_x(nx - seat_w / 2)
                patch.set_y(ny - seat_w / 2)
                lbl.set_position((nx, ny))
                centers[i] = (nx, ny)
                current_positions[i] = (nx, ny)
            fig.canvas.draw_idle()
            if frame[0] >= steps:
                active_timer[0].stop();
                active_timer[0] = None
                _refresh_overlay()

        timer = fig.canvas.new_timer(interval=16)
        timer.add_callback(tick)
        active_timer[0] = timer
        timer.start()

    def apply_mode(new_index: int):
        if active_timer[0] is not None:
            active_timer[0].stop();
            active_timer[0] = None

        mode = _COLOR_MODES[new_index]
        values = [mode["val"](s) for s in senators]

        if mode.get("dynamic_norm"):
            abs_max = max(abs(min(values)), abs(max(values)), 1e-6)
            norm = Normalize(-abs_max, abs_max)
        else:
            norm = Normalize(*mode["norm"])
        current_norm[0] = norm

        target_colors = [mode["cmap"](norm(v)) for v in values]
        start_colors = [p.get_facecolor() for p in patches_list]
        steps = 12

        def tick(frame=[0]):
            frame[0] += 1
            t = frame[0] / steps
            t = t * t * (3 - 2 * t)
            for patch, src, dst in zip(patches_list, start_colors, target_colors):
                patch.set_facecolor([s + (d - s) * t for s, d in zip(src, dst)])
            fig.canvas.draw_idle()
            if frame[0] >= steps:
                active_timer[0].stop();
                active_timer[0] = None
                mode_index[0] = new_index
                _update_cbar_and_title()

        timer = fig.canvas.new_timer(interval=16)
        timer.add_callback(tick)
        active_timer[0] = timer
        timer.start()

    apply_mode(0)

    x_right = (min(xs) + max(xs)) / 2
    y_upper = (min(ys) + max(ys)) / 2

    annot = ax.annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1e1e1e", ec="#555555", alpha=0.95),
        fontsize=7, fontfamily="monospace", color="white", zorder=10, visible=False,
    )

    def on_move(event):
        if event.inaxes != ax:
            annot.set_visible(False);
            fig.canvas.draw_idle();
            return
        for i, (cx, cy) in enumerate(centers):
            if abs(event.xdata - cx) < seat_w and abs(event.ydata - cy) < seat_w:
                annot.set_text(tooltips[i])
                annot.xy = (cx, cy)
                annot.set_position((-200 if cx > x_right else 15,
                                    -140 if cy > y_upper else 15))
                annot.set_visible(True);
                fig.canvas.draw_idle();
                return
        annot.set_visible(False);
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            apply_mode((mode_index[0] + 1) % len(_COLOR_MODES))
        elif event.key == "left":
            apply_mode((mode_index[0] - 1) % len(_COLOR_MODES))
        elif event.key.isdigit():
            idx = int(event.key) - 1
            if 0 <= idx < len(_COLOR_MODES):
                apply_mode(idx)
        elif event.key == "l":
            show_labels[0] = not show_labels[0]
            for lbl in seat_labels:
                lbl.set_visible(show_labels[0])
            fig.canvas.draw_idle()
        elif event.key == "s":
            mode = _COLOR_MODES[mode_index[0]]
            values = [mode["val"](s) for s in senators]
            target = sorted_positions(arc_pos, values)
            at_arc = all(
                abs(current_positions[i][0] - arc_pos[i][0]) < 1e-6 and
                abs(current_positions[i][1] - arc_pos[i][1]) < 1e-6
                for i in range(n)
            )
            _lerp_positions(list(current_positions), arc_pos if not at_arc else target)
        elif event.key == "c":
            overlay_mode[0] = None if overlay_mode[0] == "coalition" else "coalition"
            _refresh_overlay()
            _update_title()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    map_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model = SenateModel(n_seats=100, seed=42)
    run_viz_loop(model, draw_chamber, map_seed)
