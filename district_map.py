"""
Hexagonal district map for the Senate ABM.

Launch with:
    python district_map.py [map_seed]

Keybindings:
    Left / Right arrow  - cycle color mode
    1-6                 - jump to specific color mode
    L                   - toggle seat labels
"""

from __future__ import annotations

import math
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.colors import Normalize
from matplotlib.path import Path

from senate_model.model import SenateModel
from senate_model.agents import Senator
from senate_viz import RYG, compute_session_net_impact, build_tooltip, run_viz_loop


_HEX_DIRS = [(-1, 1), (-1, 0), (0, -1), (1, -1), (1, 0), (0, 1)]

_COLOR_MODES = [
    {"key": "approval",            "label": "Constituent Approval",                             "norm": (0, 1),  "cmap": RYG,           "val": lambda s: s.constituent_profile.approval},
    {"key": "net_impact",          "label": "Net Legislative Impact",                           "norm": (-0.5, 0.5), "cmap": RYG,       "val": None, "dynamic_norm": True},
    {"key": "reputation",          "label": "Senator Reputation",                               "norm": (0, 1),  "cmap": RYG,           "val": lambda s: s.reputation},
    {"key": "economic",            "label": "Economic Ideology (Interventionist <- -> Market)", "norm": (-1, 1), "cmap": plt.cm.PRGn,   "val": lambda s: s.constituent_profile.ideology.economic},
    {"key": "social",              "label": "Social Ideology (Progressive <- -> Traditional)",  "norm": (-1, 1), "cmap": plt.cm.RdBu_r, "val": lambda s: s.constituent_profile.ideology.social},
    {"key": "time_until_election", "label": "Sessions Until Election",                          "norm": (0, 6),  "cmap": plt.cm.Reds_r, "val": lambda s: s.time_until_election},
]


def hex_grid_positions(n: int) -> list[tuple[int, int]]:
    positions = []
    ring = 0
    while len(positions) < n:
        if ring == 0:
            positions.append((0, 0))
        else:
            x, y = ring, 0
            for dx, dy in _HEX_DIRS:
                for _ in range(ring):
                    if len(positions) >= n:
                        break
                    positions.append((x, y))
                    x += dx; y += dy
        ring += 1
    return positions[:n]


def country_shape(n_seats: int, map_seed: int) -> list[tuple[int, int]]:
    rng          = np.random.default_rng(map_seed)
    radius       = math.ceil(math.sqrt(n_seats / math.pi)) + 4
    all_positions = np.array(hex_grid_positions(radius * radius * 4))
    qs, rs       = all_positions[:, 0], all_positions[:, 1]
    pixel_pts    = np.column_stack([math.sqrt(3) * (qs + rs / 2), 1.5 * rs])

    n_harmonics = 3
    amps   = rng.uniform(0.05, 0.30, n_harmonics)
    phases = rng.uniform(0, 2 * math.pi, n_harmonics)
    freqs  = rng.integers(2, 7, n_harmonics)

    def blob_radius(angle):
        return 1.0 + sum(a * math.cos(f * angle + p) for a, p, f in zip(amps, phases, freqs))

    angles = np.linspace(0, 2 * math.pi, 360, endpoint=False)

    def count_inside(scale):
        poly  = [(math.cos(a) * blob_radius(a) * scale, math.sin(a) * blob_radius(a) * scale) for a in angles]
        shape = Path(poly + [poly[0]], closed=True)
        return shape.contains_points(pixel_pts).sum(), shape

    lo, hi = 1.0, float(radius)
    for _ in range(20):
        mid = (lo + hi) / 2
        if count_inside(mid)[0] < n_seats: lo = mid
        else: hi = mid

    _, shape = count_inside(hi)
    inside   = [tuple(p) for p in all_positions[shape.contains_points(pixel_pts)].tolist()]
    return inside[:n_seats] if len(inside) >= n_seats else hex_grid_positions(n_seats)


def axial_to_pixel(q: int, r: int, size: float) -> tuple[float, float]:
    return size * math.sqrt(3) * (q + r / 2), size * 1.5 * r


def draw_map(model: SenateModel, map_seed: int = 0) -> None:
    senators   = sorted(model.agents_by_type[Senator], key=lambda s: s.seat_id)
    n          = len(senators)
    positions  = country_shape(n, map_seed)
    hex_radius = 0.82

    impact_by_id = compute_session_net_impact(model)
    for mode in _COLOR_MODES:
        if mode["key"] == "net_impact":
            mode["val"] = lambda s, _imp=impact_by_id: _imp.get(s.unique_id, 0.0)
            break

    plt.rcParams["font.family"]    = "monospace"
    plt.rcParams["font.monospace"] = ["Consolas", "Courier New", "monospace"]
    try:
        plt.rcParams["keymap.yscale"].remove("l")
    except ValueError:
        pass

    fig     = plt.figure(figsize=(14, 10), facecolor="#2b2b2b")
    ax      = fig.add_axes([0.05, 0.05, 0.84, 0.88], facecolor="#2b2b2b")
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.65])
    ax.set_aspect("equal")
    ax.axis("off")

    mode_index   = [0]
    active_timer = [None]
    current_norm = [None]
    centers, tooltips, patches = [], [], []

    for senator, (q, r) in zip(senators, positions):
        px, py = axial_to_pixel(q, r, 1.0)
        patch  = RegularPolygon((px, py), numVertices=6, radius=hex_radius,
                                orientation=0, linewidth=0, edgecolor="none", facecolor="gray", zorder=2)
        ax.add_patch(patch)
        patches.append(patch)
        centers.append((px, py))
        tooltips.append(build_tooltip(senator, impact_by_id.get(senator.unique_id)))

    x_vals = [cx for cx, _ in centers]
    y_vals = [cy for _, cy in centers]
    pad    = hex_radius * 2
    ax.set_xlim(min(x_vals) - pad, max(x_vals) + pad)
    ax.set_ylim(min(y_vals) - pad, max(y_vals) + pad)
    ax.set_autoscale_on(False)

    sm   = plt.cm.ScalarMappable(); sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.ax.yaxis.label.set_color("white")
    cbar.ax.tick_params(colors="white")

    title = ax.set_title("", fontsize=12, color="white", pad=12)
    ax.text(0.5, -0.02, "Left/Right: cycle color mode  |  1-6: jump  |  L: toggle seat labels",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#888888")

    seat_labels = [
        ax.text(cx, cy, str(senator.seat_id), ha="center", va="center",
                fontsize=8, color="black", fontfamily="monospace", zorder=3, visible=False)
        for senator, (cx, cy) in zip(senators, centers)
    ]
    show_labels = [False]

    x_right = (min(x_vals) + max(x_vals)) / 2
    y_upper = (min(y_vals) + max(y_vals)) / 2

    def _update_cbar_and_title():
        mode = _COLOR_MODES[mode_index[0]]
        norm = current_norm[0] if current_norm[0] is not None else Normalize(*mode["norm"])
        sm.set_cmap(mode["cmap"]); sm.set_norm(norm)
        cbar.update_normal(sm)
        cbar.set_label(mode["label"])
        cbar.ax.yaxis.label.set_color("white")
        title.set_text(
            f"Senate Districts — {n} seats  |  Session {model.current_session}"
            f"  |  Map seed: {map_seed}"
            f"  |  [{mode_index[0] + 1}/{len(_COLOR_MODES)}] {mode['label']}"
        )
        ax.set_xlim(min(x_vals) - pad, max(x_vals) + pad)
        ax.set_ylim(min(y_vals) - pad, max(y_vals) + pad)
        fig.canvas.draw_idle()

    def apply_mode(new_index: int):
        if active_timer[0] is not None:
            active_timer[0].stop(); active_timer[0] = None

        mode   = _COLOR_MODES[new_index]
        values = [mode["val"](s) for s in senators]
        if mode.get("dynamic_norm"):
            abs_max = max(abs(min(values)), abs(max(values)), 1e-6)
            norm    = Normalize(-abs_max, abs_max)
        else:
            norm = Normalize(*mode["norm"])
        current_norm[0] = norm
        target_colors = [mode["cmap"](norm(v)) for v in values]
        start_colors  = [patch.get_facecolor() for patch in patches]
        steps         = 12

        def tick(frame=[0]):
            t = (frame[0] + 1) / steps
            t = t * t * (3 - 2 * t)
            for patch, src, dst in zip(patches, start_colors, target_colors):
                patch.set_facecolor([s + (d - s) * t for s, d in zip(src, dst)])
            fig.canvas.draw_idle()
            frame[0] += 1
            if frame[0] >= steps:
                active_timer[0].stop(); active_timer[0] = None
                mode_index[0] = new_index
                _update_cbar_and_title()

        timer = fig.canvas.new_timer(interval=120 // steps)
        timer.add_callback(tick)
        active_timer[0] = timer
        timer.start()

    apply_mode(0)

    annot = ax.annotate(
        "", xy=(0, 0), xytext=(15, 15), textcoords="offset points",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1e1e1e", ec="#555555", alpha=0.95),
        fontsize=7, fontfamily="monospace", color="white", zorder=10, visible=False,
    )

    def on_move(event):
        if event.inaxes != ax:
            annot.set_visible(False); fig.canvas.draw_idle(); return
        for i, (cx, cy) in enumerate(centers):
            if math.hypot(event.xdata - cx, event.ydata - cy) < hex_radius:
                annot.set_text(tooltips[i])
                annot.xy = (cx, cy)
                annot.set_position((-180 if cx > x_right else 15, -120 if cy > y_upper else 15))
                annot.set_visible(True); fig.canvas.draw_idle(); return
        annot.set_visible(False); fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            apply_mode((mode_index[0] + 1) % len(_COLOR_MODES))
        elif event.key == "left":
            apply_mode((mode_index[0] - 1) % len(_COLOR_MODES))
        elif event.key == "l":
            show_labels[0] = not show_labels[0]
            for lbl in seat_labels:
                lbl.set_visible(show_labels[0])
            fig.canvas.draw_idle()
        elif event.key.isdigit():
            idx = int(event.key) - 1
            if 0 <= idx < len(_COLOR_MODES):
                apply_mode(idx)

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show(block=True)


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    map_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    model    = SenateModel(n_seats=100, seed=42)
    run_viz_loop(model, draw_map, map_seed)
