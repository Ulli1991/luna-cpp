#!/usr/bin/env python3
"""Draws docs/figures/flow_step.png, the control flow of one hydro run, in LUNA's figure style."""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("figstyle", os.path.join(HERE, "figstyle.py"))
figstyle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(figstyle)
figstyle.apply_style()

INK, FAINT = "#222222", "#6b6b6b"
SETUP = ("#eef1f5", "#7c8796")
HYDRO = ("#dbe9f6", "#1f77b4")
MESH = ("#dff0dc", "#2ca02c")
TEST = ("#fde8d3", "#ff7f0e")


def box(ax, x, y, w, h, title, text="", tone=SETUP, lw=1.5):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.1", fc=tone[0], ec=tone[1], lw=lw, zorder=3))
    if text:
        ax.text(x, y + 0.2 * h, title, ha="center", va="center", fontsize=11, fontweight="bold", color=INK, zorder=4)
        ax.text(x, y - 0.2 * h, text, ha="center", va="center", fontsize=9, color=INK, linespacing=1.15, zorder=4)
    else:
        ax.text(x, y, title, ha="center", va="center", fontsize=11, fontweight="bold", color=INK, zorder=4)


def diamond(ax, x, y, w, h, text):
    ax.add_patch(Polygon([(x - w / 2, y), (x, y + h / 2), (x + w / 2, y), (x, y - h / 2)], closed=True, fc=TEST[0], ec=TEST[1], lw=1.5, zorder=3, joinstyle="round"))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.1, zorder=4)


def arrow(ax, start, end, label=None, at=None, bend=None, color=INK):
    if bend:
        points = [start, *bend]
        for a, b in zip(points, points[1:]):
            ax.plot(*zip(a, b), color=color, lw=1.6, solid_capstyle="round", zorder=2)
        start = points[-1]
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, lw=1.6, color=color, shrinkA=0, shrinkB=0, zorder=2))
    if label:
        ax.text(*at, label, ha="center", va="center", fontsize=10, color=FAINT, style="italic", zorder=5, bbox=dict(fc="white", ec="none", pad=1.2))


def main():
    fig, ax = plt.subplots(figsize=(15, 6.6))
    ax.set_xlim(0, 132)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.minorticks_off()
    lane = 26

    box(ax, 11, 51, 19, 6.5, "Initial state", "particles, conserved Q")
    box(ax, 11, 39.5, 19, 8.5, "Build the mesh", "voronoi::Mesh update,\nupdate_cells", MESH)
    box(ax, 11, lane, 19, 12.5, "time_step()", "primitives W, point\nvelocities w, limited\ngradients, global dt", HYDRO)
    diamond(ax, 29, lane, 12, 13, "mesh and\nintegrator")
    arrow(ax, (11, 47.75), (11, 43.75))
    arrow(ax, (11, 35.25), (11, 32.25))
    arrow(ax, (20.5, lane), (23, lane))

    y = 45
    box(ax, 68, y, 40, 7.5, "Static mesh", "half step, HLLC fluxes, Q − dt ΣF", HYDRO)
    arrow(ax, (29, lane + 6.5), (48, y), "static", (38, y + 1.5), bend=[(29, y)])
    ax.plot([88, 114, 114], [y, y, lane], color=INK, lw=1.6, solid_joinstyle="round", zorder=2)

    ax.add_patch(FancyBboxPatch((37, 16.3), 73.5, 19.2, boxstyle="round,pad=0,rounding_size=1.4", fc="#fafafa", ec="0.75", lw=1, ls=(0, (4, 3)), zorder=1))
    ax.text(38.5, 33.6, "moving mesh, two passes: the default", fontsize=10.5, color=FAINT, style="italic", va="center", zorder=4)
    box(ax, 48.5, lane - 1, 19, 11.5, "Pass 1", "fluxes on the old mesh,\nhalf applied", HYDRO)
    box(ax, 65, lane - 1, 10, 11.5, "Drift", "x + dt w", HYDRO)
    box(ax, 81, lane - 1, 16, 11.5, "Update mesh", "repair, faces,\ncell lists", MESH, lw=2.4)
    box(ax, 100, lane - 1, 18, 11.5, "Pass 2", "fluxes on the new\nmesh, half applied", HYDRO)
    arrow(ax, (35, lane), (39, lane))
    arrow(ax, (58, lane), (60, lane))
    arrow(ax, (70, lane), (73, lane))
    arrow(ax, (89, lane), (91, lane))
    ax.plot([109, 114], [lane, lane], color=INK, lw=1.6, zorder=2)

    y = 7.5
    box(ax, 56, y, 34, 7.5, "Moving, one pass", "half step, HLLC fluxes, Q − dt ΣF", HYDRO)
    box(ax, 85, y, 16, 7.5, "Drift, update\nthe mesh", "", MESH)
    arrow(ax, (29, lane - 6.5), (39, y), bend=[(29, y)])
    ax.text(29.8, 13.2, "--integrator=one-pass", fontsize=10, color=FAINT, style="italic", va="center", zorder=5, bbox=dict(fc="white", ec="none", pad=1.2))
    arrow(ax, (73, y), (77, y))
    ax.plot([93, 114, 114], [y, y, lane], color=INK, lw=1.6, solid_joinstyle="round", zorder=2)

    ax.add_patch(Circle((114, lane), 0.6, fc=INK, ec=INK, zorder=3))
    diamond(ax, 123.5, lane, 13, 12, "t < t_end?")
    arrow(ax, (114, lane), (117, lane))
    box(ax, 123.5, 45, 15, 7.5, "End of run", "snapshots")
    arrow(ax, (123.5, lane + 6), (123.5, 41.25), "no", (125.7, 36))
    ax.plot([123.5, 123.5, 11], [lane - 6, 2, 2], color=INK, lw=1.6, solid_capstyle="round", solid_joinstyle="round", zorder=2)
    ax.add_patch(FancyArrowPatch((11, 2), (11, lane - 6.25), arrowstyle="-|>", mutation_scale=15, lw=1.6, color=INK, shrinkA=0, shrinkB=0, zorder=2))
    ax.text(66, 2, "yes: the next step", ha="center", va="center", fontsize=10, color=FAINT, style="italic", zorder=5, bbox=dict(fc="white", ec="none", pad=1.2))

    for k, (tone, label) in enumerate(((HYDRO, "hydro"), (MESH, "mesh library"), (TEST, "decision"))):
        x = 46 + k * 19
        ax.add_patch(FancyBboxPatch((x, 52.2), 2.4, 1.6, boxstyle="round,pad=0,rounding_size=0.4", fc=tone[0], ec=tone[1], lw=1.2, zorder=3))
        ax.text(x + 3.2, 53, label, va="center", fontsize=10.5, color=INK)
    fig.savefig(os.path.join(HERE, "..", "docs", "figures", "flow_step.png"), dpi=200)


if __name__ == "__main__":
    main()
