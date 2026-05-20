#!/usr/bin/env python3
"""
visualize_qvalues.py
====================
Visualise RVI Q-value CSV files produced by mm1_baseline.

Usage
-----
    python visualize_qvalues.py exp2_rvi_qvalues.csv [exp3_rvi_qvalues.csv ...] [--save]

    # From the bin directory:
    C:\\Users\\yc0127198\\source\\repos\\dvi\\dvi\\.venv\\Scripts\\python.exe ^
        ..\\..\\..\\..\\visualize_qvalues.py exp2_rvi_qvalues.csv --save

CSV columns expected
--------------------
    f0, f1, top_type, opt_action, opt_serves,
    q_serve_0, q_serve_1, delta, past_dl_0, past_dl_1

    top_type   : which type is at the head of the FIFO queue (0 or 1)
    opt_serves : which type the RVI optimal policy serves (0 or 1)
    delta      : q_serve_1 - q_serve_0  (positive = type 0 cheaper = blue)
    past_dl_n  : 1 if FIL_n has exceeded its deadline

Figures produced (per CSV)
--------------------------
    Figure 1  (3 panels): FIFO policy | RVI optimal policy | delta heatmap
    Figure 2  (2 panels): Q(serve type 0)  | Q(serve type 1)  surfaces
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Colour constants (match LaTeX heatmap palette)
# ---------------------------------------------------------------------------
BLUE   = "#4472C4"   # type 0
ORANGE = "#ED7D31"   # type 1
NEAR_TIE_THRESH = 5  # |delta| below this value is highlighted as near-tie

# Two-colour discrete map: 0 -> blue, 1 -> orange
POLICY_CMAP = LinearSegmentedColormap.from_list("policy", [BLUE, ORANGE], N=2)

# Diverging map for delta: blue side = type 0 cheaper, red side = type 1 cheaper
DELTA_CMAP = "RdBu_r"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "f0", "f1", "top_type", "opt_action", "opt_serves",
        "q_serve_0", "q_serve_1", "delta", "past_dl_0", "past_dl_1",
    }
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {path} is missing columns: {sorted(missing)}")
    return df


def df_to_grid(df: pd.DataFrame, col: str) -> np.ndarray:
    """Pivot a flat (f0, f1, value) frame into a 2-D numpy grid.

    The grid is indexed as grid[f0, f1], matching imshow with origin='lower'
    so that f0 runs along the y-axis and f1 along the x-axis.
    """
    f0_max = int(df["f0"].max())
    f1_max = int(df["f1"].max())
    grid = np.full((f0_max + 1, f1_max + 1), np.nan)
    for _, row in df.iterrows():
        grid[int(row["f0"]), int(row["f1"])] = row[col]
    return grid


def deadline_lines(df: pd.DataFrame):
    """Return (dl0, dl1): pixel-boundary positions of the two deadlines.

    past_dl_n == 0 when FIL_n <= due_time_n, so the deadline boundary
    sits at  max(f_n | past_dl_n == 0) + 0.5.
    """
    def _dl(fil_col, flag_col):
        rows = df[df[flag_col] == 0]
        return float(rows[fil_col].max()) + 0.5 if not rows.empty else None

    return _dl("f0", "past_dl_0"), _dl("f1", "past_dl_1")


# ---------------------------------------------------------------------------
# Axis-decoration helpers
# ---------------------------------------------------------------------------

def add_deadline_lines(ax, dl0, dl1):
    """Draw dashed deadline lines and return (handle, label) pairs."""
    added = []
    if dl0 is not None:
        # Horizontal line: deadline for type 0 (y-axis = f0 with origin='lower')
        h = ax.axhline(dl0, color="black", linewidth=1.5, linestyle="--", alpha=0.70)
        added.append((h, f"DL type 0  (f₀={dl0 - 0.5:.0f})"))
    if dl1 is not None:
        # Vertical line: deadline for type 1 (x-axis = f1)
        h = ax.axvline(dl1, color="dimgrey", linewidth=1.5, linestyle="--", alpha=0.70)
        added.append((h, f"DL type 1  (f₁={dl1 - 0.5:.0f})"))
    return added


def add_near_tie_dots(ax, df, thresh):
    """Scatter small black dots on cells where |delta| < thresh."""
    mask = df["delta"].abs() < thresh
    sub  = df[mask]
    if sub.empty:
        return None
    sc = ax.scatter(
        sub["f1"], sub["f0"],
        s=8, c="black", marker=".", alpha=0.75, zorder=5,
        label=f"|Δ| < {thresh}  (near tie)",
    )
    return sc


def set_axis_labels(ax):
    ax.set_xlabel("FIL type 1  (f₁)", fontsize=9)
    ax.set_ylabel("FIL type 0  (f₀)", fontsize=9)


# ---------------------------------------------------------------------------
# Panel-level plotting
# ---------------------------------------------------------------------------

def plot_policy_panel(ax, grid, title, dl0, dl1):
    """Render a discrete policy grid (0 = blue/type-0, 1 = orange/type-1)."""
    ax.imshow(
        grid, origin="lower", aspect="equal",
        cmap=POLICY_CMAP, vmin=0, vmax=1,
        interpolation="nearest",
    )
    dl_items = add_deadline_lines(ax, dl0, dl1)
    ax.set_title(title, fontsize=10, fontweight="bold")
    set_axis_labels(ax)

    patch_0 = mpatches.Patch(facecolor=BLUE,   edgecolor="black", linewidth=0.4,
                              label="Serve type 0")
    patch_1 = mpatches.Patch(facecolor=ORANGE, edgecolor="black", linewidth=0.4,
                              label="Serve type 1")
    handles = [patch_0, patch_1] + [h for h, _ in dl_items]
    labels  = ["Serve type 0", "Serve type 1"] + [lb for _, lb in dl_items]
    ax.legend(handles=handles, labels=labels, fontsize=7.5,
              loc="upper left", framealpha=0.88)


def plot_delta_panel(ax, df, grid_delta, dl0, dl1):
    """Render the diverging delta = Q(serve 1) - Q(serve 0) heatmap."""
    vmax = float(np.nanmax(np.abs(grid_delta)))
    im = ax.imshow(
        grid_delta, origin="lower", aspect="equal",
        cmap=DELTA_CMAP, vmin=-vmax, vmax=vmax,
        interpolation="nearest",
    )
    dl_items = add_deadline_lines(ax, dl0, dl1)
    sc = add_near_tie_dots(ax, df, NEAR_TIE_THRESH)

    ax.set_title(
        "Δ = Q(serve 1) − Q(serve 0)\n"
        "blue = type 0 cheaper ► serve 0   |   red = type 1 cheaper ► serve 1",
        fontsize=9, fontweight="bold",
    )
    set_axis_labels(ax)

    legend_handles = [h for h, _ in dl_items]
    legend_labels  = [lb for _, lb in dl_items]
    if sc is not None:
        legend_handles.append(sc)
        legend_labels.append(f"|Δ| < {NEAR_TIE_THRESH}  (near tie)")
    if legend_handles:
        ax.legend(handles=legend_handles, labels=legend_labels,
                  fontsize=7.5, loc="upper left", framealpha=0.88)

    return im


def plot_qvalue_surface(ax, fig, grid, title, cmap, dl0, dl1):
    """Render a continuous Q-value surface with a colorbar."""
    im = ax.imshow(grid, origin="lower", aspect="equal",
                   cmap=cmap, interpolation="nearest")
    add_deadline_lines(ax, dl0, dl1)
    ax.set_title(title, fontsize=10, fontweight="bold")
    set_axis_labels(ax)
    fig.colorbar(im, ax=ax, shrink=0.82, label="cost-to-go")
    return im


# ---------------------------------------------------------------------------
# Main visualisation routine (one CSV)
# ---------------------------------------------------------------------------

def visualise(csv_path: str, save: bool):
    # Derive a clean stem for titles / output filenames
    base = os.path.splitext(os.path.basename(csv_path))[0]
    stem = base[: -len("_rvi_qvalues")] if base.endswith("_rvi_qvalues") else base

    print(f"\nLoading  {csv_path}")
    df = load_csv(csv_path)

    # Build grids -----------------------------------------------------------------
    # FIFO always serves the FIFO-head (top_type); RVI uses opt_serves
    grid_fifo  = df_to_grid(df, "top_type")
    grid_rvi   = df_to_grid(df, "opt_serves")
    grid_delta = df_to_grid(df, "delta")
    grid_q0    = df_to_grid(df, "q_serve_0")
    grid_q1    = df_to_grid(df, "q_serve_1")

    dl0, dl1 = deadline_lines(df)

    print(f"  Grid   {grid_fifo.shape[0]} x {grid_fifo.shape[1]} "
          f"(f0 max={grid_fifo.shape[0]-1}, f1 max={grid_fifo.shape[1]-1})")
    print(f"  DL0={dl0 - 0.5 if dl0 else 'n/a'}  DL1={dl1 - 0.5 if dl1 else 'n/a'}")
    print(f"  delta in [{df['delta'].min():.2f}, {df['delta'].max():.2f}]  "
          f"near-tie cells: {(df['delta'].abs() < NEAR_TIE_THRESH).sum()}")

    # ── Figure 1: FIFO | RVI | delta ─────────────────────────────────────────
    fig1, axes = plt.subplots(1, 3, figsize=(17, 5.5),
                              gridspec_kw={"wspace": 0.32})
    fig1.suptitle(
        f"Scheduling policy comparison — {stem.upper()}   "
        f"(λ=[0.25,0.25], c=[100,300], D=[3,3])",
        fontsize=12, fontweight="bold",
    )

    plot_policy_panel(axes[0], grid_fifo,
                      "FIFO policy\n(always serves head-of-queue)",
                      dl0, dl1)
    plot_policy_panel(axes[1], grid_rvi,
                      "RVI optimal policy",
                      dl0, dl1)
    im_d = plot_delta_panel(axes[2], df, grid_delta, dl0, dl1)
    fig1.colorbar(im_d, ax=axes[2], shrink=0.82, label="Δ (cost units)")

    fig1.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Figure 2: Q-value surfaces ────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.5),
                               gridspec_kw={"wspace": 0.32})
    fig2.suptitle(
        f"Q-value surfaces — {stem.upper()}",
        fontsize=12, fontweight="bold",
    )

    plot_qvalue_surface(axes2[0], fig2, grid_q0,
                        "Q(serve type 0)   [q_serve_0]",
                        "Blues_r", dl0, dl1)
    plot_qvalue_surface(axes2[1], fig2, grid_q1,
                        "Q(serve type 1)   [q_serve_1]",
                        "Oranges_r", dl0, dl1)

    fig2.tight_layout(rect=[0, 0, 1, 0.95])

    # ── Save or show ──────────────────────────────────────────────────────────
    if save:
        out_dir = os.path.dirname(os.path.abspath(csv_path))
        out1 = os.path.join(out_dir, f"{stem}_policy_comparison.png")
        out2 = os.path.join(out_dir, f"{stem}_qvalue_surfaces.png")
        fig1.savefig(out1, dpi=150, bbox_inches="tight")
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"  Saved: {out1}")
        print(f"  Saved: {out2}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise RVI Q-value CSV files from mm1_baseline / queue_rvi.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "csv_files", nargs="+",
        help="One or more CSV paths (e.g. exp2_rvi_qvalues.csv)",
    )
    parser.add_argument(
        "--save", action="store_true",
        help="Save PNG files next to each CSV instead of showing interactive windows",
    )
    args = parser.parse_args()

    for csv_path in args.csv_files:
        if not os.path.isfile(csv_path):
            print(f"WARNING: not found — {csv_path}", file=sys.stderr)
            continue
        visualise(csv_path, args.save)

    if not args.save:
        input("\nPress Enter to close all figures …")


if __name__ == "__main__":
    main()
