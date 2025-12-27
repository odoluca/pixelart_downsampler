#!/usr/bin/env python3
"""
pixelart_grid_fit_downsample_mt.py

Multiprocessing acceleration for:
- grid search scoring
- refinement pass

Note:
- Uses process parallelism for CPU-bound loops.
- Windows requires the __main__ guard (present).

Suggestions:
for 4 by 4 pixelart:
python pixelart_grid_fit_downsample.py .\victory_arcDeTriumph_2.png .\outputs\test-arc.png  --min-cell 4 --max-cell 4  --print-grid --workers 0 --jitter 1 --size-tweak 1 --addDateTag --majority-min-frac 0.65 --thr 12 --qbits 4
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image

import concurrent.futures as cf



# -----------------------------
# Utilities
# -----------------------------

def srgb_to_float(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32)

def clamp_int(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x

def quantize_rgb(rgb_u8: np.ndarray, bits: int) -> np.ndarray:
    shift = 8 - bits
    return (rgb_u8 >> shift).astype(np.int32)

def dominant_cluster_mean_and_mask(
    rgb_u8: np.ndarray,
    thr: float,
    qbits: int,
    majority_min_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    px = rgb_u8.reshape(-1, 3)
    if px.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), np.zeros((0,), dtype=bool)

    q = quantize_rgb(px, qbits)
    base = 1 << qbits
    keys = (q[:, 0] * base + q[:, 1]) * base + q[:, 2]

    counts = np.bincount(keys, minlength=base**3)
    mode_key = int(np.argmax(counts))

    sel_mode = (keys == mode_key)
    if not np.any(sel_mode):
        mean0 = px.mean(axis=0).astype(np.float32)
        return mean0, np.ones((px.shape[0],), dtype=bool)

    mode_mean = px[sel_mode].mean(axis=0).astype(np.float32)

    d = px.astype(np.float32) - mode_mean[None, :]
    dist = np.sqrt(np.sum(d * d, axis=1))
    inliers = dist <= float(thr)

    if inliers.mean() < majority_min_frac:
        inliers = sel_mode.copy()
        if inliers.mean() < majority_min_frac:
            mean_all = px.mean(axis=0).astype(np.float32)
            d2 = px.astype(np.float32) - mean_all[None, :]
            dist2 = np.sqrt(np.sum(d2 * d2, axis=1))
            med = np.median(dist2)
            mad = np.median(np.abs(dist2 - med)) + 1e-6
            inliers = dist2 <= (med + 3.0 * mad)

    final_mean = px[inliers].mean(axis=0).astype(np.float32) if np.any(inliers) else px.mean(axis=0).astype(np.float32)
    return final_mean, inliers

def dispersion_of_pixels(rgb_u8: np.ndarray, mean_rgb: np.ndarray) -> float:
    px = rgb_u8.reshape(-1, 3).astype(np.float32)
    if px.shape[0] == 0:
        return 0.0
    d = px - mean_rgb[None, :]
    return float(np.mean(np.sum(d * d, axis=1)))

def iter_cells(W: int, H: int, cell: int, ox: int, oy: int):
    cols = (W - ox) // cell
    rows = (H - oy) // cell
    for cy in range(rows):
        y0 = oy + cy * cell
        y1 = y0 + cell
        for cx in range(cols):
            x0 = ox + cx * cell
            x1 = x0 + cell
            yield cx, cy, x0, y0, x1, y1

def grid_fit_score(rgb: np.ndarray, cell: int, ox: int, oy: int, sample_step: int) -> float:
    H, W, _ = rgb.shape
    cols = (W - ox) // cell
    rows = (H - oy) // cell
    if cols <= 1 or rows <= 1:
        return float("inf")

    means = np.zeros((rows, cols, 3), dtype=np.float32)
    within = 0.0
    n_cells = 0

    for cx, cy, x0, y0, x1, y1 in iter_cells(W, H, cell, ox, oy):
        block = rgb[y0:y1:sample_step, x0:x1:sample_step, :]
        if block.size == 0:
            continue
        m = block.reshape(-1, 3).mean(axis=0)
        means[cy, cx, :] = m
        d = block.reshape(-1, 3) - m[None, :]
        within += float(np.mean(np.sum(d * d, axis=1)))
        n_cells += 1

    if n_cells == 0:
        return float("inf")

    within /= n_cells

    dh = means[:, 1:, :] - means[:, :-1, :]
    dv = means[1:, :, :] - means[:-1, :, :]
    between = 0.5 * (
        float(np.mean(np.sqrt(np.sum(dh * dh, axis=2)))) +
        float(np.mean(np.sqrt(np.sum(dv * dv, axis=2))))
    )

    eps = 1e-6
    return float(within / (between + eps))


# -----------------------------
# Multiprocessing helpers
# -----------------------------

# We use globals in workers to avoid pickling big arrays repeatedly.
_G_RGB = None
_G_RGBA = None

def _init_grid_workers(rgb: np.ndarray):
    global _G_RGB
    _G_RGB = rgb

def _init_refine_workers(rgba: np.ndarray, base_colors: np.ndarray, grid: Tuple[int,int,int],
                         thr: float, qbits: int, majority_min_frac: float, alpha_cutoff: int,
                         jitter: int, size_tweak: int, alpha_penalty: float, anchor_penalty: float):
    global _G_RGBA, _G_BASE, _G_GRID, _G_PARAMS
    _G_RGBA = rgba
    _G_BASE = base_colors
    _G_GRID = grid
    _G_PARAMS = (thr, qbits, majority_min_frac, alpha_cutoff, jitter, size_tweak, alpha_penalty, anchor_penalty)


def _score_candidate(args: Tuple[int,int,int,int]) -> Tuple[float,int,int,int]:
    # args: (cell, ox, oy, sample_step)
    cell, ox, oy, sample_step = args
    s = grid_fit_score(_G_RGB, cell=cell, ox=ox, oy=oy, sample_step=sample_step)
    return (s, cell, ox, oy)

def extract_cell_pixels(rgba_u8: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha_cutoff: int) -> np.ndarray:
    block = rgba_u8[y0:y1, x0:x1, :]
    if block.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    a = block[..., 3]
    keep = a >= alpha_cutoff
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.uint8)
    rgb = block[..., :3][keep]
    return rgb.reshape(-1, 3)

def compute_cell_color(rgba_u8: np.ndarray, x0: int, y0: int, x1: int, y1: int,
                       thr: float, qbits: int, majority_min_frac: float, alpha_cutoff: int) -> Tuple[np.ndarray, float]:
    px = extract_cell_pixels(rgba_u8, x0, y0, x1, y1, alpha_cutoff)
    if px.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), 0.0
    mean_rgb, _ = dominant_cluster_mean_and_mask(px, thr=thr, qbits=qbits, majority_min_frac=majority_min_frac)
    disp = dispersion_of_pixels(px, mean_rgb)
    return mean_rgb, disp

def _refine_row(row_idx: int) -> Tuple[int, np.ndarray]:
    """
    Refine a single output row; return (row_idx, refined_row_colors_float32[cols,3]).
    """
    rgba = _G_RGBA
    base = _G_BASE
    cell0, ox0, oy0 = _G_GRID
    thr, qbits, majority_min_frac, alpha_cutoff, jitter, size_tweak, alpha_penalty, anchor_penalty = _G_PARAMS

    H, W, _ = rgba.shape
    rows, cols, _ = base.shape

    refined_row = base[row_idx, :, :].copy()

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (1, -1), (-1, 1), (1, 1)]

    cy = row_idx
    for cx in range(cols):
        x0 = ox0 + cx * cell0
        y0 = oy0 + cy * cell0
        base_c = base[cy, cx, :]

        neighbor_colors = []
        for dx, dy in nbrs:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                neighbor_colors.append(base[ny, nx, :])
        if not neighbor_colors:
            continue
        neighbor_colors = np.stack(neighbor_colors, axis=0)

        best_obj = -float("inf")
        best_color = refined_row[cx, :]

        for ds in range(-size_tweak, size_tweak + 1):
            cell = cell0 + ds
            if cell < 1:
                continue

            for jy in range(-jitter, jitter + 1):
                for jx in range(-jitter, jitter + 1):
                    xa = clamp_int(x0 + jx, 0, max(0, W - cell))
                    ya = clamp_int(y0 + jy, 0, max(0, H - cell))
                    xb, yb = xa + cell, ya + cell

                    c, disp = compute_cell_color(
                        rgba, xa, ya, xb, yb,
                        thr=thr, qbits=qbits, majority_min_frac=majority_min_frac, alpha_cutoff=alpha_cutoff
                    )

                    d = neighbor_colors - c[None, :]
                    contrast = float(np.sum(np.sqrt(np.sum(d * d, axis=1))))

                    # Anchor penalty: discourage drifting far from the base per-cell estimate.
                    # This helps preserve thin/diagonal contours that can be lost when the
                    # optimizer slides toward a flat interior.
                    if anchor_penalty > 0.0:
                        anchor = float(np.sqrt(np.sum((c - base_c) ** 2)))  # RGB distance
                    else:
                        anchor = 0.0

                    obj = contrast - alpha_penalty * disp - anchor_penalty * anchor


                    if obj > best_obj:
                        best_obj = obj
                        best_color = c

        refined_row[cx, :] = best_color

    return (row_idx, refined_row)


# -----------------------------
# Main pipeline
# -----------------------------

@dataclass
class GridSpec:
    cell: int
    ox: int
    oy: int
    score: float

def find_best_grid_mp(
    rgb_float: np.ndarray,
    min_cell: int,
    max_cell: int,
    cell_step: int,
    offset_step: int,
    sample_step: int,
    workers: int,
) -> GridSpec:
    best = GridSpec(cell=min_cell, ox=0, oy=0, score=float("inf"))

    candidates: List[Tuple[int,int,int,int]] = []
    for cell in range(min_cell, max_cell + 1, cell_step):
        off_step = max(1, min(offset_step, cell))
        for oy in range(0, cell, off_step):
            for ox in range(0, cell, off_step):
                candidates.append((cell, ox, oy, sample_step))

    if workers == 1:
        for c in candidates:
            s = grid_fit_score(rgb_float, c[0], c[1], c[2], c[3])
            if s < best.score:
                best = GridSpec(cell=c[0], ox=c[1], oy=c[2], score=s)
        return best

    if workers <= 0:
        workers = os.cpu_count() or 1

    with cf.ProcessPoolExecutor(max_workers=workers, initializer=_init_grid_workers, initargs=(rgb_float,)) as ex:
        for s, cell, ox, oy in ex.map(_score_candidate, candidates, chunksize=32):
            if s < best.score:
                best = GridSpec(cell=cell, ox=ox, oy=oy, score=float(s))

    return best

def initial_downsample_serial(
    rgba_u8: np.ndarray,
    grid: GridSpec,
    thr: float,
    qbits: int,
    majority_min_frac: float,
    alpha_cutoff: int,
) -> np.ndarray:
    H, W, _ = rgba_u8.shape
    cell = grid.cell
    ox, oy = grid.ox, grid.oy
    cols = (W - ox) // cell
    rows = (H - oy) // cell

    out = np.zeros((rows, cols, 3), dtype=np.float32)
    for cx, cy, x0, y0, x1, y1 in iter_cells(W, H, cell, ox, oy):
        c, _ = compute_cell_color(
            rgba_u8, x0, y0, x1, y1,
            thr=thr, qbits=qbits, majority_min_frac=majority_min_frac, alpha_cutoff=alpha_cutoff
        )
        out[cy, cx, :] = c
    return out

def refine_mp(
    rgba_u8: np.ndarray,
    base_colors: np.ndarray,
    grid: GridSpec,
    thr: float,
    qbits: int,
    majority_min_frac: float,
    alpha_cutoff: int,
    jitter: int,
    size_tweak: int,
    alpha_penalty: float,
    anchor_penalty: float,
    workers: int,
) -> np.ndarray:

    if (jitter <= 0 and size_tweak <= 0) or workers == 1:
        # serial or no-op
        if jitter <= 0 and size_tweak <= 0:
            return base_colors.copy()
        # serial refine (reuse same logic row-by-row in current process)
        refined = np.zeros_like(base_colors)
        _init_refine_workers(rgba_u8, base_colors, (grid.cell, grid.ox, grid.oy),
                             thr, qbits, majority_min_frac, alpha_cutoff,
                             jitter, size_tweak, alpha_penalty, anchor_penalty)

        for r in range(base_colors.shape[0]):
            ridx, row = _refine_row(r)
            refined[ridx, :, :] = row
        return refined

    if workers <= 0:
        workers = os.cpu_count() or 1

    rows = base_colors.shape[0]
    refined = np.zeros_like(base_colors)

    with cf.ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_refine_workers,
        initargs=(
            rgba_u8, base_colors, (grid.cell, grid.ox, grid.oy),
            thr, qbits, majority_min_frac, alpha_cutoff,
            jitter, size_tweak, alpha_penalty, anchor_penalty
        )
    ) as ex:
        for ridx, row in ex.map(_refine_row, range(rows), chunksize=1):
            refined[ridx, :, :] = row

    return refined

def save_output(colors_float: np.ndarray, output_path: str):
    out_u8 = np.clip(np.rint(colors_float), 0, 255).astype(np.uint8)
    Image.fromarray(out_u8, mode="RGB").save(output_path, format="PNG")


# -----------------------------
# CLI
# -----------------------------



def parse_args():
    p = argparse.ArgumentParser(description="Fit pixel-art grid and downsample (multiprocess).")
    p.add_argument("input_png")
    p.add_argument("output_png")
    p.add_argument("--min-cell", type=int, default=2,help="Minimum grid cell size (pixels) to consider during grid fitting. "
            "Set this close to the expected pixel size in the source image to speed up search.")
    p.add_argument("--max-cell", type=int, default=48,help="Maximum grid cell size (pixels) to consider during grid fitting. "
            "For faster, more reliable fits, keep the [min,max] range tight (e.g., 10..14).")
    p.add_argument("--cell-step", type=int, default=1,help="Step size when scanning candidate cell sizes. "
            "Use 1 for accuracy; use 2–4 for speed if you are unsure and searching broadly.")
    p.add_argument("--offset-step", type=int, default=1,help="Step size for scanning x/y offsets within a cell (0..cell-1). "
            "1 is most accurate; 2–4 is faster. "
            "If you already know the grid is aligned, larger steps are usually fine.")
    p.add_argument("--sample-step", type=int, default=1,help="Stride for sampling pixels inside each cell *during grid scoring only*. "
            "1 = use all pixels (best accuracy). "
            "2–3 can speed up grid fitting on large images with minimal quality loss.")
    p.add_argument("--thr", type=float, default=18.0,help="RGB distance threshold (0..~441) used to keep pixels as inliers around the dominant "
            "color cluster within each cell. Lower = sharper edges but may drop thin/diagonal strokes; "
            "higher = more tolerant to noise/anti-aliasing but can smear. "
            "Typical: 12–22. Start at 16–18 for clean pixel art.")
    p.add_argument("--qbits", type=int, default=5,help="Quantization bits per color channel used to estimate the dominant color cluster (mode bin). "
            "Higher = more bins (more sensitive, can split similar colors); lower = coarser bins "
            "(more stable, but may merge shades). Typical: 4–6. Default 5 is a good balance."
    )
    p.add_argument("--majority-min-frac", type=float, default=0.50, help="Minimum fraction of pixels that must be retained as inliers for a cell. "
            "If the threshold-based inliers fall below this, the algorithm falls back to safer rules "
            "(dominant bin or a robust MAD filter). "
            "Lowering this (e.g., 0.35–0.45) helps preserve thin/diagonal contours; "
            "raising it makes fills more stable."
    )
    p.add_argument("--alpha-cutoff", type=int, default=8,help="Ignore pixels with alpha < cutoff (0..255) when sampling cells. "
            "Use 0 to include fully transparent pixels; keep default for sprites with transparent backgrounds."
    )
    p.add_argument("--jitter", type=int, default=0, help="Per-cell sampling window jitter radius in pixels during refinement. "
            "The algorithm tries offsets in [-jitter,+jitter] for both x and y to better align with the true grid. "
            "0 disables refinement. Typical: 1. Values >=2 can improve difficult cases but increase runtime and "
            "can cause drifting unless --anchor is used.")
    p.add_argument("--size-tweak", type=int, default=0, help=
            "Allow per-cell sampling window size adjustment by +/- this many pixels during refinement. "
            "Helps when the source grid is not perfectly uniform (capture/scale artifacts). "
            "0 disables. Typical: 0–1. Values >=2 increase runtime and risk contour loss unless anchored."
    )
    p.add_argument("--alpha", type=float, default=0.20,
                   help="Dispersion penalty weight during refinement. Higher penalizes mixed-color windows, reducing smearing "
            "and discouraging the optimizer from sampling across edges. "
            "Typical: 0.20–0.60. If edges smear, increase; if thin strokes vanish, consider lowering slightly "
            "or reducing jitter/size-tweak and using --anchor."
    )
    p.add_argument("--workers", type=int, default=0,
                   help="Number of worker processes (0=auto, 1=disable parallelism).")
    p.add_argument("--print-grid", action="store_true", help="Print the best detected grid parameters (cell size and offsets) and the fit score."
    )
    p.add_argument("--anchor", type=float, default=0.0,
        help="Anchor penalty weight during refinement. Penalizes candidates that deviate from the base (non-jittered) "
            "cell color. This prevents the per-cell optimizer from drifting into flatter regions and accidentally "
            "erasing thin/diagonal contours. "
            "0 disables. Typical: 0.4–1.0 when using jitter>=1 or size-tweak>=1."
        )
    p.add_argument("--addDateTag", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()

    img = Image.open(args.input_png).convert("RGBA")
    rgba = np.array(img, dtype=np.uint8)
    rgb_float = srgb_to_float(rgba[..., :3])

    grid = find_best_grid_mp(
        rgb_float,
        min_cell=args.min_cell,
        max_cell=args.max_cell,
        cell_step=max(1, args.cell_step),
        offset_step=max(1, args.offset_step),
        sample_step=max(1, args.sample_step),
        workers=args.workers,
    )

    if args.print_grid:
        print(f"Best grid: cell={grid.cell} ox={grid.ox} oy={grid.oy} score={grid.score:.6g}")

    base = initial_downsample_serial(
        rgba, grid,
        thr=args.thr, qbits=args.qbits,
        majority_min_frac=args.majority_min_frac,
        alpha_cutoff=args.alpha_cutoff
    )

    refined = refine_mp(
        rgba, base, grid,
        thr=args.thr, qbits=args.qbits,
        majority_min_frac=args.majority_min_frac,
        alpha_cutoff=args.alpha_cutoff,
        jitter=max(0, args.jitter),
        size_tweak=max(0, args.size_tweak),
        alpha_penalty=max(0.0, args.alpha),
        anchor_penalty=max(0.0, args.anchor),
        workers=args.workers
    )

    if args.addDateTag:
        args.output_png= args.output_png.replace(".png",f"_{datetime.now().strftime('%Y%m%d%H%M')}.png")

    save_output(refined, args.output_png)

if __name__ == "__main__":
    main()
