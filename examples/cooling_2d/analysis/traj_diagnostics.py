"""
Per-trajectory diagnostics for 2D cooling.
Builds a CSV with multiple outlier metrics — no trajectories dropped here,
just measured. Postselection rule applied separately in eval.
Writes a log file instead of stdout (HPC-friendly).
"""
import numpy as np
import os, glob, csv, logging

OUT_DIR = "cooling-2d-trial1"
CSV_OUT = os.path.join(OUT_DIR, "traj_diagnostics.csv")
LOG_OUT = os.path.join(OUT_DIR, "traj_diagnostics.log")

logging.basicConfig(
    filename=LOG_OUT, filemode="w", level=logging.INFO,
    format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

files = sorted(glob.glob(os.path.join(OUT_DIR, "traj_*.npz")))
log.info(f"Found {len(files)} trajectories")

# First pass: collect nx/ny for ensemble statistics (median/MAD per timestep)
nx_stack = []; ny_stack = []
for f in files:
    d = np.load(f, allow_pickle=True)
    nx_stack.append(d["nx"]); ny_stack.append(d["ny"])
nx_stack = np.array(nx_stack); ny_stack = np.array(ny_stack)

# robust ensemble center/scale at each timestep (ignore NaN)
nx_med = np.nanmedian(nx_stack, axis=0)
ny_med = np.nanmedian(ny_stack, axis=0)
nx_mad = np.nanmedian(np.abs(nx_stack - nx_med), axis=0) + 1e-12
ny_mad = np.nanmedian(np.abs(ny_stack - ny_med), axis=0) + 1e-12

rows = []
for idx, f in enumerate(files):
    d = np.load(f, allow_pickle=True)
    pe = d["pe"]; nx = d["nx"]; ny = d["ny"]; norm = d["norm"]
    jt = d["jump_times"]
    kerr = d["kahler_errors"] if "kahler_errors" in d else np.array([])

    # --- Criterion 5: completion (NaN-fill on abort) ---
    completed = bool(np.all(np.isfinite(nx)) and np.all(np.isfinite(ny))
                     and np.all(np.isfinite(pe)))
    # step where it aborted (first NaN), or -1 if completed
    if completed:
        abort_step = -1
    else:
        nan_idx = np.where(~np.isfinite(nx))[0]
        abort_step = int(nan_idx[0]) if len(nan_idx) else -1

    # --- magnitude metrics ---
    max_nx = float(np.nanmax(nx)); max_ny = float(np.nanmax(ny))
    final_nx = float(nx[np.isfinite(nx)][-1]) if np.any(np.isfinite(nx)) else np.nan
    final_ny = float(ny[np.isfinite(ny)][-1]) if np.any(np.isfinite(ny)) else np.nan

    # --- norm drift: MCWF norm should stay in (0,1]; flag excursions ---
    fin_norm = norm[np.isfinite(norm)]
    norm_max = float(np.max(fin_norm)) if len(fin_norm) else np.nan
    norm_min = float(np.min(fin_norm)) if len(fin_norm) else np.nan
    norm_bad = bool(norm_max > 1.1 or norm_min < 0)

    # --- jump statistics ---
    n_jumps = int(len(jt))
    if n_jumps > 1:
        gaps = np.diff(np.sort(jt))
        min_gap = float(np.min(gaps))
        # count rapid successive jumps (gap < 1 dt-ish, here < 0.01)
        n_rapid = int(np.sum(gaps < 0.01))
    else:
        min_gap = np.nan; n_rapid = 0

    # --- ensemble outlier fraction (criterion 3) ---
    # fraction of timesteps where this traj exceeds median + 5*MAD
    with np.errstate(invalid='ignore'):
        nx_z = np.abs(nx - nx_med) / nx_mad
        ny_z = np.abs(ny - ny_med) / ny_mad
    out_frac = float(np.nanmean((nx_z > 5) | (ny_z > 5)))

    # --- kahler (informational only) ---
    kerr_max = float(np.max(kerr)) if kerr.size else np.nan

    rows.append({
        "traj": idx,
        "completed": int(completed),
        "abort_step": abort_step,
        "max_nx": round(max_nx, 4),
        "max_ny": round(max_ny, 4),
        "final_nx": round(final_nx, 4) if np.isfinite(final_nx) else "",
        "final_ny": round(final_ny, 4) if np.isfinite(final_ny) else "",
        "norm_max": round(norm_max, 5) if np.isfinite(norm_max) else "",
        "norm_min": round(norm_min, 5) if np.isfinite(norm_min) else "",
        "norm_bad": int(norm_bad),
        "n_jumps": n_jumps,
        "min_jump_gap": round(min_gap, 5) if np.isfinite(min_gap) else "",
        "n_rapid_jumps": n_rapid,
        "ensemble_outlier_frac": round(out_frac, 4),
        "kahler_max": round(kerr_max, 3) if np.isfinite(kerr_max) else "",
    })

# write CSV
keys = list(rows[0].keys())
with open(CSV_OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
log.info(f"Saved {CSV_OUT}")

# summary
n_complete = sum(r["completed"] for r in rows)
n_normbad  = sum(r["norm_bad"] for r in rows)
n_rapid    = sum(1 for r in rows if r["n_rapid_jumps"] > 0)
log.info(f"\nSummary of {len(rows)} trajectories:")
log.info(f"  completed (no dt-floor abort): {n_complete}")
log.info(f"  norm excursion flagged:        {n_normbad}")
log.info(f"  has rapid successive jumps:    {n_rapid}")
log.info(f"  ensemble outlier (>5% steps):  "
      f"{sum(1 for r in rows if r['ensemble_outlier_frac'] > 0.05)}")

# suggested clean set: completed AND not norm_bad AND outlier_frac < 0.05
clean = [r["traj"] for r in rows
         if r["completed"] and not r["norm_bad"]]
log.info(f"\nSuggested clean set: {len(clean)}/{len(rows)} trajectories")
np.save(os.path.join(OUT_DIR, "clean_traj_idx.npy"), np.array(clean))
log.info(f"Saved clean indices -> {OUT_DIR}/clean_traj_idx.npy")
