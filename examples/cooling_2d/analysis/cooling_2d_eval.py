"""
2D cooling eval — postselects on clean_traj_idx.npy (completed + norm-ok).
Saves two plots (no show). HPC-friendly.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, glob

OUT_DIR = "cooling-2d-trial1"
T       = 100.0
dt      = 0.002
N_steps = int(T / dt)
t_arr   = np.linspace(0, T, N_steps + 1)

# ---------------------------------------------------------------------------
# Load clean indices (from traj_diagnostics.py)
# ---------------------------------------------------------------------------
clean_path = os.path.join(OUT_DIR, "clean_traj_idx.npy")
clean_idx  = np.load(clean_path) if os.path.exists(clean_path) else None

files = sorted(glob.glob(os.path.join(OUT_DIR, "traj_*.npz")))

pe_all=[]; nx_all=[]; ny_all=[]
for idx, f in enumerate(files):
    if clean_idx is not None and idx not in clean_idx:
        continue
    d = np.load(f, allow_pickle=True)
    pe_all.append(d["pe"]); nx_all.append(d["nx"]); ny_all.append(d["ny"])
pe_all=np.array(pe_all); nx_all=np.array(nx_all); ny_all=np.array(ny_all)
N = len(pe_all)

pe=np.nanmean(pe_all,0); pe_se=np.nanstd(pe_all,0)/np.sqrt(N)
nx=np.nanmean(nx_all,0); nx_se=np.nanstd(nx_all,0)/np.sqrt(N)
ny=np.nanmean(ny_all,0); ny_se=np.nanstd(ny_all,0)/np.sqrt(N)

# ---------------------------------------------------------------------------
# QuTiP reference
# ---------------------------------------------------------------------------
meta_path = os.path.join(OUT_DIR, "meta.npz")
has_qt = os.path.exists(meta_path)
if has_qt:
    m = np.load(meta_path, allow_pickle=True)
    t_qt  = m["t_arr"]
    pe_qt = m["pe_qt"]; nx_qt = m["nx_qt"]; ny_qt = m["ny_qt"]

# ---------------------------------------------------------------------------
# Plot 1: <n> vs t
# ---------------------------------------------------------------------------
fig1, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.fill_between(t_arr, nx-nx_se, nx+nx_se, color='r', alpha=0.25)
ax.fill_between(t_arr, ny-ny_se, ny+ny_se, color='g', alpha=0.25)
ax.plot(t_arr, nx, 'r-', lw=1.8, label=r'$\langle n_x\rangle$ TDVP')
ax.plot(t_arr, ny, 'g-', lw=1.8, label=r'$\langle n_y\rangle$ TDVP')
if has_qt:
    ax.plot(t_qt, nx_qt, 'r--', lw=1.5, alpha=0.8, label=r'$\langle n_x\rangle$ QuTiP')
    ax.plot(t_qt, ny_qt, 'g--', lw=1.5, alpha=0.8, label=r'$\langle n_y\rangle$ QuTiP')
ax.set_xlabel('t'); ax.set_ylabel(r'$\langle n\rangle$')
ax.set_title(f'2D cooling — phonon number ({N} clean trajs)')
ax.legend(); ax.grid(True, alpha=0.3)
fig1.savefig(os.path.join(OUT_DIR, 'n_vs_t.png'), dpi=150)

# ---------------------------------------------------------------------------
# Plot 2: populations vs t
# ---------------------------------------------------------------------------
fig2, ax = plt.subplots(figsize=(8,5), constrained_layout=True)
ax.fill_between(t_arr, pe-pe_se, pe+pe_se, color='b', alpha=0.25)
ax.plot(t_arr, pe,   'b-', lw=1.8, label=r'$P_e$ TDVP')
ax.plot(t_arr, 1-pe, 'k-', lw=1.8, alpha=0.7, label=r'$P_g$ TDVP')
if has_qt:
    ax.plot(t_qt, pe_qt,   'b--', lw=1.5, alpha=0.8, label=r'$P_e$ QuTiP')
    ax.plot(t_qt, 1-pe_qt, 'k--', lw=1.5, alpha=0.8, label=r'$P_g$ QuTiP')
ax.set_xlabel('t'); ax.set_ylabel('population')
ax.set_title(f'2D cooling — spin populations ({N} clean trajs)')
ax.legend(); ax.grid(True, alpha=0.3)
fig2.savefig(os.path.join(OUT_DIR, 'pops_vs_t.png'), dpi=150)

# log instead of print
with open(os.path.join(OUT_DIR, "eval.log"), "w") as fh:
    fh.write(f"clean trajs used: {N}/{len(files)}\n")
    fh.write(f"saved n_vs_t.png, pops_vs_t.png\n")
