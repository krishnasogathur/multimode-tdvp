import numpy as np
import time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joblib import Parallel, delayed
from tdvp.solver import (
    GaussianComponent, HierarchicalState, Nu,
    TDVPSolver, apply_jump_displacement, apply_jump_spin_decay, pack_state, rk4_step,
)
from tdvp.gaussians import single_mode_gaussians_expec

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Gamma   = 1.0
Delta   = 2.0
Omega   = 0.5
nu_x    = 2.0
nu_y    = 1.5
eta_x   = 0.2
eta_y   = 0.2
eta_x_se = 0.2
eta_y_se = 0.2
T       = 200.0
dt      = 0.001
NGAUSS  = 5
N_TRAJ  = 1000
R_MAX   = 0.05
SEED    = 120
alpha0_x = np.sqrt(4.0)
alpha0_y = np.sqrt(3.0)

OUT_DIR = "cooling-2d-trial1"
os.makedirs(OUT_DIR, exist_ok=True)
N_steps = int(T / dt)
t_arr   = np.linspace(0, T, N_steps + 1)

# 2 modes: k=0 is x, k=1 is y
H_terms = [
    (nu_x,    "e", "e", {0: (1, 1)}),
    (nu_y,    "e", "e", {1: (1, 1)}),
    (nu_x,    "g", "g", {0: (1, 1)}),
    (nu_y,    "g", "g", {1: (1, 1)}),
    (Delta,   "e", "e", {}),
    (Omega/2, "e", "g", {}, {0: +eta_x, 1: +eta_y}),
    (Omega/2, "g", "e", {}, {0: -eta_x, 1: -eta_y}),
]
K_terms = [(Gamma/2, "e", "e", {})]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _S(gi, gj, k, m, n):
    return single_mode_gaussians_expec(
        gi.alpha[k], gi.beta[k], gj.alpha[k], gj.beta[k], m, n)

def _amp(gi, gj):
    return np.exp(gi.kappa + gj.kappa + 1j*(gj.theta - gi.theta))

def _overlap(gi, gj):
    return _amp(gi, gj) * _S(gi, gj, 0, 0, 0) * _S(gi, gj, 1, 0, 0)

def compute_norm(psi):
    return sum(np.real(_overlap(gi, gj))
               for gs in psi.state.values() for gi in gs for gj in gs)

def compute_pe(psi):
    if "e" not in psi.state: return 0.0
    return sum(np.real(_overlap(gi, gj))
               for gi in psi.state["e"] for gj in psi.state["e"])

def compute_n(psi, k):
    v = 0.0
    for gs in psi.state.values():
        for gi in gs:
            for gj in gs:
                amp = _amp(gi, gj)
                other = 1 - k
                v += np.real(amp
                             * _S(gi, gj, k,      1, 1)
                             * _S(gi, gj, other,  0, 0))
    return v

def compute_a(psi, k):
    v = 0.0 + 0j
    for gs in psi.state.values():
        for gi in gs:
            for gj in gs:
                amp = _amp(gi, gj)
                other = 1 - k
                v += amp * _S(gi, gj, k, 0, 1) * _S(gi, gj, other, 0, 0)
    return v

def make_nus():
    return [Nu(s, p, k, kind)
            for s in ["e", "g"]
            for p in range(NGAUSS)
            for k in range(2)
            for kind in ["kappa", "theta", "x", "y", "r", "phi"]
            if not (kind in ["kappa", "theta"] and k == 1)]
    # kappa/theta only once per Gaussian (k=0), not per mode

def make_nus():
    nus = []
    for s in ["e", "g"]:
        for p in range(NGAUSS):
            nus.append(Nu(s, p, 0, "kappa"))
            nus.append(Nu(s, p, 0, "theta"))
            for k in range(2):
                nus.append(Nu(s, p, k, "x"))
                nus.append(Nu(s, p, k, "y"))
                nus.append(Nu(s, p, k, "r"))
                nus.append(Nu(s, p, k, "phi"))
    return nus

def make_sector_gaussians(ax, ay, suppress, rng):
    gaussians = []
    for _ in range(NGAUSS):
        angle  = rng.uniform(0, 2*np.pi)
        radius = rng.uniform(0, R_MAX)
        gaussians.append(GaussianComponent(
            kappa=0.0, theta=rng.uniform(0, 2*np.pi),
            x=[np.real(ax) + radius*np.cos(angle),
               np.real(ay) + radius*np.cos(angle)],
            y=[np.imag(ax) + radius*np.sin(angle),
               np.imag(ay) + radius*np.sin(angle)],
            r=[rng.uniform(0.0, 0.05), rng.uniform(0.0, 0.05)],
            phi=[0.0, 0.0]))
    nm = sum(np.real(_overlap(gi, gj))
             for gi in gaussians for gj in gaussians)
    for g in gaussians:
        g.kappa -= 0.5 * np.log(nm) + suppress
    return gaussians

def make_psi(seed):
    rng = np.random.default_rng(seed)
    psi = HierarchicalState()
    psi.add_gaussian("g", make_sector_gaussians(alpha0_x, alpha0_y, 0.0, rng))
    psi.add_gaussian("e", make_sector_gaussians(alpha0_x, alpha0_y, 3.0, rng))
    nm_tot = compute_norm(psi)
    for gs in psi.state.values():
        for g in gs: g.kappa -= 0.5 * np.log(nm_tot)
    return psi

def reset_after_jump(psi, rng):
    alpha_e_x = np.mean([g.alpha[0] for g in psi.state["e"]])
    alpha_e_y = np.mean([g.alpha[1] for g in psi.state["e"]])
    apply_jump_spin_decay(psi, "e", "g")
    # recoil kicks on both modes, random direction
    dir_x = np.random.choice([-1, 1])
    dir_y = np.random.choice([-1, 1])
    apply_jump_displacement(psi, "g", 1j * eta_x_se * dir_x, mode_k=0)
    apply_jump_displacement(psi, "g", 1j * eta_y_se * dir_y, mode_k=1)
    psi.state["e"] = make_sector_gaussians(alpha_e_x, alpha_e_y, 3.0, rng)
    nm = compute_norm(psi)
    for gs in psi.state.values():
        for g in gs: g.kappa -= 0.5 * np.log(nm)

# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
def run_trajectory(traj_idx):
    seed_rng = SEED * 1000 + traj_idx
    seed_psi = SEED * 1000 + traj_idx + 99999
    rng    = np.random.default_rng(seed_rng)
    psi    = make_psi(seed=seed_psi)
    nus    = make_nus()
    solver = TDVPSolver(psi, nus, H_terms, K_terms=K_terms)
    z      = pack_state(psi, nus)
    r      = rng.uniform()

    pe     = np.zeros(N_steps + 1)
    nx     = np.zeros(N_steps + 1)
    ny     = np.zeros(N_steps + 1)
    norm   = np.zeros(N_steps + 1)
    jump_times = []
    KAHLER_EVERY = 2000
    kahler_errors = []; kahler_steps = []

    nm = compute_norm(psi)
    pe[0]  = compute_pe(psi) / nm
    nx[0]  = compute_n(psi, 0) / nm
    ny[0]  = compute_n(psi, 1) / nm
    norm[0] = nm

    cur_dt = dt; t_now = 0.0; step = 0

    while step < N_steps:
        z_prev = z.copy()
        success = False
        for attempt in range(6):
            try:
                z_try = rk4_step(z_prev, cur_dt, solver)
                if not np.all(np.isfinite(z_try)):
                    raise ValueError("non-finite z")
                z = z_try; t_now += cur_dt; success = True
                cur_dt = min(cur_dt * 2.0, dt); break
            except Exception:
                cur_dt /= 2.0; solver.sync(z_prev)
        if not success:
            print(f"traj {traj_idx:04d} aborting at step {step}", flush=True)
            pe[step:] = np.nan; nx[step:] = np.nan; ny[step:] = np.nan
            break

        nm = compute_norm(psi)
        while nm < r:
            jump_times.append(t_now)
            reset_after_jump(psi, rng)
            nus    = make_nus()
            solver = TDVPSolver(psi, nus, H_terms, K_terms=K_terms)
            z      = pack_state(psi, nus)
            r      = rng.uniform()
            nm     = compute_norm(psi)

        while step < N_steps and t_arr[step+1] <= t_now + 1e-12:
            step += 1
            norm[step] = nm
            pe[step]   = compute_pe(psi) / nm
            nx[step]   = compute_n(psi, 0) / nm
            ny[step]   = compute_n(psi, 1) / nm
            if step % KAHLER_EVERY == 0:
                solver.eval(z)
                lam, U = np.linalg.eigh(solver.g_mat)
                mask = lam > 1e-8 * lam.max()
                U_r  = U[:, mask]
                # pseudoinverse restricted to image only
                g_pi_r = U_r @ np.diag(1.0/lam[mask]) @ U_r.T
                J    = U_r.T @ g_pi_r @ solver.Omega_mat @ U_r
                kerr = float(np.linalg.norm(J @ J + np.eye(mask.sum())))
                kahler_errors.append(kerr); kahler_steps.append(step)
                print(f"traj {traj_idx:04d}  step {step:05d}  kahler_err={kerr:.4f}", flush=True)
                # print(f"traj {traj_idx:04d}  step {step:05d}  kahler_err={kerr:.4f}", flush=True)
            if step % 200 == 0:
                print(f"traj {traj_idx:04d}  step {step:05d}  pe={pe[step]:.3f}  nx={nx[step]:.3f}  ny={ny[step]:.3f}  jumps={len(jump_times)}", flush=True)

    np.savez(os.path.join(OUT_DIR, f"traj_{traj_idx:04d}.npz"),
             pe=pe, nx=nx, ny=ny, norm=norm,
             jump_times=np.array(jump_times),
             Gamma=np.array(Gamma), Omega=np.array(Omega),
             nu_x=np.array(nu_x), nu_y=np.array(nu_y),
             eta_x=np.array(eta_x), eta_y=np.array(eta_y),
             T=np.array(T), dt=np.array(dt), NGAUSS=np.array(NGAUSS),
             traj_idx=np.array(traj_idx),
             kahler_errors=np.array(kahler_errors), kahler_steps=np.array(kahler_steps))
    return len(jump_times)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print(f"Gamma={Gamma}, Delta={Delta}, Omega={Omega}, nu_x={nu_x}, nu_y={nu_y}")
print(f"eta_x={eta_x}, eta_y={eta_y}, N_TRAJ={N_TRAJ}, dt={dt}, T={T}")
t0 = time.time()
n_jumps_all = Parallel(n_jobs=-1, verbose=5)(
    delayed(run_trajectory)(traj) for traj in range(N_TRAJ))
elapsed = time.time() - t0
n_jumps_all = np.array(n_jumps_all)
print(f"Done in {elapsed:.1f}s  avg_jumps={n_jumps_all.mean():.1f}")

# ---------------------------------------------------------------------------
# QuTiP 2D mcsolve
# ---------------------------------------------------------------------------
import qutip as qt
N_fock = 30
a    = qt.destroy(N_fock); b = qt.destroy(N_fock)
Id2  = qt.qeye(2); IdN = qt.qeye(N_fock)
sp   = qt.sigmap(); sm = qt.sigmam()
def S(op):  return qt.tensor(op, IdN, IdN)
def Ax(op): return qt.tensor(Id2, op, IdN)
def Ay(op): return qt.tensor(Id2, IdN, op)
see  = S(sp*sm)
Dx_pos = qt.displace(N_fock, 1j*eta_x); Dx_neg = Dx_pos.dag()
Dy_pos = qt.displace(N_fock, 1j*eta_y); Dy_neg = Dy_pos.dag()
Dx_se_pos = qt.displace(N_fock, 1j*eta_x_se); Dx_se_neg = Dx_se_pos.dag()
Dy_se_pos = qt.displace(N_fock, 1j*eta_y_se); Dy_se_neg = Dy_se_pos.dag()
H_qt = (nu_x * Ax(a.dag()*a) + nu_y * Ay(b.dag()*b) + Delta * see
      + Omega/2 * (S(sp)*Ax(Dx_pos)*Ay(Dy_pos) + S(sm)*Ax(Dx_neg)*Ay(Dy_neg)))
c_ops = [
    np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_pos) * Ay(Dy_se_pos),
    np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_neg) * Ay(Dy_se_pos),
    np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_pos) * Ay(Dy_se_neg),
    np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_neg) * Ay(Dy_se_neg),
]
psi0_qt = qt.tensor(qt.basis(2, 0), qt.coherent(N_fock, alpha0_x), qt.coherent(N_fock, alpha0_y))
e_ops   = [see, Ax(a.dag()*a), Ay(b.dag()*b)]
result  = qt.mcsolve(H_qt, psi0_qt, t_arr, c_ops, e_ops=e_ops, ntraj=500,
                     options={"nsteps": 100000})
pe_qt = np.array([result.expect[i][0] for i in range(500)]).mean(axis=0)
nx_qt = np.array([result.expect[i][1] for i in range(500)]).mean(axis=0)
ny_qt = np.array([result.expect[i][2] for i in range(500)]).mean(axis=0)
print("QuTiP done.")

np.savez(os.path.join(OUT_DIR, "meta.npz"),
    t_arr=t_arr, pe_qt=pe_qt, nx_qt=nx_qt, ny_qt=ny_qt,
    n_jumps_all=n_jumps_all,
    Gamma=np.array(Gamma), Omega=np.array(Omega), Delta=np.array(Delta),
    nu_x=np.array(nu_x), nu_y=np.array(nu_y),
    eta_x=np.array(eta_x), eta_y=np.array(eta_y),
    T=np.array(T), dt=np.array(dt), N_TRAJ=np.array(N_TRAJ))
print(f"Saved to {OUT_DIR}/")