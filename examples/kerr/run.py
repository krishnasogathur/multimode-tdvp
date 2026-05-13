# """
# Yb-174 1S0->1P1 imaging in tweezer.
# Tweezer V(x) = -V0 exp(-x²/2σ²) ≈ harmonic + quartic + sextic
# Imaging: alternating beams, sequence:
#   [200ns R] [400ns L] [400ns R] [200ns L]  (total 1200ns ≈ 220/Γ)

# H = ω·a†a + c4·:(a+a†)^4: + c6·:(a+a†)^6:
#   + (Ω/2)(σ+ D(±i·η_LD) + h.c.)  ← sign flips with beam direction

# K = (Γ/2) σ_ee
# Jump operator: c± = sqrt(Γ/2) σ- D(±i·η_LD)  (random ± per emission, half rate each)
# """
# import numpy as np
# import time, os, sys
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# from joblib import Parallel, delayed
# from numpyclassesmultimode import (
#     GaussianComponent, HierarchicalState, Nu,
#     TDVPSolver, apply_jump_displacement, apply_jump_spin_decay,
#     pack_state, rk4_step,
# )
# from helper import single_mode_gaussians_expec

# # ---------------------------------------------------------------------------
# # Physical parameters → natural units (Gamma = 1)
# # ω/Γ = 1/159.943 (Yb-174 1S0->1P1: Γ=2π×29.1MHz, ω_trap=2π×182kHz)
# # All H coefficients in units of ℏΓ
# # ---------------------------------------------------------------------------
# Gamma_phys  = 1.0           # by definition
# omega       = 1.0/159.943   # ≈ 6.25e-3
# Omega_drive = 4.4           # Ω/Γ = 4.4 (s ≈ 40)
# eta_LD      = 0.199
# c4          = -7.4176e-07   # = (-1.1864e-04) * (ω/Γ)
# c6          = 2.3466e-10    # = (3.7533e-08)  * (ω/Γ)

# # 1200 ns × Γ_phys = 1200e-9 × 2π·29.1e6 ≈ 219.4 Γ-units
# T_total   = 219.4
# seq_ns    = [(200, +1), (400, -1), (400, +1), (200, -1)]
# ns_per_Gamma = T_total / 1200.0
# seq_omega = [(dur*ns_per_Gamma, sign) for dur, sign in seq_ns]

# # Sim params
# dt     = 1e-3
# N_TRAJ = 200
# NGAUSS = 5
# R_MAX  = 0.05
# SEED   = 42
# alpha0 = 0.0 + 0j  # ground vibrational state


# OUT_DIR = "tweezer-imaging-trial1"
# os.makedirs(OUT_DIR, exist_ok=True)

# N_steps = int(T_total / dt)
# t_arr   = np.linspace(0, T_total, N_steps + 1)

# est_per_step = 150e-6 * 3   # ~450us with anharmonic terms
# est_per_traj = N_steps * est_per_step
# est_total = est_per_traj * N_TRAJ / 20   # assume 20 cores
# print(f"Est runtime: {est_total/60:.1f} min ({N_steps} steps × {N_TRAJ} trajs / 20 cores)")

# # exit(0)
# # ---------------------------------------------------------------------------
# # Build H_terms: time-independent part + time-dependent laser
# # Time-dependent kick eta = ±eta_LD switched in piecewise constant chunks.
# # Easiest implementation: rebuild solver with new H at each segment.
# # ---------------------------------------------------------------------------
# def H_static_terms():
#     """ω·a†a + c4·:(a+a†)^4: + c6·:(a+a†)^6: in both sectors."""
#     # (a+a†)^4 normal-ordered:
#     # + a†⁴ + 4a†³a + 6a†²a² + 4a†a³ + a⁴ + 6a†² + 12a†a + 6a² + 3
#     quartic_no = [
#         (1, (4,0)), (4, (3,1)), (6, (2,2)), (4, (1,3)), (1, (0,4)),
#         (6, (2,0)), (12, (1,1)), (6, (0,2)), (3, (0,0))]
#     # (a+a†)^6 normal-ordered:
#     sextic_no = [
#         (1,(6,0)),(6,(5,1)),(15,(4,2)),(20,(3,3)),(15,(2,4)),(6,(1,5)),(1,(0,6)),
#         (15,(4,0)),(60,(3,1)),(90,(2,2)),(60,(1,3)),(15,(0,4)),
#         (45,(2,0)),(90,(1,1)),(45,(0,2)),
#         (15,(0,0))]
#     terms = []
#     for sigma in ("e", "g"):
#         terms.append((omega, sigma, sigma, {0: (1,1)}))
#         for coef, (m,n) in quartic_no:
#             if (m,n) == (0,0):
#                 terms.append((c4*coef, sigma, sigma, {}))
#             else:
#                 terms.append((c4*coef, sigma, sigma, {0: (m,n)}))
#         for coef, (m,n) in sextic_no:
#             if (m,n) == (0,0):
#                 terms.append((c6*coef, sigma, sigma, {}))
#             else:
#                 terms.append((c6*coef, sigma, sigma, {0: (m,n)}))
#     return terms

# def H_terms_with_laser(beam_sign):
#     """Add laser drive with displacement of given sign."""
#     terms = H_static_terms()
#     eta_signed = beam_sign * eta_LD
#     terms.append((Omega_drive/2, "e", "g", {}, +eta_signed))
#     terms.append((Omega_drive/2, "g", "e", {}, -eta_signed))
#     return terms

# K_terms = [(Gamma_phys/2, "e", "e", {})]

# # ---------------------------------------------------------------------------
# # Helpers (same as 1D cooling)
# # ---------------------------------------------------------------------------
# def _S(gi, gj, m, n):
#     return single_mode_gaussians_expec(
#         gi.alpha[0], gi.beta[0], gj.alpha[0], gj.beta[0], m, n)
# def _amp(gi, gj):
#     return np.exp(gi.kappa + gj.kappa + 1j*(gj.theta - gi.theta))
# def compute_norm(psi):
#     return sum(np.real(_amp(gi, gj) * _S(gi, gj, 0, 0))
#                for gs in psi.state.values() for gi in gs for gj in gs)
# def compute_pe(psi):
#     if "e" not in psi.state: return 0.0
#     return sum(np.real(_amp(gi, gj) * _S(gi, gj, 0, 0))
#                for gi in psi.state["e"] for gj in psi.state["e"])
# def expec_op(psi, ops_list):
#     """ops_list: list of (coef, m, n). Computes Σ coef * <ψ|a†^m a^n|ψ>."""
#     v = 0.0
#     for gs in psi.state.values():
#         for gi in gs:
#             for gj in gs:
#                 amp = _amp(gi, gj)
#                 for coef, m, n in ops_list:
#                     v += np.real(coef * amp * _S(gi, gj, m, n))
#     return v
# def compute_n(psi):
#     return expec_op(psi, [(1.0, 1, 1)])
# def compute_T(psi):
#     """T = (ω/4)(2a†a + 1 - a² - a†²). In ω units: (a†a + 1/2 - (a²+a†²)/2)."""
#     # NO: a†a, 1, -a²/2, -a†²/2 (already normal ordered)
#     return omega * (expec_op(psi, [(1.0, 1, 1), (0.5, 0, 0),
#                                     (-0.5, 2, 0), (-0.5, 0, 2)]))
# def compute_V(psi):
#     """V = (V₀/2σ²)x² - (V₀/8σ⁴)x⁴ + (V₀/48σ⁶)x⁶ - V₀
#         in ω units: harmonic part absorbs into ω·a†a together with kinetic.
#        So V_anharm part:  c4·:(a+a†)^4: + c6·:(a+a†)^6:
#        Plus harmonic V alone (without kinetic): (ω/4)(2a†a + 1 + a² + a†²)
#     """
#     V_harm = omega * (expec_op(psi, [(1.0, 1, 1), (0.5, 0, 0),
#                                       (0.5, 2, 0), (0.5, 0, 2)]))
#     quartic_no = [
#         (1,4,0),(4,3,1),(6,2,2),(4,1,3),(1,0,4),
#         (6,2,0),(12,1,1),(6,0,2),(3,0,0)]
#     sextic_no = [
#         (1,6,0),(6,5,1),(15,4,2),(20,3,3),(15,2,4),(6,1,5),(1,0,6),
#         (15,4,0),(60,3,1),(90,2,2),(60,1,3),(15,0,4),
#         (45,2,0),(90,1,1),(45,0,2),
#         (15,0,0)]
#     V_quart = expec_op(psi, [(c4*c, m, n) for c,m,n in quartic_no])
#     V_sext  = expec_op(psi, [(c6*c, m, n) for c,m,n in sextic_no])
#     return V_harm + V_quart + V_sext   # (constant -V0 dropped)

# # ---------------------------------------------------------------------------
# # psi setup
# # ---------------------------------------------------------------------------
# def make_nus():
#     return [Nu(s, p, 0, kind)
#             for s in ["e", "g"]
#             for p in range(NGAUSS)
#             for kind in ["kappa", "theta", "x", "y", "r", "phi"]]

# def make_sector_gaussians(alpha_c, suppress, rng):
#     gs = []
#     for _ in range(NGAUSS):
#         ang = rng.uniform(0, 2*np.pi); rad = rng.uniform(0, R_MAX)
#         gs.append(GaussianComponent(
#             kappa=0.0, theta=rng.uniform(0, 2*np.pi),
#             x=[np.real(alpha_c) + rad*np.cos(ang)],
#             y=[np.imag(alpha_c) + rad*np.sin(ang)],
#             r=[rng.uniform(0.0, 0.05)], phi=[0.0]))
#     nm = sum(np.real(np.exp(gi.kappa + gj.kappa + 1j*(gj.theta - gi.theta)) *
#               single_mode_gaussians_expec(gi.alpha[0], gi.beta[0],
#                                           gj.alpha[0], gj.beta[0], 0, 0))
#              for gi in gs for gj in gs)
#     for g in gs: g.kappa -= 0.5*np.log(nm) + suppress
#     return gs

# def make_psi(seed):
#     rng = np.random.default_rng(seed)
#     # sample initial coherent state from thermal distribution
#     nbar = 1.83
#     alpha_init = (rng.normal(0, np.sqrt(nbar)) + 1j*rng.normal(0, np.sqrt(nbar))) / np.sqrt(2)
#     psi = HierarchicalState()
#     psi.add_gaussian("g", make_sector_gaussians(alpha_init, 0.0, rng))
#     psi.add_gaussian("e", make_sector_gaussians(alpha_init, 3.0, rng))
#     nm = compute_norm(psi)
#     for gs in psi.state.values():
#         for g in gs: g.kappa -= 0.5*np.log(nm)
#     return psi

# def reset_after_jump(psi, rng):
#     """Jump c± = sqrt(Γ/2) σ- D(±i·η_LD) — random ± direction."""
#     alpha_e = np.mean([g.alpha[0] for g in psi.state["e"]])
#     direction = rng.choice([-1, +1])
#     apply_jump_spin_decay(psi, "e", "g")
#     apply_jump_displacement(psi, "g", 1j * eta_LD * direction)
#     psi.state["e"] = make_sector_gaussians(alpha_e, 3.0, rng)
#     nm = compute_norm(psi)
#     for gs in psi.state.values():
#         for g in gs: g.kappa -= 0.5*np.log(nm)

# # ---------------------------------------------------------------------------
# # Trajectory
# # ---------------------------------------------------------------------------
# def run_trajectory(traj_idx):
#     rng = np.random.default_rng(SEED*1000 + traj_idx)
#     psi = make_psi(seed=SEED*1000 + traj_idx + 99999)
#     nus = make_nus()

#     # Build segment boundaries in time
#     seg_bounds = np.cumsum([dur for dur, _ in seq_omega])  # cumulative end times
#     seg_signs  = [s for _, s in seq_omega]
#     cur_seg    = 0
#     solver     = TDVPSolver(psi, nus, H_terms_with_laser(seg_signs[0]), K_terms=K_terms)
#     z          = pack_state(psi, nus)
#     r          = rng.uniform()

#     pe   = np.zeros(N_steps + 1)
#     n_ph = np.zeros(N_steps + 1)
#     E    = np.zeros(N_steps + 1)
#     T_   = np.zeros(N_steps + 1)
#     V_   = np.zeros(N_steps + 1)
#     nm   = compute_norm(psi)
#     pe[0]   = compute_pe(psi)/nm
#     n_ph[0] = compute_n(psi)/nm
#     T_[0]   = compute_T(psi)/nm
#     V_[0]   = compute_V(psi)/nm
#     E[0]    = T_[0] + V_[0]
#     jump_times = []

#     cur_dt = dt; t_now = 0.0; step = 0
#     while step < N_steps:
#         # update segment if needed
#         while cur_seg < len(seg_bounds) - 1 and t_now >= seg_bounds[cur_seg]:
#             cur_seg += 1
#             solver = TDVPSolver(psi, nus, H_terms_with_laser(seg_signs[cur_seg]),
#                                 K_terms=K_terms)
#             z = pack_state(psi, nus)

#         z_prev = z.copy(); ok = False
#         for _ in range(6):
#             try:
#                 z_try = rk4_step(z_prev, cur_dt, solver)
#                 if not np.all(np.isfinite(z_try)): raise ValueError
#                 z = z_try; t_now += cur_dt; ok = True
#                 cur_dt = min(cur_dt*2, dt); break
#             except Exception:
#                 cur_dt /= 2; solver.sync(z_prev)
#         if not ok:
#             print(f"traj {traj_idx} aborting at step {step}", flush=True)
#             for arr in [pe, n_ph, E, T_, V_]: arr[step:] = np.nan
#             break

#         nm = compute_norm(psi)
#         while nm < r:
#             jump_times.append(t_now)
#             reset_after_jump(psi, rng)
#             nus = make_nus()
#             solver = TDVPSolver(psi, nus, H_terms_with_laser(seg_signs[cur_seg]),
#                                 K_terms=K_terms)
#             z = pack_state(psi, nus); r = rng.uniform(); nm = compute_norm(psi)

#         while step < N_steps and t_arr[step+1] <= t_now + 1e-12:
#             step += 1
#             pe[step]   = compute_pe(psi)/nm
#             n_ph[step] = compute_n(psi)/nm
#             T_[step]   = compute_T(psi)/nm
#             V_[step]   = compute_V(psi)/nm
#             E[step]    = T_[step] + V_[step]
#             if step % 200 == 0:
#                 print(time.time())
#                 print(f"traj {traj_idx:03d} step {step} pe={pe[step]:.3f} "
#                       f"n={n_ph[step]:.2f} E={E[step]:.2f} jumps={len(jump_times)}",
#                       flush=True)

#     np.savez(os.path.join(OUT_DIR, f"traj_{traj_idx:03d}.npz"),
#              pe=pe, n_phot=n_ph, E=E, T=T_, V=V_,
#              jump_times=np.array(jump_times),
#              Gamma=Gamma_phys, Omega=Omega_drive, eta=eta_LD,
#              T_tot=T_total, dt=dt, NGAUSS=NGAUSS, traj_idx=traj_idx)
#     return len(jump_times)

# # ---------------------------------------------------------------------------
# # Run
# # ---------------------------------------------------------------------------
# if __name__ == "__main__":
#     print(f"omega=1, Gamma/omega={Gamma_phys}, Omega/omega={Omega_drive}")
#     print(f"eta_LD={eta_LD}, c4={c4:.2e}, c6={c6:.2e}")
#     print(f"T_total={T_total:.3f} (ω-units), N_steps={N_steps}, N_TRAJ={N_TRAJ}")
#     t0 = time.time()
#     n_jumps_all = Parallel(n_jobs=-1, verbose=5)(
#         delayed(run_trajectory)(t) for t in range(N_TRAJ))
#     print(f"Done in {time.time()-t0:.1f}s, avg jumps={np.mean(n_jumps_all):.1f}")




"""
Yb-174 1S0->1P1 imaging in tweezer.
Tweezer V(x) = -V0 exp(-x²/2σ²) ≈ harmonic + quartic + sextic
Imaging: alternating beams, sequence:
  [200ns R] [400ns L] [400ns R] [200ns L]  (total 1200ns ≈ 220/Γ)

H = ω·a†a + c4·:(a+a†)^4: + c6·:(a+a†)^6:
  + (Ω/2)(σ+ D(±i·η_LD) + h.c.)  ← sign flips with beam direction

K = (Γ/2) σ_ee
Jump operator: c± = sqrt(Γ/2) σ- D(±i·η_LD)  (random ± per emission, half rate each)
"""
import numpy as np
import time, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joblib import Parallel, delayed
from numpyclassesmultimode import (
    GaussianComponent, HierarchicalState, Nu,
    TDVPSolver, apply_jump_displacement, apply_jump_spin_decay,
    pack_state, rk4_step,
)
from helper import single_mode_gaussians_expec

# ---------------------------------------------------------------------------
# Physical parameters → natural units (Gamma = 1)
# ω/Γ = 1/159.943 (Yb-174 1S0->1P1: Γ=2π×29.1MHz, ω_trap=2π×182kHz)
# All H coefficients in units of ℏΓ
# ---------------------------------------------------------------------------
Gamma_phys  = 1.0           # by definition
omega       = 1.0/159.943   # ≈ 6.25e-3
Omega_drive = 4.4           # Ω/Γ = 4.4 (s ≈ 40)
eta_LD      = 0.199
c4          = -7.4176e-07   # = (-1.1864e-04) * (ω/Γ)
c6          = 2.3466e-10    # = (3.7533e-08)  * (ω/Γ)

# 1200 ns × Γ_phys = 1200e-9 × 2π·29.1e6 ≈ 219.4 Γ-units
T_total   = 219.4 / 8
seq_ns    = [(25, +1), (50, -1), (50, +1), (25, -1)]
ns_per_Gamma = T_total / 150.0
seq_omega = [(dur*ns_per_Gamma, sign) for dur, sign in seq_ns]

# Sim params
dt     = 1e-3
N_TRAJ = 200
NGAUSS = 5
R_MAX  = 0.05
SEED   = 100
alpha0 = 0.0 + 0j  # ground vibrational state

OUT_DIR = "tweezer-imaging-trial2"
os.makedirs(OUT_DIR, exist_ok=True)

N_steps = int(T_total / dt)
t_arr   = np.linspace(0, T_total, N_steps + 1)

# ---------------------------------------------------------------------------
# Build H_terms: time-independent part + time-dependent laser
# Time-dependent kick eta = ±eta_LD switched in piecewise constant chunks.
# Easiest implementation: rebuild solver with new H at each segment.
# ---------------------------------------------------------------------------
def H_static_terms():
    """ω·a†a + c4·:(a+a†)^4: in both sectors.
    Sextic dropped (negligible <n><30, 16 terms saved).
    Constants dropped (no dynamics effect)."""
    # (a+a†)^4 normal-ordered, drop constant and merge a†a into omega term:
    # original: a†⁴+4a†³a+6a†²a²+4a†a³+a⁴ + 6a†²+12a†a+6a² + 3
    # drop constant 3, absorb 12·c4·a†a into ω → ω' = ω + 12·c4 (negligible)
    omega_eff = omega + 12*c4
    quartic_no = [
        (1, (4,0)), (4, (3,1)), (6, (2,2)), (4, (1,3)), (1, (0,4)),
        (6, (2,0)),             (6, (0,2))]
    terms = []
    for sigma in ("e", "g"):
        terms.append((omega_eff, sigma, sigma, {0: (1,1)}))
        for coef, (m,n) in quartic_no:
            terms.append((c4*coef, sigma, sigma, {0: (m,n)}))
    return terms

def H_terms_with_laser(beam_sign):
    """Add laser drive with displacement of given sign."""
    terms = H_static_terms()
    eta_signed = beam_sign * eta_LD
    terms.append((Omega_drive/2, "e", "g", {}, +eta_signed))
    terms.append((Omega_drive/2, "g", "e", {}, -eta_signed))
    return terms

K_terms = [(Gamma_phys/2, "e", "e", {})]

# ---------------------------------------------------------------------------
# Helpers (same as 1D cooling)
# ---------------------------------------------------------------------------
def _S(gi, gj, m, n):
    return single_mode_gaussians_expec(
        gi.alpha[0], gi.beta[0], gj.alpha[0], gj.beta[0], m, n)
def _amp(gi, gj):
    return np.exp(gi.kappa + gj.kappa + 1j*(gj.theta - gi.theta))
def compute_norm(psi):
    return sum(np.real(_amp(gi, gj) * _S(gi, gj, 0, 0))
               for gs in psi.state.values() for gi in gs for gj in gs)
def compute_pe(psi):
    if "e" not in psi.state: return 0.0
    return sum(np.real(_amp(gi, gj) * _S(gi, gj, 0, 0))
               for gi in psi.state["e"] for gj in psi.state["e"])
def expec_op(psi, ops_list):
    """ops_list: list of (coef, m, n). Computes Σ coef * <ψ|a†^m a^n|ψ>."""
    v = 0.0
    for gs in psi.state.values():
        for gi in gs:
            for gj in gs:
                amp = _amp(gi, gj)
                for coef, m, n in ops_list:
                    v += np.real(coef * amp * _S(gi, gj, m, n))
    return v
def compute_n(psi):
    return expec_op(psi, [(1.0, 1, 1)])
def compute_T(psi):
    """T = (ω/4)(2a†a + 1 - a² - a†²). In ω units: (a†a + 1/2 - (a²+a†²)/2)."""
    # NO: a†a, 1, -a²/2, -a†²/2 (already normal ordered)
    return omega * (expec_op(psi, [(1.0, 1, 1), (0.5, 0, 0),
                                    (-0.5, 2, 0), (-0.5, 0, 2)]))
def compute_V(psi):
    """V = (V₀/2σ²)x² - (V₀/8σ⁴)x⁴ + (V₀/48σ⁶)x⁶ - V₀
        in ω units: harmonic part absorbs into ω·a†a together with kinetic.
       So V_anharm part:  c4·:(a+a†)^4: + c6·:(a+a†)^6:
       Plus harmonic V alone (without kinetic): (ω/4)(2a†a + 1 + a² + a†²)
    """
    V_harm = omega * (expec_op(psi, [(1.0, 1, 1), (0.5, 0, 0),
                                      (0.5, 2, 0), (0.5, 0, 2)]))
    quartic_no = [
        (1,4,0),(4,3,1),(6,2,2),(4,1,3),(1,0,4),
        (6,2,0),(12,1,1),(6,0,2),(3,0,0)]
    V_quart = expec_op(psi, [(c4*c, m, n) for c,m,n in quartic_no])
    return V_harm + V_quart   # sextic and constant -V0 dropped

# ---------------------------------------------------------------------------
# psi setup
# ---------------------------------------------------------------------------
def make_nus():
    return [Nu(s, p, 0, kind)
            for s in ["e", "g"]
            for p in range(NGAUSS)
            for kind in ["kappa", "theta", "x", "y", "r", "phi"]]

def make_sector_gaussians(alpha_c, suppress, rng):
    gs = []
    for _ in range(NGAUSS):
        ang = rng.uniform(0, 2*np.pi); rad = rng.uniform(0, R_MAX)
        gs.append(GaussianComponent(
            kappa=0.0, theta=rng.uniform(0, 2*np.pi),
            x=[np.real(alpha_c) + rad*np.cos(ang)],
            y=[np.imag(alpha_c) + rad*np.sin(ang)],
            r=[rng.uniform(0.0, 0.05)], phi=[0.0]))
    nm = sum(np.real(np.exp(gi.kappa + gj.kappa + 1j*(gj.theta - gi.theta)) *
              single_mode_gaussians_expec(gi.alpha[0], gi.beta[0],
                                          gj.alpha[0], gj.beta[0], 0, 0))
             for gi in gs for gj in gs)
    for g in gs: g.kappa -= 0.5*np.log(nm) + suppress
    return gs

def make_psi(seed):
    rng = np.random.default_rng(seed)
    # sample initial coherent state from thermal distribution
    nbar = 1.83
    alpha_init = (rng.normal(0, np.sqrt(nbar)) + 1j*rng.normal(0, np.sqrt(nbar))) / np.sqrt(2)
    psi = HierarchicalState()
    psi.add_gaussian("g", make_sector_gaussians(alpha_init, 0.0, rng))
    psi.add_gaussian("e", make_sector_gaussians(alpha_init, 3.0, rng))
    nm = compute_norm(psi)
    for gs in psi.state.values():
        for g in gs: g.kappa -= 0.5*np.log(nm)
    return psi

def reset_after_jump(psi, rng):
    """Jump c± = sqrt(Γ/2) σ- D(±i·η_LD) — random ± direction."""
    alpha_e = np.mean([g.alpha[0] for g in psi.state["e"]])
    direction = rng.choice([-1, +1])
    apply_jump_spin_decay(psi, "e", "g")
    apply_jump_displacement(psi, "g", 1j * eta_LD * direction)
    psi.state["e"] = make_sector_gaussians(alpha_e, 3.0, rng)
    nm = compute_norm(psi)
    for gs in psi.state.values():
        for g in gs: g.kappa -= 0.5*np.log(nm)

# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------
def run_trajectory(traj_idx):
    rng = np.random.default_rng(SEED*1000 + traj_idx)
    psi = make_psi(seed=SEED*1000 + traj_idx + 99999)
    nus = make_nus()

    # Build segment boundaries in time
    seg_bounds = np.cumsum([dur for dur, _ in seq_omega])  # cumulative end times
    seg_signs  = [s for _, s in seq_omega]
    cur_seg    = 0
    solver     = TDVPSolver(psi, nus, H_terms_with_laser(seg_signs[0]), K_terms=K_terms)
    z          = pack_state(psi, nus)
    r          = rng.uniform()

    pe   = np.zeros(N_steps + 1)
    n_ph = np.zeros(N_steps + 1)
    E    = np.zeros(N_steps + 1)
    T_   = np.zeros(N_steps + 1)
    V_   = np.zeros(N_steps + 1)
    nm   = compute_norm(psi)
    pe[0]   = compute_pe(psi)/nm
    n_ph[0] = compute_n(psi)/nm
    T_[0]   = compute_T(psi)/nm
    V_[0]   = compute_V(psi)/nm
    E[0]    = T_[0] + V_[0]
    jump_times = []

    cur_dt = dt; t_now = 0.0; step = 0
    while step < N_steps:
        # update segment if needed
        while cur_seg < len(seg_bounds) - 1 and t_now >= seg_bounds[cur_seg]:
            cur_seg += 1
            solver = TDVPSolver(psi, nus, H_terms_with_laser(seg_signs[cur_seg]),
                                K_terms=K_terms)
            z = pack_state(psi, nus)

        z_prev = z.copy(); ok = False
        for _ in range(6):
            try:
                z_try = rk4_step(z_prev, cur_dt, solver)
                if not np.all(np.isfinite(z_try)): raise ValueError
                z = z_try; t_now += cur_dt; ok = True
                cur_dt = min(cur_dt*2, dt); break
            except Exception:
                cur_dt /= 2; solver.sync(z_prev)
        if not ok:
            print(f"traj {traj_idx} aborting at step {step}", flush=True)
            for arr in [pe, n_ph, E, T_, V_]: arr[step:] = np.nan
            break

        nm = compute_norm(psi)
        while nm < r:
            jump_times.append(t_now)
            reset_after_jump(psi, rng)
            nus = make_nus()
            solver = TDVPSolver(psi, nus, H_terms_with_laser(seg_signs[cur_seg]),
                                K_terms=K_terms)
            z = pack_state(psi, nus); r = rng.uniform(); nm = compute_norm(psi)

        while step < N_steps and t_arr[step+1] <= t_now + 1e-12:
            step += 1
            pe[step]   = compute_pe(psi)/nm
            n_ph[step] = compute_n(psi)/nm
            T_[step]   = compute_T(psi)/nm
            V_[step]   = compute_V(psi)/nm
            E[step]    = T_[step] + V_[step]
            if step % 200 == 0:
                print(time.time())
                print(f"traj {traj_idx:03d} step {step} pe={pe[step]:.3f} "
                      f"n={n_ph[step]:.2f} E={E[step]:.2f} jumps={len(jump_times)}",
                      flush=True)

    np.savez(os.path.join(OUT_DIR, f"traj_{traj_idx:03d}.npz"),
             pe=pe, n_phot=n_ph, E=E, T=T_, V=V_,
             jump_times=np.array(jump_times),
             Gamma=Gamma_phys, Omega=Omega_drive, eta=eta_LD,
             T_tot=T_total, dt=dt, NGAUSS=NGAUSS, traj_idx=traj_idx)
    return len(jump_times)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"omega=1, Gamma/omega={Gamma_phys}, Omega/omega={Omega_drive}")
    print(f"eta_LD={eta_LD}, c4={c4:.2e}, c6={c6:.2e}")
    print(f"T_total={T_total:.3f} (ω-units), N_steps={N_steps}, N_TRAJ={N_TRAJ}")
    t0 = time.time()
    n_jumps_all = Parallel(n_jobs=-1, verbose=5)(
        delayed(run_trajectory)(t) for t in range(N_TRAJ))
    print(f"Done in {time.time()-t0:.1f}s, avg jumps={np.mean(n_jumps_all):.1f}")