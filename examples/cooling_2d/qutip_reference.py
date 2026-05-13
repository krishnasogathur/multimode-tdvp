# """
# 2D laser cooling of a trapped ion.
# Two motional modes (x, y) + two-level atom.

# H = nu_x a†a + nu_y b†b - Delta s_ee
#   + (Omega/2)(s+ D_x(i*eta_x) D_y(i*eta_y) + h.c.)

# Jump: sqrt(Gamma) * s- * D_x(i*eta_x_se) * D_y(i*eta_y_se)  (recoil on emission)
# """
# import numpy as np
# import qutip as qt
# import matplotlib.pyplot as plt

# # ---------------------------------------------------------------------------
# # Parameters
# # ---------------------------------------------------------------------------
# Gamma   = 1.0
# Delta   = 2.0
# Omega   = 0.5
# nu_x    = 2.0
# nu_y    = 1.5    # different frequency for y mode
# eta_x   = 0.2   # Lamb-Dicke x (absorption)
# eta_y   = 0.2   # Lamb-Dicke y (absorption)
# eta_x_se = 0.2  # recoil x (emission)
# eta_y_se = 0.2  # recoil y (emission)
# T       = 50.0
# dt      = 0.1
# ntraj   = 100
# N_fock  = 20    # truncation per mode (small for speed)

# t_arr = np.linspace(0, T, int(T/dt) + 1)

# # ---------------------------------------------------------------------------
# # Operators — space: spin x mode_x x mode_y
# # ---------------------------------------------------------------------------
# sx  = qt.sigmax(); sy = qt.sigmay(); sz = qt.sigmaz()
# sp  = qt.sigmap(); sm = qt.sigmam()
# Id2 = qt.qeye(2)
# IdN = qt.qeye(N_fock)

# a   = qt.destroy(N_fock)   # mode x
# b   = qt.destroy(N_fock)   # mode y

# # tensor order: spin, mode_x, mode_y
# def S(op):  return qt.tensor(op, IdN, IdN)
# def Ax(op): return qt.tensor(Id2, op, IdN)
# def Ay(op): return qt.tensor(Id2, IdN, op)

# x_op = a + a.dag()
# y_op = b + b.dag()

# # Dx_pos = (1j * eta_x   * x_op).expm()
# Dx_pos = qt.displace(N_fock, 1j * eta_x)
# Dx_neg = Dx_pos.dag()
# # Dy_pos = (1j * eta_y   * y_op).expm()
# Dy_pos = qt.displace(N_fock, 1j * eta_y)

# Dy_neg = Dy_pos.dag()

# # Dx_se_pos = (1j * eta_x_se * x_op).expm()
# Dx_se_pos = qt.displace(N_fock, 1j * eta_x_se)
# Dx_se_neg = Dx_se_pos.dag()
# # Dy_se_pos = (1j * eta_y_se * y_op).expm()
# Dy_se_pos = qt.displace(N_fock, 1j * eta_y_se)
# Dy_se_neg = Dy_se_pos.dag()

# see = S(sp * sm)

# H = (nu_x  * Ax(a.dag() * a)
#    + nu_y  * Ay(b.dag() * b)
#    + Delta * see
#    + Omega/2 * (S(sp) * Ax(Dx_pos) * Ay(Dy_pos)
#               + S(sm) * Ax(Dx_neg) * Ay(Dy_neg)))

# # 4 jump operators (±x recoil, ±y recoil, split equally)
# c_ops = [
#     np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_pos) * Ay(Dy_se_pos),
#     np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_neg) * Ay(Dy_se_pos),
#     np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_pos) * Ay(Dy_se_neg),
#     np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_neg) * Ay(Dy_se_neg),
# ]

# # initial state: |e, 0_x, 0_y>  (atom excited, both modes in vacuum)
# alpha0_x = np.sqrt(4)
# alpha0_y = np.sqrt(3)
# psi0 = qt.tensor(qt.basis(2, 1),   # excited state (fix indexing)
#                  qt.coherent(N_fock, alpha0_x),
#                  qt.coherent(N_fock, alpha0_y))
# e_ops = [see,
#          Ax(a.dag() * a),
#          Ay(b.dag() * b)]

# # ---------------------------------------------------------------------------
# # Run mcsolve
# # ---------------------------------------------------------------------------
# print(f"Running mcsolve: {ntraj} trajs, T={T}, N_fock={N_fock}")
# result = qt.mcsolve(H, psi0, t_arr, c_ops, e_ops=e_ops, ntraj=ntraj,
#                     options={"nsteps": 100000})

# pe_qt = result.expect[0]
# nx_qt = result.expect[1]
# ny_qt = result.expect[2]
# print(f"Done. nx_final={nx_qt[-1]:.4f}, ny_final={ny_qt[-1]:.4f}")

# # ---------------------------------------------------------------------------
# # Plot
# # ---------------------------------------------------------------------------
# fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
# fig.suptitle(f"2D laser cooling  nu_x={nu_x} nu_y={nu_y} eta={eta_x} Delta={Delta} Omega={Omega}")

# axs[0].plot(t_arr, pe_qt, 'b-'); axs[0].set_ylabel("Pe"); axs[0].set_xlabel("t"); axs[0].grid(True)
# axs[1].plot(t_arr, nx_qt, 'r-', label="nx"); axs[1].plot(t_arr, ny_qt, 'g-', label="ny")
# axs[1].set_ylabel("<n>"); axs[1].set_xlabel("t"); axs[1].legend(); axs[1].grid(True)
# axs[2].plot(nx_qt, ny_qt, 'k-'); axs[2].set_xlabel("nx"); axs[2].set_ylabel("ny"); axs[2].grid(True)
# axs[2].set_title("phase space")

# plt.savefig("qutip_cooling_2d.png", dpi=150)
# print("Saved qutip_cooling_2d.png")
# plt.show()






"""
2D laser cooling of a trapped ion.
Two motional modes (x, y) + two-level atom.

H = nu_x a†a + nu_y b†b - Delta s_ee
  + (Omega/2)(s+ D_x(i*eta_x) D_y(i*eta_y) + h.c.)

Jump: sqrt(Gamma) * s- * D_x(i*eta_x_se) * D_y(i*eta_y_se)  (recoil on emission)
"""
import numpy as np
import qutip as qt
import scipy.sparse as sp
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CSR helper — version-agnostic (QuTiP 4 and 5)
# ---------------------------------------------------------------------------
import scipy.sparse as sp

_QUTIP_V5 = qt.__version__.split(".")[0] >= "5"

def to_csr(qobj: qt.Qobj) -> qt.Qobj:
    """Force the internal storage of a Qobj to CSR sparse format."""
    if _QUTIP_V5:
        return qobj.to("csr")
    # QuTiP 4: qobj.data is a scipy sparse matrix
    qobj.data = sp.csr_matrix(qobj.data)
    return qobj

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
Gamma    = 1.0
Delta    = 2.0
Omega    = 0.5
nu_x     = 2.0
nu_y     = 1.5
eta_x    = 0.2
eta_y    = 0.2
eta_x_se = 0.2
eta_y_se = 0.2
T        = 200.0
dt       = 0.1
ntraj    = 250
N_fock   = 20

t_arr = np.linspace(0, T, int(T / dt) + 1)

# ---------------------------------------------------------------------------
# Operators — space: spin x mode_x x mode_y
# ---------------------------------------------------------------------------
sx  = qt.sigmax(); sy = qt.sigmay(); sz = qt.sigmaz()
sp_ = qt.sigmap(); sm = qt.sigmam()
Id2 = qt.qeye(2)
IdN = qt.qeye(N_fock)

a = qt.destroy(N_fock)   # mode x
b = qt.destroy(N_fock)   # mode y

def S(op):  return qt.tensor(op, IdN, IdN)
def Ax(op): return qt.tensor(Id2, op, IdN)
def Ay(op): return qt.tensor(Id2, IdN, op)

# --- displacement operators — convert to CSR immediately after construction
# qt.displace() calls expm() internally, which may return a dense-backed Qobj
Dx_pos    = to_csr(qt.displace(N_fock,  1j * eta_x))
Dx_neg    = to_csr(Dx_pos.dag())
Dy_pos    = to_csr(qt.displace(N_fock,  1j * eta_y))
Dy_neg    = to_csr(Dy_pos.dag())

Dx_se_pos = to_csr(qt.displace(N_fock,  1j * eta_x_se))
Dx_se_neg = to_csr(Dx_se_pos.dag())
Dy_se_pos = to_csr(qt.displace(N_fock,  1j * eta_y_se))
Dy_se_neg = to_csr(Dy_se_pos.dag())

see = S(sp_ * sm)

# --- Hamiltonian — build, then enforce CSR on the full operator
H = to_csr(
    nu_x  * Ax(a.dag() * a)
  + nu_y  * Ay(b.dag() * b)
  + Delta * see
  + Omega/2 * (S(sp_) * Ax(Dx_pos) * Ay(Dy_pos)
             + S(sm)  * Ax(Dx_neg) * Ay(Dy_neg))
)

# --- collapse operators — each one CSR
c_ops = [
    to_csr(np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_pos) * Ay(Dy_se_pos)),
    to_csr(np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_neg) * Ay(Dy_se_pos)),
    to_csr(np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_pos) * Ay(Dy_se_neg)),
    to_csr(np.sqrt(Gamma/4) * S(sm) * Ax(Dx_se_neg) * Ay(Dy_se_neg)),
]

# --- quick sanity check: print density of H
nnz    = H.data.nnz if hasattr(H.data, "nnz") else H.data.to_array().astype(bool).sum()
total  = np.prod(H.shape)
print(f"H sparsity: {nnz}/{total} non-zeros ({100*nnz/total:.2f}%)")

# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------
alpha0_x = np.sqrt(4)
alpha0_y = np.sqrt(3)
psi0 = qt.tensor(
    qt.basis(2, 1),
    qt.coherent(N_fock, alpha0_x),
    qt.coherent(N_fock, alpha0_y),
)

e_ops = [see, Ax(a.dag() * a), Ay(b.dag() * b)]

# ---------------------------------------------------------------------------
# Run mcsolve
# ---------------------------------------------------------------------------
print(f"Running mcsolve: {ntraj} trajs, T={T}, N_fock={N_fock}")
result = qt.mcsolve(H, psi0, t_arr, c_ops, e_ops=e_ops, ntraj=ntraj,
                    options={"nsteps": 100000})

pe_qt = result.expect[0]
nx_qt = result.expect[1]
ny_qt = result.expect[2]
print(f"Done. nx_final={nx_qt[-1]:.4f}, ny_final={ny_qt[-1]:.4f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axs = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
fig.suptitle(
    f"2D laser cooling  nu_x={nu_x} nu_y={nu_y} "
    f"eta={eta_x} Delta={Delta} Omega={Omega}"
)

axs[0].plot(t_arr, pe_qt, 'b-'); axs[0].set_ylabel("Pe"); axs[0].set_xlabel("t"); axs[0].grid(True)
axs[1].plot(t_arr, nx_qt, 'r-', label="nx")
axs[1].plot(t_arr, ny_qt, 'g-', label="ny")
axs[1].set_ylabel("<n>"); axs[1].set_xlabel("t"); axs[1].legend(); axs[1].grid(True)
axs[2].plot(nx_qt, ny_qt, 'k-')
axs[2].set_xlabel("nx"); axs[2].set_ylabel("ny"); axs[2].grid(True)
axs[2].set_title("phase space")

plt.savefig("qutip_cooling_2d.png", dpi=150)
print("Saved qutip_cooling_2d.png")
plt.show()