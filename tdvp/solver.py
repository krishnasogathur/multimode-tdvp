"""
Numpy-backed TDVP solver for hierarchical Gaussian states.
Supports multi-sector (spin) states with cross-sector H_terms.

H_terms format: (coeff, sigma_bra, sigma_ket, ops_dict)
Old 2-tuple (coeff, ops_dict) is auto-converted to ("g","g") upon construction.

Physics of cross-sector extension:
  - Overlap matrix g_mat / Omega remain block-diagonal in sigma
    because spin sectors are orthogonal: <g_sigma | g_sigma'> = delta_{sigma,sigma'}
  - Force vector F[nu_a] receives contributions from H terms where sigma_ket = sigma(nu_a)
    using a cross-sector S tensor S[i_bra, j_ket, k, m, n]
"""

import numpy as np
from dataclasses import dataclass
from .gaussians import single_mode_gaussians_expec


# ---------------------------------------------------------------------------
# Frontend — used for init and observables only; class overhead is negligible compared to overlap and force computations.
# ---------------------------------------------------------------------------

class GaussianComponent:
    def __init__(self, kappa, theta, x, y, r, phi):
        self.kappa = np.float64(kappa)
        self.theta = np.float64(theta)
        self.x   = np.array(x,   dtype=float)
        self.y   = np.array(y,   dtype=float)
        self.r   = np.array(r,   dtype=float)
        self.phi = np.array(phi, dtype=float)

    @property
    def alpha(self): return self.x + 1j * self.y

    @property
    def beta(self):  return self.r * np.exp(1j * self.phi)


class HierarchicalState:
    def __init__(self):
        self.state = {}   # sigma -> list[GaussianComponent]

    def add_spin_sector(self, sigma, gaussians=None):
        self.state[sigma] = gaussians if gaussians is not None else []

    def add_gaussian(self, sigma, gaussian):
        if sigma not in self.state:
            self.add_spin_sector(sigma)
        if isinstance(gaussian, (list, tuple)):
            self.state[sigma].extend(gaussian)
        else:
            self.state[sigma].append(gaussian)


@dataclass(frozen=True)
class Nu:
    sigma: object
    p:     int
    k:     int
    kind:  str   # "kappa","theta","x","y","r","phi"


# ---------------------------------------------------------------------------
# H_terms normaliser - call once at startup or in TDVPSolver.__init__
# ---------------------------------------------------------------------------

def normalise_H_terms(H_terms):
    """
    Accept (coeff, ops), (coeff, sb, sk, ops), or (coeff, sb, sk, ops, eta).
    Returns list of (coeff, sigma_bra, sigma_ket, ops_dict, eta).
    eta is the Lamb-Dicke displacement parameter: the bosonic factor is D(i*eta) = exp(i*eta*(a+a†)).
    eta=0 recovers the standard case.
    """
    out = []
    for term in H_terms:
        if len(term) == 2:
            coeff, ops = term
            out.append((coeff, "g", "g", ops, 0.0))
        elif len(term) == 4:
            coeff, sb, sk, ops = term
            out.append((coeff, sb, sk, ops, 0.0))
        elif len(term) == 5:
            out.append(tuple(term))
        else:
            raise ValueError(f"Unexpected H_terms entry (len={len(term)}): {term}")
    return out


# ---------------------------------------------------------------------------
# pack — called once at the start to get z0 (initial state vector)
# ---------------------------------------------------------------------------

def pack_state(psi, nus):
    z = np.zeros(len(nus), dtype=float)
    for i, nu in enumerate(nus):
        g = psi.state[nu.sigma][nu.p]
        if   nu.kind == "kappa": z[i] = g.kappa
        elif nu.kind == "theta": z[i] = g.theta
        elif nu.kind == "x":     z[i] = g.x[nu.k]
        elif nu.kind == "y":     z[i] = g.y[nu.k]
        elif nu.kind == "r":     z[i] = g.r[nu.k]
        elif nu.kind == "phi":   z[i] = g.phi[nu.k]
    return z


# ---------------------------------------------------------------------------
# Layout - built once from psi + nus at solver init
# ---------------------------------------------------------------------------

def _build_layout(psi, nus):
    kinds  = np.array([nu.kind  for nu in nus])
    sigmas = np.array([nu.sigma for nu in nus])
    p_idx  = np.array([nu.p     for nu in nus], dtype=int)
    k_idx  = np.array([nu.k     for nu in nus], dtype=int)
    N_nu   = len(nus)

    masks = {kind: (kinds == kind)
             for kind in ["kappa", "theta", "x", "y", "r", "phi"]}

    unique_sigmas = list(dict.fromkeys(nu.sigma for nu in nus))

    sigma_nu_idx = {
        sigma: np.where(sigmas == sigma)[0]
        for sigma in unique_sigmas
    }

    sigma_layout = {}
    for sigma in unique_sigmas:
        gaussians = psi.state[sigma]
        N_g     = len(gaussians)
        N_modes = len(gaussians[0].x)

        kappa_z = np.full(N_g,            -1, dtype=int)
        theta_z = np.full(N_g,            -1, dtype=int)
        x_z     = np.full((N_g, N_modes), -1, dtype=int)
        y_z     = np.full((N_g, N_modes), -1, dtype=int)
        r_z     = np.full((N_g, N_modes), -1, dtype=int)
        phi_z   = np.full((N_g, N_modes), -1, dtype=int)

        for i, nu in enumerate(nus):
            if nu.sigma != sigma:
                continue
            if   nu.kind == "kappa": kappa_z[nu.p]       = i
            elif nu.kind == "theta": theta_z[nu.p]       = i
            elif nu.kind == "x":     x_z[nu.p, nu.k]     = i
            elif nu.kind == "y":     y_z[nu.p, nu.k]     = i
            elif nu.kind == "r":     r_z[nu.p, nu.k]     = i
            elif nu.kind == "phi":   phi_z[nu.p, nu.k]   = i

        sigma_layout[sigma] = {
            "N_g":          N_g,
            "N_modes":      N_modes,
            "kappa_z":      kappa_z,
            "theta_z":      theta_z,
            "x_z":          x_z,
            "y_z":          y_z,
            "r_z":          r_z,
            "phi_z":        phi_z,
            "kappa_frozen": np.array([g.kappa      for g in gaussians]),
            "theta_frozen": np.array([g.theta      for g in gaussians]),
            "x_frozen":     np.array([g.x.copy()   for g in gaussians]),
            "y_frozen":     np.array([g.y.copy()   for g in gaussians]),
            "r_frozen":     np.array([g.r.copy()   for g in gaussians]),
            "phi_frozen":   np.array([g.phi.copy() for g in gaussians]),
        }

    return {
        "N_nu":          N_nu,
        "p_idx":         p_idx,
        "k_idx":         k_idx,
        "masks":         masks,
        "unique_sigmas": unique_sigmas,
        "sigma_nu_idx":  sigma_nu_idx,
        "sigma_layout":  sigma_layout,
    }


def _arrays_from_z(z, layout, sigma, global_kappa_max=None):
    """Slices z into per-sigma parameter arrays with kappa_shift for stability.

    global_kappa_max: if provided (multi-sector case), use it as the shift
    reference so amplitudes across sectors are consistently scaled.
    Leave as None for single-sector runs (per-sector shift, backward compat).
    """
    sl      = layout["sigma_layout"][sigma]
    N_g     = sl["N_g"];  N_modes = sl["N_modes"]

    def _get_scalar(key):
        idx  = sl[key + "_z"]
        out  = sl[key + "_frozen"].copy()
        mask = idx >= 0
        if mask.any():
            out[mask] = z[idx[mask]]
        return out

    def _get_modal(key):
        idx  = sl[key + "_z"]
        out  = sl[key + "_frozen"].copy()
        mask = idx >= 0
        if mask.any():
            out[mask] = z[idx[mask]]
        return out

    kappa = _get_scalar("kappa")
    if global_kappa_max is not None:
        kappa_shift = kappa - global_kappa_max
    else:
        kappa_shift = kappa - kappa.max()
    theta = _get_scalar("theta")
    x           = _get_modal("x")
    y           = _get_modal("y")
    r           = _get_modal("r")
    phi         = _get_modal("phi")

    return {
        "N_g":     N_g,
        "N_modes": N_modes,
        "kappa":   kappa,
        "theta":   theta,
        "x":       x,  "y": y,  "r": r,  "phi": phi,
        "alpha":   x + 1j * y,
        "beta":    r * np.exp(1j * phi),
        "c_conj":  np.exp(kappa_shift - 1j * theta),
        "c":       np.exp(kappa_shift + 1j * theta),
    }


# ---------------------------------------------------------------------------
# S tensor — same sector
# ---------------------------------------------------------------------------

def _infer_mn_max(H_terms): 
    ''' Originally used this which bruteforce computed all overlaps just to be careful. 
    later realised we can be smart about what overlaps we compute.'''
    m_max = n_max = 0
    for term in H_terms:
        ops = term[3]
        for k, (m, n) in ops.items():
            m_max = max(m_max, m);  n_max = max(n_max, n)
    return m_max + 3, n_max + 3

def _infer_mn_pairs(H_terms):
    """Minimal set of (m,n) pairs needed by overlap matrix and force vector.
    Keeps dense tensor shape but only fills needed slices - 1.6-2.2x speedup."""
    pairs = set()
    # _build_overlap_matrix always needs full 0..2 x 0..2 (W_FULL tensor)
    for m in range(3):
        for n in range(3):
            pairs.add((m, n))
    # _make_op_slices needs (m,n),(m+1,n),(m,n-1),(m+2,n),(m+1,n-1),(m,n-2)
    for term in H_terms:
        ops = term[3]
        for k, (m, n) in ops.items():
            for dm, dn in [(0,0),(1,0),(0,-1),(2,0),(1,-1),(0,-2)]:
                mm, nn = m+dm, n+dn
                if mm >= 0 and nn >= 0:
                    pairs.add((mm, nn))
    pairs.add((0,1)); pairs.add((1,0))  # needed for <a> type overlaps
    m_max = max(m for m,n in pairs) + 1
    n_max = max(n for m,n in pairs) + 1
    return sorted(pairs), m_max, n_max


def _build_S_tensor(arr, m_max, n_max, mn_pairs=None):
    """Shape: (N_g, N_g, N_modes, m_max, n_max).
    If mn_pairs given, only fills those slices (sparse fill - faster)."""
    N_g     = arr["N_g"]
    N_modes = arr["N_modes"]
    alpha   = arr["alpha"]
    beta    = arr["beta"]

    a_i = alpha[:, None, :]
    b_i = beta[:, None, :]
    a_j = alpha[None, :, :]
    b_j = beta[None, :, :]

    _vf = np.vectorize(single_mode_gaussians_expec, otypes=[complex])

    S = np.zeros((N_g, N_g, N_modes, m_max, n_max), dtype=complex)
    it = mn_pairs if mn_pairs is not None else [(m,n) for m in range(m_max) for n in range(n_max)]
    for (m, n) in it:
        S[:, :, :, m, n] = _vf(a_i, b_i, a_j, b_j, m, n)
    return S


# ---------------------------------------------------------------------------
# Cross-sector S tensor — bra from sigma_bra, ket from sigma_ket
# ---------------------------------------------------------------------------

def _build_cross_S_tensor(arr_bra, arr_ket, m_max, n_max, mn_pairs=None):
    """
    Shape: (N_g_bra, N_g_ket, N_modes, m_max, n_max).
    If mn_pairs given, only fills those slices (sparse fill — faster).
    """
    N_g_bra = arr_bra["N_g"]
    N_g_ket = arr_ket["N_g"]
    N_modes = arr_bra["N_modes"]
    alpha_bra = arr_bra["alpha"]
    beta_bra  = arr_bra["beta"]
    alpha_ket = arr_ket["alpha"]
    beta_ket  = arr_ket["beta"]

    a_i = alpha_bra[:, None, :]
    b_i = beta_bra[:, None, :]
    a_j = alpha_ket[None, :, :]
    b_j = beta_ket[None, :, :]

    _vf = np.vectorize(single_mode_gaussians_expec, otypes=[complex])

    S = np.zeros((N_g_bra, N_g_ket, N_modes, m_max, n_max), dtype=complex)
    it = mn_pairs if mn_pairs is not None else [(m,n) for m in range(m_max) for n in range(n_max)]
    for (m, n) in it:
        S[:, :, :, m, n] = _vf(a_i, b_i, a_j, b_j, m, n)
    return S


# ---------------------------------------------------------------------------
# Tangent coefficients
# ---------------------------------------------------------------------------

_W_FULL = np.zeros((3, 3, 3, 3), dtype=complex)
_W_FULL[0,0,0,0]=1; _W_FULL[1,1,0,0]=1; _W_FULL[2,2,0,0]=2
_W_FULL[1,0,0,1]=1; _W_FULL[2,1,0,1]=2
_W_FULL[0,1,1,0]=1; _W_FULL[1,2,1,0]=2
_W_FULL[1,1,1,1]=1; _W_FULL[2,2,1,1]=4
_W_FULL[0,2,2,0]=1; _W_FULL[2,0,0,2]=1
_W_FULL[1,2,2,1]=1; _W_FULL[2,1,1,2]=1
_W_FULL[2,2,2,2]=1


def _tangent_coeffs(layout, arr, sigma):
    idx    = layout["sigma_nu_idx"][sigma]
    masks  = layout["masks"]
    p_idx  = layout["p_idx"]
    k_idx  = layout["k_idx"]
    N_nu   = layout["N_nu"]
    alpha  = arr["alpha"];  r_arr = arr["r"];  phi_arr = arr["phi"]

    f = np.zeros(N_nu, dtype=complex)
    g = np.zeros(N_nu, dtype=complex)
    h = np.zeros(N_nu, dtype=complex)

    def _m(kind): return idx[masks[kind][idx]]

    f[_m("kappa")] = 1.0
    f[_m("theta")] = 1j

    xm = _m("x")
    if len(xm):
        _a = alpha[p_idx[xm], k_idx[xm]]
        et = np.tanh(r_arr[p_idx[xm], k_idx[xm]]) * np.exp(1j * phi_arr[p_idx[xm], k_idx[xm]])
        f[xm] = et * np.conj(_a) - np.real(_a)
        g[xm] = 1.0 - et

    ym = _m("y")
    if len(ym):
        _a = alpha[p_idx[ym], k_idx[ym]]
        et = np.tanh(r_arr[p_idx[ym], k_idx[ym]]) * np.exp(1j * phi_arr[p_idx[ym], k_idx[ym]])
        f[ym] = -et * (1j * np.conj(_a)) - np.imag(_a)
        g[ym] = 1j * (1.0 + et)

    rm = _m("r")
    if len(rm):
        _a   = alpha[p_idx[rm], k_idx[rm]]
        _r   = r_arr[p_idx[rm], k_idx[rm]]
        _phi = phi_arr[p_idx[rm], k_idx[rm]]
        _h   = 0.5 / np.cosh(_r)**2 * np.exp(1j * _phi)
        f[rm] = -0.5 * np.tanh(_r) + np.conj(_a)**2 * _h
        g[rm] = -2.0 * np.conj(_a) * _h
        h[rm] = _h

    pm = _m("phi")
    if len(pm):
        _a   = alpha[p_idx[pm], k_idx[pm]]
        _r   = r_arr[p_idx[pm], k_idx[pm]]
        _phi = phi_arr[p_idx[pm], k_idx[pm]]
        _h   = 0.5j * np.tanh(_r) * np.exp(1j * _phi)
        f[pm] = np.conj(_a)**2 * _h
        g[pm] = -2.0 * np.conj(_a) * _h
        h[pm] = _h

    return f, g, h


# ---------------------------------------------------------------------------
# Overlap matrix — block-diagonal in sigma (cross-sector = 0 by orthogonality)
# ---------------------------------------------------------------------------

def _build_overlap_matrix(layout, arrs, S_tensors, f_all, g_all, h_all):
    N_nu  = layout["N_nu"]
    p_idx = layout["p_idx"]
    k_idx = layout["k_idx"]
    S_mat = np.zeros((N_nu, N_nu), dtype=complex)

    for sigma in layout["unique_sigmas"]:
        idx     = layout["sigma_nu_idx"][sigma]
        arr     = arrs[sigma]
        S_t     = S_tensors[sigma]
        N_modes = arr["N_modes"]

        pi = p_idx[idx];  ki = k_idx[idx];  n_loc = len(idx)
        T   = np.stack([f_all[idx], g_all[idx], h_all[idx]], axis=1)
        amp = arr["c_conj"][pi][:, None] * arr["c"][pi][None, :]
        base_S_full = S_t[:, :, :, 0, 0]
        block       = np.zeros((n_loc, n_loc), dtype=complex)
        all_modes   = np.arange(N_modes)

        for k_val in np.unique(ki):
            a_loc  = np.where(ki == k_val)[0]
            pa_sub = pi[a_loc]
            S_sub  = S_t[pa_sub[:, None], pa_sub[None, :], k_val, :3, :3]
            T_sub  = T[a_loc]
            amp_sub = amp[np.ix_(a_loc, a_loc)]
            X      = np.einsum("pqmn,abmn->abpq", _W_FULL, S_sub)
            tang   = np.einsum("ap,bq,abpq->ab", np.conj(T_sub), T_sub, X)
            other  = all_modes[all_modes != k_val]
            if len(other) > 0:
                prod_excl = np.prod(
                    base_S_full[pa_sub[:, None], pa_sub[None, :], :][:, :, other],
                    axis=2)
            else:
                prod_excl = np.ones((len(a_loc), len(a_loc)), dtype=complex)
            block[np.ix_(a_loc, a_loc)] += amp_sub * prod_excl * tang

        if N_modes > 1:
            for ka in np.unique(ki):
                for kb in np.unique(ki):
                    if ka == kb:
                        continue
                    a_loc  = np.where(ki == ka)[0]
                    b_loc  = np.where(ki == kb)[0]
                    pa_sub = pi[a_loc];  pb_sub = pi[b_loc]
                    S_ka   = S_t[pa_sub[:, None], pb_sub[None, :], ka, 0, :3]
                    S_kb   = S_t[pa_sub[:, None], pb_sub[None, :], kb, :3, 0]
                    bra    = np.einsum("at,abt->ab", np.conj(T[a_loc]), S_ka)
                    ket    = np.einsum("bt,abt->ab", T[b_loc], S_kb)
                    other  = all_modes[(all_modes != ka) & (all_modes != kb)]
                    if len(other) > 0:
                        prod_excl = np.prod(
                            base_S_full[pa_sub[:, None], pb_sub[None, :], :][:, :, other],
                            axis=2)
                    else:
                        prod_excl = np.ones((len(a_loc), len(b_loc)), dtype=complex)
                    block[np.ix_(a_loc, b_loc)] += (
                        amp[np.ix_(a_loc, b_loc)] * prod_excl * bra * ket)

        S_mat[np.ix_(idx, idx)] = block

    return S_mat


# ---------------------------------------------------------------------------
# Force vector — cross-sector H_terms supported
# ---------------------------------------------------------------------------

def _make_op_slices(S_t, ops, N_modes):
    """
    S_t shape: (N_g_bra, N_g_ket, N_modes, m_max, n_max).
    Returns sf, sg, sh each of shape (N_modes, N_g_bra, N_g_ket).
    """
    sf  = np.zeros((N_modes, S_t.shape[0], S_t.shape[1]), dtype=complex)
    sg  = np.zeros_like(sf)
    sh  = np.zeros_like(sf)
    for k in range(N_modes):
        m, n   = ops.get(k, (0, 0))
        sf[k]  = S_t[:, :, k, m, n]
        sg[k]  = S_t[:, :, k, m+1, n]
        if n > 0: sg[k] += n * S_t[:, :, k, m, n-1]
        sh[k]  = S_t[:, :, k, m+2, n]
        if n > 0: sh[k] += 2*n * S_t[:, :, k, m+1, n-1]
        if n > 1: sh[k] += n*(n-1) * S_t[:, :, k, m, n-2]
    return sf, sg, sh


def _build_force_vector(layout, arrs, S_tensors, cross_S_tensors,
                        H_terms, f_all, g_all, h_all,
                        displaced_S_tensors=None):
    """
    H_terms: list of (coeff, sigma_bra, sigma_ket, ops_dict, eta).

    Physics:
        F[nu_a] = <psi| H |d_{nu_a} psi>
        For term (coeff, sigma_bra, sigma_ket, ops, eta):
          - Contributes to nus in sigma_ket  (derivative acts on ket)
          - Uses c_conj from sigma_bra, c from sigma_ket
          - eta != 0: bosonic factor is D(i*eta), shifts alpha_ket by i*eta,
            adds phase exp(i*eta*Re(alpha_ket)), modifies tangent coefficients.
    """
    N_nu  = layout["N_nu"]
    p_idx = layout["p_idx"]
    k_idx = layout["k_idx"]
    F     = np.zeros(N_nu, dtype=complex)

    for coeff, sigma_bra, sigma_ket, ops, eta in H_terms:
        if sigma_bra not in arrs or sigma_ket not in arrs:
            continue

        arr_bra = arrs[sigma_bra]
        arr_ket = arrs[sigma_ket]

        # nus in sigma_ket receive this contribution
        idx     = layout["sigma_nu_idx"][sigma_ket]
        N_modes = arr_ket["N_modes"]
        pi      = p_idx[idx]
        ki      = k_idx[idx]
        fa      = f_all[idx];  ga = g_all[idx];  ha = h_all[idx]

        if eta != 0.0 and displaced_S_tensors is not None:
            # bra-side displacement: ⟨g_bra|D(iη) = e^{iη Re(α_bra)} ⟨g_bra_disp|
            # ket and tangent vectors unchanged — no modification to f,g,h needed.
            disp_key   = (sigma_bra, sigma_ket,
                          tuple(sorted(eta.items())) if isinstance(eta, dict) else eta)
            S_t        = displaced_S_tensors[disp_key]["S_t"]
            c_bra_disp = displaced_S_tensors[disp_key]["c_bra_disp"]
            amp        = c_bra_disp[:, None] * arr_ket["c"][pi][None, :]
            fa_use, ga_use, ha_use = fa, ga, ha
        else:
            # standard case
            if sigma_bra == sigma_ket:
                S_t = S_tensors[sigma_bra]
            else:
                S_t = cross_S_tensors[(sigma_bra, sigma_ket)]
            amp         = arr_bra["c_conj"][:, None] * arr_ket["c"][pi][None, :]
            fa_use, ga_use, ha_use = fa, ga, ha

        sf, sg, sh = _make_op_slices(S_t, ops, N_modes)

        if N_modes == 1:
            tang = (fa_use[None,:]*sf[0][:,pi]
                  + ga_use[None,:]*sg[0][:,pi]
                  + ha_use[None,:]*sh[0][:,pi])
            F[idx] += coeff * np.sum(amp * tang, axis=0)
        else:
            all_modes = np.arange(N_modes)
            for k0 in np.unique(ki):
                loc    = np.where(ki == k0)[0]
                pi_k0  = pi[loc]
                other  = all_modes[all_modes != k0]
                if len(other) > 0:
                    prod_excl = np.ones((arr_bra["N_g"], len(loc)), dtype=complex)
                    for om in other:
                        prod_excl *= sf[om][:, pi_k0]
                else:
                    prod_excl = np.ones((arr_bra["N_g"], len(loc)), dtype=complex)
                tang = (fa_use[loc][None,:]*sf[k0][:,pi_k0]
                      + ga_use[loc][None,:]*sg[k0][:,pi_k0]
                      + ha_use[loc][None,:]*sh[k0][:,pi_k0])
                F[idx[loc]] += coeff * np.sum(amp[:,loc] * prod_excl * tang, axis=0)

    return F


# ---------------------------------------------------------------------------
# Observable helpers
# ---------------------------------------------------------------------------

_current_S_tensors: dict = {}


def single_mode_expectation(g1, g2, k, m, n):
    sigma = getattr(g1, "_sigma", None)
    if sigma is not None and sigma in _current_S_tensors:
        S_t = _current_S_tensors[sigma]
        if m < S_t.shape[3] and n < S_t.shape[4]:
            return S_t[g1._idx, g2._idx, k, m, n]
    b1 = g1.beta[k] * (1 if g1.r[k] >= 0 else -1)
    b2 = g2.beta[k] * (1 if g2.r[k] >= 0 else -1)
    return single_mode_gaussians_expec(g1.alpha[k], b1, g2.alpha[k], b2, m, n)


def _tag_gaussians(psi):
    for sigma, gaussians in psi.state.items():
        for idx, g in enumerate(gaussians):
            g._idx   = idx
            g._sigma = sigma


def _sync_psi_from_z(psi, layout, z):
    for sigma in layout["unique_sigmas"]:
        sl = layout["sigma_layout"][sigma]
        for p, g in enumerate(psi.state[sigma]):
            if sl["kappa_z"][p] >= 0: g.kappa = float(z[sl["kappa_z"][p]])
            if sl["theta_z"][p] >= 0: g.theta = float(z[sl["theta_z"][p]])
            for k in range(sl["N_modes"]):
                if sl["x_z"][p, k]   >= 0: g.x[k]   = float(z[sl["x_z"][p, k]])
                if sl["y_z"][p, k]   >= 0: g.y[k]   = float(z[sl["y_z"][p, k]])
                if sl["r_z"][p, k]   >= 0: g.r[k]   = float(z[sl["r_z"][p, k]])
                if sl["phi_z"][p, k] >= 0: g.phi[k] = float(z[sl["phi_z"][p, k]])
    _tag_gaussians(psi)


# ---------------------------------------------------------------------------
# TDVPSolver
# ---------------------------------------------------------------------------

class TDVPSolver:
    """
    McLachlan variational principle solver.

    Minimises ||R||² where R = (ż_μ v_μ + iH|ψ⟩ + K|ψ⟩).
    Setting ∂||R||²/∂ż_μ = 0 gives:

        g_μν ż_ν = -2 Im[C_μ] - 2 Re[D_μ]

    where C_μ = <ψ|H|v_μ>,  D_μ = <ψ|K|v_μ>,  g_μν = 2 Re[<v_μ|v_ν>].

    For closed (Hermitian) dynamics set K_terms=[] (default).
    For open dynamics pass K_terms = (1/2) Σ_m c†_m c_m in the same
    format as H_terms.
    """

    def __init__(self, psi, nus, H_terms, K_terms=None, pinv_thresh=1e-12):
        self.psi          = psi
        self.H_terms      = normalise_H_terms(H_terms)
        self.K_terms      = normalise_H_terms(K_terms) if K_terms else []
        self.pinv_thresh  = pinv_thresh
        self.N            = len(nus)
        self.g_mat        = None
        self.cond         = None
        # auto-select Dirac-Frenkel for pure Hermitian (no K_terms)
        self._use_df      = not bool(self.K_terms)

        self.layout  = _build_layout(psi, nus)

        # infer needed (m,n) pairs from both H and K terms
        all_terms = self.H_terms + self.K_terms
        self.mn_pairs, self.m_max, self.n_max = _infer_mn_pairs(all_terms)

        # cross-sector pairs needed for H and K
        self._cross_pairs = set()
        for coeff, sb, sk, ops, eta in all_terms:
            if sb != sk:
                self._cross_pairs.add((sb, sk))

    def _global_kappa_max(self, z):
        if not self._cross_pairs:
            return None
        raw_kappas = []
        for sigma in self.layout["unique_sigmas"]:
            sl = self.layout["sigma_layout"][sigma]
            for p in range(sl["N_g"]):
                idx = sl["kappa_z"][p]
                raw_kappas.append(z[idx] if idx >= 0 else sl["kappa_frozen"][p])
        return float(np.max(raw_kappas)) if raw_kappas else None

    def eval(self, z):
        global _current_S_tensors

        global_km = self._global_kappa_max(z)
        arrs = {sigma: _arrays_from_z(z, self.layout, sigma, global_km)
                for sigma in self.layout["unique_sigmas"]}

        S_tensors = {sigma: _build_S_tensor(arrs[sigma], self.m_max, self.n_max, self.mn_pairs)
                     for sigma in arrs}
        _current_S_tensors = S_tensors

        cross_S_tensors = {}
        for (sb, sk) in self._cross_pairs:
            if sb in arrs and sk in arrs:
                cross_S_tensors[(sb, sk)] = _build_cross_S_tensor(
                    arrs[sb], arrs[sk], self.m_max, self.n_max, self.mn_pairs)

        # displaced S tensors for terms with eta != 0
        # ⟨g_bra|D(iη) = e^{iη Re(α_bra)} ⟨g_bra with α→α-iη|
        # So displace bra alpha by -iη, put phase on c_conj_bra.
        # Ket and tangent vectors are completely unchanged.
        # key: (sigma_bra, sigma_ket, eta) -> {"S_t": ..., "c_bra_disp": ...}
        displaced_S_tensors = {}
        for coeff, sb, sk, ops, eta in (self.H_terms + self.K_terms):
            if eta != 0.0 and sb in arrs and sk in arrs:
                eta_key  = tuple(sorted(eta.items())) if isinstance(eta, dict) else eta
                disp_key = (sb, sk, eta_key)
                if disp_key not in displaced_S_tensors:
                    arr_b        = arrs[sb]
                    arr_k        = arrs[sk]
                    alpha_b_orig = arr_b["alpha"]            # (N_g_bra, N_modes)
                    eta_dict     = eta if isinstance(eta, dict) else {0: eta}
                    alpha_b_disp = alpha_b_orig.copy()
                    phase_bra    = np.ones(arr_b["N_g"], dtype=complex)
                    for k_mode, eta_k in eta_dict.items():
                        alpha_b_disp[:, k_mode] -= 1j * eta_k
                        phase_bra *= np.exp(1j * eta_k * np.real(alpha_b_orig[:, k_mode]))
                    c_bra_disp   = arr_b["c_conj"] * phase_bra
                    arr_b_disp   = {
                        "N_g":     arr_b["N_g"],
                        "N_modes": arr_b["N_modes"],
                        "alpha":   alpha_b_disp,
                        "beta":    arr_b["beta"],
                    }
                    displaced_S_tensors[disp_key] = {
                        "S_t":       _build_cross_S_tensor(arr_b_disp, arr_k,
                                                           self.m_max, self.n_max, self.mn_pairs),
                        "c_bra_disp": c_bra_disp,
                    }

        _sync_psi_from_z(self.psi, self.layout, z)

        f_all = np.zeros(self.N, dtype=complex)
        g_all = np.zeros(self.N, dtype=complex)
        h_all = np.zeros(self.N, dtype=complex)
        for sigma in self.layout["unique_sigmas"]:
            f, g, h = _tangent_coeffs(self.layout, arrs[sigma], sigma)
            f_all += f;  g_all += g;  h_all += h

        # metric g_μν = 2 Re[<v_μ|v_ν>]
        S_mat      = _build_overlap_matrix(self.layout, arrs, S_tensors, f_all, g_all, h_all)
        self.g_mat = 2.0 * np.real(S_mat)
        self.Omega = 2.0 * np.imag(S_mat)
        self.cond  = np.linalg.cond(self.g_mat)

        # McLachlan rhs = -2 Im[C_μ] - 2 Re[D_μ]
        # where C_μ = <ψ|H|v_μ> = F_H,  D_μ = <ψ|K|v_μ> = F_K
        F_H = _build_force_vector(self.layout, arrs, S_tensors, cross_S_tensors,
                                  self.H_terms, f_all, g_all, h_all,
                                  displaced_S_tensors=displaced_S_tensors)
        # Dirac-Frenkel (Hermitian): rhs = -2 Re[F_H]  (energy-conserving, Omega solve)
        # McLachlan (open):          rhs = -2 Im[F_H]  (+ K correction below)
        if self._use_df:
            rhs = -2.0 * np.real(F_H)
        else:
            rhs = -2.0 * np.imag(F_H)

        # non-Hermitian part: K convention is K = L†L, rhs -= Re[F_K]
        if self.K_terms:
            F_K  = _build_force_vector(self.layout, arrs, S_tensors, cross_S_tensors,
                                       self.K_terms, f_all, g_all, h_all,
                                       displaced_S_tensors=displaced_S_tensors)
            rhs -= 2.0 * np.real(F_K)

        return self._solve(rhs)

    def _solve(self, rhs):
        if self._use_df:
            return self._solve_df(rhs)
        return self._solve_mclachlan(rhs)

    def _solve_mclachlan(self, rhs):
        """McLachlan: g-pseudoinverse via eigendecomposition."""
        lam, U = np.linalg.eigh(self.g_mat)
        mask   = lam > self.pinv_thresh
        U_r    = U[:, mask]
        lam_r  = lam[mask]
        return U_r @ ((U_r.T @ rhs) / lam_r)

    def _solve_df(self, rhs):
        """Dirac-Frenkel: Omega-based solve for Hermitian dynamics.
        Projects to image of g, then SVD-pseudoinverts Omega in that subspace.
        """
        lam, U = np.linalg.eigh(self.g_mat)
        mask   = lam > self.pinv_thresh
        U_r    = U[:, mask]
        O_r    = U_r.T @ self.Omega @ U_r
        Uo, so, Vo = np.linalg.svd(O_r)
        cut    = 1e-7 * (so[0] if len(so) else 1.0)
        so_inv = np.where(np.abs(so) > cut, 1.0/so, 0.0)
        O_inv  = (Vo.T * so_inv) @ Uo.T
        return U_r @ O_inv @ (U_r.T @ rhs)

    def sync(self, z):
        global _current_S_tensors
        global_km = self._global_kappa_max(z)
        arrs = {sigma: _arrays_from_z(z, self.layout, sigma, global_km)
                for sigma in self.layout["unique_sigmas"]}
        S_tensors = {sigma: _build_S_tensor(arrs[sigma], self.m_max, self.n_max, self.mn_pairs)
                     for sigma in arrs}
        _current_S_tensors = S_tensors
        _sync_psi_from_z(self.psi, self.layout, z)


# ---------------------------------------------------------------------------
# Jump helpers — analytic parameter updates for operators inside the manifold
# ---------------------------------------------------------------------------

def apply_jump_cavity_loss(psi, sigma, kappa_rate):
    """
    Jump c = sqrt(kappa_rate) * a  (cavity loss).
    Stays inside manifold: a|α⟩ = α|α⟩.
    Updates kappa and theta analytically. Eq. (36) of Bond et al. 2024.
    """
    for g in psi.state[sigma]:
        alpha_p = g.alpha[0]
        if np.abs(alpha_p) < 1e-300:
            g.kappa = -1e6
            continue
        new_kappa = np.log(np.sqrt(np.abs(kappa_rate)) * np.exp(g.kappa) * np.abs(alpha_p))
        new_theta = np.angle(kappa_rate * np.exp(1j * g.theta + g.kappa) * alpha_p)
        g.kappa = new_kappa
        g.theta = new_theta


def apply_jump_spin_decay(psi, sigma_e, sigma_g):
    """
    Jump c = sqrt(gamma) * sigma_-  (spin decay e->g).
    Stays inside manifold: just moves Gaussians from e-sector to g-sector.
    Post-jump state is pure g-sector with bosonic states from e-sector.
    """
    psi.state[sigma_g] = [
        GaussianComponent(g.kappa, g.theta, g.x.copy(), g.y.copy(),
                          g.r.copy(), g.phi.copy())
        for g in psi.state[sigma_e]
    ]
    for g in psi.state[sigma_e]:
        g.kappa = -1e6    # suppress e-sector


def apply_jump_displacement(psi, sigma, eta, mode_k=0):
    """
    Jump involving displacement D(eta) on mode k (e.g. laser recoil).
    D(eta)|α⟩ = e^{i·phase}|α+eta⟩ — shifts each Gaussian center analytically.
    Always stays inside manifold for any eta.
    """
    for g in psi.state[sigma]:
        g.x[mode_k] += np.real(eta)
        g.y[mode_k] += np.imag(eta)


# ---------------------------------------------------------------------------
# RK4
# ---------------------------------------------------------------------------

def rk4_step(z, dt, solver):
    k1 = solver.eval(z)
    k2 = solver.eval(z + 0.5*dt*k1)
    k3 = solver.eval(z + 0.5*dt*k2)
    k4 = solver.eval(z + dt*k3)
    z_new = z + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    solver.sync(z_new)
    return z_new


def adaptive_rk4_step(z, dt, solver, dt_min=1e-4, dt_max=1e-1):
    z_new = rk4_step(z, dt, solver)
    if   solver.cond > 1e7: dt *= 0.7
    elif solver.cond < 1e4: dt *= 1.05
    return z_new, max(dt_min, min(dt, dt_max))