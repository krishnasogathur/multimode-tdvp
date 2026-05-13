

import numpy as np
from qutip import destroy, squeeze, displace, basis
import time
from math import factorial
# from math import factorial
# from scipy.special import comb
import cmath 


from numba import njit

@njit(cache=True,)
def _comb(n, k):
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result

@njit(cache=True,)
def _factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

@njit(cache=True,)
def hermite_phys(n, x):
    if n == 0:
        return 1.0 + 0.0j
    if n == 1:
        return 2.0 * x

    Hnm2 = 1.0 + 0.0j
    Hnm1 = 2.0 * x

    for k in range(1, n):
        Hn = 2.0 * x * Hnm1 - 2.0 * k * Hnm2
        Hnm2, Hnm1 = Hnm1, Hn

    # recursion can be cached; make it a lookup table maybe
    return Hnm1

def gaussian_derivative_closed_form(x, y, c, m, n):
    val = -x**2 - y**2 - c*x*y
    # print("aaaaaaaaaaaaaaaaaaa" , x**2, y**2, c*x*y)
    F = np.exp(val) # tough to compute apparently, can

    total = 0.0 + 0.0j

    for r in range(m + 1):
        k = m - r
        if n - k < 0:
            continue

        term = (
            _comb(m, r)
            * ((c/2) ** k)
            * (2 ** k)
            * _factorial(n) / _factorial(n - k)
            * hermite_phys(n - k, y + 0.5*c * x)
            * ((-1) ** r)
            * hermite_phys(r, x + 0.5*c * y)
        )

        total += term

    return total * F

# def hermite_poly_sum(x,y,c,m,n):
#     total = 0.0 + 0.0j

#     for r in range(m + 1):
#         k = m - r
#         if n - k < 0:
#             continue

#         term = (
#             comb(m, r, exact=True)
#             * ((c/2) ** k)
#             * (2 ** k)
#             * factorial(n) / factorial(n - k)
#             * hermite_phys(n - k, y + 0.5*c * x)
#             * ((-1) ** r)
#             * hermite_phys(r, x + 0.5*c * y)
#         )

#         total += term
#     return total


@njit(cache=True,)
def hermite_poly_sum(x,y,c,m,n):
    total = 0.0 + 0.0j

    for r in range(m + 1):
        k = m - r
        if n - k < 0:
            continue

        term = (
            _comb(m, r)
            * ((c/2) ** k)
            * (2 ** k)
            * _factorial(n) / _factorial(n - k)
            * hermite_phys(n - k, y + 0.5*c * x)
            * ((-1) ** r)
            * hermite_phys(r, x + 0.5*c * y)
        )

        total += term
    return total

@njit(cache=True,)
def gaussian_derivative_closed_form_without_F(x, y, c, m, n):
   
    total = hermite_poly_sum(x, y, c, m, n)

    return total

''' 
slightly tricky working with above func bc F overflows quite easily. I'll simply compute F outside
'''


@njit(cache=True,)
def gaussian_state_overlap(z1, beta1, z2, beta2):
    """
    <z1,beta1 | z2,beta2> using Zhang–Feng–Gilmore formula
    """

    r1, r2 = abs(beta1), abs(beta2)
    t1, t2 = np.angle(beta1), np.angle(beta2)

    # sigma_21
    if np.abs(r1 - r2) < 1e-10 and np.abs(t1 - t2) < 1e-10:
        # print("r1, r2 and t1, t2 are very close; using simplified formula for sigma_21")
        sigma21 = 1.0  # no squeezing difference, so sigma_21 simplifies to 1
    else:
        sigma21 = (
            np.cosh(r2)*np.cosh(r1)
            - np.exp(1j*(t2-t1))*np.sinh(r2)*np.sinh(r1)
        )

    # eta_21 and eta_12
    eta21 = (
        (z2 - z1)*np.cosh(r2)
        - (np.conj(z2) - np.conj(z1))*np.exp(1j*t2)*np.sinh(r2)
    )

    eta12 = (
        (z1 - z2)*np.cosh(r1)
        - (np.conj(z1) - np.conj(z2))*np.exp(1j*t1)*np.sinh(r1)
    )

    exponent = (
        (eta21 * np.conj(eta12)) / (2*sigma21)
        + 0.5*(z2*np.conj(z1) - np.conj(z2)*z1)
    )

    return np.exp(exponent) / np.sqrt(sigma21)


    

def psi_overlap(psi1, psi2): # probably not that useful outside of for computing norm so I will keep it
    """
    Compute <psi1 | psi2> using hierarchical representation.
    """

    overlap = 0.0 + 0.0j

    for sigma in psi1.state:
        if sigma not in psi2.state:
            continue  # orthogonal spin sectors

        gaussians_1 = psi1.state[sigma]
        gaussians_2 = psi2.state[sigma]

        for g1 in gaussians_1:
            c1 = np.exp(g1.kappa - 1j * g1.theta)  

            for g2 in gaussians_2:
                c2 = np.exp(g2.kappa + 1j * g2.theta) 

                prod = 1.0 + 0.0j
                for k in range(len(g1.x)):
                    z1 = g1.alpha[k]   
                    z2 = g2.alpha[k]   

                    beta1 = 0.0  # no squeezing for now
                    beta2 = 0.0

                    prod *= gaussian_state_overlap(z1, beta1, z2, beta2)
                    # multiplying products to find multi mode gaussian overlap

                overlap += c1 * c2 * prod

    return overlap


# import numpy as np
# from math import comb, factorial

def single_mode_gaussians_expec_same_state(a, b, m, n):
    """
    Same-state expectation using single Hermite-type sum
    with (-1)^(n-k) structure as requested.
    """

    result = 0.0 + 0.0j

    abs_alpha_sq = np.abs(a)**2

    for k in range(0, min(m, n) + 1):

        coeff = (
            comb(n, k)
            * factorial(n) / factorial(n - k)
        )

        sign = (-1)**(n - k)

        term = (
            coeff
            * sign
            * (abs_alpha_sq)**(-k)
        )

        result += term

    prefactor = (2 * a)**n * (2 * np.conj(a))**m

    return prefactor * result


@njit(cache=True,)
def single_mode_gaussians_expec(a1, b1, a2, b2, m, n):

    if m == 0 and n == 0:
        # print("m,n both zero, using gaussian state overlap")

        return gaussian_state_overlap(a1, b1, a2, b2)
    
    if np.abs(b1) < 1e-10 and np.abs(b2) < 1e-10:
        # print("both squeezing parameters are zero, using simplified formula")
        return np.conj(a1)**m * (a2)**n * gaussian_state_overlap(a1, b1, a2, b2) 
    
    # print(" one of m, n nonzero and at least one squeezing parameter nonzero, using general formula with \n\n ", "a1=", a1, "b1=", b1, "a2=", a2, "b2=", b2, "m=", m, "n=", n)

#    if np.abs(b1 - b2) < 1e-10:
#        return single_mode_gaussians_expec_same_squeez(a1, b1, a2, m, n)    # Preprocessing
    alpha = a2 - a1
    r1 = np.abs(b1)
    t1 = np.angle(b1)
    r2 = np.abs(b2)
    t2 = np.angle(b2)
    
    u1 = np.cosh(r1)
    v1 = np.sinh(r1) * np.exp(1j * t1)
    u2 = np.cosh(r2)
    v2 = np.sinh(r2) * np.exp(1j * t2)
    
    v1_s = np.conj(v1)
    
    g_coeff = u2*(alpha*v1_s - np.conj(alpha)*u1) + v1_s*(alpha*u2 - np.conj(alpha)*v2)
    gs_coeff = -v2*(alpha*v1_s - np.conj(alpha)*u1) - u1*(alpha*u2 - np.conj(alpha)*v2)
    g_sq_coeff = u2*v1_s
    gs_sq_coeff = u1*v2
    g_gs_coeff = -(u1*u2 + v1_s*v2)
    const = (alpha*v1_s - np.conj(alpha)*u1) * (alpha*u2 - np.conj(alpha)*v2)
    
    # exit(0)
    ''' 
    for later: maybe need a separate func for when squeezing is zero.
    '''
    sigma_21_inv = 0.5/(u2*u1 - v2*v1_s)
    
    g_coeff *= sigma_21_inv
    gs_coeff *= sigma_21_inv
    g_sq_coeff *= sigma_21_inv
    gs_sq_coeff *= sigma_21_inv
    g_gs_coeff *= sigma_21_inv
    const *= sigma_21_inv
    
    g_coeff += 0.5*np.conj(a1 + a2)
    gs_coeff += -0.5*(a1 + a2)
    const += 0.5*(a2*np.conj(a1) - np.conj(a2)*a1)
    g_gs_coeff += 0.5
    
    a = g_sq_coeff
    b = gs_sq_coeff
    c = g_gs_coeff
    d = g_coeff
    e = gs_coeff
    f = const
    

    x_scale = np.sqrt(-a)
    y_scale = np.sqrt(-b)


    xy_scale = x_scale * y_scale
    xy_scale_inv = 1.0 / xy_scale
    c_inv_xy = c * xy_scale_inv
    inv_x = 1.0 / x_scale
    inv_y = 1.0 / y_scale
    d_inv_x = d * inv_x
    e_inv_y = e * inv_y
    
    # if np.abs(c_inv_xy**2 - 4) < 1e-1:
    #     print("aaaaaaaaaaa" , np.abs(b1), np.abs(b2), np.abs(r1-r2), np.abs(t1 - t2), m , n
    #     )
        
    
    den = 4.0 - c_inv_xy**2
    
    x0 = (-2.0*d_inv_x - c_inv_xy*e_inv_y) / den
    y0 = (-2.0*e_inv_y - c_inv_xy*d_inv_x) / den
    
    c_eff = -c_inv_xy
    
    const_new = (
        f
        - x0**2
        - y0**2
        - c_eff*x0*y0
        - d_inv_x*x0
        - e_inv_y*y0
    )
    
    gamma_new = x0
    gamma_star_new = y0
    
    # prefactor = x_scale**m * y_scale**n * np.sqrt(2.0*sigma_21_inv) * np.exp(const_new)
    # if debug:
        # derivative = gaussian_derivative_closed_form_without_F(gamma_new, gamma_star_new, c_eff, m, n) * np.exp(f - const_new)
    # else:
        # derivative = gaussian_derivative_closed_form(gamma_new, gamma_star_new, c_eff, m, n)

    # new prefactor and derivative; directly evaluating F(x0, y0) contribution from F(Gamma, Gamma*). hopefully this helps prevent overflow issues
    prefactor = x_scale**m * y_scale**n * np.sqrt(2.0*sigma_21_inv) 
    if np.abs(f) > 1e8:
        # print("f blowing up, exiting")
        # exit(0)
        raise Exception("f blowing up")
    derivative = gaussian_derivative_closed_form_without_F(gamma_new, gamma_star_new, c_eff, m, n) * np.exp(f)

    return prefactor * derivative



def gaussians_expec(g1, g2, k=0, m=0, n=0):
    c1_conj = np.exp(g1.kappa - 1j * g1.theta)
    c2 = np.exp(g2.kappa + 1j * g2.theta)
    a1, b1 = g1.alpha[k], g1.beta[k]
    a2, b2 = g2.alpha[k], g2.beta[k]

    # print("squeezing" , b1, b2)
    return c1_conj * c2 * single_mode_gaussians_expec(a1, b1, a2, b2, m, n)


''' 
simply wanted to benchmark the new derivative impl; seems to be working fine.
'''
# a1 = 0.9 + 0.4j
# b1 = 0.4 + 0.8j
# a2 = 0.3 + 0.1j
# b2 = 0.5 + 0.6j
# m = 3
# n = 2

# res = single_mode_gaussians_expec(a1, b1, a2, b2, m, n)


# ana = res



# ''' end here'''

# # benchmarking vs qutip

# N = 40   

# start_time = time.time()
# a = destroy(N)
# adag = a.dag()
# # a = destroy(N)
# vac = basis(N, 0)

# # NOTE: QuTiP squeeze convention differs by a minus sign
# S1 = squeeze(N, -b1)
# S2 = squeeze(N, -b2)
# D1 = displace(N, a1)
# D2 = displace(N, a2)
# Dg = displace(N, 0)

# psi1 = D1 * S1 * vac
# psi2 = D2 * S2 * vac

# # exact = (psi1.dag() * Dg * psi2)
# operator = (adag ** m) * (a ** n)
# exact = (psi1.dag() * operator * psi2)
# end_time = time.time()
# time_qutip = end_time - start_time
# # -----------------------------
# # comparison
# # -----------------------------
# print("Analytic:", ana)
# print("QuTiP:   ", exact)
# print("Diff:    ", exact - ana)
# print("Abs err: ", abs(exact - ana))
# print("Time taken for QuTiP calculation: %.6f seconds" % time_qutip)
