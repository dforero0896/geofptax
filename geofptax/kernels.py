"""
Cosmological perturbation theory kernels and integration utilities.

This module provides JAX-compatible implementations of standard perturbation theory
(SPT) kernels for large-scale structure calculations, including redshift-space
distortions and geometric corrections. It also includes numerical integration
routines for computing bispectrum multipoles and power spectra.

The module is designed for efficient computation using JAX's automatic differentiation
and just-in-time compilation capabilities.

Main features:
- Standard SPT kernels (F2, G2, Z1, Z2)
- Redshift-space distortion modeling
- Alcock-Paczynski effect corrections
- Finger-of-God damping
- GEO-FPT geometric corrections
- Numerical integration with Gauss-Legendre quadrature
- Bispectrum multipole calculations
- Sugiyama estimator implementation

Functions are organized into:
1. Numerical integration utilities
2. SPT kernel definitions
3. Bispectrum integrands and multipole calculations
4. Sugiyama estimator implementation
5. Power spectrum 1-loop corrections

All functions are JAX-compatible and support vectorization via `jax.vmap`.
"""

import jax
import jax.numpy as jnp
#from quadax import quadgk
import numpy as np
from functools import partial
from .constants import *



@jax.jit
def jax_leggauss(x):
    """
    Compute Gauss-Legendre quadrature nodes and weights using JAX-compatible methods.
    
    This function computes the nodes (roots of Legendre polynomials) and weights
    for Gauss-Legendre quadrature using Newton's method with Chebyshev nodes as
    initial guesses.
    
    Parameters
    ----------
    x : jnp.ndarray
        Array of indices or placeholder values (typically `jnp.arange(n)`).
        The length determines the number of quadrature points.
    
    Returns
    -------
    x_nodes : jnp.ndarray
        Quadrature nodes in the interval [-1, 1].
    w_weights : jnp.ndarray
        Quadrature weights corresponding to each node.
    
    Notes
    -----
    - Uses Newton iteration (5 iterations typically sufficient for convergence)
    - Based on the Golub-Welsch algorithm for computing Gaussian quadrature rules
    - Fully JAX-compatible and JIT-compilable
    
    Examples
    --------
    >>> n = 10
    >>> x, w = jax_leggauss(jnp.arange(n))
    >>> print(x.shape, w.shape)  # Both have shape (10,)
    """
    # Use Newton's method to find roots of Legendre polynomials
    # Initial guess (Chebyshev nodes)
    n = x.shape[0]
    x = jnp.cos(jnp.pi * (x + 0.5) / n)
    
    # Newton iteration
    def body_fun(*val):
        x, _ = val
        P, P_prime = legendre_recurrence(x, n)
        dx = P / P_prime
        x_new = x - dx
        return x_new, None
    
    x, _ = jax.lax.scan(body_fun, x, None, length=5)  # 5 iterations typically sufficient
    
    # Compute weights
    _, P_prime = legendre_recurrence(x, n)
    w = 2.0 / ((1.0 - x**2) * P_prime**2)
    return x, w

def legendre_recurrence(x, n):
    """
    Compute Legendre polynomial and its derivative using recurrence relations.
    
    Parameters
    ----------
    x : jnp.ndarray
        Points at which to evaluate the Legendre polynomial.
    n : int
        Degree of the Legendre polynomial.
    
    Returns
    -------
    P : jnp.ndarray
        Values of the Legendre polynomial of degree n at points x.
    dP : jnp.ndarray
        Values of the derivative of the Legendre polynomial at points x.
    
    Notes
    -----
    - Uses the standard three-term recurrence relation for Legendre polynomials
    - Efficient for moderate values of n (n < 1000)
    - Fully JAX-compatible
    
    Examples
    --------
    >>> x = jnp.linspace(-1, 1, 100)
    >>> P, dP = legendre_recurrence(x, 5)
    """
    P0 = jnp.ones_like(x)
    P1 = x
    dP0 = jnp.zeros_like(x)
    dP1 = jnp.ones_like(x)
    
    for k in range(1, n):
        P = ((2*k + 1)*x*P1 - k*P0)/(k + 1)
        dP = ((2*k + 1)*(P1 + x*dP1) - k*dP0)/(k + 1)
        P0, P1 = P1, P
        dP0, dP1 = dP1, dP
    
    return P1, dP1

@partial(jax.jit, static_argnames = ('f', 'nx', 'ny'))
def integrate_2d_gauss(f, xmin, xmax, ymin, ymax, nx=20, ny=20):
    """
    Compute a 2D integral using vectorized Gauss-Legendre quadrature.

    This function performs double integration of a function f(x,y) over the rectangular domain
    [xmin, xmax] × [ymin, ymax] using Gauss-Legendre quadrature rules. The implementation
    is fully JAX-compatible and optimized for parallel evaluation.

    Parameters
    ----------
    f : callable
        A JAX-compatible function of two variables to be integrated.
        Signature must be f(x, y) where x and y are JAX arrays.
    xmin : float
        Lower bound of integration for the x-axis.
    xmax : float
        Upper bound of integration for the x-axis.
    ymin : float
        Lower bound of integration for the y-axis.
    ymax : float
        Upper bound of integration for the y-axis.
    nx : int, optional
        Number of Gauss-Legendre quadrature points for x-axis. Default is 20.
    ny : int, optional
        Number of Gauss-Legendre quadrature points for y-axis. Default is 20.

    Returns
    -------
    float
        The computed value of the integral ∫∫ f(x,y) dx dy over the domain.

    Notes
    -----
    - The implementation uses JAX-native operations and is fully JIT-compilable.
    - For smooth functions, nx=ny=20 typically gives machine precision.
    - The quadrature weights/nodes are computed using the Golub-Welsch algorithm.
    - The function handles vectorized inputs efficiently using JAX's vmap.

    Examples
    --------
    >>> def f(x, y):
    ...     return jnp.sin(x) * jnp.cos(y)
    >>> integrate_2d_gauss(f, 0, 1, 0, 1)  # Integrate sin(x)cos(y) over [0,1]×[0,1]
    Array(0.354036, dtype=float32)

    See Also
    --------
    weights_leggauss : The underlying 1D quadrature implementation
    jax.vmap : Used for vectorized function evaluation
    """
    # Get quadrature points and weights
    x, wx = jax_leggauss(jnp.arange(nx))
    y, wy = jax_leggauss(jnp.arange(ny))
    
    # Rescale from [-1, 1] to [xmin, xmax] × [ymin, ymax]
    x_scaled = 0.5*(xmax - xmin)*x + 0.5*(xmax + xmin)
    y_scaled = 0.5*(ymax - ymin)*y + 0.5*(ymax + ymin)
    
    # Create 2D grid of points
    X, Y = jnp.meshgrid(x_scaled, y_scaled, indexing='ij')
    
    # Vectorized function evaluation
    F = jax.vmap(jax.vmap(f, (0, 0)), (0, 0))(X, Y)
    
    # Compute integral
    integral = jnp.sum(wx[:, None] * wy[None, :] * F)
    return integral * 0.25*(xmax - xmin)*(ymax - ymin)

    
def cosab(ka, kb, kc):
    """
    Compute the cosine of the angle between vectors ka and kb using the law of cosines.
    
    Given vectors with magnitudes ka and kb and the magnitude of their difference kc,
    compute the cosine of the angle between ka and kb.

    Parameters
    ----------
    ka : float or jnp.ndarray
        Magnitude of the first vector.
    kb : float or jnp.ndarray
        Magnitude of the second vector.
    kc : float or jnp.ndarray
        Magnitude of the third vector (resultant of ka and kb).

    Returns
    -------
    float or jnp.ndarray
        Cosine of the angle between vectors ka and kb.
    
    Notes
    -----
    - Derived from the law of cosines: kc² = ka² + kb² - 2*ka*kb*cos(θ)
    - Used extensively in perturbation theory kernels
    
    Examples
    --------
    >>> ka, kb, kc = 1.0, 1.0, jnp.sqrt(2.0)
    >>> cosab(ka, kb, kc)  # Should be cos(90°) = 0
    Array(0., dtype=float32)
    """
    return (kc * kc - ka * ka - kb * kb) / (2 * ka * kb)

def f2_ker(ka, kb, kc):
    """
    Compute the second-order SPT kernel F2 for matter density perturbations.
    
    The F2 kernel describes the second-order contribution to the density field
    in standard perturbation theory.

    Parameters
    ----------
    ka : float or jnp.ndarray
        Magnitude of the first wavevector.
    kb : float or jnp.ndarray
        Magnitude of the second wavevector.
    kc : float or jnp.ndarray
        Magnitude of the resultant wavevector.

    Returns
    -------
    float or jnp.ndarray
        Value of the F2 kernel.
    
    Notes
    -----
    - F2 kernel is symmetric in its arguments: F2(k₁, k₂) = F2(k₂, k₁)
    - The kernel is scale-free and depends only on wavevector magnitudes and angles
    - Expression: F2 = 5/7 + 1/2*cosθ*(k₁/k₂ + k₂/k₁) + 2/7*cos²θ
    
    Examples
    --------
    >>> f2_ker(1.0, 1.0, jnp.sqrt(2.0))  # Equal magnitudes, 90° angle
    Array(0.5714286, dtype=float32)
    """
    cab = cosab(ka, kb, kc)
    return 5. / 7. + 0.5 * cab * (ka / kb + kb / ka) + 2. / 7. * cab**2

def interpol_ker(a, fi_a_val, a_val=jnp.array([1. / (1 + 2.), 1. / (1 + 1.), 1. / (1 + 0.5)])):
    """
    Interpolate the kernel values at a given scale factor.
    
    Linearly interpolates kernel values from predefined scale factors to
    a target scale factor a.

    Parameters
    ----------
    a : float
        Scale factor at which to interpolate.
    fi_a_val : jnp.ndarray
        Array of kernel values at predefined scale factors.
    a_val : jnp.ndarray, optional
        Predefined scale factors for interpolation. Default is [1/3, 1/2, 2/3]
        corresponding to redshifts z=2, 1, 0.5.

    Returns
    -------
    jnp.ndarray
        Interpolated kernel values at the given scale factor.
    
    Examples
    --------
    >>> a = 0.5
    >>> fi_vals = jnp.array([1.0, 1.5, 2.0])
    >>> interpol_ker(a, fi_vals)
    Array(1.75, dtype=float32)
    """
    return jnp.interp(a, a_val, fi_a_val)

interpol_ker = jax.vmap(interpol_ker, in_axes = (None, 0))

def g2_ker(ka, kb, kc):
    """
    Compute the second-order SPT kernel G2 for velocity divergence perturbations.
    
    The G2 kernel describes the second-order contribution to the velocity
    divergence field in standard perturbation theory.

    Parameters
    ----------
    ka : float or jnp.ndarray
        Magnitude of the first wavevector.
    kb : float or jnp.ndarray
        Magnitude of the second wavevector.
    kc : float or jnp.ndarray
        Magnitude of the resultant wavevector.

    Returns
    -------
    float or jnp.ndarray
        Value of the G2 kernel.
    
    Notes
    -----
    - G2 kernel is symmetric in its arguments: G2(k₁, k₂) = G2(k₂, k₁)
    - Related to the F2 kernel but with different coefficients
    - Expression: G2 = 3/7 + 1/2*cosθ*(k₁/k₂ + k₂/k₁) + 4/7*cos²θ
    
    Examples
    --------
    >>> g2_ker(1.0, 1.0, jnp.sqrt(2.0))
    Array(0.42857143, dtype=float32)
    """
    cab = cosab(ka, kb, kc)
    return 3. / 7. + 0.5 * cab * (ka / kb + kb / ka) + 4. / 7. * cab**2

def z1_ker(mu, cosm_par):
    """
    Compute the first-order redshift-space SPT kernel Z1.
    
    The Z1 kernel describes linear redshift-space distortions, incorporating
    both linear bias and Kaiser effects.

    Parameters
    ----------
    mu : float or jnp.ndarray
        Cosine of the angle between the wavevector and the line of sight.
    cosm_par : jnp.ndarray
        Cosmological parameters array, where cosm_par[4] is the linear bias (b₁)
        and cosm_par[1] is the growth rate (f).

    Returns
    -------
    float or jnp.ndarray
        Value of the Z1 kernel.
    
    Notes
    -----
    - Z1 = b₁ + f*μ² where μ = k·ẑ/|k|
    - b₁: linear galaxy bias
    - f: linear growth rate f ≈ Ωₘ(z)^γ with γ ≈ 0.55
    
    Examples
    --------
    >>> mu = 0.5
    >>> cosm_par = jnp.array([...])  # Contains b₁ at index 3, f at index 0
    >>> z1_ker(mu, cosm_par)
    """
    b1, ff = cosm_par[3], cosm_par[0]
    return b1 + ff * mu**2


def z2_ker(ka, kb, kc, fkern, gkern, mua, mub, cosm_par):
    """
    Compute the second-order redshift-space SPT kernel Z2.
    
    The Z2 kernel describes second-order redshift-space distortions,
    incorporating bias terms, velocity divergence, and nonlinear RSD effects.

    Parameters
    ----------
    ka : float or jnp.ndarray
        Magnitude of the first wavevector.
    kb : float or jnp.ndarray
        Magnitude of the second wavevector.
    kc : float or jnp.ndarray
        Magnitude of the resultant wavevector.
    fkern : float or jnp.ndarray
        F2 kernel value.
    gkern : float or jnp.ndarray
        G2 kernel value.
    mua : float or jnp.ndarray
        Cosine of the angle between ka and the line of sight.
    mub : float or jnp.ndarray
        Cosine of the angle between kb and the line of sight.
    cosm_par : jnp.ndarray
        Cosmological parameters array, where:
        - cosm_par[3] is the linear bias (b₁)
        - cosm_par[0] is the growth rate (f)
        - cosm_par[4] is the second-order bias (b₂)
        - cosm_par[5] is the tidal bias (bs)

    Returns
    -------
    float or jnp.ndarray
        Value of the Z2 kernel.
    
    Notes
    -----
    - Full expression includes terms proportional to b₁, f, f², b₂, and bₛ
    - bₛ is the tidal bias term: bₛ = -4/7*(b₁ - 1)
    - S₂ kernel: S₂ = cos²θ - 1/3
    
    Examples
    --------
    >>> ka, kb, kc = 1.0, 1.0, jnp.sqrt(2.0)
    >>> fkern = f2_ker(ka, kb, kc)
    >>> gkern = g2_ker(ka, kb, kc)
    >>> mua, mub = 0.5, -0.5
    >>> cosm_par = jnp.array([...])  # Contains b₁, f, b₂ at appropriate indices
    >>> z2_ker(ka, kb, kc, fkern, gkern, mua, mub, cosm_par)
    """
    cab = cosab(ka, kb, kc)
    
    b1, ff, b2, bs = cosm_par[3], cosm_par[0], cosm_par[4], cosm_par[5]

    ksq = jnp.sqrt(ka**2 + kb**2 + 2 * ka * kb * cab)  # modulus of vector sum k1 + k2
    mu12 = (ka * mua + kb * mub) / ksq
    
    # TODO: relax condition on bs and make it a parameter.
    
    s2 = cab**2 - 1.0 / 3.0  # S_2 kernel

    b1_terms = b1 * (fkern + 0.5 * ff * mu12 * ksq * (mua / ka + mub / kb))
    g_term = ff * mu12**2 * gkern
    fsq_term = 0.5 * ff**2 * mu12 * ksq * mua * mub * (mub / ka + mua / kb)
    b_terms = 0.5 * (b2 + bs * s2)

    return b1_terms + g_term + fsq_term + b_terms



def geo_fac(ka, kb, kc, af, hh):
    """
    Compute the GEO-FPT factor multiplying Z2_SPT to obtain Z2_GEO.
    
    GEO-FPT (Geometrical Fitting formula for Perturbation Theory) provides
    a phenomenological correction to improve the accuracy of perturbation
    theory predictions for the bispectrum.

    Parameters
    ----------
    ka : float or jnp.ndarray
        Magnitude of the first wavevector.
    kb : float or jnp.ndarray
        Magnitude of the second wavevector.
    kc : float or jnp.ndarray
        Magnitude of the resultant wavevector.
    af : jnp.ndarray
        Array of coefficients for the GEO-FPT factor.
        Typically has 5 elements: [f₁, f₂, f₃, f₄, f₅]
    hh : float
        Hubble parameter (normalization factor).

    Returns
    -------
    float or jnp.ndarray
        Value of the GEO-FPT factor.
    
    Notes
    -----
    - The factor is computed as: f₁ + f₂*(cos_med/cos_min) + f₃*(cos_max/cos_min)
      + f₄*area + f₅*area²
    - Uses Heron's formula to compute triangle area
    - cos_max, cos_med, cos_min are cosines ordered by magnitude
    
    Examples
    --------
    >>> ka, kb, kc = 0.1, 0.1, 0.1  # Equilateral triangle
    >>> af = jnp.array([1.0, 0.1, 0.1, 0.01, 0.001])
    >>> hh = 0.7
    >>> geo_fac(ka, kb, kc, af, hh)
    """
    # Determine kmax, kmed, kmin
    k = jnp.array([ka, kb, kc])
    kmax = jnp.max(k, axis = 0)
    kmin = jnp.min(k, axis = 0)
    kmed = jnp.sum(k, axis = 0) - kmax - kmin
    
    
    # Compute cosines
    cosmax = (kmed**2 + kmin**2 - kmax**2) / (2 * kmed * kmin)
    cosmed = (kmax**2 + kmin**2 - kmed**2) / (2 * kmax * kmin)
    cosmin = (kmax**2 + kmed**2 - kmin**2) / (2 * kmax * kmed)

    # Compute area using Heron's formula
    perim = (ka + kb + kc) / 2
    area = jnp.sqrt(perim * (perim - ka) * (perim - kb) * (perim - kc)) / (hh**2 * 0.001)

    # Compute extra term
    extra = af[0] + af[1] * cosmed / cosmin + af[2] * cosmax / cosmin + af[3] * area + af[4] * area**2
    
    return extra

def geo_fac_damped(ka, kb, kc, af, hh, k_damp=0.12, width=5.):
    """
    GEO-FPT factor with smooth damping beyond a specified scale.
    
    Applies a Gaussian damping to the GEO-FPT corrections beyond k_damp,
    keeping only the constant term f₁ at high k.

    Parameters
    ----------
    ka : float or jnp.ndarray
        Magnitude of the first wavevector.
    kb : float or jnp.ndarray
        Magnitude of the second wavevector.
    kc : float or jnp.ndarray
        Magnitude of the resultant wavevector.
    af : jnp.ndarray
        Array of coefficients for the GEO-FPT factor.
    hh : float
        Hubble parameter (normalization factor).
    k_damp : float, optional
        Damping scale (h/Mpc). Default is 0.12.
    width : float, optional
        Width of the damping transition. Default is 0.03.

    Returns
    -------
    float or jnp.ndarray
        Damped GEO-FPT factor.
    
    Notes
    -----
    - Damping factor: exp(-((k_avg - k_damp)/width)²)
    - Only the constant term f₁ is kept at high k
    - Useful for avoiding unphysical behavior at small scales
    
    Examples
    --------
    >>> ka, kb, kc = 0.2, 0.2, 0.2  # High k triangle
    >>> af = jnp.array([1.0, 0.1, 0.1, 0.01, 0.001])
    >>> hh = 0.7
    >>> geo_fac_damped(ka, kb, kc, af, hh, k_damp=0.12, width=0.03)
    """
    k_avg = (ka + kb + kc) / 3
    
    # Compute standard GEO-FPT factor
    standard = geo_fac(ka, kb, kc, af, hh)
    
    # Damping factor: 1 for k << k_damp, 0 for k >> k_damp
    #damp = jnp.exp(-((k_avg - k_damp)/width)**2)
    damp = (jax.nn.sigmoid(-width * (ka - k_damp))) * (jax.nn.sigmoid(-width * (kb - k_damp))) * (jax.nn.sigmoid(-width * (kc - k_damp)))
    
    # Apply damping to corrections (keep f₁ term)
    f1_only = af[0]  # Just the constant term
    corrections = standard - f1_only
    
    return f1_only + damp * corrections

def geo_fac_pade(ka, kb, kc, af, hh, A_peak=6.235):
    """
    GEO-FPT factor with Padé [2/2] approximant for area terms.
    
    The area term is constructed to:
    1. Match the polynomial f4*A + f5*A^2 at low A (small k).
    2. Asymptote to a constant at high A (large k) such that the 
       FULL GEO-FPT factor asymptotes to 1.0. 
       (This ensures the geometric correction smoothly turns off at high k, 
       leaving the standard SPT kernel).
    3. Have a strictly positive denominator (no real roots) to prevent divergence.
    """
    # Unpack coefficients
    f1, f2, f3, f4, f5 = af
    
    # Compute triangle area (normalized)
    perim = (ka + kb + kc) / 2
    area_sq = perim * (perim - ka) * (perim - kb) * (perim - kc)
    area = jnp.sqrt(jnp.maximum(area_sq, 1e-20)) / (hh**2 * 0.001)
    
    # Compute shape terms (cosines of angles)
    k = jnp.array([ka, kb, kc])
    kmax = jnp.max(k, axis=0)
    kmin = jnp.min(k, axis=0)
    kmed = jnp.sum(k, axis=0) - kmax - kmin
    
    cosmax = (kmed**2 + kmin**2 - kmax**2) / (2 * kmed * kmin)
    cosmed = (kmax**2 + kmin**2 - kmed**2) / (2 * kmax * kmin)
    cosmin = (kmax**2 + kmed**2 - kmin**2) / (2 * kmax * kmed)
    
    # Shape contribution (independent of area A)
    shape_term = (f1 + 
                  f2 * jnp.where(jnp.abs(cosmin) > 1e-10, cosmed/cosmin, 0.0) + 
                  f3 * jnp.where(jnp.abs(cosmin) > 1e-10, cosmax/cosmin, 0.0))
    
    # We want the FULL factor (shape_term + area_terms) to asymptote to 1.0 as A -> inf.
    # Therefore, the area_terms must asymptote to (1.0 - shape_term).
    A_target = 1.0 - shape_term
    
    # Padé [2/2] form: (p1*A + p2*A^2) / (1 + q1*A + q2*A^2)
    # 1. Low A matches f4*A + f5*A^2 => p1 = f4, p2 = f5 + f4*q1
    # 2. High A asymptotes to A_target => q2 = p2 / A_target
    # 3. No real roots => q1^2 - 4*q2 < 0
    
    p1 = f4
    
    # Choosing q1 = 2*f4 minimizes the discriminant to -4*(f4^2 + f5)/A_target.
    # As long as f4^2 + f5 > 0 and A_target > 0, the denominator never crosses zero.
    q1 = 2.0 * f4
    p2 = f5 + f4 * q1
    
    # Protect against A_target <= 0 (which would make q2 negative and cause poles).
    # If A_target <= 0, we fallback to a small positive value (effectively damping to 0).
    A_target_safe = jnp.maximum(A_target, 1e-6)
    
    q2 = p2 / A_target_safe
    
    # Padé approximant for area terms
    area_terms = (p1 * area + p2 * area**2) / (1.0 + q1 * area + q2 * area**2)
    
    # Complete GEO-FPT factor
    extra = shape_term + area_terms
    
    return extra

# Vectorized versions of the functions
g2_ker_vec = jax.vmap(g2_ker, (0, 0, 0))
z1_ker_vec = jax.vmap(z1_ker, (0, None))
z2_ker_vec = jax.vmap(z2_ker, (0, 0, 0, 0, 0, 0, 0, None))
geo_fac_vec = jax.vmap(geo_fac, (0, 0, 0, None, None))


@jax.jit
def bkeff_r_scalar(mua_m, phi, tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp):
    """
    Compute the integrand for the effective bispectrum in redshift space.
    
    This is the core integrand function for computing redshift-space bispectrum
    multipoles, including all physical effects: RSD, AP, FoG, and GEO-FPT corrections.

    Parameters
    ----------
    mua_m : float or jnp.ndarray
        Cosine of the angle between ka and the line of sight in real space.
    phi : float or jnp.ndarray
        Azimuthal angle (angle between planes defined by ka and kb).
    tr : tuple or jnp.ndarray
        Triangle side lengths (ka_m, kb_m, kc_m) in real space.
    cosm_par : jnp.ndarray
        Cosmological parameters array containing:
        - cosm_par[1]: α_∥ (parallel AP parameter)
        - cosm_par[2]: α_⟂ (perpendicular AP parameter)
        - cosm_par[0]: f (growth rate)
        - cosm_par[3]: b₁ (linear bias)
        - cosm_par[4]: b₂ (quadratic bias)
        - cosm_par[5]: bs (tidal bias)
        - cosm_par[8]: σ_FoG (Finger-of-God damping scale)
    pk_in : tuple or jnp.ndarray
        Power spectrum values at ka_m, kb_m, kc_m.
    sig_fog : float
        Finger-of-God damping factor.
    log_km : jnp.ndarray
        Logarithm of wavevector magnitudes for interpolation.
    log_pkm : jnp.ndarray
        Logarithm of power spectrum values for interpolation.
    af : jnp.ndarray
        Array of coefficients for the GEO-FPT factor.
    mp : int
        Multipole index (0 for monopole, 1 for quadrupole, etc.).

    Returns
    -------
    float or jnp.ndarray
        Value of the integrand.
    
    Notes
    -----
    - Applies Alcock-Paczynski scaling to transform from real to redshift space
    - Includes triangle validity checks
    - Computes all necessary kernels (Z1, Z2, F2, G2) with GEO-FPT corrections
    - Applies FoG damping
    - Returns zero for invalid triangles
    
    Examples
    --------
    >>> mua_m = 0.5
    >>> phi = jnp.pi/4
    >>> tr = (0.1, 0.1, 0.1)  # Equilateral triangle
    >>> cosm_par = jnp.array([...])  # Cosmological parameters
    >>> pk_in = (1.0, 1.0, 1.0)  # Power spectrum values
    >>> sig_fog = 5.0
    >>> log_km = jnp.log10(jnp.linspace(0.01, 0.3, 100))
    >>> log_pkm = jnp.log10(pk_interp(10**log_km))
    >>> af = jnp.array([1.0, 0.1, 0.1, 0.01, 0.001])
    >>> mp = 0  # Monopole
    >>> bkeff_r_scalar(mua_m, phi, tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp)
    """
    ka_m, kb_m, kc_m = tr
    pka, pkb, pkc = pk_in
    spline_me = lambda logk: jnp.interp(logk, log_km, log_pkm)
    
    alpa, alpe = cosm_par[1], cosm_par[2]
    Fsq = 1.0 / (alpa / alpe)**2
    b1, ff = cosm_par[3], cosm_par[0]
    A_P = cosm_par[6]  # Power spectrum shot noise
    A_B = cosm_par[7]  # Bispectrum shot noise
    cab_m = cosab(ka_m, kb_m, kc_m)
    mub_m = mua_m * cab_m - jnp.sqrt((1.0 - mua_m**2) * (1.0 - cab_m**2)) * jnp.cos(phi)
    muc_m = (-ka_m * mua_m - kb_m * mub_m) / kc_m
    mua_m = jnp.broadcast_to(mua_m, muc_m.shape)
    
    ka = ka_m * jnp.sqrt(1.0 + mua_m**2 * (Fsq - 1.0)) / alpe
    kb = kb_m * jnp.sqrt(1.0 + mub_m**2 * (Fsq - 1.0)) / alpe
    kc = kc_m * jnp.sqrt(1.0 + muc_m**2 * (Fsq - 1.0)) / alpe
    hh = 1.0
    
    # Discard invalid triangles
    valid = (kb + ka - kc >= hh * 1.1 * 2 * jnp.pi / 1000.0) & \
            (ka + kc - kb >= hh * 1.1 * 2 * jnp.pi / 1000.0) & \
            (kb + kc - ka >= hh * 1.1 * 2 * jnp.pi / 1000.0)

    mua = mua_m * alpe / (alpa * jnp.sqrt(1.0 + mua_m**2 * (Fsq - 1.0)))
    mub = mub_m * alpe / (alpa * jnp.sqrt(1.0 + mub_m**2 * (Fsq - 1.0)))
    muc = muc_m * alpe / (alpa * jnp.sqrt(1.0 + muc_m**2 * (Fsq - 1.0)))
    
    
    eff_fact = geo_fac(ka, kb, kc, af, hh)
    
    D_fog = 1. / (1 + 0.5 * ((ka * mua)**2 +  (kb * mub)**2 + (kc * muc)**2)**2 * (sig_fog / hh)**4)**2

    

    z1_1 = z1_ker(mua, cosm_par)
    z1_2 = z1_ker(mub, cosm_par)
    z1_3 = z1_ker(muc, cosm_par)

    f2k_12 = f2_ker(ka, kb, kc)
    f2k_23 = f2_ker(kc, kb, ka)
    f2k_13 = f2_ker(ka, kc, kb)

    g2k_12 = g2_ker(ka, kb, kc)
    g2k_23 = g2_ker(kc, kb, ka)
    g2k_13 = g2_ker(ka, kc, kb)

    z2_12 = z2_ker(ka, kb, kc, f2k_12, g2k_12, mua, mub, cosm_par) * eff_fact
    z2_23 = z2_ker(kb, kc, ka, f2k_23, g2k_23, mub, muc, cosm_par) * eff_fact
    z2_13 = z2_ker(ka, kc, kb, f2k_13, g2k_13, mua, muc, cosm_par) * eff_fact
    leg = jnp.select([mp == 1, mp == 2, mp == 3], 
                     [5 * (3 * mua**2 - 1) / 2, 5 * (3 * mub**2 - 1) / 2, 5 * (3 * muc**2 - 1) / 2],
                     1.)
    
    
    pka = 10**spline_me(jnp.log10(ka))
    pkb = 10**spline_me(jnp.log10(kb))
    pkc = 10**spline_me(jnp.log10(kc))

    result = leg * D_fog * (z1_1 * z1_2 * z2_12 * pka * pkb +
                            z1_3 * z1_2 * z2_23 * pkc * pkb +
                            z1_1 * z1_3 * z2_13 * pka * pkc) / (2 * jnp.pi * alpa**2 * alpe**4)


    shot_noise = ((b1 * A_B + 2.0 * A_P * ff * mua**2) * z1_1 * pka +
                  (b1 * A_B + 2.0 * A_P * ff * mub**2) * z1_2 * pkb +
                  (b1 * A_B + 2.0 * A_P * ff * muc**2) * z1_3 * pkc +
                  A_P**2)

    #return valid * result #jnp.where(valid, result, 0.0)
    result += leg * shot_noise / (2 * jnp.pi * alpa**2 * alpe**4)
    return valid * result

bkeff_r_vmap = jax.vmap(bkeff_r_scalar, in_axes=(0, 0, None, None, None, None, None, None, None, None))

def integrate_bkeff_r(tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp, xmin, xmax, num_points):
    """
    Perform 2D integration of the effective bispectrum integrand using Gauss-Legendre quadrature.
    
    Integrates over μ (cosine of angle with line of sight) and φ (azimuthal angle)
    to compute bispectrum multipoles.

    Parameters
    ----------
    tr : tuple or jnp.ndarray
        Triangle side lengths (ka_m, kb_m, kc_m) in real space.
    cosm_par : jnp.ndarray
        Cosmological parameters array.
    pk_in : tuple or jnp.ndarray
        Power spectrum values at ka_m, kb_m, kc_m.
    sig_fog : float
        Finger-of-God damping factor.
    log_km : jnp.ndarray
        Logarithm of wavevector magnitudes for interpolation.
    log_pkm : jnp.ndarray
        Logarithm of power spectrum values for interpolation.
    af : jnp.ndarray
        Array of coefficients for the GEO-FPT factor.
    mp : int
        Multipole index (0 for monopole, 1 for quadrupole, etc.).
    xmin : tuple
        Lower bounds of integration (mua_min, phi_min).
    xmax : tuple
        Upper bounds of integration (mua_max, phi_max).
    num_points : int, optional
        Number of Gauss-Legendre points for mua and phi axes. Default is 20.

    Returns
    -------
    float
        Result of the 2D integration.
    
    Notes
    -----
    - Uses Gauss-Legendre quadrature for high accuracy
    - Integration domain: μ ∈ [-1, 1], φ ∈ [0, 2π]
    - Calls `integrate_2d_gauss` for the actual integration
    
    Examples
    --------
    >>> tr = (0.1, 0.1, 0.1)  # Equilateral triangle
    >>> cosm_par = jnp.array([...])
    >>> pk_in = (1.0, 1.0, 1.0)
    >>> sig_fog = 5.0
    >>> log_km = jnp.log10(jnp.linspace(0.01, 0.3, 100))
    >>> log_pkm = jnp.log10(pk_interp(10**log_km))
    >>> af = jnp.array([1.0, 0.1, 0.1, 0.01, 0.001])
    >>> mp = 0
    >>> xmin = (-1.0, 0.0)
    >>> xmax = (1.0, 2*jnp.pi)
    >>> integrate_bkeff_r(tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp, xmin, xmax, 20)
    """
    nx , ny = num_points, num_points
    def integrand(mua, phi):
        return bkeff_r_scalar(mua, phi, tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp)

    return integrate_2d_gauss(integrand, xmin[0], xmax[0], xmin[1], xmax[1], nx=nx, ny=ny)

def _integrate_bkeff_r(tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp, xmin, xmax, num_points):
    """
    Perform 2D integration of the effective bispectrum integrand using trapezoidal rule.
    
    Alternative implementation using trapezoidal integration on a regular grid.
    Less accurate than Gauss-Legendre but simpler.

    Parameters
    ----------
    tr : tuple or jnp.ndarray
        Triangle side lengths (ka_m, kb_m, kc_m) in real space.
    cosm_par : jnp.ndarray
        Cosmological parameters array.
    pk_in : tuple or jnp.ndarray
        Power spectrum values at ka_m, kb_m, kc_m.
    sig_fog : float
        Finger-of-God damping factor.
    log_km : jnp.ndarray
        Logarithm of wavevector magnitudes for interpolation.
    log_pkm : jnp.ndarray
        Logarithm of power spectrum values for interpolation.
    af : jnp.ndarray
        Array of coefficients for the GEO-FPT factor.
    mp : int
        Multipole index (0 for monopole, 1 for quadrupole, etc.).
    xmin : tuple
        Lower bounds of integration (mua_min, phi_min).
    xmax : tuple
        Upper bounds of integration (mua_max, phi_max).
    num_points : int
        Number of points for the integration grid.

    Returns
    -------
    float
        Result of the 2D integration.
    
    Notes
    -----
    - Uses regular grid and trapezoidal rule
    - Vectorized evaluation using `bkeff_r_vmap`
    - Less accurate than Gauss-Legendre for smooth integrands
    
    See Also
    --------
    integrate_bkeff_r : Gauss-Legendre version (preferred)
    """
    mua_grid = jnp.linspace(xmin[0], xmax[0], num_points)
    phi_grid = jnp.linspace(xmin[1], xmax[1], num_points)
    mua_mesh, phi_mesh = jnp.meshgrid(mua_grid, phi_grid, indexing='ij')
    

    # Vectorize over grid points
    integrand_values = bkeff_r_vmap(mua_mesh.ravel(), phi_mesh.ravel(), tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp)

    # Reshape and integrate using trapezoidal rule
    integrand_values = integrand_values.reshape(mua_mesh.shape)
    integral = jnp.trapezoid(jnp.trapezoid(integrand_values, phi_grid, axis=1), mua_grid)
    return integral


vec_integrate_bkeff_r = jax.jit(jax.vmap(integrate_bkeff_r, in_axes = (0, None, 0, None, None, None, None, None, None, None, None)), static_argnames = ('num_points',))



def ext_bk_mp(tr, tr2, tr3, tr4, log_km, log_pkm, cosm_par, redshift, fi_vals=F_VALS_FULL, num_points=50):
    """
    Compute the effective bispectrum multipoles for a given set of triangles and cosmological parameters.
    
    Computes four multipoles (monopole and three quadrupoles) for four different
    triangle configurations.

    Parameters
    ----------
    tr : jnp.ndarray
        Array of triangle side lengths (ka, kb, kc) for the monopole calculation.
    tr2 : jnp.ndarray
        Array of triangle side lengths (ka, kb, kc) for the first multipole calculation.
    tr3 : jnp.ndarray
        Array of triangle side lengths (ka, kb, kc) for the second multipole calculation.
    tr4 : jnp.ndarray
        Array of triangle side lengths (ka, kb, kc) for the third multipole calculation.
    log_km : jnp.ndarray
        Logarithm of wavevector magnitudes for interpolation.
    log_pkm : jnp.ndarray
        Logarithm of power spectrum values for interpolation.
    cosm_par : jnp.ndarray
        Cosmological parameters array.
    redshift : float
        Redshift at which to compute the bispectrum.
    fi_vals : jnp.ndarray, optional
        Array of kernel values for interpolation. Default is F_VALS_FULL.
    num_points : int, optional
        Number of points for the integration grid. Default is 50.

    Returns
    -------
    tuple
        A tuple containing the bispectrum monopole (bk0), first multipole (bk200),
        second multipole (bk020), and third multipole (bk002).
    
    Notes
    -----
    - Multipole indices: 0=monopole, 1=quadrupole for ka, 2=quadrupole for kb, 3=quadrupole for kc
    - All triangles should correspond to the same physical configuration
    - Uses vectorized integration `vec_integrate_bkeff_r` for efficiency
    
    Examples
    --------
    >>> tr = jnp.array([[0.1, 0.1, 0.1]])  # Equilateral triangle for monopole
    >>> tr2 = jnp.array([[0.1, 0.1, 0.1]])  # Same triangle for first quadrupole
    >>> tr3 = jnp.array([[0.1, 0.1, 0.1]])  # Same triangle for second quadrupole
    >>> tr4 = jnp.array([[0.1, 0.1, 0.1]])  # Same triangle for third quadrupole
    >>> log_km = jnp.log10(jnp.linspace(0.01, 0.3, 100))
    >>> log_pkm = jnp.log10(pk_interp(10**log_km))
    >>> cosm_par = jnp.array([...])
    >>> redshift = 0.5
    >>> bk0, bk200, bk020, bk002 = ext_bk_mp(tr, tr2, tr3, tr4, log_km, log_pkm, cosm_par, redshift)
    """
    a_t = 1.0 / (1.0 + redshift)

    # Interpolate the kernel values
    af = interpol_ker(a_t, fi_vals)

    # Define the interpolation function for the power spectrum
    spline_me = lambda logk: jnp.interp(logk, log_km, log_pkm)

    # Finger-of-God damping factor
    sig_fog = cosm_par[8]

    # Integration limits
    xmin = [-1.0, 0.0]
    xmax = [1.0, 2 * jnp.pi]

    # Compute the bispectrum monopole
    pk_in = 10**spline_me(jnp.log10(tr))
    bk0 = vec_integrate_bkeff_r(tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, 0, xmin, xmax, num_points)

    # Compute the first multipole
    pk_in = 10**spline_me(jnp.log10(tr2))
    bk200 = vec_integrate_bkeff_r(tr2, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, 1, xmin, xmax, num_points)

    # Compute the second multipole
    pk_in = 10**spline_me(jnp.log10(tr3))
    bk020 = vec_integrate_bkeff_r(tr3, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, 2, xmin, xmax, num_points)

    # Compute the third multipole
    pk_in = 10**spline_me(jnp.log10(tr4))
    bk002 = vec_integrate_bkeff_r(tr4, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, 3, xmin, xmax, num_points)

    return bk0, bk200, bk020, bk002


@partial(jax.jit, static_argnames=('num_points',))
def bk_multip(tr, tr2, tr3, tr4, kp, pk, cosm_par, redshift, num_points=10, fi_vals=F_VALS_FULL):
    """
    Compute the bispectrum multipoles for a given set of triangles, power spectrum, and cosmological parameters.
    
    High-level wrapper that computes all four multipoles and returns them in a dictionary.

    Parameters
    ----------
    tr : jnp.ndarray
        Array of triangle side lengths (ka, kb, kc) for the monopole calculation.
    tr2 : jnp.ndarray
        Array of triangle side lengths (ka, kb, kc) for the first multipole calculation.
    tr3 : jnp.ndarray
        Array of triangle side lengths (ka, kb, kc) for the second multipole calculation.
    tr4 : jnp.ndarray
        Array of triangle side lengths (ka, kb, kc) for the third multipole calculation.
    kp : jnp.ndarray
        Array of wavevector magnitudes for the power spectrum.
    pk : jnp.ndarray
        Array of power spectrum values corresponding to kp.
    cosm_par : jnp.ndarray
        Cosmological parameters array.
    redshift : float
        Redshift at which to compute the bispectrum.
    num_points : int, optional
        Number of points for the integration grid. Default is 50.
    fi_vals : jnp.ndarray, optional
        Array of kernel values for interpolation. Default is F_VALS_FULL.

    Returns
    -------
    dict
        A dict containing the bispectrum monopole (000), first multipole (200),
        second multipole (020), and third multipole (002).
    
    Notes
    -----
    - Keys: '000' (monopole), '200' (quadrupole for ka), '020' (quadrupole for kb), '002' (quadrupole for kc)
    - Calls `ext_bk_mp` internally
    - JIT-compiled for performance
    
    Examples
    --------
    >>> tr = jnp.array([[0.1, 0.1, 0.1]])
    >>> tr2 = tr3 = tr4 = tr
    >>> kp = jnp.linspace(0.01, 0.3, 100)
    >>> pk = linear_power_spectrum(kp, cosm_par)
    >>> cosm_par = jnp.array([...])
    >>> redshift = 0.5
    >>> result = bk_multip(tr, tr2, tr3, tr4, kp, pk, cosm_par, redshift)
    >>> bk0 = result['000']
    >>> bk200 = result['200']
    """
    # Compute the bispectrum multipoles
    bk0, bk200, bk020, bk002 = ext_bk_mp(
        tr, tr2, tr3, tr4, jnp.log10(kp), jnp.log10(pk), cosm_par, redshift, num_points=num_points, fi_vals=fi_vals
    )

    return {'000':bk0, '200':bk200, '020':bk020, '002':bk002}


######### Sugiyama estimator #################
# Basically taken from FolpsD https://github.com/alejandroaviles/folpsD/blob/main/FOLPSD.py

def geo_fac_sugiyama_simple(k1, k2, x12, af, hh=1.0, geo_expansion = 'poly'):
    """
    Compute GEO-FPT factor for given triangle geometry (Sugiyama version).
    
    Simplified version for the Sugiyama estimator that only needs k1, k2,
    and the cosine between them.

    Parameters
    ----------
    k1, k2 : float or jnp.ndarray
        First two wavevector magnitudes (real space)
    x12 : float or jnp.ndarray
        Cosine of angle between k1 and k2
    af : jnp.ndarray
        GEO-FPT coefficients
    hh : float, optional
        Hubble parameter normalization

    Returns
    -------
    eff_fact : float or jnp.ndarray
        GEO-FPT correction factor
    
    Notes
    -----
    - Computes k3 internally using law of cosines
    - Calls the standard `geo_fac` function
    """
    # Compute third side
    k3 = jnp.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * x12)
    
    # Call original geo_fac
    if geo_expansion == 'pade':
        return geo_fac_pade(k1, k2, k3, af, hh)
    elif geo_expansion == 'poly':
        return geo_fac(k1, k2, k3, af, hh)
    else:
        raise ValueError(f"GeoFPT expansion {geo_expansion} not recognized.")


# Vectorized version
geo_fac_sugiyama_simple_vec = jax.vmap(geo_fac_sugiyama_simple, 
                                       in_axes=(0, 0, 0, None, None, None))


def bkeff_sugiyama(k1, k2, x12, mu1, phi, cosm_par, pk_interp, 
                   log_km, log_pkm, af, mp=None, hh=1.0, geo_expansion = 'poly'):
    """
    JAX-compatible Sugiyama bispectrum integrand.
    
    Computes the bispectrum integrand for the Sugiyama estimator, including
    all physical effects (RSD, AP, FoG, GEO-FPT, shot noise).

    Parameters
    ----------
    k1, k2 : float
        First two wavevector magnitudes (real space)
    x12 : float
        Cosine of angle between k1 and k2
    mu1 : float
        Cosine of angle between k1 and line of sight
    phi : float
        Azimuthal angle
    cosm_par : jnp.ndarray
        Cosmological parameters array
    pk_interp : callable
        Power spectrum interpolation function
    log_km : jnp.ndarray
        Logarithm of wavevector magnitudes for interpolation
    log_pkm : jnp.ndarray
        Logarithm of power spectrum values for interpolation
    af : jnp.ndarray
        GEO-FPT coefficients
    mp : int, optional
        Multipole index (not used in Sugiyama, for compatibility)
    hh : float, optional
        Hubble parameter normalization

    Returns
    -------
    float
        Value of the bispectrum integrand
    
    Notes
    -----
    - Based on FOLPSD implementation (https://github.com/alejandroaviles/folpsD)
    - Includes shot noise terms with amplitudes A_P and A_B
    - All control flow replaced with JAX operations for JIT compatibility
    """
    # Extract parameters
    alpa, alpe = cosm_par[1], cosm_par[2]
    b1, ff = cosm_par[3], cosm_par[0]
    sig_fog = cosm_par[8]  # σ_B
    A_P = cosm_par[6]      # Power spectrum shot noise
    A_B = cosm_par[7]      # Bispectrum shot noise
    
    Fsq = 1.0 / (alpa / alpe)**2
    
    # 1. Compute triangle geometry and AP transforms
    k3_m = jnp.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * x12)
    #mu2_m = mu1 * x12 - jnp.sqrt((1.0 - mu1**2) * (1.0 - x12**2)) * jnp.cos(phi)
    mu2_m = mu1 * x12 + jnp.sqrt((1.0 - mu1**2) * (1.0 - x12**2)) * jnp.cos(phi)
    muc_m = (-k1 * mu1 - k2 * mu2_m) / k3_m
    
    # AP transforms to redshift space
    k1_rs = k1 * jnp.sqrt(1.0 + mu1**2 * (Fsq - 1.0)) / alpe
    k2_rs = k2 * jnp.sqrt(1.0 + mu2_m**2 * (Fsq - 1.0)) / alpe
    k3_rs = k3_m * jnp.sqrt(1.0 + muc_m**2 * (Fsq - 1.0)) / alpe
    
    # Transform mu's to redshift space
    mu1_rs = mu1 * alpe / (alpa * jnp.sqrt(1.0 + mu1**2 * (Fsq - 1.0)))
    mu2_rs = mu2_m * alpe / (alpa * jnp.sqrt(1.0 + mu2_m**2 * (Fsq - 1.0)))
    muc_rs = muc_m * alpe / (alpa * jnp.sqrt(1.0 + muc_m**2 * (Fsq - 1.0)))
    
    # 2. Compute GEO-FPT factor
    eff_fact = geo_fac_sugiyama_simple(k1, k2, x12, af, hh, geo_expansion)
    
    # 3. Apply FoG damping (using σ_B = cosm_par[9])
    k_par_sq_sum = (k1_rs * mu1_rs)**2 + (k2_rs * mu2_rs)**2 + (k3_rs * muc_rs)**2
    D_fog = 1. / (1 + 0.5 * k_par_sq_sum**2 * (sig_fog / hh)**4)**2
    
    # 4. Compute Z1 kernels
    z1_1 = z1_ker(mu1_rs, cosm_par)
    z1_2 = z1_ker(mu2_rs, cosm_par)
    z1_3 = z1_ker(muc_rs, cosm_par)

    
    
    # 5. Compute F2 and G2 kernels
    f2k_12 = f2_ker(k1_rs, k2_rs, k3_rs)
    f2k_23 = f2_ker(k2_rs, k3_rs, k1_rs)
    f2k_13 = f2_ker(k1_rs, k3_rs, k2_rs)
    
    g2k_12 = g2_ker(k1_rs, k2_rs, k3_rs)
    g2k_23 = g2_ker(k2_rs, k3_rs, k1_rs)
    g2k_13 = g2_ker(k1_rs, k3_rs, k2_rs)

    
    # 6. Compute Z2 kernels with GEO-FPT
    z2_12 = z2_ker(k1_rs, k2_rs, k3_rs, f2k_12, g2k_12, mu1_rs, mu2_rs, cosm_par) * eff_fact
    z2_23 = z2_ker(k2_rs, k3_rs, k1_rs, f2k_23, g2k_23, mu2_rs, muc_rs, cosm_par) * eff_fact
    z2_13 = z2_ker(k1_rs, k3_rs, k2_rs, f2k_13, g2k_13, mu1_rs, muc_rs, cosm_par) * eff_fact
    #jax.debug.print("GEO fac = {}", eff_fact)
    # 7. Get power spectra
    spline_me = lambda logk: jnp.interp(logk, log_km, log_pkm)
    pk1 = 10**spline_me(jnp.log10(k1_rs))
    pk2 = 10**spline_me(jnp.log10(k2_rs))
    pk3 = 10**spline_me(jnp.log10(k3_rs))
    
    # 8. Form tree-level bispectrum terms
    B12 = 2 * z1_1 * z1_2 * z2_12 * pk1 * pk2
    B23 = 2 * z1_2 * z1_3 * z2_23 * pk2 * pk3
    B31 = 2 * z1_1 * z1_3 * z2_13 * pk1 * pk3

    
    
    tree_level = B12 + B23 + B31

    A_B -= 1 
    
    # 9. Compute shot noise terms (ALWAYS compute, even if A_P=0, A_B=0)
    # This avoids conditionals and is JAX-friendly
    shot1 = (b1 * A_B + 2.0 * A_P * ff * mu1_rs**2) * z1_1 * pk1
    shot2 = (b1 * A_B + 2.0 * A_P * ff * mu2_rs**2) * z1_2 * pk2
    shot3 = (b1 * A_B + 2.0 * A_P * ff * muc_rs**2) * z1_3 * pk3
    
    shot_noise = shot1 + shot2 + shot3 + A_P**2
    
    # 10. Combine: tree-level gets FoG damping, shot noise does not
    # The expression handles zero shot noise automatically when A_P=0, A_B=0
    #result = (D_fog * tree_level + shot_noise) / (2 * jnp.pi * alpa**2 * alpe**4)
    result = (D_fog * tree_level + shot_noise) / (alpa**2 * alpe**4)
    
    # 11. Apply validity mask (triangle inequality)
    #valid = (k2_rs + k1_rs - k3_rs >= hh * 1.1 * 2 * jnp.pi / 1000.0) & \
    #        (k1_rs + k3_rs - k2_rs >= hh * 1.1 * 2 * jnp.pi / 1000.0) & \
    #        (k2_rs + k3_rs - k1_rs >= hh * 1.1 * 2 * jnp.pi / 1000.0)
    
    #return jnp.where(valid, result, 0.0)
    return result
# Vectorized version
bkeff_sugiyama_vec = jax.vmap(bkeff_sugiyama, 
                             in_axes=(None, None, 0, 0, 0, None, None, None, None, None, None, None, None))


def compute_basis_grid(x_pts, mu_pts, phi_pts):
    """
    Precompute ALL basis functions on the angular grid for Sugiyama estimator.
    
    The Sugiyama estimator expands the bispectrum in a basis of 6 functions.
    This function precomputes all basis functions on a 3D grid for efficient
    integration.

    Parameters
    ----------
    x_pts : jnp.ndarray
        Grid points for x = cosθ₁₂ (angle between k1 and k2)
    mu_pts : jnp.ndarray
        Grid points for μ₁ (cosine of angle between k1 and LOS)
    phi_pts : jnp.ndarray
        Grid points for φ (azimuthal angle)

    Returns
    -------
    jnp.ndarray
        Array of shape (6, N_x, N_mu, N_phi) containing the 6 basis functions
        evaluated at all grid points.
    
    Notes
    -----
    - Basis functions: 1, P₁(x), P₂(x), P₂(μ₁), P₂(μ₂), P₁(x)P₁(μ₁)
    - Normalized according to Sugiyama et al. (2019)
    - Efficient when all 6 coefficients are needed simultaneously
    """
    # Create meshgrid
    X, M, P = jnp.meshgrid(x_pts, mu_pts, phi_pts, indexing='ij')
    
    # Precompute common terms
    sqrt1_mu2 = jnp.sqrt(1.0 - M**2)
    sqrt1_x2 = jnp.sqrt(1.0 - X**2)
    cosphi = jnp.cos(P)
    cos2phi = jnp.cos(2 * P)
    
    # Compute each basis function
    b000 = 1.0 / (8 * jnp.pi) * jnp.ones_like(X)
    
    b110 = (-3 * jnp.sqrt(3) * X) / (8 * jnp.pi)
    
    b220 = (5 * jnp.sqrt(5) / (16 * jnp.pi)) * (-1.0 + 3.0 * X**2)
    
    b202 = (5 * jnp.sqrt(5) / (16 * jnp.pi)) * (-1.0 + 3.0 * M**2)
    
    b022 = (5 * jnp.sqrt(5) / (32 * jnp.pi)) * (
        (-1.0 + 3.0 * M**2) * (-1.0 + 3.0 * X**2) +
        12.0 * M * sqrt1_mu2 * X * sqrt1_x2 * cosphi +
        3.0 * (1.0 - M**2) * (1.0 - X**2) * cos2phi
    )
    
    # Corrected b112 to match angdep_integrands:
    # second term uses 3 * sqrt(3) instead of 6
    b112 = (3 * jnp.sqrt(2.5) / (8 * jnp.pi)) * (
        jnp.sqrt(3) * (-1.0 + 3.0 * M**2) * X +
        3.0 * jnp.sqrt(3) * M * sqrt1_mu2 * sqrt1_x2 * cosphi
    )

    b222 = (
        25 * jnp.sqrt(70) * (
            1 - 3 * M**2 * X**2
            - 3 * M * X * sqrt1_mu2 * sqrt1_x2 * cosphi
            - 1.5 * (1 - M**2) * (1 - X**2) * (1 - cos2phi)
        )
    ) / (112 * jnp.pi)
    
    return jnp.stack([b000, b110, b220, b202, b022, b112, b222], axis=0)



def compute_sugiyama_multipoles(k1k2_pairs, log_km, log_pkm, cosm_par, redshift,
                                      fi_vals=F_VALS_FULL, num_points=50, geo_expansion = 'poly'):
    """
    Compute Sugiyama coefficients using scalar functions and proper vmap.
    
    Main function for computing the 6 Sugiyama multipole coefficients for
    a set of (k1, k2) pairs.

    Parameters
    ----------
    k1k2_pairs : jnp.ndarray
        Array of shape (N_pairs, 2) containing (k1, k2) values for each triangle.
    log_km : jnp.ndarray
        Logarithm of wavevector magnitudes for power spectrum interpolation.
    log_pkm : jnp.ndarray
        Logarithm of power spectrum values for interpolation.
    cosm_par : jnp.ndarray
        Cosmological parameters array.
    redshift : float
        Redshift at which to compute the bispectrum.
    fi_vals : jnp.ndarray, optional
        Array of kernel values for interpolation. Default is F_VALS_FULL.
    num_points : int, optional
        Number of Gauss-Legendre points per angular dimension. Default is 50.

    Returns
    -------
    jnp.ndarray
        Array of shape (6, N_pairs) containing the 6 Sugiyama coefficients
        for each triangle pair.
    
    Notes
    -----
    - Implements the Sugiyama estimator from arXiv:1903.09172
    - Uses 3D Gauss-Legendre quadrature over (x, μ, φ)
    - Returns coefficients: [B000, B110, B220, B202, B022, B112, B222]
    - H-factors from FOLPSD are applied for correct normalization
    """
    # Setup
    a_t = 1.0 / (1.0 + redshift)
    af = interpol_ker(a_t, fi_vals)
    
    # Create quadrature grids
    x_pts, x_wts = jax_leggauss(jnp.arange(num_points))
    x_pts = 0.5 * (x_pts + 1) * 2 - 1  # Transform to [-1, 1]
    
    mu_pts, mu_wts = jax_leggauss(jnp.arange(num_points))
    mu_pts = 0.5 * (mu_pts + 1) * 2 - 1  # Transform to [-1, 1]
    
    phi_pts, phi_wts = jax_leggauss(jnp.arange(num_points))
    phi_pts = 0.5 * (phi_pts + 1) * (2 * jnp.pi)  # Transform to [0, 2π]
    phi_wts = phi_wts * jnp.pi  # Adjust weights for [0, 2π] interval
    
    # Create meshgrids
    X, M, P = jnp.meshgrid(x_pts, mu_pts, phi_pts, indexing='ij')
    Wx, Wm, Wp = jnp.meshgrid(x_wts, mu_wts, phi_wts, indexing='ij')
    W_total = Wx * Wm * Wp
    
    # Precompute basis grid
    basis_grid = compute_basis_grid(x_pts, mu_pts, phi_pts)  # shape: (6, Nx, Nmu, Nphi)
    
    # Power spectrum interpolation function
    def pk_interp(k):
        return 10**jnp.interp(jnp.log10(k), log_km, log_pkm)
    
    # Process each triangle pair
    def process_pair(k1k2):
        k1, k2 = k1k2
        
        # Flatten the grid for vectorized computation
        x_flat = X.ravel()
        mu_flat = M.ravel()
        phi_flat = P.ravel()
        w_flat = W_total.ravel()
        
        # Vectorized computation using vmap
        # Note: we vmap over the flattened grid, not over k1, k2
        B_flat = bkeff_sugiyama_vec(
            k1, k2, x_flat, mu_flat, phi_flat,
            cosm_par, pk_interp, log_km, log_pkm, af, None, 1.0, geo_expansion,
        )
        
        # Reshape back to 3D
        B_3d = B_flat.reshape(X.shape)
        
        # Compute coefficients by integrating with basis functions
        coeffs = jnp.zeros(7)
        for i in range(7):
            # Multiply by basis function and integrate
            integrand = B_3d * basis_grid[i]
            coeffs = coeffs.at[i].set(jnp.sum(integrand * W_total))
        
        return coeffs
    
    # Vectorize over triangle pairs
    all_coeffs = jax.vmap(process_pair)(k1k2_pairs)
    
    # Apply normalization factors (H factors from FOLPSD)
    H_factors = jnp.array([1.0, -1.0/jnp.sqrt(3.0), 1.0/jnp.sqrt(5.0),
                           1.0/jnp.sqrt(5.0), 1.0/jnp.sqrt(5.0), 
                           jnp.sqrt(2.0/15.0), -2 / jnp.sqrt(70)])
    
    normalized_coeffs = all_coeffs * H_factors
    
    return normalized_coeffs.T

def bk_sugiyama_multip(k1, k2, kp, pk, cosm_par, redshift, num_points=10, fi_vals=F_VALS_FULL,
                       geo_expansion = 'poly'):
    """
    Compute Sugiyama multipole coefficients for given (k1, k2) pairs.
    
    High-level wrapper that returns Sugiyama coefficients as a dictionary.

    Parameters
    ----------
    k1 : jnp.ndarray
        Array of first wavevector magnitudes.
    k2 : jnp.ndarray
        Array of second wavevector magnitudes.
    kp : jnp.ndarray
        Array of wavevector magnitudes for the power spectrum.
    pk : jnp.ndarray
        Array of power spectrum values corresponding to kp.
    cosm_par : jnp.ndarray
        Cosmological parameters array.
    redshift : float
        Redshift at which to compute the bispectrum.
    num_points : int, optional
        Number of Gauss-Legendre points per angular dimension. Default is 50.
    fi_vals : jnp.ndarray, optional
        Array of kernel values for interpolation. Default is F_VALS_FULL.

    Returns
    -------
    dict
        Dictionary with keys: '000', '110', '220', '202', '022', '112', '222'
        containing the corresponding Sugiyama coefficients.
    
    Examples
    --------
    >>> k1 = jnp.array([0.1, 0.2, 0.3])
    >>> k2 = jnp.array([0.1, 0.2, 0.3])
    >>> kp = jnp.linspace(0.01, 0.5, 100)
    >>> pk = linear_power_spectrum(kp, cosm_par)
    >>> result = bk_sugiyama_multip(k1, k2, kp, pk, cosm_par, redshift=0.5)
    >>> b000 = result['000']
    >>> b110 = result['110']
    """
    k1k2_pairs=jnp.vstack([k1,k2]).T
    bk = compute_sugiyama_multipoles(k1k2_pairs, jnp.log10(kp), jnp.log10(pk), cosm_par, 
                                          redshift, fi_vals=fi_vals, 
                                          num_points=num_points,
                                          geo_expansion = geo_expansion)
    labels = ['000', '110', '220', '202', '022', '112', '222']
    return dict(zip(labels, bk))



@jax.jit
def pt_kernel(k, q, wq):
    """Vectorized kernel calculation using Gauss-Legendre quadrature."""
    jq = q**2 * wq / (4. * jnp.pi**2)
    x = q / k[:, None]
    
    def kernel_ff(x):
        # 1. Protect against x = 0 and x = -1 (division by zero)
        # We replace 0 with 1.0 for the safe evaluation of the internal branches.
        x_safe = jnp.where(jnp.abs(x) < 1e-15, 1.0, x)
        denom_safe = jnp.where(jnp.abs(x_safe + 1.0) < 1e-15, 1.0, x_safe + 1.0)
        
        # 2. Protect the argument of the logarithm from being exactly 0
        # When x=1, (x-1)/(x+1) = 0, and log(0) = -Inf. 
        # Clamping to 1e-15 prevents the 0 * -Inf = NaN issue in JAX.
        log_arg = jnp.abs((x_safe - 1.0) / denom_safe)
        log_arg_safe = jnp.maximum(log_arg, 1e-15)
        
        # 3. Compute the main branch safely
        term1 = 6.0 / x_safe**2 - 79.0 + 50.0 * x_safe**2 - 21.0 * x_safe**4
        term2 = 0.75 * (1.0 / x_safe - x_safe)**3 * (2.0 + 7.0 * x_safe**2) * 2.0 * jnp.log(log_arg_safe)
        
        toret = (term1 + term2) / 504.0
        
        # 4. Apply asymptotic expansion for x > 10 safely
        mask_large = x > 10.0
        toret_large = -61.0/630.0 + 2.0/105.0/x_safe**2 - 10.0/1323.0/x_safe**4
        toret = jnp.where(mask_large, toret_large, toret)
        
        # 5. Apply Taylor expansion near x = 1 safely
        dx = x - 1.0
        mask_small = jnp.abs(dx) < 0.01
        toret_small = -11.0/126.0 + dx/126.0 - 29.0/252.0*dx**2
        toret = jnp.where(mask_small, toret_small, toret)
        
        # 6. Final division by x^2 safely
        # Prevents division by zero if x=0. 
        x_sq_safe = jnp.maximum(x**2, 1e-30)
        
        return toret / x_sq_safe
    return 2 * jq * kernel_ff(x)

@partial(jax.jit, static_argnames = ('n_gauss'))
def pt_pk_1loop(k, q, wq, pk_q, kernel13_d, n_gauss=20):
    """
    Compute 1-loop power spectrum using Gauss-Legendre quadrature.
    
    Parameters
    ----------
    k : jnp.ndarray
        Output wavenumbers
    q : jnp.ndarray
        Integration wavenumbers
    pk_q : jnp.ndarray
        Linear power spectrum at q
    n_gauss : int
        Number of Gauss-Legendre points for angular integration
        
    Returns
    -------
    jnp.ndarray
        1-loop power spectrum
    """
   

    k11 = k
    k = k[:, None]
    jq = q**2 * wq / (4. * np.pi**2)
    # Get q quadrature points and weights
    

    mus, wmus = jax_leggauss(jnp.arange(n_gauss))

    # Compute P22
    #jax.debug.print("sizes {} {} {}", q.shape, pk_q.shape, k11.shape)
    pk_k = jnp.interp(k11, q, pk_q)

    def get_pk22_dd(mu, wmu):
        kdq = k * q * mu  # k \cdot q
        kq2 = k**2 - 2. * kdq + q**2  # |k - q|^2
        qdkq = kdq - q**2   # k \cdot (k - q)
        F2_d = 5. / 7. + 1. / 2. * qdkq * (1. / q**2 + 1. / kq2) + 2. / 7. * qdkq**2 / (q**2 * kq2)
        pk_kq = jnp.interp(kq2**0.5, q, pk_q, left=0., right=0.)
        jq_pk_q_pk_kq = jq * pk_q * pk_kq
        return 2 * wmu * jnp.sum(F2_d**2 * jq_pk_q_pk_kq, axis=-1)

    pk22_dd = jnp.sum(jax.vmap(get_pk22_dd)(mus, wmus), axis=0)
    pk11 = pk_k
    pk13_dd = 2. * jnp.sum(kernel13_d * pk_q, axis=-1) * pk_k
    return pk11, pk22_dd,  pk13_dd

@jax.jit
def weights_trapz(x):
    """Return weights for trapezoidal integration."""
    if x.size == 0:
        return np.array(1.)
    if x.size == 1:
        return np.ones(x.size)
    if x.size == 2:
        return np.ones(x.size) / 2. * (x[1] - x[0])
    return jnp.insert(x[2:] - x[:-2], jnp.array([0, len(x) - 1]), jnp.array([x[1] - x[0], x[-1] - x[-2]])) / 2.

