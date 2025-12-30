import jax
import jax.numpy as jnp
#from quadax import quadgk
import numpy as np
from functools import partial
from .constants import *



@jax.jit
def jax_leggauss(x):
    """JAX-compatible Gauss-Legendre quadrature weights and nodes."""
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
    """Compute Legendre polynomial and its derivative using recurrence."""
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
    """
    return (kc * kc - ka * ka - kb * kb) / (2 * ka * kb)

def f2_ker(ka, kb, kc):
    """
    Compute the second-order SPT kernel F2.

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
    """
    cab = cosab(ka, kb, kc)
    return 5. / 7. + 0.5 * cab * (ka / kb + kb / ka) + 2. / 7. * cab**2

def interpol_ker(a, fi_a_val, a_val=jnp.array([1. / (1 + 2.), 1. / (1 + 1.), 1. / (1 + 0.5)])):
    """
    Interpolate the kernel values at a given scale factor.

    Parameters
    ----------
    a : float
        Scale factor at which to interpolate.
    fi_a_val : jnp.ndarray
        Array of kernel values at predefined scale factors.
    a_val : jnp.ndarray, optional
        Predefined scale factors for interpolation. Default is [1/3, 1/2, 2/3].

    Returns
    -------
    jnp.ndarray
        Interpolated kernel values at the given scale factor.
    """
    return jnp.interp(a, a_val, fi_a_val)

interpol_ker = jax.vmap(interpol_ker, in_axes = (None, 0))

def g2_ker(ka, kb, kc):
    """
    Compute the second-order SPT kernel G2.

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
    """
    cab = cosab(ka, kb, kc)
    return 3. / 7. + 0.5 * cab * (ka / kb + kb / ka) + 4. / 7. * cab**2

def z1_ker(mu, cosm_par):
    """
    Compute the first-order redshift-space SPT kernel Z1.

    Parameters
    ----------
    mu : float or jnp.ndarray
        Cosine of the angle between the wavevector and the line of sight.
    cosm_par : jnp.ndarray
        Cosmological parameters array, where cosm_par[4] is the linear bias (b1)
        and cosm_par[1] is the growth rate (f).

    Returns
    -------
    float or jnp.ndarray
        Value of the Z1 kernel.
    """
    b1, ff = cosm_par[4], cosm_par[1]
    return b1 + ff * mu**2


def z2_ker(ka, kb, kc, fkern, gkern, mua, mub, cosm_par):
    """
    Compute the second-order redshift-space SPT kernel Z2.

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
        Cosmological parameters array, where cosm_par[4] is the linear bias (b1),
        cosm_par[1] is the growth rate (f), and cosm_par[5] is the second-order bias (b2).

    Returns
    -------
    float or jnp.ndarray
        Value of the Z2 kernel.
    """
    cab = cosab(ka, kb, kc)
    
    b1, ff, b2 = cosm_par[4], cosm_par[1], cosm_par[5]

    ksq = jnp.sqrt(ka**2 + kb**2 + 2 * ka * kb * cab)  # modulus of vector sum k1 + k2
    mu12 = (ka * mua + kb * mub) / ksq
    
    # TODO: relax condition on bs and make it a parameter.
    bs = -4.0 / 7.0 * (b1 - 1.0)
    s2 = cab**2 - 1.0 / 3.0  # S_2 kernel

    b1_terms = b1 * (fkern + 0.5 * ff * mu12 * ksq * (mua / ka + mub / kb))
    g_term = ff * mu12**2 * gkern
    fsq_term = 0.5 * ff**2 * mu12 * ksq * mua * mub * (mub / ka + mua / kb)
    b_terms = 0.5 * (b2 + bs * s2)

    return b1_terms + g_term + fsq_term + b_terms



def geo_fac(ka, kb, kc, af, hh):
    """
    Compute the GEO-FPT factor multiplying Z2_SPT to obtain Z2_GEO.

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

    Returns
    -------
    float or jnp.ndarray
        Value of the GEO-FPT factor.
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


# Vectorized versions of the functions
g2_ker_vec = jax.vmap(g2_ker, (0, 0, 0))
z1_ker_vec = jax.vmap(z1_ker, (0, None))
z2_ker_vec = jax.vmap(z2_ker, (0, 0, 0, 0, 0, 0, 0, None))
geo_fac_vec = jax.vmap(geo_fac, (0, 0, 0, None, None))


@jax.jit
def bkeff_r_scalar(mua_m, phi, tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp):
    """
    Compute the integrand for the effective bispectrum in redshift space.

    Parameters
    ----------
    mua_m : float or jnp.ndarray
        Cosine of the angle between ka and the line of sight in real space.
    phi : float or jnp.ndarray
        Azimuthal angle.
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

    Returns
    -------
    float or jnp.ndarray
        Value of the integrand.
    """
    ka_m, kb_m, kc_m = tr
    pka, pkb, pkc = pk_in
    spline_me = lambda logk: jnp.interp(logk, log_km, log_pkm)
    
    alpa, alpe = cosm_par[2], cosm_par[3]
    Fsq = 1.0 / (alpa / alpe)**2

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

    return valid * result #jnp.where(valid, result, 0.0)


bkeff_r_vmap = jax.vmap(bkeff_r_scalar, in_axes=(0, 0, None, None, None, None, None, None, None, None))

def integrate_bkeff_r(tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp, xmin, xmax, num_points):
    """
    Perform 2D integration of the effective bispectrum integrand using Gauss-Legendre quadrature.

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
    """
    nx , ny = num_points, num_points
    def integrand(mua, phi):
        return bkeff_r_scalar(mua, phi, tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp)

    return integrate_2d_gauss(integrand, xmin[0], xmax[0], xmin[1], xmax[1], nx=nx, ny=ny)

def _integrate_bkeff_r(tr, cosm_par, pk_in, sig_fog, log_km, log_pkm, af, mp, xmin, xmax, num_points):
    """
    Perform 2D integration of the effective bispectrum integrand.

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
    """
    a_t = 1.0 / (1.0 + redshift)

    # Interpolate the kernel values
    af = interpol_ker(a_t, fi_vals)

    # Define the interpolation function for the power spectrum
    spline_me = lambda logk: jnp.interp(logk, log_km, log_pkm)

    # Finger-of-God damping factor
    sig_fog = cosm_par[9]

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
    """
    # Compute the bispectrum multipoles
    bk0, bk200, bk020, bk002 = ext_bk_mp(
        tr, tr2, tr3, tr4, jnp.log10(kp), jnp.log10(pk), cosm_par, redshift, num_points=num_points, fi_vals=fi_vals
    )

    return {'000':bk0, '200':bk200, '020':bk020, '002':bk002}


######### Sugiyama estimator #################
# Basically taken from FolpsD https://github.com/alejandroaviles/folpsD/blob/main/FOLPSD.py

def geo_fac_sugiyama_simple(k1, k2, x12, af, hh=1.0):
    """
    Compute GEO-FPT factor for given triangle geometry.
    
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
    """
    # Compute third side
    k3 = jnp.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * x12)
    
    # Call original geo_fac
    return geo_fac(k1, k2, k3, af, hh)


# Vectorized version
geo_fac_sugiyama_simple_vec = jax.vmap(geo_fac_sugiyama_simple, 
                                       in_axes=(0, 0, 0, None, None))


def bkeff_sugiyama(k1, k2, x12, mu1, phi, cosm_par, pk_interp, 
                   log_km, log_pkm, af, mp=None, hh=1.0):
    """
    JAX-compatible Sugiyama bispectrum integrand.
    All control flow replaced with JAX operations.
    """
    # Extract parameters
    alpa, alpe = cosm_par[2], cosm_par[3]
    b1, ff = cosm_par[4], cosm_par[1]
    sig_fog = cosm_par[9]  # σ_B
    A_P = cosm_par[6]      # Power spectrum shot noise
    A_B = cosm_par[8]      # Bispectrum shot noise
    
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
    eff_fact = geo_fac_sugiyama_simple(k1, k2, x12, af, hh)
    
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
                             in_axes=(None, None, 0, 0, 0, None, None, None, None, None, None, None))


def compute_basis_grid(x_pts, mu_pts, phi_pts):
    """
    Precompute ALL basis functions on the angular grid.
    
    Returns array of shape (6, N_x, N_mu, N_phi)
    Efficient when we need all 6 coefficients anyway.
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
    
    b112 = (3 * jnp.sqrt(2.5) / (8 * jnp.pi)) * (
        jnp.sqrt(3) * (-1.0 + 3.0 * M**2) * X +
        6.0 * M * sqrt1_mu2 * sqrt1_x2 * cosphi
    )
    
    return jnp.stack([b000, b110, b220, b202, b022, b112], axis=0)

def _compute_sugiyama_multipoles(k1k2_pairs, log_km, log_pkm, cosm_par, redshift,
                                fi_vals=F_VALS_FULL, num_points=50):
    """
    Compute all 6 Sugiyama coefficients with integrated shot noise.
    
    Note: Shot noise is now integrated directly, not applied post-hoc.
    """
    # Setup
    a_t = 1.0 / (1.0 + redshift)
    af = interpol_ker(a_t, fi_vals)
    
    # Angular grid setup (same as before)
    x_pts, x_wts = jax_leggauss(jnp.arange(num_points))
    x_pts = 0.5 * (x_pts + 1) * 2 - 1
    
    mu_pts, mu_wts = jax_leggauss(jnp.arange(num_points))
    mu_pts = 0.5 * (mu_pts + 1) * 2 - 1
    
    phi_pts, phi_wts = jax_leggauss(jnp.arange(num_points))
    phi_pts = 0.5 * (phi_pts + 1) * (2 * jnp.pi)
    phi_wts = phi_wts * jnp.pi
    
    # Create meshgrids
    X, M, P = jnp.meshgrid(x_pts, mu_pts, phi_pts, indexing='ij')
    Wx, Wm, Wp = jnp.meshgrid(x_wts, mu_wts, phi_wts, indexing='ij')
    W_total = Wx * Wm * Wp
    
    # Precompute basis grid
    basis_grid = compute_basis_grid(x_pts, mu_pts, phi_pts)
    
    # Vectorized computation
    def compute_for_pair(k1k2):
        k1, k2 = k1k2
        k1_exp = jnp.full_like(X, k1)
        k2_exp = jnp.full_like(X, k2)
        
        # Compute bispectrum WITH integrated shot noise
        B = bkeff_sugiyama(k1_exp, k2_exp, X, M, P, cosm_par,
                          lambda k: 10**jnp.interp(jnp.log10(k), log_km, log_pkm),
                          log_km, log_pkm, af, mp=None)
        
        # Integrate with basis functions
        coeffs = jnp.zeros(6)
        for i in range(6):
            #coeffs = coeffs.at[i].set(jnp.sum(B * basis_grid[i] * W_total))
            # Step 1: Integrate over φ (multiply by 2 like in FOLPSD)
            # phi_wts shape: (Nφ,), broadcast to (1, 1, Nφ)
            phi_integrand = B * basis_grid[i] * phi_wts[None, None, :]
            int_phi = 2.0 * jnp.sum(phi_integrand, axis=2)  # Shape: (Nx, Nμ)
            
            # Step 2: Integrate over μ
            # mu_wts shape: (Nμ,), broadcast to (1, Nμ)
            mu_integrand = int_phi * mu_wts[None, :]
            int_mu = jnp.sum(mu_integrand, axis=1)  # Shape: (Nx,)
            
            # Step 3: Integrate over x
            # x_wts shape: (Nx,)
            int_all = jnp.sum(int_mu * x_wts)
            
            coeffs = coeffs.at[i].set(int_all)
        
        return coeffs
    
    # Vectorize over all pairs
    all_coeffs = jax.vmap(compute_for_pair)(k1k2_pairs)
    
    # Apply normalization factors
    H_factors = jnp.array([1.0, -1.0/jnp.sqrt(3.0), 1.0/jnp.sqrt(5.0),
                           1.0/jnp.sqrt(5.0), 1.0/jnp.sqrt(5.0), 
                           jnp.sqrt(2.0/15.0)])
    
    normalized_coeffs = all_coeffs * H_factors
    
    return normalized_coeffs.T

def compute_sugiyama_multipoles(k1k2_pairs, log_km, log_pkm, cosm_par, redshift,
                                      fi_vals=F_VALS_FULL, num_points=50):
    """
    Compute Sugiyama coefficients using scalar functions and proper vmap.
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
            cosm_par, pk_interp, log_km, log_pkm, af, None, 1.0
        )
        
        # Reshape back to 3D
        B_3d = B_flat.reshape(X.shape)
        
        # Compute coefficients by integrating with basis functions
        coeffs = jnp.zeros(6)
        for i in range(6):
            # Multiply by basis function and integrate
            integrand = B_3d * basis_grid[i]
            coeffs = coeffs.at[i].set(jnp.sum(integrand * W_total))
        
        return coeffs
    
    # Vectorize over triangle pairs
    all_coeffs = jax.vmap(process_pair)(k1k2_pairs)
    
    # Apply normalization factors (H factors from FOLPSD)
    H_factors = jnp.array([1.0, -1.0/jnp.sqrt(3.0), 1.0/jnp.sqrt(5.0),
                           1.0/jnp.sqrt(5.0), 1.0/jnp.sqrt(5.0), 
                           jnp.sqrt(2.0/15.0)])
    
    normalized_coeffs = all_coeffs * H_factors
    
    return normalized_coeffs.T

def bk_sugiyama_multip(k1, k2, kp, pk, cosm_par, redshift, num_points=10, fi_vals=F_VALS_FULL):
    
    k1k2_pairs=jnp.vstack([k1,k2]).T
    bk = compute_sugiyama_multipoles(k1k2_pairs, jnp.log10(kp), jnp.log10(pk), cosm_par, 
                                          redshift, fi_vals=fi_vals, 
                                          num_points=num_points)
    labels = ['000', '110', '220', '202', '022', '112']
    return dict(zip(labels, bk))



def pt_kernel(k, q, wq):
    jq = q**2 * wq / (4. * np.pi**2)
    k = k[:, None]
    x = q / k
    # Integral of F3(q, -q, k) over mu cosine angle between k and q
    def kernel_ff(x):
        x = np.array(x)
        toret = (6. / x**2 - 79. + 50. * x**2 - 21. * x**4 + 0.75 * (1. / x - x)**3 * (2. + 7. * x**2) * 2 * np.log(np.abs((x - 1.) / (x + 1.)))) / 504.
        mask = x > 10.
        toret[mask] = - 61. / 630. + 2. / 105. / x[mask]**2 - 10. / 1323. / x[mask]**4
        dx = x - 1.
        mask = np.abs(dx) < 0.01
        toret[mask] = - 11. / 126. + dx[mask] / 126. - 29. / 252. * dx[mask]**2
        return toret / x**2

    return 2 * jq * kernel_ff(x)


@jax.jit
def pt_pk_1loop(k, q, wq, pk_q, kernel13_d):
    # We could have a speed-up with FFTlog, see https://arxiv.org/pdf/1603.04405.pdf
    k11 = k
    k = k[:, None]
    jq = q**2 * wq / (4. * jnp.pi**2)

    
    mus, wmus = jax_leggauss(jnp.arange(20))
    mus = 0.5 * (mus + 1) * 2 - 1

    # Compute P22
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
    pk_dd = pk11 + pk22_dd + pk13_dd
    return pk_dd

def weights_trapz(x):
    #From desilike.utils
    """Return weights for trapezoidal integration."""
    if x.size == 0:
        return np.array(1.)
    if x.size == 1:
        return np.ones(x.size)
    if x.size == 2:
        return np.ones(x.size) / 2. * (x[1] - x[0])
    return jnp.insert(x[2:] - x[:-2], jnp.array([0, len(x) - 1]), jnp.array([x[1] - x[0], x[-1] - x[-2]])) / 2.



