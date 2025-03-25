import jax
import jax.numpy as jnp
#from quadax import quadgk
import numpy as np
from functools import partial
from .constants import *



def integrate_1d(f, a, b, args = ()):
    """
    Perform 1D integration using quadgk.
    
    Args:
        f: Function to integrate.
        a, b: Lower and upper limits of integration.
    
    Returns:
        The result of the 1D integral.
    """
    result, _ = quadgk(f, [a, b], args = args)
    return result


def integrate_2d(f, xmin, xmax, ymin, ymax, args = ()):
    """
    2D integrator using nested 1D integrations with quadgk.
    
    Args:
        f: Function to integrate, f(x, y).
        xmin, xmax: Limits of integration for the x-axis.
        ymin, ymax: Limits of integration for the y-axis.
    
    Returns:
        The result of the 2D integral.
    """
    # Define a function to integrate f(x, y) over y for a fixed x
    def integrate_y(x):
        return integrate_1d(lambda y, *args: f(x, y, *args), ymin, ymax, args = args)
    
    
    integral_2d = integrate_1d(integrate_y, xmin, xmax)
    
    return integral_2d
    
    
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
    kmax = jnp.max(k)
    kmin = jnp.min(k)
    kmed = jnp.sum(k) - kmax - kmin

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
vec_integrate = jax.jit(jax.vmap(integrate_2d, in_axes = (None, None, None, None, None, (0, None, 0, None, None, None, None, None))), static_argnames = ('f',))



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
def bk_multip(tr, tr2, tr3, tr4, kp, pk, cosm_par, redshift, num_points=50, fi_vals=F_VALS_FULL):
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
    tuple
        A tuple containing the bispectrum monopole (bk0), first multipole (bk200),
        second multipole (bk020), and third multipole (bk002).
    """
    # Compute the bispectrum multipoles
    bk0, bk200, bk020, bk002 = ext_bk_mp(
        tr, tr2, tr3, tr4, jnp.log10(kp), jnp.log10(pk), cosm_par, redshift, num_points=num_points, fi_vals=fi_vals
    )

    return bk0, bk200, bk020, bk002


