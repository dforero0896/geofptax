# File: test_geo_factor.py
"""
Standalone test script for GEO-FPT factor k-dependence analysis.
This helps diagnose the unexpected k³ trend in bispectrum calculations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

# Set JAX to use double precision for better numerical stability
jax.config.update("jax_enable_x64", True)

# ============================================================================
# 1. DEFINE THE GEO-FPT FUNCTIONS
# ============================================================================

def geo_fac_original(ka, kb, kc, af, hh=1.0):
    """
    Original GEO-FPT factor implementation.
    
    Parameters:
    -----------
    ka, kb, kc : float or jnp.ndarray
        Triangle side lengths (k1, k2, k3)
    af : jnp.ndarray
        GEO-FPT coefficients [af0, af1, af2, af3, af4]
    hh : float
        Hubble parameter normalization
    
    Returns:
    --------
    geo_factor : float or jnp.ndarray
        GEO-FPT correction factor
    """
    # Determine kmax, kmed, kmin
    k = jnp.array([ka, kb, kc])
    kmax = jnp.max(k, axis=0)
    kmin = jnp.min(k, axis=0)
    kmed = jnp.sum(k, axis=0) - kmax - kmin
    
    # Compute cosines
    cosmax = (kmed**2 + kmin**2 - kmax**2) / (2 * kmed * kmin)
    cosmed = (kmax**2 + kmin**2 - kmed**2) / (2 * kmax * kmin)
    cosmin = (kmax**2 + kmed**2 - kmin**2) / (2 * kmax * kmed)
    
    # Compute area using Heron's formula
    perim = (ka + kb + kc) / 2
    area = jnp.sqrt(perim * (perim - ka) * (perim - kb) * (perim - kc)) / (hh**2 * 0.001)
    
    # Compute GEO-FPT factor
    extra = af[0] + af[1] * cosmed / cosmin + af[2] * cosmax / cosmin + af[3] * area + af[4] * area**2
    
    return extra


def geo_fac_safe(ka, kb, kc, af, hh=1.0, normalize_area=True, area_scale=0.01):
    """
    Safe GEO-FPT factor with optional area normalization.
    
    Parameters:
    -----------
    normalize_area : bool
        If True, normalize area by typical scale to keep it ~O(1)
    area_scale : float
        Normalization scale for area (typical k_nl²)
    
    Returns:
    --------
    geo_factor : float or jnp.ndarray
        GEO-FPT correction factor
    """
    # Determine kmax, kmed, kmin
    k = jnp.array([ka, kb, kc])
    kmax = jnp.max(k, axis=0)
    kmin = jnp.min(k, axis=0)
    kmed = jnp.sum(k, axis=0) - kmax - kmin
    
    # Compute cosines with safety
    cosmax = (kmed**2 + kmin**2 - kmax**2) / (2 * kmed * kmin + 1e-10)
    cosmed = (kmax**2 + kmin**2 - kmed**2) / (2 * kmax * kmin + 1e-10)
    cosmin = (kmax**2 + kmed**2 - kmin**2) / (2 * kmax * kmed + 1e-10)
    
    # Clamp cosines to valid range
    cosmax = jnp.clip(cosmax, -1.0, 1.0)
    cosmed = jnp.clip(cosmed, -1.0, 1.0)
    cosmin = jnp.clip(cosmin, -1.0, 1.0)
    
    # Compute area using Heron's formula (safe version)
    perim = (ka + kb + kc) / 2
    area_term = perim * (perim - ka) * (perim - kb) * (perim - kc)
    area_term = jnp.where(area_term < 0, 0.0, area_term)  # Avoid negative due to numerical errors
    area = jnp.sqrt(area_term)
    
    # Normalize area
    if normalize_area:
        area_norm = area / (hh**2 * area_scale)
    else:
        area_norm = area / (hh**2 * 0.001)  # Original scaling
    
    # Compute GEO-FPT factor with safety
    cosmin_safe = jnp.where(jnp.abs(cosmin) < 1e-10, 1.0, cosmin)  # Avoid division by zero
    extra = af[0] + af[1] * cosmed / cosmin_safe + \
                     af[2] * cosmax / cosmin_safe + \
                     af[3] * area_norm + af[4] * area_norm**2
    
    # Safety: if result is extreme, return 1
    extra = jnp.where((extra > 1e10) | (extra < 1e-10) | jnp.isnan(extra), 1.0, extra)
    
    return extra


def geo_fac_sugiyama_simple(k1, k2, x12, af, hh=1.0, safe=True, **kwargs):
    """
    GEO-FPT factor for Sugiyama estimator.
    
    Parameters:
    -----------
    k1, k2 : float
        First two wavevector magnitudes
    x12 : float
        Cosine of angle between k1 and k2
    af : jnp.ndarray
        GEO-FPT coefficients
    hh : float
        Hubble parameter
    safe : bool
        Use safe version of geo_fac
    
    Returns:
    --------
    geo_factor : float
    """
    k3 = jnp.sqrt(k1**2 + k2**2 + 2 * k1 * k2 * x12)
    
    if safe:
        return geo_fac_safe(k1, k2, k3, af, hh, **kwargs)
    else:
        return geo_fac_original(k1, k2, k3, af, hh)


# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def fit_power_law(k_vals: jnp.ndarray, f_vals: jnp.ndarray) -> Tuple[float, float]:
    """
    Fit f(k) ~ A * k^p using linear regression in log space.
    
    Returns:
    --------
    p : float
        Power law index
    A : float
        Amplitude
    """
    # Filter out zeros and negative values
    mask = (f_vals > 0) & (k_vals > 0)
    k_masked = k_vals[mask]
    f_masked = f_vals[mask]
    
    if len(k_masked) < 3:
        return 0.0, 1.0
    
    log_k = jnp.log(k_masked)
    log_f = jnp.log(f_masked)
    
    # Linear regression: log_f = p * log_k + log_A
    # p = cov(log_k, log_f) / var(log_k)
    log_k_mean = jnp.mean(log_k)
    log_f_mean = jnp.mean(log_f)
    
    cov = jnp.sum((log_k - log_k_mean) * (log_f - log_f_mean))
    var = jnp.sum((log_k - log_k_mean)**2)
    
    p = cov / var
    log_A = log_f_mean - p * log_k_mean
    A = jnp.exp(log_A)
    
    return float(p), float(A)


def analyze_k_dependence(k_vals: np.ndarray, f_vals: np.ndarray, label: str) -> dict:
    """
    Analyze k-dependence of a function.
    
    Returns:
    --------
    analysis : dict
        Contains power law fit, derivatives, etc.
    """
    k_vals_jax = jnp.array(k_vals)
    f_vals_jax = jnp.array(f_vals)
    
    # Fit power law
    p, A = fit_power_law(k_vals_jax, f_vals_jax)
    
    # Compute logarithmic derivative at each point
    log_deriv = np.gradient(np.log(f_vals), np.log(k_vals))
    
    # Find average in different k ranges
    mask_low = k_vals < 0.03
    mask_mid = (k_vals >= 0.03) & (k_vals < 0.1)
    mask_high = k_vals >= 0.1
    
    avg_deriv_low = np.mean(log_deriv[mask_low]) if np.any(mask_low) else 0.0
    avg_deriv_mid = np.mean(log_deriv[mask_mid]) if np.any(mask_mid) else 0.0
    avg_deriv_high = np.mean(log_deriv[mask_high]) if np.any(mask_high) else 0.0
    
    return {
        'label': label,
        'power_law_index': p,
        'amplitude': A,
        'log_derivative': log_deriv,
        'avg_deriv_low_k': avg_deriv_low,
        'avg_deriv_mid_k': avg_deriv_mid,
        'avg_deriv_high_k': avg_deriv_high,
        'min_value': float(np.min(f_vals)),
        'max_value': float(np.max(f_vals)),
        'variation': float(np.max(f_vals) / np.min(f_vals)) if np.min(f_vals) > 0 else float('inf')
    }


def print_analysis(analysis: dict):
    """Print analysis results in a readable format."""
    print(f"\n{analysis['label']}:")
    print(f"  Power law index (overall): {analysis['power_law_index']:.4f}")
    print(f"  Amplitude: {analysis['amplitude']:.4e}")
    print(f"  Average logarithmic derivative:")
    print(f"    Low k (<0.03): {analysis['avg_deriv_low_k']:.4f}")
    print(f"    Mid k (0.03-0.1): {analysis['avg_deriv_mid_k']:.4f}")
    print(f"    High k (>0.1): {analysis['avg_deriv_high_k']:.4f}")
    print(f"  Min value: {analysis['min_value']:.4e}")
    print(f"  Max value: {analysis['max_value']:.4e}")
    print(f"  Variation (max/min): {analysis['variation']:.4e}")
    
    if abs(analysis['power_law_index']) > 1.0:
        print(f"  WARNING: Strong k-dependence! Factor ~ k^{analysis['power_law_index']:.2f}")
    if analysis['variation'] > 10:
        print(f"  WARNING: Large variation across k-range!")


# ============================================================================
# 3. TEST FUNCTIONS
# ============================================================================

def test_different_af_coefficients():
    """Test how different af coefficients affect k-dependence."""
    print("\n" + "="*80)
    print("TESTING DIFFERENT af COEFFICIENTS")
    print("="*80)
    
    k_test = np.logspace(-2, 0, 50)  # k from 0.01 to 1.0 h/Mpc
    
    # Different af coefficient scenarios
    af_scenarios = {
        'Constant only (af[0]=1)': [1.0, 0.0, 0.0, 0.0, 0.0],
        'Small area term (af[3]=0.1)': [1.0, 0.0, 0.0, 0.1, 0.0],
        'Moderate area term (af[3]=1.0)': [1.0, 0.0, 0.0, 1.0, 0.0],
        'Large area term (af[3]=10.0)': [1.0, 0.0, 0.0, 10.0, 0.0],
        'Area² term (af[4]=1.0)': [1.0, 0.0, 0.0, 0.0, 1.0],
        'Both area terms (af[3]=1.0, af[4]=0.1)': [1.0, 0.0, 0.0, 1.0, 0.1],
        'Typical from literature': [1.0, 0.1, 0.01, 0.5, 0.05],
    }
    
    results = {}
    
    for name, af in af_scenarios.items():
        af_array = jnp.array(af)
        factors = []
        
        for k in k_test:
            factor = float(geo_fac_sugiyama_simple(k, k, 0.5, af_array, safe=False))
            factors.append(factor)
        
        results[name] = analyze_k_dependence(k_test, np.array(factors), name)
        print_analysis(results[name])
    
    # Plot all scenarios
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear-linear plot
    for name, result in results.items():
        factors = []
        for k in k_test:
            af_array = jnp.array(af_scenarios[name])
            factor = float(geo_fac_sugiyama_simple(k, k, 0.5, af_array, safe=False))
            factors.append(factor)
        
        axes[0, 0].plot(k_test, factors, label=name)
    
    axes[0, 0].set_xlabel('k [h/Mpc]')
    axes[0, 0].set_ylabel('geo_factor')
    axes[0, 0].set_title('GEO-FPT factor (linear scale)')
    axes[0, 0].legend(fontsize=8, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log-log plot
    for name, result in results.items():
        factors = []
        for k in k_test:
            af_array = jnp.array(af_scenarios[name])
            factor = float(geo_fac_sugiyama_simple(k, k, 0.5, af_array, safe=False))
            factors.append(factor)
        
        axes[0, 1].loglog(k_test, factors, label=name)
    
    axes[0, 1].set_xlabel('k [h/Mpc]')
    axes[0, 1].set_ylabel('geo_factor')
    axes[0, 1].set_title('GEO-FPT factor (log-log scale)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Power law indices
    names = list(results.keys())
    indices = [results[name]['power_law_index'] for name in names]
    
    axes[1, 0].barh(names, indices)
    axes[1, 0].axvline(0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Power law index (p in k^p)')
    axes[1, 0].set_title('Strength of k-dependence')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Variation
    variations = [results[name]['variation'] for name in names]
    axes[1, 1].barh(names, variations, color='orange')
    axes[1, 1].set_xlabel('Variation (max/min)')
    axes[1, 1].set_title('Dynamic range across k')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('plots/geo_factor_af_coefficients.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def test_different_triangle_shapes():
    """Test GEO-FPT factor for different triangle configurations."""
    print("\n" + "="*80)
    print("TESTING DIFFERENT TRIANGLE SHAPES")
    print("="*80)
    
    k_test = np.logspace(-2, 0, 50)
    
    # Triangle configurations
    configs = {
        'Equilateral (k1=k2=k3)': lambda k: (k, k, k),
        'Squeezed (k1=k2=k, k3=0.01)': lambda k: (k, k, 0.01),
        'Isosceles (k1=k2=k, k3=1.5*k)': lambda k: (k, k, 1.5*k),
        'Right angle (x12=0)': lambda k: (k, k, np.sqrt(2)*k),
        'Linear (x12=1)': lambda k: (k, k, 2*k),
    }
    
    # Use a typical af
    af = jnp.array([1.0, 0.1, 0.01, 0.5, 0.05])
    
    results = {}
    
    for name, triangle_func in configs.items():
        factors = []
        
        for k in k_test:
            k1, k2, k3 = triangle_func(k)
            factor = float(geo_fac_original(k1, k2, k3, af, 1.0))
            factors.append(factor)
        
        results[name] = analyze_k_dependence(k_test, np.array(factors), name)
        print_analysis(results[name])
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # All shapes on same plot
    for name, result in results.items():
        factors = []
        triangle_func = configs[name]
        
        for k in k_test:
            k1, k2, k3 = triangle_func(k)
            factor = float(geo_fac_original(k1, k2, k3, af, 1.0))
            factors.append(factor)
        
        axes[0, 0].plot(k_test, factors, label=name)
    
    axes[0, 0].set_xlabel('k [h/Mpc]')
    axes[0, 0].set_ylabel('geo_factor')
    axes[0, 0].set_title('GEO-FPT factor for different triangle shapes')
    axes[0, 0].legend(fontsize=8, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log-log plot
    for name, result in results.items():
        factors = []
        triangle_func = configs[name]
        
        for k in k_test:
            k1, k2, k3 = triangle_func(k)
            factor = float(geo_fac_original(k1, k2, k3, af, 1.0))
            factors.append(factor)
        
        axes[0, 1].loglog(k_test, factors, label=name)
    
    axes[0, 1].set_xlabel('k [h/Mpc]')
    axes[0, 1].set_ylabel('geo_factor')
    axes[0, 1].set_title('Log-log scale')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Power law indices by shape
    names = list(results.keys())
    indices = [results[name]['power_law_index'] for name in names]
    
    axes[1, 0].barh(names, indices)
    axes[1, 0].axvline(0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Power law index (p in k^p)')
    axes[1, 0].set_title('k-dependence by triangle shape')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Area calculation breakdown
    areas = []
    area_squares = []
    
    for name, triangle_func in configs.items():
        k_mid = 0.1  # Typical scale
        k1, k2, k3 = triangle_func(k_mid)
        
        perim = (k1 + k2 + k3) / 2
        area = np.sqrt(perim * (perim - k1) * (perim - k2) * (perim - k3))
        areas.append(area)
        area_squares.append(area**2)
    
    x = np.arange(len(names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, areas, width, label='Area')
    axes[1, 1].bar(x + width/2, area_squares, width, label='Area²')
    axes[1, 1].set_xlabel('Triangle shape')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Area calculation at k=0.1 h/Mpc')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plots/geo_factor_triangle_shapes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def test_area_normalization():
    """Test the effect of area normalization."""
    print("\n" + "="*80)
    print("TESTING AREA NORMALIZATION")
    print("="*80)
    
    k_test = np.logspace(-2, 0, 50)
    
    # af with significant area terms
    af = jnp.array([1.0, 0.0, 0.0, 1.0, 0.1])
    
    # Different normalization strategies
    strategies = {
        'Original (no normalization)': {'safe': False},
        'Safe with k_nl²=0.01 normalization': {'safe': True, 'normalize_area': True, 'area_scale': 0.01},
        'Safe with k_nl²=0.1 normalization': {'safe': True, 'normalize_area': True, 'area_scale': 0.1},
        'Safe with k_nl²=1.0 normalization': {'safe': True, 'normalize_area': True, 'area_scale': 1.0},
    }
    
    results = {}
    
    for name, params in strategies.items():
        factors = []
        
        for k in k_test:
            factor = float(geo_fac_sugiyama_simple(k, k, 0.5, af, hh=1.0, **params))
            factors.append(factor)
        
        results[name] = analyze_k_dependence(k_test, np.array(factors), name)
        print_analysis(results[name])
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Linear plot
    for name, result in results.items():
        factors = []
        params = strategies[name]
        
        for k in k_test:
            factor = float(geo_fac_sugiyama_simple(k, k, 0.5, af, hh=1.0, **params))
            factors.append(factor)
        
        axes[0, 0].plot(k_test, factors, label=name)
    
    axes[0, 0].set_xlabel('k [h/Mpc]')
    axes[0, 0].set_ylabel('geo_factor')
    axes[0, 0].set_title('Effect of area normalization')
    axes[0, 0].legend(fontsize=8, loc='best')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log-log plot
    for name, result in results.items():
        factors = []
        params = strategies[name]
        
        for k in k_test:
            factor = float(geo_fac_sugiyama_simple(k, k, 0.5, af, hh=1.0, **params))
            factors.append(factor)
        
        axes[0, 1].loglog(k_test, factors, label=name)
    
    axes[0, 1].set_xlabel('k [h/Mpc]')
    axes[0, 1].set_ylabel('geo_factor')
    axes[0, 1].set_title('Log-log scale')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Power law indices
    names = list(results.keys())
    indices = [results[name]['power_law_index'] for name in names]
    
    axes[1, 0].barh(names, indices)
    axes[1, 0].axvline(0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Power law index (p in k^p)')
    axes[1, 0].set_title('k-dependence with different normalizations')
    axes[1, 0].grid(True, alpha=0.3, axis='x')
    
    # Actual area values (at k=0.1)
    k_ref = 0.1
    area_values = {}
    
    for name, params in strategies.items():
        factor = float(geo_fac_sugiyama_simple(k_ref, k_ref, 0.5, af, hh=1.0, **params))
        area_values[name] = factor - af[0]  # Subtract constant term
    
    names_area = list(area_values.keys())
    values = list(area_values.values())
    
    axes[1, 1].barh(names_area, values, color='green')
    axes[1, 1].axvline(0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].set_xlabel('Contribution from area terms at k=0.1')
    axes[1, 1].set_title('Area term contribution (geo_factor - af[0])')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('plots/geo_factor_area_normalization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def test_expected_vs_actual_F_VALS():
    """Test with actual F_VALS_FULL from the code."""
    print("\n" + "="*80)
    print("TESTING WITH ACTUAL F_VALS_FULL (SIMULATED)")
    print("="*80)
    
    # Simulate F_VALS_FULL based on typical values
    # Shape: (5, 3) for [af0, af1, af2, af3, af4] at 3 scale factors
    F_VALS_FULL = jnp.array([
        [0.95, 0.97, 0.99],      # af[0]: constant term
        [0.05, 0.03, 0.01],      # af[1]: cosmed/cosmin
        [0.01, 0.005, 0.001],    # af[2]: cosmax/cosmin
        [0.5, 0.3, 0.1],         # af[3]: area term
        [0.05, 0.02, 0.005],     # af[4]: area² term
    ])
    
    print("Simulated F_VALS_FULL:")
    for i, name in enumerate(['af[0]', 'af[1]', 'af[2]', 'af[3]', 'af[4]']):
        print(f"  {name}: {F_VALS_FULL[i, 0]:.4f} at a=1/3, "
              f"{F_VALS_FULL[i, 1]:.4f} at a=1/2, "
              f"{F_VALS_FULL[i, 2]:.4f} at a=2/3")
    
    # Interpolation function
    def interpol_ker(a, fi_vals):
        a_val = jnp.array([1./3., 1./2., 2./3.])
        return jnp.interp(a, a_val, fi_vals)
    
    # Test at different redshifts
    redshifts = [0.0, 0.5, 1.0]
    k_test = np.logspace(-2, 0, 50)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for idx, redshift in enumerate(redshifts):
        a_t = 1.0 / (1.0 + redshift)
        af = interpol_ker(a_t, F_VALS_FULL)
        
        print(f"\nRedshift z={redshift} (a={a_t:.3f}):")
        print(f"  Interpolated af: {af}")
        
        # Test different triangle configurations
        configs = {
            'Equilateral': (k_test, k_test, 0.5),
            'Squeezed': (k_test, k_test * 0.1, 0.5),
            'Stretched': (k_test, k_test * 2.0, 0.5),
        }
        
        for config_name, (k1_vals, k2_vals, x12) in configs.items():
            factors = []
            
            for k1, k2 in zip(k1_vals, k2_vals):
                factor = float(geo_fac_sugiyama_simple(k1, k2, x12, af, safe=False))
                factors.append(factor)
            
            analysis = analyze_k_dependence(k1_vals, np.array(factors), 
                                          f"z={redshift}, {config_name}")
            
            # Plot
            row = idx // 3
            col = idx % 3
            
            axes[row, col].plot(k1_vals, factors, label=config_name)
            
            print(f"  {config_name}: p = {analysis['power_law_index']:.3f}")
        
        axes[row, col].set_xlabel('k1 [h/Mpc]')
        axes[row, col].set_ylabel('geo_factor')
        axes[row, col].set_title(f'z={redshift} (a={a_t:.3f})')
        axes[row, col].legend(fontsize=8)
        axes[row, col].grid(True, alpha=0.3)
        axes[row, col].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('plots/geo_factor_actual_F_VALS.png', dpi=150, bbox_inches='tight')
    plt.show()


def diagnostic_script_for_bispectrum_trend():
    """
    Diagnostic to understand why bispectrum shows k³ trend.
    """
    print("\n" + "="*80)
    print("DIAGNOSTIC: UNDERSTANDING k³ TREND IN BISPECTRUM")
    print("="*80)
    
    # Typical tree-level bispectrum scaling: B_tree ~ P(k)²
    # For P(k) ~ k^n (n ≈ -2 at small scales, -3 at large scales)
    # So B_tree ~ k^{2n} ≈ k^{-4} to k^{-6}
    
    # If GEO-FPT factor adds ~k^p, then total B ~ k^{2n + p}
    # For k³ trend: 2n + p ≈ 3
    
    print("\n1. TREE-LEVEL SCALING:")
    print("   B_tree(k) ~ [P(k)]² ~ k^{2n}")
    print("   For typical P(k):")
    print("     - Linear regime (large scales): n ≈ 1 to 0")
    print("     - Nonlinear regime: n ≈ -2 to -3")
    print("     - So B_tree ~ k^{-4} to k^{-6} at small scales")
    
    print("\n2. OBSERVED TREND:")
    print("   You observe: B_total ~ k³")
    print("   This means: B_total = GEO-FPT × B_tree ~ k³")
    print("   So: k^{p} × k^{2n} ~ k³")
    print("   => p + 2n ≈ 3")
    
    print("\n3. SOLVING FOR GEO-FPT SCALING:")
    n_values = [-1, -2, -3]
    for n in n_values:
        p_needed = 3 - 2*n
        print(f"   If P(k) ~ k^{n}, then GEO-FPT must be ~ k^{p_needed}")
    
    print("\n4. TESTING HYPOTHESIS:")
    print("   Let's measure actual GEO-FPT scaling...")
    
    # Test with typical values
    k_test = np.logspace(-2, 0, 50)
    af = jnp.array([1.0, 0.0, 0.0, 1.0, 0.1])  # Significant area terms
    
    factors = []
    for k in k_test:
        factor = float(geo_fac_sugiyama_simple(k, k, 0.5, af, safe=False))
        factors.append(factor)
    
    analysis = analyze_k_dependence(k_test, np.array(factors), "Test case")
    p_measured = analysis['power_law_index']
    
    print(f"\n   Measured GEO-FPT scaling: ~ k^{p_measured:.3f}")
    
    # What P(k) scaling would this imply?
    print("\n5. IMPLIED P(k) SCALING FROM YOUR DATA:")
    print("   From p + 2n ≈ 3:")
    for assumed_B_scaling in [3.0]:  # Your observed
        n_implied = (assumed_B_scaling - p_measured) / 2
        print(f"   If B_total ~ k^{assumed_B_scaling} and GEO-FPT ~ k^{p_measured:.3f}")
        print(f"   Then P(k) ~ k^{n_implied:.3f}")
    
    print("\n6. RECOMMENDATIONS:")
    print("   A. Check if af[3] and af[4] are too large")
    print("   B. Try normalizing area by k_nl²")
    print("   C. Test with af[3]=af[4]=0 (no area terms)")
    print("   D. Verify P(k) input scaling")
    
    return p_measured


# ============================================================================
# 4. MAIN FUNCTION
# ============================================================================

def main():
    """Run all tests."""
    import os
    
    # Create output directory
    os.makedirs('plots', exist_ok=True)
    
    print("\n" + "="*80)
    print("GEO-FPT FACTOR k-DEPENDENCE ANALYSIS")
    print("="*80)
    
    # Run tests
    test_different_af_coefficients()
    test_different_triangle_shapes()
    test_area_normalization()
    test_expected_vs_actual_F_VALS()
    
    # Run diagnostic
    p_measured = diagnostic_script_for_bispectrum_trend()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nKey findings:")
    print("1. The GEO-FPT factor's k-dependence is dominated by area terms (af[3], af[4])")
    print("2. Area ~ k² for fixed triangle shapes")
    print("3. af[3]*area gives ~k² dependence, af[4]*area² gives ~k⁴ dependence")
    print(f"4. Your measured GEO-FPT scaling: ~k^{p_measured:.3f}")
    print("\nRecommendations:")
    print("• Set af[3]=af[4]=0 for initial testing")
    print("• Use area normalization (safe version with normalize_area=True)")
    print("• Verify F_VALS_FULL values are reasonable")
    print("• Check if the 0.001 factor in area calculation is correct")


if __name__ == "__main__":
    main()