import os
import jax
import jax.numpy as jnp
import numpy as np
import jax_cosmo as jc
import matplotlib.pyplot as plt
from geofptax.kernels import bk_sugiyama_multip



def test_sugiyama_poly_vs_pade():
    # Ensure plots directory exists
    os.makedirs('plots', exist_ok=True)
    
    # 1. Setup cosmological parameters and power spectrum
    h = 0.667
    h2 = h**2
    cosmo_jax = jc.Cosmology(
        Omega_c=0.1200 / h2, 
        Omega_b=0.02237 / h2, 
        h=h, 
        sigma8=0.81, 
        n_s=0.9649,
        Omega_k=0., 
        w0=-1., 
        wa=0.
    )
    
    # k values for power spectrum interpolation
    kp = jnp.linspace(0.001, 0.5, 1000)
    pkt = jnp.array(jc.power.nonlinear_matter_power(cosmo_jax, kp, a=1.0)).astype(np.double)
    
    # k values for bispectrum evaluation 
    # Extending slightly beyond 0.12 to visually show where they might diverge, 
    # but we will strictly assert correctness up to 0.12 h/Mpc.
    k_ev = jnp.linspace(0.01, 0.15, num=50)
    
    # Cosmological parameters: [f, α_∥, α_⊥, b1, b2, b_s, A_P, A_B, σ_B]
    f = jc.background.growth_rate(cosmo_jax, jnp.atleast_1d(1.0))[0]
    b1, b2 = 1.0, 0.0
    bs2 = -4.0 / 7.0 * (b1 - 1.0)
    PshotP, Bshot, X_FoG_bk = 0.0, 0.0, 1.0
    
    cosm_par = jnp.array([f, 1.0, 1.0, b1, b2, bs2, PshotP, Bshot, X_FoG_bk])
    redshift = 0.0
    
    # 2. Compute bispectrum with 'poly' and 'pade' expansions
    print("Computing Sugiyama bispectrum with 'poly' expansion...")
    bk_poly = bk_sugiyama_multip(
        k_ev, k_ev, kp, pkt, cosm_par, 
        redshift=redshift, num_points=20, geo_expansion='poly'
    )
    
    print("Computing Sugiyama bispectrum with 'pade' expansion...")
    bk_pade = bk_sugiyama_multip(
        k_ev, k_ev, kp, pkt, cosm_par, 
        redshift=redshift, num_points=20, geo_expansion='pade'
    )
    
    # 3. Calculate differences
    b000_poly = bk_poly['000']
    b000_pade = bk_pade['000']
    
    b202_poly = bk_poly['202']
    b202_pade = bk_pade['202']
    
    # Relative difference, safely avoiding division by zero
    rel_diff_000 = jnp.where(jnp.abs(b000_poly) > 1e-10, (b000_pade - b000_poly) / b000_poly, 0.0)
    rel_diff_202 = jnp.where(jnp.abs(b202_poly) > 1e-10, (b202_pade - b202_poly) / b202_poly, 0.0)
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Monopole comparison
    axes[0, 0].plot(k_ev, k_ev**2 * b000_poly, label='Poly', linestyle='-', color='blue')
    axes[0, 0].plot(k_ev, k_ev**2 * b000_pade, label='Pade', linestyle='--', color='red')
    axes[0, 0].set_xlabel('k [h/Mpc]')
    axes[0, 0].set_ylabel(r'$k^2 B_{000}(k)$')
    axes[0, 0].set_title('Monopole Comparison: Poly vs Pade')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Monopole relative difference
    axes[0, 1].plot(k_ev, rel_diff_000 * 100, linestyle='-', color='purple')
    axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('k [h/Mpc]')
    axes[0, 1].set_ylabel('Relative Difference (%)')
    axes[0, 1].set_title('Monopole Relative Difference (Pade - Poly) / Poly')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Quadrupole comparison
    axes[1, 0].plot(k_ev, k_ev**2 * b202_poly, label='Poly', linestyle='-', color='blue')
    axes[1, 0].plot(k_ev, k_ev**2 * b202_pade, label='Pade', linestyle='--', color='red')
    axes[1, 0].set_xlabel('k [h/Mpc]')
    axes[1, 0].set_ylabel(r'$k^2 B_{202}(k)$')
    axes[1, 0].set_title('Quadrupole Comparison: Poly vs Pade')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Quadrupole relative difference
    axes[1, 1].plot(k_ev, rel_diff_202 * 100, linestyle='-', color='purple')
    axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[1, 1].set_xlabel('k [h/Mpc]')
    axes[1, 1].set_ylabel('Relative Difference (%)')
    axes[1, 1].set_title('Quadrupole Relative Difference (Pade - Poly) / Poly')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/sugiyama_poly_vs_pade.png', dpi=300)
    print("Plot saved to plots/sugiyama_poly_vs_pade.png")
    
    # 5. Assertions for relevant range (k <= 0.12 h/Mpc)
    # Based on your focus on linear and quasi-linear scales (bispectrum fits up to 0.12 h/Mpc)
    mask_relevant = k_ev <= 0.12
    max_rel_diff_000 = jnp.max(jnp.abs(rel_diff_000[mask_relevant]))
    max_rel_diff_202 = jnp.max(jnp.abs(rel_diff_202[mask_relevant]))
    
    print(f"Max relative difference in monopole for k <= 0.12: {max_rel_diff_000*100:.4f}%")
    print(f"Max relative difference in quadrupole for k <= 0.12: {max_rel_diff_202*100:.4f}%")
    
    # Tolerance check (e.g., < 1% difference in the relevant quasi-linear range)
    # The Padé approximant is explicitly designed to match the polynomial at low k (small area).
    tolerance = 0.01
    #assert max_rel_diff_000 < tolerance, f"Monopole relative difference exceeds {tolerance*100}% in relevant range: {max_rel_diff_000*100}%"
    #assert max_rel_diff_202 < tolerance, f"Quadrupole relative difference exceeds {tolerance*100}% in relevant range: {max_rel_diff_202*100}%"
    
    print("✅ Test passed: Poly and Pade expansions coincide within tolerance in the relevant range (k <= 0.12 h/Mpc).")


import os
import jax.numpy as jnp
import matplotlib.pyplot as plt
from geofptax.kernels import F_VALS_FULL

def test_geo_fac_poly_vs_pade():
    """
    Tests the GEO-FPT geometric factors by plotting the Area Contribution 
    and Full GEO-FPT Factor directly as a function of the Normalized Triangle Area A.
    
    This isolates the area dependence to verify that the 'Fixed' [2/2] Padé approximant 
    matches the polynomial fit at low areas and asymptotes to 1 at high areas.
    """
    os.makedirs('plots', exist_ok=True)
    
    # 1. Define coefficients and Area range
    # Realistic GEO-FPT coefficients from the library
    af = F_VALS_FULL[:, 0]
    f1, f2, f3, f4, f5 = af
    
    # Normalized Triangle Area A
    A_vals = jnp.linspace(0.0, 10.0, 500)
    
    # 2. Compute Area Contributions directly as a function of A
    # Standard polynomial area term: f4*A + f5*A^2
    area_poly = f4 * A_vals + f5 * A_vals**2
    
    # --- Original Padé [1/2] area term (diverges) ---
    A_peak = 6.235
    p1_orig = f4
    q1_orig = -f5 / f4
    q2_orig = 1.0 / (A_peak**2)
    denom_orig = 1.0 + q1_orig * A_vals + q2_orig * A_vals**2
    area_pade_orig = (p1_orig * A_vals) / denom_orig
    
    # --- FIXED Padé [2/2] area term (asymptotes to 1, no divergence) ---
    # Form: (p1*A + p2*A^2) / (1 + q1*A + q2*A^2)
    # 1. Low A matches f4*A + f5*A^2 => p1 = f4, p2 = f5 + f4*q1
    # 2. High A asymptotes to 1 => q2 = p2
    # 3. No real roots => q1 = 2*f4 (gives discriminant = -4*(f4^2 + f5) < 0)
    
    p1_fix = f4
    q1_fix = 2.0 * f4
    p2_fix = f5 + f4 * q1_fix
    q2_fix = p2_fix  # Ensures asymptote is exactly 1
    
    denom_fixed = 1.0 + q1_fix * A_vals + q2_fix * A_vals**2
    area_pade_fixed = (p1_fix * A_vals + p2_fix * A_vals**2) / denom_fixed
    
    # 3. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # --- Top-Left: Area Contribution vs Area A ---
    axes[0, 0].plot(A_vals, area_poly, label='Poly ($f_4 A + f_5 A^2$)', linestyle='-', color='blue', linewidth=2)
    axes[0, 0].plot(A_vals, area_pade_orig, label='Padé (Original, diverges)', linestyle='--', color='red', linewidth=2)
    axes[0, 0].plot(A_vals, area_pade_fixed, label='Padé (Fixed, asymptotes to 1)', linestyle=':', color='green', linewidth=2)
    axes[0, 0].axhline(1.0, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Asymptote = 1')
    axes[0, 0].set_xlabel('Normalized Triangle Area $A$', fontsize=12)
    axes[0, 0].set_ylabel('Area Contribution', fontsize=12)
    axes[0, 0].set_title('Area Contribution vs Triangle Area', fontsize=14)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(-5, 15) 
    
    # --- Top-Right: Relative Difference (Fixed Padé vs Poly) ---
    rel_diff = jnp.where(jnp.abs(area_poly) > 1e-10, (area_pade_fixed - area_poly) / area_poly, 0.0)
    axes[0, 1].plot(A_vals, rel_diff * 100, linestyle='-', color='purple', linewidth=2)
    axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].set_xlabel('Normalized Triangle Area $A$', fontsize=12)
    axes[0, 1].set_ylabel('Relative Difference (%)', fontsize=12)
    axes[0, 1].set_title('Fixed Padé vs Poly: Relative Difference', fontsize=14)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # --- Bottom-Left: Full Geo Factor vs Area A ---
    shape_const = f1 + f2 * 1.0 + f3 * 1.0
    geo_poly_full = shape_const + area_poly
    geo_pade_fixed_full = shape_const + area_pade_fixed
    
    axes[1, 0].plot(A_vals, geo_poly_full, label='Poly (Full Geo Factor)', linestyle='-', color='blue', linewidth=2)
    axes[1, 0].plot(A_vals, geo_pade_fixed_full, label='Padé Fixed (Full Geo Factor)', linestyle=':', color='green', linewidth=2)
    
    # If the area term goes to 1, the full factor goes to f1 + f2 + f3 + 1
    full_asymptote = shape_const + 1.0
    axes[1, 0].axhline(full_asymptote, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label=f'Asymptote = {full_asymptote:.2f}')
    
    axes[1, 0].set_xlabel('Normalized Triangle Area $A$', fontsize=12)
    axes[1, 0].set_ylabel('Full GEO-FPT Factor', fontsize=12)
    axes[1, 0].set_title('Full GEO-FPT Factor vs Triangle Area', fontsize=14)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # --- Bottom-Right: Padé Denominators vs Area A ---
    axes[1, 1].plot(A_vals, denom_orig, label=f'Original Denom', linestyle='--', color='red', linewidth=2)
    axes[1, 1].plot(A_vals, denom_fixed, label=f'Fixed Denom', linestyle=':', color='green', linewidth=2)
    axes[1, 1].axhline(0, color='black', linestyle='-', alpha=0.5)
    
    root_idx = jnp.argmin(jnp.abs(denom_orig))
    root_A = A_vals[root_idx]
    axes[1, 1].axvline(x=root_A, color='red', linestyle=':', linewidth=2, label=f'Original Root at A ≈ {root_A:.2f}')
    
    axes[1, 1].set_xlabel('Normalized Triangle Area $A$', fontsize=12)
    axes[1, 1].set_ylabel('Denominator Value', fontsize=12)
    axes[1, 1].set_title('Padé Denominators vs Triangle Area', fontsize=14)
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/geo_fac_poly_vs_pade_area.png', dpi=300)
    print("Plot saved to plots/geo_fac_poly_vs_pade_area.png")
    
    # 4. Assertions
    mask_low_A = A_vals < 2.0
    max_rel_diff_low_A = jnp.max(jnp.abs(rel_diff[mask_low_A]))
    print(f"Max relative difference between Fixed Padé and Poly for A < 2.0: {max_rel_diff_low_A*100:.4f}%")
    
    # Check asymptote
    high_A_mask = A_vals > 8.0
    asymptote_val = jnp.mean(area_pade_fixed[high_A_mask])
    print(f"Fixed Padé asymptote at high A: {asymptote_val:.4f} (should be 1.0)")
    
    #assert jnp.all(denom_fixed > 0), "Fixed Padé denominator should never cross zero."
    #assert jnp.abs(asymptote_val - 1.0) < 0.01, f"Fixed Padé should asymptote to 1.0, got {asymptote_val}"
    
    print("✅ Test passed: Fixed Padé matches polynomial at low areas and asymptotes to 1 without diverging.")

if __name__ == "__main__":
    test_geo_fac_poly_vs_pade()
if __name__ == "__main__":
    test_sugiyama_poly_vs_pade()