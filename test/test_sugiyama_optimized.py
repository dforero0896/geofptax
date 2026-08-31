import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import jax_cosmo as jc
import matplotlib.pyplot as plt

# Import original functions
from geofptax.kernels import (
    bk_sugiyama_multip,
    jax_leggauss,
    interpol_ker,
    geo_fac,
    bkeff_sugiyama,
    __compute_sugiyama_multipoles
)

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

def bk_sugiyama_multip_old(k1, k2, kp, pk, cosm_par, redshift, num_points=20, geo_expansion='poly'):
    """Wrapper to match the API of the original bk_sugiyama_multip"""
    k1k2_pairs = jnp.vstack([k1, k2]).T
    bk = __compute_sugiyama_multipoles(
        k1k2_pairs, jnp.log10(kp), jnp.log10(pk), cosm_par,
        redshift, num_points=num_points, geo_expansion=geo_expansion
    )
    labels = ['000', '110', '220', '202', '022', '112', '222']
    return dict(zip(labels, bk))

# =====================================================================
# 3. SETUP REALISTIC COSMOLOGY & POWER SPECTRUM
# =====================================================================
print("Setting up realistic cosmology and power spectrum...")
h = 0.667
h2 = h**2
cosmo_jax = jc.Cosmology(
    Omega_c=0.1200 / h2, Omega_b=0.02237 / h2, h=h, sigma8=0.81, n_s=0.9649,
    Omega_k=0., w0=-1., wa=0.
)

# Evaluation k-values (isosceles triangles: k1 = k2)
k_ev = jnp.linspace(0.01, 0.3, num=30)

# Realistic power spectrum grid
kp = jnp.logspace(-3, 1, 500)
redshift = 0.5
nlmpk = jnp.array(jc.power.nonlinear_matter_power(cosmo_jax, kp, a=1. / (1 + redshift))).astype(jnp.float64)

# Cosmological parameters: [f, α_∥, α_⊥, b1, b2, bs, A_P, A_B, σ_B]
cosm_par = jnp.array([0.7627, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 5.0])

# =====================================================================
# 4. RUN BOTH IMPLEMENTATIONS WITH PROPER TIMING
# =====================================================================
num_points_test = 50  # Lowered for faster test execution
num_warmup = 2  # Number of warmup runs to trigger JIT compilation
num_timed = 10   # Number of timed runs for averaging

print(f"\nWarming up ORIGINAL function ({num_warmup} runs)...")
for i in range(num_warmup):
    _ = bk_sugiyama_multip_old(
        k_ev, k_ev, kp, nlmpk, cosm_par, redshift=redshift,
        num_points=num_points_test, geo_expansion='poly'
    )
    jax.block_until_ready(_['000'])
print("Warmup complete.")

print(f"\nTiming ORIGINAL function ({num_timed} runs)...")
times_orig = []
for i in range(num_timed):
    tic = time.time()
    bk_orig = bk_sugiyama_multip_old(
        k_ev, k_ev, kp, nlmpk, cosm_par, redshift=redshift,
        num_points=num_points_test, geo_expansion='poly'
    )
    jax.block_until_ready(bk_orig['000'])
    times_orig.append(time.time() - tic)
time_orig = np.mean(times_orig)
print(f"Original time: {time_orig:.4f} s (±{np.std(times_orig):.4f} s)")

print(f"\nWarming up OPTIMIZED function ({num_warmup} runs)...")
for i in range(num_warmup):
    _ = bk_sugiyama_multip(
        k_ev, k_ev, kp, nlmpk, cosm_par, redshift=redshift,
        num_points=num_points_test, geo_expansion='poly'
    )
    jax.block_until_ready(_['000'])
print("Warmup complete.")

print(f"\nTiming OPTIMIZED function ({num_timed} runs)...")
times_opt = []
for i in range(num_timed):
    tic = time.time()
    bk_opt = bk_sugiyama_multip(
        k_ev, k_ev, kp, nlmpk, cosm_par, redshift=redshift,
        num_points=num_points_test, geo_expansion='poly'
    )
    jax.block_until_ready(bk_opt['000'])
    times_opt.append(time.time() - tic)
time_opt = np.mean(times_opt)
print(f"Optimized time: {time_opt:.4f} s (±{np.std(times_opt):.4f} s)")

# =====================================================================
# 5. PLOT COMPARISON
# =====================================================================
print("\nGenerating comparison plots...")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

k_ev2 = k_ev**2

# --- Monopole (000) ---
axes[0, 0].plot(k_ev, k_ev2 * bk_orig['000'], label='Original', marker='o', markersize=4)
axes[0, 0].plot(k_ev, k_ev2 * bk_opt['000'], label='Optimized', marker='x', markersize=4, linestyle='--')
axes[0, 0].set_ylabel(r'$k^2 B_{000}$', fontsize=12)
axes[0, 0].set_title('Monopole ($B_{000}$)', fontsize=14)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[1, 0].plot(k_ev, bk_opt['000'] / bk_orig['000'], color='red', label='Ratio (Opt/Orig)')
axes[1, 0].axhline(1.0, color='black', linestyle='--', alpha=0.5)
axes[1, 0].set_ylabel('Ratio', fontsize=12)
axes[1, 0].set_xlabel(r'$k \, [h/\mathrm{Mpc}]$', fontsize=12)
axes[1, 0].set_ylim(0.99, 1.01)
axes[1, 0].grid(True, alpha=0.3)

# --- Quadrupole (202) ---
axes[0, 1].plot(k_ev, k_ev2 * bk_orig['202'], label='Original', marker='o', markersize=4)
axes[0, 1].plot(k_ev, k_ev2 * bk_opt['202'], label='Optimized', marker='x', markersize=4, linestyle='--')
axes[0, 1].set_ylabel(r'$k^2 B_{202}$', fontsize=12)
axes[0, 1].set_title('Quadrupole ($B_{202}$)', fontsize=14)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 1].plot(k_ev, bk_opt['202'] / bk_orig['202'], color='red', label='Ratio (Opt/Orig)')
axes[1, 1].axhline(1.0, color='black', linestyle='--', alpha=0.5)
axes[1, 1].set_ylabel('Ratio', fontsize=12)
axes[1, 1].set_xlabel(r'$k \, [h/\mathrm{Mpc}]$', fontsize=12)
axes[1, 1].set_ylim(0.99, 1.01)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = "plots/sugiyama_comparison.png"
fig.savefig(plot_path, dpi=300)
print(f"✅ Plot saved to {plot_path}")
plt.show()

print(f"\nSpeedup: {time_orig/time_opt:.2f}x")
print("Note: The optimized version uses drastically less RAM because it avoids")
print("materializing the (Nx, Nmu, Nphi) coordinate meshes and the 4D basis grid.")