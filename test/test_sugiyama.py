from jax import config
#config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
import jax_cosmo as jc
from geofptax.kernels import bk_multip, bk_sugiyama_multip
import matplotlib.pyplot as plt
import time


fig, ax = plt.subplots(3, 3, figsize = (15,15))

P_template = np.loadtxt('./test_data/Perturbation_theory_non-linear_Quijote_z05_f.dat')
cosm_par = np.array([P_template[0,-1],0.7627,1.,1.,1.,0.,1.,4.,1.,5.])
# [ $\sigma_8$ , $f$, $\alpha_\parallel$, $\alpha_\bot$, $b_1$ , $b_2$ , $A_P$, $\sigma_P$, $A_B$, $\sigma_B$]



redshift = 0.5
k = np.array(P_template[:,0])
kp = np.array(P_template)[:-2,0]
h = 0.667
cosmo_jax = jc.Cosmology(Omega_c=0.1200 / h**2, Omega_b=0.02237 / h**2, h=h, sigma8 = cosm_par[0], n_s=0.9649,
                    Omega_k=0., w0=-1., wa=0.)
nlmpk = np.array(jc.power.nonlinear_matter_power(cosmo_jax, k, a=1. / (1 + 0.))[:-2]).astype(np.double)
tic = time.time()




k_ev = np.linspace(0.01,0.2, num = 40) 
k_ev_bk=np.vstack([k_ev,k_ev]).T   # List of pairs of k. (B=B(k1,k2))
k1k2pairs=k_ev_bk

bk_sugi = jax.jit(bk_sugiyama_multip, static_argnames = ("num_points",))(k_ev, k_ev, kp,nlmpk,cosm_par,redshift=0.5, num_points = 50)
#bk_sugi = bk_sugiyama_multip(k_ev, k_ev, kp,nlmpk,cosm_par,redshift=0.5, num_points = 50)
print(list(map(np.shape, bk_sugi)))

print(f"Took {time.time() - tic} s", flush = True)
_ax = ax.ravel()
_ax[0].plot(k_ev, k_ev**2 * bk_sugi[0])
_ax[0].plot(k_ev, k_ev**2 * bk_sugi[4])
#for i in range(len(bk_sugi)):
#    _ax[i].plot(k_ev, k_ev**3 * bk_sugi[i])

fig.tight_layout()
fig.savefig("plots/test_sugiyama.png", dpi=300)


dbk_dcosmo_fun = jax.jit(jax.jacfwd(lambda x: bk_sugiyama_multip(k_ev, k_ev, kp,nlmpk,x,redshift=0.5, num_points = 50), ))

dbk_dcosmo = dbk_dcosmo_fun(cosm_par)
print(dbk_dcosmo.shape)

def safe_div(x, y):
    mask = jnp.abs(y) < 1e-2
    return jnp.where(mask, 0., x / jnp.where(mask, 1., y))

_ax[1].plot(k_ev, safe_div(dbk_dcosmo[0,:,2], bk_sugi[0]))
_ax[2].plot(k_ev, safe_div(dbk_dcosmo[4,:,2], bk_sugi[4]))

_ax[2].set(ylim=(-5, 20))
fig.tight_layout()
fig.savefig("plots/test_sugiyama.png", dpi=300)


def loss_fun(x):
    bk = bk_sugiyama_multip(k_ev, k_ev, kp,nlmpk,x,redshift=0.5, num_points = 50)
    return (bk**2).mean()


grad_fun = jax.jit(jax.grad(loss_fun))


print(grad_fun(cosm_par))

