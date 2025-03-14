from jax import config
#config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
import jax_cosmo as jc
from geofptax.kernels import bk_multip
from geofptax.ckernels import bk_multip as bk_multip_c
import matplotlib.pyplot as plt
import time


inp_dv = np.load('./test_data/quijote_dvs.npz')
tr0 = inp_dv['quij_tr0']
tr020 = inp_dv['quij_tr020']
mean = inp_dv['quij_meanz05']
bk_sn = inp_dv['quij_snz05']
fig, ax = plt.subplots(3, 3, figsize = (15,15))

#Load the PTcool template for the power spectrum, in this case the Nseries fiducial one
P_template = np.loadtxt('./test_data/Perturbation_theory_non-linear_Quijote_z05_f.dat')

#cosmological parameters, ordered as seen in the Readme file
cosm_par = np.array([P_template[0,-1],0.7627,1.,1.,1.,0.,1.,4.,1.,5.])
# [ $\sigma_8$ , $f$, $\alpha_\parallel$, $\alpha_\bot$, $b_1$ , $b_2$ , $A_P$, $\sigma_P$, $A_B$, $\sigma_B$]
print(cosm_par)

#compute non-linear matter power spectrum
#nlmpk = Pdd(P_template,jnp.array(P_template[:,0]),cosm_par)[:-2]
redshift = 0.5
k = np.array(P_template[:,0])
kp = np.array(P_template)[:-2,0]
h = 0.667
cosmo_jax = jc.Cosmology(Omega_c=0.1200 / h**2, Omega_b=0.02237 / h**2, h=h, sigma8 = cosm_par[0], n_s=0.9649,
                    Omega_k=0., w0=-1., wa=0.)
nlmpk = np.array(jc.power.nonlinear_matter_power(cosmo_jax, k, a=1. / (1 + 0.))[:-2]).astype(np.double)
print(nlmpk.shape)
tic = time.time()
bk0, bk200, bk020, bk002 = bk_multip_c(tr0,tr0,tr020,tr020,kp,nlmpk,cosm_par,redshift=0.5,fit_full=1)
print(f"Took {time.time() - tic} s", flush = True)


#compute the Poisson shot-noise from the measuremenents
sn = bk_sn-mean
sn0 = sn[:bk0.size]
sn200 = sn[bk0.size:2*bk0.size]
sn020 = sn[2*bk0.size:]


mean0 = mean[:bk0.size]
mean200 = mean[bk0.size:2*bk0.size]
mean020 = mean[2*bk0.size:]

#apply the shot-noise correction with the A_B parameter
A_B = cosm_par[8]/(cosm_par[2]*cosm_par[3]*cosm_par[3])**2
bk0 += (A_B-1.)*sn0
bk200 += (A_B-1.)*sn200
bk020 += (A_B-1.)*sn020

bks = (bk0, bk200, bk020)
means = (mean0, mean200, mean020)
c_ratio = []
for i in range(3):
    ax[0,i].plot(bks[i],label=r'GEO-FPT')
    ax[0,i].plot(means[i],label=r'Quijote')
    ax[1,i].plot(bks[i]/means[i],label=r'GEO-FPT')
    c_ratio.append(bks[i]/means[i])
    ax[2,i].plot(bks[i]/means[i] / c_ratio[-1] - 1,label=r'GEO-FPT')
fig.savefig("plots/test_bispec.png", dpi=300)


# Compute bispectrum

for num_points in [11, 20 ,50 ,100]:
    bk0, bk200, bk020, bk002 = bk_multip(tr0,tr0,tr020,tr020,kp,nlmpk,cosm_par,redshift=0.5, num_points = num_points)
    tic = time.time()
    bk0, bk200, bk020, bk002 = bk_multip(tr0,tr0,tr020,tr020,kp,nlmpk,cosm_par,redshift=0.5, num_points = num_points)
    print(f"Took {time.time() - tic} s", flush = True)


    #compute the Poisson shot-noise from the measuremenents
    sn = bk_sn-mean
    sn0 = sn[:bk0.size]
    sn200 = sn[bk0.size:2*bk0.size]
    sn020 = sn[2*bk0.size:]

    #apply the shot-noise correction with the A_B parameter
    A_B = cosm_par[8]/(cosm_par[2]*cosm_par[3]*cosm_par[3])**2
    bk0 += (A_B-1.)*sn0
    bk200 += (A_B-1.)*sn200
    bk020 += (A_B-1.)*sn020

    bks = (bk0, bk200, bk020)
    for i in range(3):
        ax[0,i].plot(bks[i],label=f'geofptax {num_points}')
        ax[1,i].plot(bks[i]/means[i],label=f'geofptax {num_points}')
        ax[2,i].plot(bks[i]/means[i] / c_ratio[i] - 1,label=f'geofptax {num_points}')
    fig.savefig("plots/test_bispec.png", dpi=300)

for i in range(3):
    ax[0,i].legend()
    ax[1,i].legend()
    ax[2,i].legend()
    ax[0,i].set(title = "B model")
    ax[1,i].set(title = "ratio = B model / measurement")
    ax[2,i].set(title = "ratio geofptax / GEO-FPT - 1")
fig.tight_layout()
fig.savefig("plots/test_bispec.png", dpi=300)


def bk_model(tr, tr2,tr3,tr4, kp, pk, cosm_par, redshift, fi_vals, num_points = 50):
    bk0, bk200, bk020, bk002 = bk_multip(tr, tr2,tr3,tr4, kp, pk, cosm_par, redshift, num_points = 50, fi_vals = fi_vals)
    A_B = cosm_par[8]/(cosm_par[2]*cosm_par[3]*cosm_par[3])**2
    bk0 += (A_B-1.)*sn0
    bk200 += (A_B-1.)*sn200
    bk020 += (A_B-1.)*sn020

    return bk0, bk200, bk020

def likelihood(cosm_par, geo_params):
    bk0, bk200, bk020 = bk_model(tr0,tr0,tr020,tr020,kp,nlmpk,cosm_par,redshift=0.5, fi_vals = geo_params, num_points = num_points)
    return jnp.mean((bk0 - means[0])**2 / means[0]**2)


bk_grad = jax.jacfwd(bk_model, argnums = (6,))
from geofptax.constants import F_VALS_FULL
dbk0, dbk200, dbk020 = bk_grad(tr0,tr0,tr020,tr020,kp,nlmpk,cosm_par,redshift=0.5, num_points = num_points, fi_vals = F_VALS_FULL)


print(dbk0)

dlike = jax.grad(likelihood, argnums = (0,1))

print(dlike(cosm_par, F_VALS_FULL))


