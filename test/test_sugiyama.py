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




k_ev = np.linspace(0.01,0.12, num = 40) 
k_ev_bk=np.vstack([k_ev,k_ev]).T   # List of pairs of k. (B=B(k1,k2))
k1k2pairs=k_ev_bk

bk_sugi = jax.jit(bk_sugiyama_multip, static_argnames = ("num_points",))(k_ev, k_ev, kp,nlmpk,cosm_par,redshift=0.5, num_points = 50)
#bk_sugi = bk_sugiyama_multip(k_ev, k_ev, kp,nlmpk,cosm_par,redshift=0.5, num_points = 50)
print(list(map(np.shape, bk_sugi)))

print(f"Took {time.time() - tic} s", flush = True)
_ax = ax.ravel()
_ax[0].plot(k_ev, k_ev**2 * bk_sugi['000'])
_ax[0].plot(k_ev, k_ev**2 * bk_sugi['202'])
#for i in range(len(bk_sugi)):
#    _ax[i].plot(k_ev, k_ev**3 * bk_sugi[i])

fig.tight_layout()
fig.savefig("plots/test_sugiyama.png", dpi=300)


dbk_dcosmo_fun = jax.jit(jax.jacfwd(lambda x: bk_sugiyama_multip(k_ev, k_ev, kp,nlmpk,x,redshift=0.5, num_points = 50), ))

dbk_dcosmo = dbk_dcosmo_fun(cosm_par)
print(dbk_dcosmo)

def safe_div(x, y):
    mask = jnp.abs(y) < 1e-2
    return jnp.where(mask, 0., x / jnp.where(mask, 1., y))

_ax[1].plot(k_ev, safe_div(dbk_dcosmo['000'][:,2], bk_sugi['000']))
_ax[2].plot(k_ev, safe_div(dbk_dcosmo['202'][:,2], bk_sugi['202']))

_ax[2].set(ylim=(-5, 20))
fig.tight_layout()
fig.savefig("plots/test_sugiyama.png", dpi=300)




from geofptax.kernels import weights_trapz, pt_kernel, pt_pk_1loop
from geofptax import folps
inputpkT = np.loadtxt('test_data/pk_linear.txt', unpack = True)
print(inputpkT.shape)
kt = inputpkT[0]
q = kt
ktmin, ktmax = min(kk.min() for kk in kt) * 0.7, max(kk.max() for kk in kt) * 1.3
kt = jnp.linspace(ktmin, ktmax, 100)
kt = jnp.linspace(0.0001, 0.55, 1000)
wq = weights_trapz(q)
kernel = pt_kernel(kt, q, wq)
pkt = pt_pk_1loop(kt, q, wq, inputpkT[1], kernel)

_ax[3].plot(kt, kt * pkt, label = "1loop")
_ax[3].plot(kp,kp * nlmpk, label = "at 0.5")
_ax[3].plot(jnp.interp(kp, inputpkT[0], inputpkT[0]),  
                jnp.interp(kp, inputpkT[0], inputpkT[0]) * jnp.interp(kp, inputpkT[0], inputpkT[1]), label = 'linear')

fig.savefig("plots/test_sugiyama.png", dpi=300)


#bias parameters

b1, b2, bs2,b3nl = 1,0,0,0;                 
bs2 = -4.0 / 7.0 * (b1 - 1.0)
#EFT parameters
alpha0, alpha2, alpha4 = 0,0,0    # PEFT(k) = (alpha0 + alpha2*mu^2+ alpha4*mu^4)  k^2 Plin(k)
ctilde = 0               #NLO counterterm
X_FoG = 1.  # uses a Lorentzian Damping 1/(1+x^2), with x = X_FoG f sigma_v mu. 
X_FoG_bk=1.
#Stochatics parameters
# Noise is Pshot = PshotP * ( alphashot0 + alphashot2*(k*mu)**2 )
alphashot0 = 0;          
alphashot2 = 0;            
PshotP = 1.    # =1/barn.  Poissonian shot noise
Bshot = 1.
cosmo_jax = jc.Cosmology(Omega_c=0.1200 / h**2, Omega_b=0.02237 / h**2, h=h, sigma8 = 0.81, n_s=0.9649,
                    Omega_k=0., w0=-1., wa=0.)

print(cosmo_jax.sigma8,jc.background.growth_rate(cosmo_jax, jnp.atleast_1d(1.)))
cosm_par = jnp.array([cosmo_jax.sigma8,jc.background.growth_rate(cosmo_jax, jnp.atleast_1d(1.))[0],1.,1.,b1,b2,PshotP,X_FoG,Bshot,X_FoG_bk])

# [ $\sigma_8$ , $f$, $\alpha_\parallel$, $\alpha_\bot$, $b_1$ , $b_2$ , $A_P$, $\sigma_P$, $A_B$, $\sigma_B$]

nlmpk = np.array(jc.power.nonlinear_matter_power(cosmo_jax, kp, a=1. / (1 + 0.))).astype(np.double)
_ax[3].plot(kp,kp * nlmpk, label = 'EH nl')

lmpk = np.array(jc.power.linear_matter_power(cosmo_jax, kp, a=1. / (1 + 0.))).astype(np.double)
_ax[3].plot(kp,kp * lmpk, label = 'EH')

pkt = np.array(jc.power.nonlinear_matter_power(cosmo_jax, kt, a=1. / (1 + 0.))).astype(np.double)
loaded_data = np.loadtxt('test_data/Carol_monopole.txt')
k_ev_C = loaded_data[:, 0]
B000_C = loaded_data[:, 1]

B202_C_old = np.loadtxt('test_data/bk202_hit.txt')
B202_k_old = np.loadtxt('test_data/k_new.txt')

B202_C = np.interp(k_ev_C, B202_k_old, B202_C_old)


bk_sugi = jax.jit(bk_sugiyama_multip, static_argnames = ("num_points",))(k_ev_C, k_ev_C, kt,pkt,cosm_par,redshift=0., num_points = 20)

_ax[4].plot(k_ev_C, k_ev_C**2 * bk_sugi['000'], label = "GEO")

_ax[5].plot(k_ev_C, k_ev_C**2 * B000_C, label = 'Ref')
_ax[5].plot(k_ev_C, k_ev_C**2 * bk_sugi['000'], label = "GEO")

_ax[6].plot(k_ev_C, B000_C /bk_sugi['000'] )

[a.legend() for a in _ax]




fig.savefig("plots/test_sugiyama.png", dpi=300)


def loss_fun(x):
    bk = bk_sugiyama_multip(k_ev_C, k_ev_C, kp,nlmpk,x,redshift=0.5, num_points = 20)
    return ((jnp.log(bk['000']) - jnp.log(B000_C))**2).mean()


grad_fun = jax.jit(jax.grad(loss_fun))


print(grad_fun(cosm_par))


import optax
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

def train_bispectrum_model(
    initial_params,  # Initial cosm_par array
    k_ev_C,          # k values for bispectrum
    kp,              # k values for power spectrum
    nlmpk,           # power spectrum values
    B000_C,          # Target bispectrum values
    redshift=0.5,
    num_points=20,
    learning_rate=1e-3,
    num_iterations=1000,
    patience=50,      # For early stopping
    min_delta=1e-6,   # Minimum change for early stopping
    clip_grad_norm=1.0,  # Gradient clipping
    loss_history_interval=10,  # How often to print loss
    param_names=None  # Optional parameter names for logging
):
    """
    Train cosmological parameters to fit bispectrum data.
    
    Parameters:
    -----------
    initial_params : jnp.ndarray
        Initial cosmological parameters [σ8, f, α_∥, α_⊥, b1, b2, A_P, σ_P, A_B, σ_B]
    k_ev_C : jnp.ndarray
        Wavevector magnitudes for bispectrum evaluation
    kp : jnp.ndarray
        Wavevector magnitudes for power spectrum interpolation
    nlmpk : jnp.ndarray
        Nonlinear matter power spectrum at kp
    B000_C : jnp.ndarray
        Target bispectrum monopole values
    redshift : float
        Redshift for calculation
    num_points : int
        Number of integration points for Sugiyama estimator
    learning_rate : float
        Initial learning rate
    num_iterations : int
        Maximum number of training iterations
    patience : int
        Patience for early stopping (iterations without improvement)
    min_delta : float
        Minimum improvement for early stopping
    clip_grad_norm : float
        Maximum gradient norm for clipping
    loss_history_interval : int
        How often to print loss history
    param_names : list of str, optional
        Names of parameters for logging
        
    Returns:
    --------
    dict with keys:
        'params' : jnp.ndarray
            Optimized parameters
        'loss_history' : list
            History of loss values
        'grad_history' : list
            History of gradient norms
        'best_params' : jnp.ndarray
            Parameters at best loss
        'best_loss' : float
            Best loss achieved
    """
    
    # Ensure inputs are JAX arrays
    initial_params = jnp.asarray(initial_params)
    k_ev_C = jnp.asarray(k_ev_C)
    kp = jnp.asarray(kp)
    nlmpk = jnp.asarray(nlmpk)
    B000_C = jnp.asarray(B000_C)
    
    # Define the loss function
    @jax.jit
    def loss_fn(params):
        bk = bk_sugiyama_multip(
            k_ev_C, k_ev_C, kp, nlmpk, params, 
            redshift=redshift, num_points=num_points
        )
        # Mean squared error on monopole
        mask = k_ev_C < 0.11
        mse = jnp.mean(((jnp.log(bk['000']) - jnp.log(B000_C)) * mask) ** 2)
        mse += jnp.mean(((jnp.log(bk['202']) - jnp.log(B202_C)) * mask) ** 2)
        return mse
    
    # Create gradient function
    grad_fn = jax.jit(jax.grad(loss_fn))
    
    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_grad_norm),
        optax.adamw(learning_rate=learning_rate)
    )
    
    # Initialize optimizer state
    opt_state = optimizer.init(initial_params)
    
    # Training variables
    params = initial_params
    best_params = params
    best_loss = float('inf')
    patience_counter = 0
    
    # History tracking
    loss_history = []
    grad_norm_history = []
    param_history = []
    
    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING BISPECTRUM MODEL")
    print(f"{'='*60}")
    print(f"Initial parameters: {initial_params}")
    if param_names:
        print("\nParameter meanings:")
        for i, name in enumerate(param_names):
            print(f"  params[{i}] = {name}")
    
    print(f"\nStarting training for {num_iterations} iterations...")
    print(f"Learning rate: {learning_rate}")
    print(f"Early stopping patience: {patience}")
    print(f"{'='*60}")
    
    for iteration in range(num_iterations):
        # Compute loss and gradient
        loss = loss_fn(params)
        grads = grad_fn(params)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Track gradient norm
        grad_norm = jnp.linalg.norm(grads)
        
        # Store history
        loss_history.append(float(loss))
        grad_norm_history.append(float(grad_norm))
        param_history.append(np.array(params))
        
        # Check for improvement
        if loss < best_loss - min_delta:
            best_loss = float(loss)
            best_params = params.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if iteration % loss_history_interval == 0:
            print(f"Iteration {iteration:4d}: Loss = {loss:.6e}, "
                  f"Grad norm = {grad_norm:.3e}, "
                  f"Patience = {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at iteration {iteration}")
            break
    
    # Final report
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Final loss: {loss_history[-1]:.6e}")
    print(f"Best loss: {best_loss:.6e}")
    print(f"Iterations: {len(loss_history)}")
    
    if param_names:
        print("\nParameter changes (initial -> final -> best):")
        for i, name in enumerate(param_names):
            init = initial_params[i]
            final = params[i]
            best = best_params[i]
            print(f"{name:15s}: {init:8.4f} -> {final:8.4f} -> {best:8.4f}")
    
    # Plot training history
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss history
    axes[0, 0].semilogy(loss_history)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Gradient norm history
    axes[0, 1].semilogy(grad_norm_history)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_title('Gradient Norm History')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Parameter evolution (first 4 parameters)
    param_history_array = np.array(param_history)
    #for i in range(min(4, len(initial_params))):
    param_labels = ['sigma8', 'f', 'qpar', 'qper', 'b1', 'b2', 'Ap', 'Sp', 'Ab', 'Sb']
    for i in range(len(initial_params)):
        axes[1, 0].plot(param_history_array[:, i], 
                       label=f'{param_labels[i]}' + (f' ({param_names[i]})' if param_names else ''))
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Parameter Value')
    axes[1, 0].set_title('Parameter Evolution')
    axes[1, 0].legend(loc='best', fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final comparison plot
    # Compute bispectrum with initial and final parameters
    bk_initial = bk_sugiyama_multip(
        k_ev_C, k_ev_C, kp, nlmpk, initial_params, 
        redshift=redshift, num_points=num_points
    )['000']
    
    bk_final = bk_sugiyama_multip(
        k_ev_C, k_ev_C, kp, nlmpk, params, 
        redshift=redshift, num_points=num_points
    )
    
    bk_best = bk_sugiyama_multip(
        k_ev_C, k_ev_C, kp, nlmpk, best_params, 
        redshift=redshift, num_points=num_points
    )
    
    axes[1, 1].plot(k_ev_C, k_ev_C**2 * B000_C, 'k-', label='Target', linewidth=2)
    axes[1, 1].plot(k_ev_C, k_ev_C**2 * B202_C, 'k-', label='Target', linewidth=2)
    axes[1, 1].plot(k_ev_C, k_ev_C**2 * bk_initial, 'r--', label='Initial', alpha=0.7)
    axes[1, 1].plot(k_ev_C, k_ev_C**2 * bk_final['000'], 'b-', label='Final', alpha=0.7)
    axes[1, 1].plot(k_ev_C, k_ev_C**2 * bk_best['000'], 'g:', label='Best', alpha=0.9, linewidth=2)
    axes[1, 1].plot(k_ev_C, k_ev_C**2 * bk_final['202'], 'b-', label='Final', alpha=0.7)
    axes[1, 1].plot(k_ev_C, k_ev_C**2 * bk_best['202'], 'g:', label='Best', alpha=0.9, linewidth=2)
    axes[1, 1].set_xlabel('k [h/Mpc]')
    axes[1, 1].set_ylabel('k² B(k)')
    axes[1, 1].set_title('Bispectrum Comparison')
    axes[1, 1].legend(loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/bispectrum_training.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Return results
    return {
        'params': params,
        'best_params': best_params,
        'best_loss': best_loss,
        'loss_history': loss_history,
        'grad_history': grad_norm_history,
        'param_history': param_history_array,
        'bk_initial': bk_initial,
        'bk_final': bk_final,
        'bk_best': bk_best
    }


# Advanced version with learning rate scheduling and parameter constraints
def train_bispectrum_model_advanced(
    initial_params,
    k_ev_C,
    kp,
    nlmpk,
    B000_C,
    redshift=0.5,
    num_points=20,
    learning_rate=1e-3,
    num_iterations=1000,
    patience=50,
    min_delta=1e-6,
    clip_grad_norm=1.0,
    schedule_type='cosine',  # 'cosine', 'exponential', 'constant', 'warmup_cosine'
    warmup_steps=100,
    param_constraints=None,  # Dict with bounds: {'min': [...], 'max': [...]}
    param_mask=None,         # Which parameters to optimize (others fixed)
    param_names=None,
    verbose=True
):
    """
    Advanced training with learning rate scheduling and parameter constraints.
    """
    
    # Ensure inputs are JAX arrays
    initial_params = jnp.asarray(initial_params)
    k_ev_C = jnp.asarray(k_ev_C)
    kp = jnp.asarray(kp)
    nlmpk = jnp.asarray(nlmpk)
    B000_C = jnp.asarray(B000_C)
    
    # Create parameter mask if provided
    if param_mask is not None:
        param_mask = jnp.asarray(param_mask, dtype=jnp.bool_)
        trainable_params = initial_params[param_mask]
    else:
        param_mask = jnp.ones_like(initial_params, dtype=jnp.bool_)
        trainable_params = initial_params
    
    # Define the loss function with parameter constraints
    def loss_fn(trainable_params):
        # Reconstruct full parameter vector
        if param_mask is not None:
            params_full = initial_params.at[param_mask].set(trainable_params)
        else:
            params_full = trainable_params
        
        # Apply parameter constraints if provided
        if param_constraints is not None:
            min_vals = jnp.array(param_constraints.get('min', -jnp.inf * jnp.ones_like(params_full)))
            max_vals = jnp.array(param_constraints.get('max', jnp.inf * jnp.ones_like(params_full)))
            params_full = jnp.clip(params_full, min_vals, max_vals)
        
        # Compute bispectrum
        bk = bk_sugiyama_multip(
            k_ev_C, k_ev_C, kp, nlmpk, params_full, 
            redshift=redshift, num_points=num_points
        )
        
        # Mean squared error
        mse = jnp.mean((bk['000'] - B000_C) ** 2)
        
        return mse
    
    # Create gradient function
    grad_fn = jax.jit(jax.grad(loss_fn))
    
    # Create learning rate schedule
    if schedule_type == 'cosine':
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=num_iterations
        )
    elif schedule_type == 'exponential':
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=num_iterations // 10,
            decay_rate=0.95
        )
    elif schedule_type == 'warmup_cosine':
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=num_iterations,
            end_value=learning_rate * 0.01
        )
    else:  # constant
        schedule = learning_rate
    
    # Initialize optimizer with schedule
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_grad_norm),
        optax.adamw(learning_rate=schedule)
    )
    
    # Initialize optimizer state
    opt_state = optimizer.init(trainable_params)
    
    # Training variables
    params = trainable_params
    best_params = params.copy()
    best_loss = float('inf')
    patience_counter = 0
    
    # History tracking
    loss_history = []
    grad_norm_history = []
    lr_history = []
    param_history = []
    
    if verbose:
        print(f"\n{'='*60}")
        print("ADVANCED BISPECTRUM TRAINING")
        print(f"{'='*60}")
        print(f"Training {jnp.sum(param_mask)}/{len(initial_params)} parameters")
        print(f"Schedule: {schedule_type}")
        print(f"Learning rate: {learning_rate}")
        print(f"Max iterations: {num_iterations}")
        print(f"{'='*60}")
    
    # Training loop
    for iteration in range(num_iterations):
        # Compute loss and gradient
        loss = loss_fn(params)
        grads = grad_fn(params)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Track metrics
        grad_norm = jnp.linalg.norm(grads)
        current_lr = schedule(iteration) if callable(schedule) else schedule
        
        loss_history.append(float(loss))
        grad_norm_history.append(float(grad_norm))
        lr_history.append(float(current_lr))
        
        # Reconstruct full parameters for history
        if param_mask is not None:
            params_full = initial_params.at[param_mask].set(params)
        else:
            params_full = params
        param_history.append(np.array(params_full))
        
        # Check for improvement
        if loss < best_loss - min_delta:
            best_loss = float(loss)
            best_params = params.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if verbose and iteration % max(1, num_iterations // 20) == 0:
            print(f"Iter {iteration:4d}: Loss = {loss:.3e}, "
                  f"Grad = {grad_norm:.2e}, "
                  f"LR = {current_lr:.2e}, "
                  f"Patience = {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping at iteration {iteration}")
            break
    
    # Reconstruct final full parameters
    if param_mask is not None:
        final_params_full = initial_params.at[param_mask].set(params)
        best_params_full = initial_params.at[param_mask].set(best_params)
    else:
        final_params_full = params
        best_params_full = best_params
    
    if verbose:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Final loss: {loss_history[-1]:.6e}")
        print(f"Best loss: {best_loss:.6e}")
        print(f"Iterations: {len(loss_history)}")
        
        if param_names:
            print("\nParameter values:")
            for i, name in enumerate(param_names):
                init = initial_params[i]
                final = final_params_full[i]
                best = best_params_full[i]
                trainable = "✓" if (param_mask is None or param_mask[i]) else "✗"
                print(f"{trainable} {name:15s}: {init:8.4f} -> {final:8.4f} -> {best:8.4f}")
    
    # Plot training history
    if verbose:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Loss history
        axes[0, 0].semilogy(loss_history)
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss (MSE)')
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gradient norm
        axes[0, 1].semilogy(grad_norm_history)
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].set_title('Gradient Norm')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 2].plot(lr_history)
        axes[0, 2].set_xlabel('Iteration')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Parameter evolution
        param_history_array = np.array(param_history)
        n_params = len(initial_params)
        n_to_plot = min(6, n_params)
        
        for i in range(n_to_plot):
            axes[1, 0].plot(param_history_array[:, i], 
                           label=f'param[{i}]' + (f' ({param_names[i]})' if param_names else ''))
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].set_title('Parameter Evolution')
        axes[1, 0].legend(loc='best', fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Final comparison
        bk_initial = bk_sugiyama_multip(
            k_ev_C, k_ev_C, kp, nlmpk, initial_params, 
            redshift=redshift, num_points=num_points
        )['000']
        
        bk_final = bk_sugiyama_multip(
            k_ev_C, k_ev_C, kp, nlmpk, final_params_full, 
            redshift=redshift, num_points=num_points
        )['000']
        
        bk_best = bk_sugiyama_multip(
            k_ev_C, k_ev_C, kp, nlmpk, best_params_full, 
            redshift=redshift, num_points=num_points
        )['000']
        
        axes[1, 1].plot(k_ev_C, k_ev_C**2 * B000_C, 'k-', label='Target', linewidth=2)
        axes[1, 1].plot(k_ev_C, k_ev_C**2 * bk_initial, 'r--', label='Initial', alpha=0.7)
        axes[1, 1].plot(k_ev_C, k_ev_C**2 * bk_final, 'b-', label='Final', alpha=0.7)
        axes[1, 1].plot(k_ev_C, k_ev_C**2 * bk_best, 'g:', label='Best', alpha=0.9, linewidth=2)
        axes[1, 1].set_xlabel('k [h/Mpc]')
        axes[1, 1].set_ylabel('k² B(k)')
        axes[1, 1].set_title('Bispectrum Comparison')
        axes[1, 1].legend(loc='best')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Residual plot
        axes[1, 2].plot(k_ev_C, bk_initial - B000_C, 'r--', label='Initial', alpha=0.7)
        axes[1, 2].plot(k_ev_C, bk_final - B000_C, 'b-', label='Final', alpha=0.7)
        axes[1, 2].plot(k_ev_C, bk_best - B000_C, 'g:', label='Best', alpha=0.9, linewidth=2)
        axes[1, 2].axhline(0, color='k', linestyle='-', alpha=0.3)
        axes[1, 2].set_xlabel('k [h/Mpc]')
        axes[1, 2].set_ylabel('Residual B(k)')
        axes[1, 2].set_title('Residuals (Predicted - Target)')
        axes[1, 2].legend(loc='best')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/bispectrum_training_advanced.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    return {
        'params': final_params_full,
        'best_params': best_params_full,
        'best_loss': best_loss,
        'loss_history': loss_history,
        'grad_history': grad_norm_history,
        'lr_history': lr_history,
        'param_history': param_history_array
    }


# Example usage:
def run_example_training():
    """
    Example of how to use the training functions.
    """
    # Assuming you have these variables defined:
    # k_ev_C, kp, nlmpk, B000_C
    
    # Define parameter names for logging
    param_names = [
        'σ8', 'f', 'α_∥', 'α_⊥', 'b1', 'b2', 
        'A_P', 'σ_P', 'A_B', 'σ_B'
    ]
    
    # Define initial parameters (your cosm_par)
    # This should be your starting guess
    initial_params = jnp.array([0.81, 0.8, 1.0, 1.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Run basic training
    print("Running basic training...")
    results_basic = train_bispectrum_model(
        initial_params=initial_params,
        k_ev_C=k_ev_C,
        kp=kp,
        nlmpk=nlmpk,
        B000_C=B000_C,
        redshift=0.5,
        num_points=20,
        learning_rate=1e-3,
        num_iterations=500,
        patience=30,
        param_names=param_names
    )
    
    # Run advanced training with constraints
    print("\n\nRunning advanced training with constraints...")
    
    # Define which parameters to optimize (e.g., only bias parameters)
    param_mask = jnp.array([False, False, False, False, True, True, False, False, False, False])
    
    # Define parameter bounds (optional)
    param_constraints = {
        'min': jnp.array([0.5, 0.3, 0.8, 0.8, 0.5, -5.0, 0.0, 0.0, 0.0, 0.0]),
        'max': jnp.array([1.2, 1.2, 1.2, 1.2, 3.0, 5.0, 10.0, 10.0, 10.0, 10.0])
    }
    
    results_advanced = train_bispectrum_model_advanced(
        initial_params=initial_params,
        k_ev_C=k_ev_C,
        kp=kp,
        nlmpk=nlmpk,
        B000_C=B000_C,
        redshift=0.5,
        num_points=20,
        learning_rate=1e-3,
        num_iterations=500,
        patience=30,
        schedule_type='warmup_cosine',
        param_mask=param_mask,
        param_constraints=param_constraints,
        param_names=param_names
    )
    
    return results_basic, results_advanced


# If you want to use stochastic gradient descent (SGD) instead:
def create_sgd_optimizer(learning_rate=1e-3, momentum=0.9):
    """Create SGD optimizer with momentum."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.sgd(learning_rate=learning_rate, momentum=momentum)
    )


# For L-BFGS or other second-order methods (requires more memory):
def train_with_lbfgs(initial_params, loss_fn, max_iterations=100):
    """
    Example using L-BFGS (requires jaxopt or custom implementation).
    Note: This is more memory-intensive but can converge faster.
    """
    try:
        from jaxopt import LBFGS
        
        optimizer = LBFGS(fun=loss_fn, maxiter=max_iterations, tol=1e-6)
        result = optimizer.run(initial_params)
        
        return {
            'params': result.params,
            'loss': result.state.value,
            'iterations': result.state.iter_num
        }
        
    except ImportError:
        print("jaxopt not installed. Install with: pip install jaxopt")
        return None


# Quick utility to test gradients
def test_gradient(initial_params, loss_fn, eps=1e-5):
    """
    Test gradient computation by comparing with finite differences.
    """
    grad_fn = jax.grad(loss_fn)
    analytic_grad = grad_fn(initial_params)
    
    # Finite difference approximation
    fd_grad = jnp.zeros_like(initial_params)
    for i in range(len(initial_params)):
        params_plus = initial_params.at[i].set(initial_params[i] + eps)
        params_minus = initial_params.at[i].set(initial_params[i] - eps)
        
        loss_plus = loss_fn(params_plus)
        loss_minus = loss_fn(params_minus)
        
        fd_grad = fd_grad.at[i].set((loss_plus - loss_minus) / (2 * eps))
    
    # Compute relative error
    rel_error = jnp.abs(analytic_grad - fd_grad) / (jnp.abs(analytic_grad) + jnp.abs(fd_grad) + 1e-10)
    max_rel_error = jnp.max(rel_error)
    
    print(f"Gradient test: Max relative error = {max_rel_error:.2e}")
    if max_rel_error > 1e-3:
        print("WARNING: Gradient might be incorrect!")
    
    return analytic_grad, fd_grad, rel_error


# 1. Basic training
results = train_bispectrum_model(
    initial_params=cosm_par,  # Your initial parameters
    k_ev_C=k_ev_C,
    kp=kp,
    nlmpk=nlmpk,
    B000_C=B000_C,
    redshift=0.,
    num_points=20,
    learning_rate=1e-2,
    num_iterations=1000
)

# 2. Access results
optimized_params = results['params']
best_params = results['best_params']
loss_history = results['loss_history']

# 3. Test with optimized parameters
bk_optimized = bk_sugiyama_multip(
    k_ev_C, k_ev_C, kp, nlmpk, optimized_params,
    redshift=0.5, num_points=20
)

# 4. Compare with target
plt.figure(figsize=(10, 6))
plt.plot(k_ev_C, k_ev_C**2 * B000_C, 'k-', label='Target', linewidth=2)
plt.plot(k_ev_C, k_ev_C**2 * bk_optimized['000'], 'b--', label='Optimized', linewidth=2)
plt.xlabel('k [h/Mpc]')
plt.ylabel('k² B(k)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("plots/test_tree.png")

