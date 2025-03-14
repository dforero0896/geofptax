import jax
import jax.numpy as jnp
from quadax import quadgk
from jax import vmap
from scipy.integrate import dblquad
import time

# Define the 1D integrator using quadgk
def integrate_1d(f, a, b):
    """
    Perform 1D integration using quadgk.
    
    Args:
        f: Function to integrate.
        a, b: Lower and upper limits of integration.
    
    Returns:
        The result of the 1D integral.
    """
    result, _ = quadgk(f, [a, b])
    return result

# Define the 2D integrator using nested 1D integrations
def integrate_2d(f, xmin, xmax, ymin, ymax):
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
        return integrate_1d(lambda y: f(x, y), ymin, ymax)
    
    # Integrate the result over x
    integral_2d = integrate_1d(integrate_y, xmin, xmax)
    
    return integral_2d

integrate_1d = jax.jit(integrate_1d, static_argnames = ('f',))
integrate_2d = jax.jit(integrate_2d, static_argnames = ('f',))


# Define the function to integrate
@jax.jit
def f(x, y):
    return x**2 + y**2
f(0,0)
# Perform the 2D integration using quadax.quadgk
start_time = time.time()
result_quadax = integrate_2d(f, xmin=0, xmax=1, ymin=0, ymax=1)
quadax_time = time.time() - start_time

# Perform the 2D integration using scipy.integrate.dblquad
start_time = time.time()
result_scipy, _ = dblquad(lambda y, x: f(x, y), 0, 1, lambda x: 0, lambda x: 1)
scipy_time = time.time() - start_time

# Print the results and computation times
print("2D Integral Result (quadax.quadgk):", result_quadax)
print("Computation Time (quadax.quadgk):", quadax_time, "seconds")
print("2D Integral Result (scipy.dblquad):", result_scipy)
print("Computation Time (scipy.dblquad):", scipy_time, "seconds")