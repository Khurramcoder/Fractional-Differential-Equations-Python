import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
import numpy as np

# Define the symbolic variables
t, u, q = sp.symbols('t u q')

# Define the functions u(t) and v(t)
def u_function(t, q_value):
    if t == 0:
        return 0
    integrand = (t - u)**(q_value - 1) * sp.cos(u**2)
    result, _ = quad(sp.lambdify(u, integrand.subs(q, q_value)), 0, t)
    return result / sp.gamma(q_value)

def v_function(t, q_value):
    if t == 0:
        return 0
    integrand = (t - u)**(q_value - 1) * sp.sin(u**2)
    result, _ = quad(sp.lambdify(u, integrand.subs(q, q_value)), 0, t)
    return result / sp.gamma(q_value)

# Create a list of time values
t_values = np.linspace(0, 10, 100)

# Create lists to store u(t) and v(t) values
u_values = [u_function(t_val, 1/2) for t_val in t_values]
v_values = [v_function(t_val, 1/2) for t_val in t_values]

# Plot u(t) and v(t)
plt.plot(u_values, v_values, color='black')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

