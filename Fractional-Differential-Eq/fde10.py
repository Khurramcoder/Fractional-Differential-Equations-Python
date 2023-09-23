import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Define symbols
t, u, x, y = sp.symbols('t u x y')

# Define parameters
a = 1/2
q = 1/2

# Define the integrals
integral_cos = sp.integrate((t - u)**(a - 1) * sp.cos(u**2), (u, 0, t))
integral_sin = sp.integrate((t - u)**(a - 1) * sp.sin(u**2), (u, 0, t))

# Define parametric functions u(t) and v(t)
u_t = (1 / sp.gamma(q)) * integral_cos
v_t = (1 / sp.gamma(q)) * integral_sin

# Create arrays for t values
t_values = np.linspace(0, 4, 100)

# Evaluate u(t) and v(t) for the given t values
u_values = [u_t.subs(t, val).evalf() for val in t_values]
v_values = [v_t.subs(t, val).evalf() for val in t_values]

# Compute the corresponding x, y, and z values
x_values = [np.cos(u) * np.sin(v) for u, v in zip(u_values, v_values)]
y_values = [np.sin(u) * np.sin(v) for u, v in zip(u_values, v_values)]
z_values = [np.cos(v) for v in v_values]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the curve
ax.plot(x_values, y_values, z_values, color='black')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
