import sympy as sp
import matplotlib.pyplot as plt
from sympy.solvers import solve

# Define symbolic variables
t, u, a, b, c, x, y, z = sp.symbols('t u a b c x y z')

# Define the functions x(t) and y(t)
x = 1/sp.gamma(a) * sp.integrate((t - u)**(a - 1) * sp.cos(u**2), (u, 0, t))
y = 1/sp.gamma(a) * sp.integrate((t - u)**(a - 1) * sp.sin(u**2), (u, 0, t))

# Define the equation for the plane
plane_eq = x/a + y/b + z/c - 1

# Solve for z
z_expr = solve(plane_eq, z)[0]

# Create a list of time values
t_values = [i/10 for i in range(41)]  # Time values from 0 to 4 with step 0.1

# Calculate x, y, and z values for the curve
x_values = [x.subs(t, t_val).subs(a, 1) for t_val in t_values]
y_values = [y.subs(t, t_val).subs(a, 1) for t_val in t_values]
z_values = [z_expr.subs({t: t_val, a: 1, b: 1, c: 1}) for t_val in t_values]

# Plot the space curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_values, y_values, z_values, color='black')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Space Curve')
plt.show()

