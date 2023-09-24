import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

# Define the fractional differential equation function
def fractional_diff_eq(t, y):
    alpha = 0.5  # Fractional order
    dydt = np.zeros_like(y)
    dydt[0] = 1.0  # Initial condition
    for k in range(1, len(y)):
        dydt[k] = y[k-1] / gamma(k + alpha)
    return dydt

# Define the gamma function
def gamma(x):
    from scipy.special import gamma
    return gamma(x)

# Time span for the solution curve
t_span = (0, 5)
# Initial condition
y0 = np.array([1.0])

# Solve the fractional differential equation using solve_ivp
solution = solve_ivp(fractional_diff_eq, t_span, y0, t_eval=np.linspace(0, 5, 1000))

# Plot the solution curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(solution.t, solution.y[0])
plt.xlabel('Time')
plt.ylabel('Solution')
plt.title('Solution Curve of Fractional Differential Equation')

# Now, let's create a surface plot
# Define the range of time and alpha values
t_range = np.linspace(0, 5, 100)
alpha_range = np.linspace(0.1, 1.0, 100)

# Create a grid of (t, alpha) values
T, Alpha = np.meshgrid(t_range, alpha_range)

# Calculate the corresponding solutions using your fractional differential equation
Z = np.zeros_like(T)
for i in range(len(alpha_range)):
    for j in range(len(t_range)):
        alpha = alpha_range[i]
        t = t_range[j]
        y0 = np.array([1.0])
        sol = solve_ivp(fractional_diff_eq, t_span, y0, t_eval=[t])  # Use the full time span
        Z[i, j] = sol.y[0][-1]

# Create a 3D surface plot
plt.subplot(1, 2, 2)
plt.xlabel('Time')
plt.ylabel('Alpha')
plt.title('Surface Plot of Fractional Differential Equation Solution')
ax = plt.gca(projection='3d')  # Use 'projection' here
ax.plot_surface(T, Alpha, Z, cmap='viridis')

plt.tight_layout()
plt.show()


