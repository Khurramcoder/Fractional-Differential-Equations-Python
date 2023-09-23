import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.special import gamma

# Define the Lorenz system of differential equations
def LorenzSystem(t, xyz):
    x, y, z = xyz
    dxdt = 10 * (y - x)
    dydt = 28 * x - y - x * z
    dzdt = x * y - (8 / 3) * z
    return [dxdt, dydt, dzdt]

# Parameters
a = 0.998
d = 0.01
tmax = 10000
h = (d)**a / gamma(a + 1)

# Initial conditions
xyz = np.array([1.0, 1.0, 1.0])

# Arrays to store the results
t_values = [0]
xyz_values = [xyz]

# Time-stepping using the Runge-Kutta method
for n in range(tmax):
    k1 = h * np.array(LorenzSystem(n * d, xyz))
    k2 = h * np.array(LorenzSystem((n + 0.5) * d, xyz + 0.5 * k1))
    k3 = h * np.array(LorenzSystem((n + 0.5) * d, xyz + 0.5 * k2))
    k4 = h * np.array(LorenzSystem((n + 1) * d, xyz + k3))
    xyz = xyz + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    t_values.append((n + 1) * d)
    xyz_values.append(xyz)

# Convert results to numpy arrays
t_values = np.array(t_values)
xyz_values = np.array(xyz_values)

# Interpolation
interp_t = np.linspace(0, tmax * d, num=10000)
interp_xyz = np.zeros((10000, 3))

for i in range(3):
    interp_xyz[:, i] = interp1d(t_values, xyz_values[:, i], kind='cubic')(interp_t)

# Plot the Lorenz attractor without interpolation
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(xyz_values[:, 0], xyz_values[:, 1], xyz_values[:, 2], lw=0.5, color='blue')
ax1.set_xlabel('x(t)')
ax1.set_ylabel('y(t)')
ax1.set_zlabel('z(t)')
ax1.set_title('Lorenz Attractor (No Interpolation)')

# Plot the Lorenz attractor with colorful interpolation
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')
colors = plt.cm.viridis(np.linspace(0, 1, len(interp_t)))  # Use a colormap
ax2.scatter(interp_xyz[:, 0], interp_xyz[:, 1], interp_xyz[:, 2], c=colors, s=1, cmap='viridis')
ax2.set_xlabel('x(t)')
ax2.set_ylabel('y(t)')
ax2.set_zlabel('z(t)')
ax2.set_title('Colorful Lorenz Attractor (With Interpolation)')

plt.show()

