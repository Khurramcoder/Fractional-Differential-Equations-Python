import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def f(t, x, y, z):
    return 40 * (y - x)

def g(t, x, y, z):
    return x * (28 - 40) + 28 * y - x * z

def p(t, x, y, z):
    return x * y - 2 * z

a = 0.998
d = 0.01
tmax = 10000
x = np.zeros(tmax+1)
y = np.zeros(tmax+1)
z = np.zeros(tmax+1)

x[0] = 0
y[0] = 1
z[0] = 0

h = (d**a) / np.math.gamma(a + 1)

for n in range(tmax):
    k1 = h * f(n*d, x[n], y[n], z[n])
    l1 = h * g(n*d, x[n], y[n], z[n])
    m1 = h * p(n*d, x[n], y[n], z[n])

    k2 = h * f(n*d + h/2, x[n] + k1/2, y[n] + l1/2, z[n] + m1/2)
    l2 = h * g(n*d + h/2, x[n] + k1/2, y[n] + l1/2, z[n] + m1/2)
    m2 = h * p(n*d + h/2, x[n] + k1/2, y[n] + l1/2, z[n] + m1/2)

    k3 = h * f(n*d + h/2, x[n] + k2/2, y[n] + l2/2, z[n] + m2/2)
    l3 = h * g(n*d + h/2, x[n] + k2/2, y[n] + l2/2, z[n] + m2/2)
    m3 = h * p(n*d + h/2, x[n] + k2/2, y[n] + l2/2, z[n] + m2/2)

    k4 = h * f(n*d + h, x[n] + k3, y[n] + l3, z[n] + m3)
    l4 = h * g(n*d + h, x[n] + k3, y[n] + l3, z[n] + m3)
    m4 = h * p(n*d + h, x[n] + k3, y[n] + l3, z[n] + m3)

    x[n + 1] = x[n] + (k1 + 2*k2 + 2*k3 + k4) / 6
    y[n + 1] = y[n] + (l1 + 2*l2 + 2*l3 + l4) / 6
    z[n + 1] = z[n] + (m1 + 2*m2 + 2*m3 + m4) / 6

sos = np.column_stack((x, y, z))

# Plot the Chua Attractor in 3D before interpolation
fig_before = plt.figure()
ax_before = fig_before.add_subplot(111, projection='3d')
ax_before.plot(x, y, z, c='blue', alpha=0.5)
ax_before.set_title("Chua Attractor (Before Interpolation)")
ax_before.set_xlabel("X")
ax_before.set_ylabel("Y")
ax_before.set_zlabel("Z")

# Interpolation
t = np.arange(0, tmax+1, 1)
interp_functions = [interp1d(t, sos[:, i], kind='cubic') for i in range(3)]

# Plot the interpolated solution with a colormap
t_interp = np.linspace(0, tmax, 10000)
sos_interp = np.column_stack((interp_functions[0](t_interp), interp_functions[1](t_interp), interp_functions[2](t_interp)))
colors = cm.viridis(np.linspace(0, 1, len(t_interp)))

fig_after = plt.figure()
ax_after = fig_after.add_subplot(111, projection='3d')
ax_after.scatter(sos_interp[:, 0], sos_interp[:, 1], sos_interp[:, 2], c=colors, alpha=0.5)
ax_after.set_title("Chua Attractor (After Interpolation) with Colorful Trajectory")
ax_after.set_xlabel("X")
ax_after.set_ylabel("Y")
ax_after.set_zlabel("Z")

plt.show()

