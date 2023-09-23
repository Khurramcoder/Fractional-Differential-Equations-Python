import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from mpl_toolkits.mplot3d import Axes3D  # Import 3D plotting module

def f(t, x, y, z):
    return -x - 5*y - z*y

def g(t, x, y, z):
    return -85*x - y - x*z

def p(t, x, y, z):
    return 0.5*z + x*y + 1

a = 0.998
d = 0.001
tmax = 10000
x = np.zeros(tmax+1)
y = np.zeros(tmax+1)
z = np.zeros(tmax+1)

x[0] = 8
y[0] = 2
z[0] = 1

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
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot
ax.plot(x, y, z)
ax.set_title("Fractional Volta Attractor")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Interpolation
t = np.arange(0, tmax+1, 1)
interp_functions = [interp1d(t, sos[:, i], kind='cubic') for i in range(3)]

# Plot the interpolated solution
t_interp = np.linspace(1, tmax, 10000)
sos_interp = np.column_stack((interp_functions[0](t_interp), interp_functions[1](t_interp), interp_functions[2](t_interp)))

fig_interp = plt.figure()
ax_interp = fig_interp.add_subplot(111, projection='3d')  # Create a 3D subplot
ax_interp.plot(sos_interp[:, 0], sos_interp[:, 1], sos_interp[:, 2])
ax_interp.set_title("Fractional Volta Attractor with Interpolation")
ax_interp.set_xlabel("X")
ax_interp.set_ylabel("Y")
ax_interp.set_zlabel("Z")

plt.show()


