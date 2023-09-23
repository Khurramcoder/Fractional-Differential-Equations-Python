import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def f(t, x, y, z):
    return -y - z

def g(t, x, y, z):
    return x + 0.2 * y

def p(t, x, y, z):
    return 0.2 + z * (x - 8)

a = 0.998
d = 0.01
x0, y0, z0 = 1, 1, 1
tmax = 2000
h = d ** a / gamma(a + 1)

n_steps = int(tmax / d)
x = np.zeros(n_steps + 1)
y = np.zeros(n_steps + 1)
z = np.zeros(n_steps + 1)

x[0] = x0
y[0] = y0
z[0] = z0

for n in range(n_steps):
    k1 = h * f(n * d, x[n], y[n], z[n])
    l1 = h * g(n * d, x[n], y[n], z[n])
    m1 = h * p(n * d, x[n], y[n], z[n])

    k2 = h * f((n * d) + (h / 2), x[n] + (k1 / 2), y[n] + (l1 / 2), z[n] + (m1 / 2))
    l2 = h * g((n * d) + (h / 2), x[n] + (k1 / 2), y[n] + (l1 / 2), z[n] + (m1 / 2))
    m2 = h * p((n * d) + (h / 2), x[n] + (k1 / 2), y[n] + (l1 / 2), z[n] + (m1 / 2))

    k3 = h * f((n * d) + (h / 2), x[n] + (k2 / 2), y[n] + (l2 / 2), z[n] + (m2 / 2))
    l3 = h * g((n * d) + (h / 2), x[n] + (k2 / 2), y[n] + (l2 / 2), z[n] + (m2 / 2))
    m3 = h * p((n * d) + (h / 2), x[n] + (k2 / 2), y[n] + (l2 / 2), z[n] + (m2 / 2))

    k4 = h * f((n * d) + h, x[n] + k3, y[n] + l3, z[n] + m3)
    l4 = h * g((n * d) + h, x[n] + k3, y[n] + l3, z[n] + m3)
    m4 = h * p((n * d) + h, x[n] + k3, y[n] + l3, z[n] + m3)

    x[n + 1] = x[n] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    y[n + 1] = y[n] + (1 / 6) * (l1 + 2 * l2 + 2 * l3 + l4)
    z[n + 1] = z[n] + (1 / 6) * (m1 + 2 * m2 + 2 * m3 + m4)

# Plot the Rössler attractor
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
