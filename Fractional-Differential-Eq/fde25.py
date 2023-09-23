import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def f(t, x, y, z):
    return y

def g(t, x, y, z):
    return z

def p(t, x, y, z):
    return -x - y - np.sign(1 + 4 * y)

a = 0.998
d = 0.01
x = [-0.5]
y = [0]
z = [0]
tmax = 2000
h = (d**a) / np.math.gamma(a + 1)

for n in range(tmax):
    k1 = h * f(n * d, x[n], y[n], z[n])
    l1 = h * g(n * d, x[n], y[n], z[n])
    m1 = h * p(n * d, x[n], y[n], z[n])
    k2 = h * f(n * d + h/2, x[n] + k1/2, y[n] + l1/2, z[n] + m1/2)
    l2 = h * g(n * d + h/2, x[n] + k1/2, y[n] + l1/2, z[n] + m1/2)
    m2 = h * p(n * d + h/2, x[n] + k1/2, y[n] + l1/2, z[n] + m1/2)
    k3 = h * f(n * d + h/2, x[n] + k2/2, y[n] + l2/2, z[n] + m2/2)
    l3 = h * g(n * d + h/2, x[n] + k2/2, y[n] + l2/2, z[n] + m2/2)
    m3 = h * p(n * d + h/2, x[n] + k2/2, y[n] + l2/2, z[n] + m2/2)
    k4 = h * f(n * d + h, x[n] + k3, y[n] + l3, z[n] + m3)
    l4 = h * g(n * d + h, x[n] + k3, y[n] + l3, z[n] + m3)
    m4 = h * p(n * d + h, x[n] + k3, y[n] + l3, z[n] + m3)
    x.append(x[n] + (k1 + 2 * k2 + 2 * k3 + k4) / 6)
    y.append(y[n] + (l1 + 2 * l2 + 2 * l3 + l4) / 6)
    z.append(z[n] + (m1 + 2 * m2 + 2 * m3 + m4) / 6)

# Plot without iteration
fig1 = plt.figure(figsize=(8, 6))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(x, y, z, linewidth=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_title('Without Iteration')

# Interpolation for iteration
t_values = np.arange(0, tmax+1)
x_interp = interp1d(t_values, x, kind='cubic')
y_interp = interp1d(t_values, y, kind='cubic')
z_interp = interp1d(t_values, z, kind='cubic')

# Iteration
t_values_new = np.linspace(0, tmax, num=2000)
x_iter = x_interp(t_values_new)
y_iter = y_interp(t_values_new)
z_iter = z_interp(t_values_new)

# Plot with Iteration
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')
ax2.plot(x_iter, y_iter, z_iter, linewidth=0.5)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_title('With Iteration')

plt.show()
