import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad

def x(u, v):
    return u * np.cos(v)

def y(u, v):
    return u * np.sin(v)

def z(u, v):
    return -u

def u_integral(t, q):
    def integrand(s):
        return (t - s)**(q - 1) * np.cos(s**2)
    result, _ = quad(integrand, 0, t)
    return result / gamma(q)

def v_integral(t, q):
    def integrand(s):
        return (t - s)**(q - 1) * np.sin(s**2)
    result, _ = quad(integrand, 0, t)
    return result / gamma(q)

q = 0.5  # Initial value of q
u_values = np.linspace(-1, 1, 100)
v_values = np.linspace(0, 2 * np.pi, 100)
U, V = np.meshgrid(u_values, v_values)

X = x(U, V)
Y = y(U, V)
Z = z(U, V)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
ax.set_box_aspect([1, 1, 1])

t_values = np.linspace(0, 10, 100)
X2 = x(u_integral(t_values, q), v_integral(t_values, q))
Y2 = y(u_integral(t_values, q), v_integral(t_values, q))
Z2 = z(u_integral(t_values, q), v_integral(t_values, q))

ax.plot(X2, Y2, Z2, color='red')

plt.show()
