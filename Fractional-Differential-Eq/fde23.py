import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Define the parameters and initial conditions
a = 0.998
b = 0.5
x0 = 1.0
y0 = 0.0
h = (0.2)**a / np.math.gamma(a + 1)
l = (0.2)**b / np.math.gamma(b + 1)
n = 10000

# Create arrays to store t, x, and y values
t = np.zeros(n+1)
x = np.zeros(n+1)
y = np.zeros(n+1)

# Set initial values
t[0] = 0.0
x[0] = x0
y[0] = y0

# Define the differential equations
def f(t, x, y):
    return y

def g(t, x, y):
    return -x + 0.25 * (1 - x**2) * y

# Perform the RK4 integration
for i in range(n):
    K1 = f(t[i], x[i], y[i])
    L1 = g(t[i], x[i], y[i])
    K2 = f(t[i] + h, x[i] + h*K1, y[i] + h*L1)
    L2 = g(t[i] + l, x[i] + l*K1, y[i] + l*L1)
    K3 = f(t[i] + h/2, x[i] + h*K2/2, y[i] + h*L2/2)
    L3 = g(t[i] + l/2, x[i] + l*K2/2, y[i] + l*L2/2)
    K4 = f(t[i] + h, x[i] + h*K3, y[i] + h*L3)
    L4 = g(t[i] + l, x[i] + l*K3, y[i] + l*L3)
    
    x[i + 1] = x[i] + (K1 + 2*K2 + 2*K3 + K4) * h / 6
    y[i + 1] = y[i] + (L1 + 2*L2 + 2*L3 + L4) * l / 6
    t[i + 1] = t[i] + 0.2

# Plot the solution
plt.figure(figsize=(8, 4))
plt.subplot(121)
plt.plot(x, y)
plt.title("RK4 Solution of Modified Fractional Duffing System")
plt.xlabel("x")
plt.ylabel("y")

# Interpolation
sol = np.column_stack((x[1:], y[1:]))
t_interp = np.linspace(1, 1000, 1000)
interp_functions = [interp1d(t[1:], sol[:, i], kind='cubic') for i in range(2)]

# Plot the interpolated solution
plt.subplot(122)
plt.plot(interp_functions[0](t_interp), interp_functions[1](t_interp))
plt.title("Interpolated Solution")
plt.xlabel("x")
plt.ylabel("y")

plt.tight_layout()
plt.show()
