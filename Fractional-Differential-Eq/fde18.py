import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Define the function f
def f(t, y):
    return y - 2 * t / y

# Parameters
delta = 0.1
alpha = 1/2

# Initialize arrays
t = np.zeros(11)
y = np.zeros(11)

# Initial conditions
t[0] = 0
y[0] = 1

# Calculate values
for k in range(5):
    t[k + 1] = t[k] + delta
    y[k + 1] = y[k] + delta**alpha / gamma(alpha + 1) * f(t[k], y[k])

# Display values
for k in range(6):
    print(t[k], y[k])

# Plot the solution
plt.plot(t[:6], y[:6], marker='o', linestyle='-')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Define the function f
def f(t, y):
    return (y - t) / (y + t)

# Parameters
delta = 0.1
alpha = 1/2

# Initialize arrays
t = np.zeros(11)
y = np.zeros(11)

# Initial conditions
t[0] = 0
y[0] = 1

# Calculate values
for k in range(10):
    t[k + 1] = t[k] + delta
    y[k + 1] = y[k] + delta**alpha / gamma(alpha + 1) * f(t[k], y[k])

# Display values
for k in range(11):
    print(t[k], y[k])

# Plot the solution
plt.plot(t, y, marker='o', linestyle='-')
plt.xlabel('t')
plt.ylabel('y')
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Define the functions f and g
def f(t, x, y):
    return (y - x) / t

def g(t, x, y):
    return (y + x) / t

# Parameters
delta = 0.2
alpha = 1/2

# Initialize arrays
t = np.zeros(7)
x = np.zeros(7)
y = np.zeros(7)

# Initial conditions
t[0] = 1
x[0] = 1
y[0] = 1

# Calculate values
for k in range(6):
    t[k + 1] = t[k] + delta
    x[k + 1] = x[k] + delta**alpha / gamma(alpha + 1) * f(t[k], x[k], y[k])
    y[k + 1] = y[k] + delta**alpha / gamma(alpha + 1) * g(t[k], x[k], y[k])

# Display values
for k in range(6):
    print(t[k + 1], x[k + 1], y[k + 1])

# Plot the solutions
plt.plot(t[1:], x[1:], marker='o', linestyle='-', label='x(t)')
plt.plot(t[1:], y[1:], marker='x', linestyle='--', label='y(t)')
plt.xlabel('t')
plt.ylabel('Values')
plt.legend()
plt.grid(True)
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Define the Lorenz system of differential equations
def LorenzSystem(y, t):
    x, y, z = y
    dxdt = x - 10 * h * (x - y)
    dydt = y + h * (28 * x - y - x * z)
    dzdt = z + h * (x * y - (8 / 3) * z)
    return [dxdt, dydt, dzdt]

# Parameters
d = 0.001  # Reduced step size
a = 0.98
h = d**a / gamma(a + 1)

# Initial conditions
y0 = [0.0, 1.0, 0.0]

# Time points
t = np.arange(1, 50001)  # Extended integration range

# Solve the system of differential equations with the default solver
sol = odeint(LorenzSystem, y0, t, rtol=1e-6, atol=1e-6)

# Plot the Lorenz attractor
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol[:, 0], sol[:, 1], sol[:, 2], lw=0.5, color='blue')
ax.set_xlabel('x(t)')
ax.set_ylabel('y(t)')
ax.set_zlabel('z(t)')
ax.set_title('Lorenz Attractor')
plt.show()





