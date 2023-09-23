import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# Define constants
alpha = 3/4

# Initialize arrays
a = np.zeros(10001)
b = np.zeros(10001)

# Initial values
a[0] = 0.1
b[0] = 0

# Define the f(n) function
def f(n):
    return gamma(n * alpha + 1) / gamma((n + 1) * alpha + 1)

# Calculate a and b values with a smaller step size and more iterations
for n in range(10000):
    a[n + 1] = 3.5 * f(n) * (b[n] - np.sum(b[k] * b[n - k] for k in range(n + 1)))
    b[n + 1] = 4 * f(n) * (a[n] - np.sum(a[k] * a[n - k] for k in range(n + 1)))

# Create a parametric plot
t = np.arange(1, 10001)
plt.figure(figsize=(8, 6))
plt.plot(a[1:], b[1:])
plt.title('Lotka Attractor')
plt.xlabel('a(t)')
plt.ylabel('b(t)')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()



