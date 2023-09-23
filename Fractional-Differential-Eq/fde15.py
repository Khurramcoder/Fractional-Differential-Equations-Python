import numpy as np
from scipy.special import gamma

# Set the initial values
a = np.zeros(6)
b = np.zeros(6)
c = np.zeros(6)
alpha = 0.95

a[0] = 0.1
b[0] = 0.1
c[0] = 0.1

# Define the gamma function
def f(n):
    return gamma(n * alpha + 1) / gamma((n + 1) * alpha + 1)

# Calculate a[1], b[1], and c[1]
a[1] = f(0) * 10 * (b[0] - a[0])
b[1] = f(0) * (28 * a[0] - a[0] * c[0] - b[0])
c[1] = f(0) * (a[0] * b[0] - 8/3 * c[0])

# Calculate a[2], b[2], and c[2] through a loop
for n in range(4):  # Adjust the loop bounds
    a[n + 2] = f(n + 1) * 10 * (b[n + 1] - a[n + 1])
    b[n + 2] = f(n + 1) * (28 * a[n + 1] - np.sum(a[k] * b[n + 1 - k] for k in range(n + 2)) - b[n + 1])
    c[n + 2] = f(n + 1) * (np.sum(a[k] * b[n + 1 - k] for k in range(n + 2)) - 8/3 * c[n + 1])

# Print the results
for n in range(6):
    print(f'n={n}: a={a[n]:.5f}, b={b[n]:.5f}, c={c[n]:.5f}')
    
    
    
import numpy as np
from scipy.special import gamma

# Set initial values
alpha = 0.995
a = np.zeros(6)
b = np.zeros(6)
c = np.zeros(6)

a[0] = 0.1
b[0] = 0.1
c[0] = 0.1

# Define the gamma function
def f(n):
    return gamma(n * alpha + 1) / gamma((n + 1) * alpha + 1)

# Calculate a[1], b[1], and c[1]
a[1] = f(0) * 10 * (b[0] - a[0])
b[1] = f(0) * (28 * a[0] - a[0] * c[0] - b[0])
c[1] = f(0) * (a[0] * b[0] - 8/3 * c[0])

# Calculate a[2], b[2], and c[2] through a loop
for n in range(1, 5):  # Adjust the loop bounds
    a[n + 1] = 10 * f(n) * (b[n] - a[n])
    b[n + 1] = f(n) * (28 * a[n] - np.sum(a[k] * b[n - k] for k in range(n + 1)))
    c[n + 1] = f(n) * (np.sum(a[k] * b[n - k] for k in range(n + 1)) - 8/3 * c[n])

# Print the results
for n in range(6):
    print(f'n={n}: a={a[n]:.5f}, b={b[n]:.5f}, c={c[n]:.5f}')


