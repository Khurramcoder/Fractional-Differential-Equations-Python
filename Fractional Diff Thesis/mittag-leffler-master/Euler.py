import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from mittag_leffler import ml

# The inputs
h = 1/10
t = [0]
y = [0]
alpha = 0.15
tfinal = 1
N = int(np.ceil((tfinal - t[0]) / h))

# Define the Mittag-Leffler function manually or use the 'mittag_leffler' package if available
def mittag_leffler(a, b, t):
    result = np.zeros_like(t)
    for i in range(len(t)):
        result[i] = ml(a, b, -t[i]**a)
    return result

# Exact Solution
t_exact = np.linspace(0, tfinal, N+1)
Exact = t_exact**4 * mittag_leffler(alpha, 5, t_exact**alpha)

# Fractional-Order ODE
def f(t, y):
    return -y + (1/gamma(5-alpha)) * t**(4-alpha)

# The Fractional Forward Euler Method
for n in range(N):
    t.append(t[-1] + h)
    sum_term = 0
    for j in range(n+1):
        sum_term += ((n-j+1)**alpha - (n-j)**alpha) * f(t[j], y[j])
    y.append(y[0] + (h**alpha / gamma(alpha+1)) * sum_term)

# Absolute Errors
Errors = np.abs(Exact - y)
Last_Error = Errors[-1]

print(f"Last Error: {Last_Error}")

# Plotting
plt.figure()
plt.plot(t_exact, Exact, label="Exact Solution")
plt.plot(t, y, label="Approximation")
plt.xlabel("t")
plt.ylabel("y(t)")
plt.legend()
plt.grid(True)
plt.show()





