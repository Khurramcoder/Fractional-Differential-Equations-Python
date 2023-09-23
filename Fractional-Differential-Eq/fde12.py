import numpy as np
import matplotlib.pyplot as plt

# Define A and b matrices
A = np.array([[5/7, 10/17, 20/39],
              [10/17, 1/2, 4/9],
              [20/39, 4/9, 2/5]])
b = np.array([15/23, 6/11, 12/25])

# Compute least squares solution
C = np.linalg.lstsq(A, b, rcond=None)[0]

# Define functions f(t) and g(t)
def f(t):
    return t**(1/3)

def g(t):
    return 0.3335*t**(1/5) + 0.9676*t**(1/2) - 0.3027*t**(3/4)

# Plot f(t) and g(t)
t = np.linspace(0, 1, 100)
plt.plot(t, f(t), label='f(t)')
plt.plot(t, g(t), label='g(t)')
plt.xlabel('t')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate |f(t) - g(t)| for t in [0, 1] in steps of 0.2
for t_val in np.arange(0, 1.2, 0.2):
    diff = np.abs(f(t_val) - g(t_val))
    print(f'|f({t_val}) - g({t_val})| = {diff:.5f}')
