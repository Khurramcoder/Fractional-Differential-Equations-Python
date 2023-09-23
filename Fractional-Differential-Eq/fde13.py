import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Define the symbols and variables
a, C1, C2, C3, t = sp.symbols('a C1 C2 C3 t')

# Define the equations
a11 = sp.gamma(2/5 + 1) / sp.gamma(2/5 + 1 + a)
a12 = sp.gamma(1/5 + 1/2 + 1) / sp.gamma(1/5 + 1/2 + 1 + a)
a13 = sp.gamma(1/5 + 3/4 + 1) / sp.gamma(1/5 + 3/4 + 1 + a)
a21 = a12
a22 = 1 / sp.gamma(2 + a)
a23 = sp.gamma(1/2 + 3/4 + 1) / sp.gamma(1/2 + 3/4 + 1 + a)
a31 = a13
a32 = a23
a33 = sp.gamma(3/2 + 1) / sp.gamma(3/2 + 1 + a)
b1 = sp.gamma(1/3 + 1/5 + 1) / sp.gamma(1/3 + 1/5 + 1 + a)
b2 = sp.gamma(1/3 + 1/2 + 1) / sp.gamma(1/3 + 1/2 + 1 + a)
b3 = sp.gamma(1/3 + 3/4 + 1) / sp.gamma(1/3 + 3/4 + 1 + a)
ec1 = C1 * a11 + C2 * a12 + C3 * a13 - b1
ec2 = C1 * a21 + C2 * a22 + C3 * a23 - b2
ec3 = C1 * a31 + C2 * a32 + C3 * a33 - b3

# Solve the equations
solution = sp.solve((ec1, ec2, ec3), (C1, C2, C3))

# Extract the values of C1, C2, and C3 from the solution
C1_val = solution[C1]
C2_val = solution[C2]
C3_val = solution[C3]

# Define functions f(t) and g(t)
def f(t):
    return t**(1/3)

def g(t):
    return C1_val*t**(1/5) + C2_val*t**(1/2) + C3_val*t**(3/4)

# Create a list of t values
t_values = np.arange(0, 1.2, 0.2)

import math

# Calculate and print |f(t) - g(t)| for different values of t
for t_val in t_values:
    diff = sp.Abs(f(t_val) - g(t_val))
    if not (diff == sp.oo or sp.Eq(diff, sp.nan)):
        if diff != 0:
            print(f'|f({t_val}) - g({t_val})| = {diff.evalf():.5f}')
        else:
            print(f'|f({t_val}) - g({t_val})| = 0.00000')
    else:
        print(f'|f({t_val}) - g({t_val})| is undefined or infinite.')

# Plot f(t) and g(t)
t_values = np.linspace(0, 1, 100)
f_values = [f(t_val) for t_val in t_values]
g_values = [g(t_val) for t_val in t_values]

plt.plot(t_values, f_values, label='f(t)')
plt.plot(t_values, g_values, label='g(t)')
plt.xlabel('t')
plt.ylabel('Value')
plt.legend()
plt.show()
