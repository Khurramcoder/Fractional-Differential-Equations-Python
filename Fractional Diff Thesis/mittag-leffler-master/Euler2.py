import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

# The inputs
h = 1/10
t = [0]
x = [1.2]
y = [4.2]
alpha = 0.85
tfinal = 1
N = int(np.ceil((tfinal - t[0]) / h))

# Exact Solution
def mlf(alpha, beta, t):
    return t**beta / gamma(beta + 1)

tExact = np.linspace(0, tfinal, N+1)
xExact = (1/5) * mlf(alpha, 1, (tExact**alpha)) + mlf(alpha, 1, (-2*tExact**alpha))
yExact = (1/5) * mlf(alpha, 1, (tExact**alpha)) + 4 * mlf(alpha, 1, (-2*tExact**alpha))

# Fractional-Order System of ODES
def f1(t, x, y):
    return 2*x - y

def f2(t, x, y):
    return 4*x - 3*y

# The Fractional Forward Euler Method for a System of ODES
for n in range(N):
    t_new = t[-1] + h
    t.append(t_new)
    
    x_new = x[0] + ((h**alpha) / gamma(alpha+1)) * np.sum(((n+1-np.arange(1, n+2))**alpha - (n-np.arange(1, n+2))**alpha) * f1(t[-1], x[-1], y[-1]))
    x.append(x_new)
    
    y_new = y[0] + ((h**alpha) / gamma(alpha+1)) * np.sum(((n+1-np.arange(1, n+2))**alpha - (n-np.arange(1, n+2))**alpha) * f2(t[-1], x[-1], y[-1]))
    y.append(y_new)

# Absolute Errors
xErrors = np.abs(xExact - x)
yErrors = np.abs(yExact - y)

xLast_Error = xErrors[-1]
yLast_Error = yErrors[-1]
xMax_Error = np.max(xErrors)
yMax_Error = np.max(yErrors)

print("xLast_Error:", xLast_Error)
print("yLast_Error:", yLast_Error)
print("xMax_Error:", xMax_Error)
print("yMax_Error:", yMax_Error)

# Plotting the results
plt.figure()
plt.plot(tExact, xExact, label="Exact x(t)")
plt.plot(tExact, yExact, label="Exact y(t)")
plt.plot(t, x, label="Approximated x(t)", linestyle='dashed')
plt.plot(t, y, label="Approximated y(t)", linestyle='dashed')
plt.xlabel("t")
plt.ylabel("x(t) and y(t)")
plt.legend()
plt.grid()
plt.show()


