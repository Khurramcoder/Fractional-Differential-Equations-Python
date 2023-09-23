import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln, gamma

# Suppress overflow warnings for np.power
np.seterr(over='ignore')

a = 2
b = 1
alpha = 1.5

t = np.arange(0, 20.001, 0.001)  # for time step 0.001 and computational time 20 sec

def safe_power(x, n):
    # Calculate x^n with safety checks to prevent overflow
    if n % 2 == 0:
        return np.where(x >= 0, np.power(x, n), np.power(-x, n))
    else:
        return np.power(x, n)

def mlf(alpha, beta, t):
    # Calculate the Mittag-Leffler function using the integral representation
    integral = 0
    for n in range(1000):  # Number of terms for approximation
        power_term = safe_power(t, n)
        gamma_term = gamma(alpha * n + beta)
        invalid_mask = np.logical_or(np.isinf(power_term), np.isnan(gamma_term))
        if invalid_mask.any():
            break  # Exit loop if overflow or invalid value
        term = power_term / gamma_term
        integral += term
    return integral

y = (1/a) * safe_power(t, alpha) * mlf(alpha, alpha + 1, (-b/a) * safe_power(t, alpha))

plt.plot(t, y)
plt.xlabel('Time [sec]')
plt.ylabel('y(t)')
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt

a2 = 0.8
a1 = 0.5
a0 = 1.0
alpha2 = 2.2
alpha1 = 0.9
h = 0.05
TSim = 35

n = round(TSim / h)
cp1 = 1
cp2 = 1
Y0 = 0
u = 1.0

c1 = np.zeros(n)
c2 = np.zeros(n)

for j in range(n):
    c1[j] = (1 - (1 + alpha1) / (j + 1)) * cp1
    c2[j] = (1 - (1 + alpha2) / (j + 1)) * cp2
    cp1 = c1[j]
    cp2 = c2[j]

Y = np.zeros(n)
Y[0] = Y0

def memo(r, c, k):
    temp = 0
    for j in range(k):
        temp += c[j] * r[k - j - 1]
    return temp

for i in range(1, n):
    Y[i] = (u - (a2) * h**(-alpha2) * memo(Y, c2, i) -
            (a1) * h**(-alpha1) * memo(Y, c1, i)) / (a2 / (h**alpha2) + a1 / (h**alpha1) + a0)

T = np.arange(0, TSim, h)  # Adjusted to match the length of Y

plt.plot(T, Y)
plt.xlabel('Time [sec]')
plt.ylabel('y(t)')
plt.grid(True)
plt.show()


import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Bagley-Torvik equation: alpha*D^2 y(t) + A*D^alpha y(t) + B*y(t) = C*F(t)
alpha = 1.5
A = 1
B = 0.5
C = 1

# Define the time interval and step size
T = np.arange(0, 40.05, 0.05)

# Define the input signal F(t)
F = np.piecewise(T, [T <= 1, T > 1], [8, 8])

# Function representing the Bagley-Torvik equation
def bagley_torvik(y, t):
    # Compute the derivatives
    dydt = np.zeros_like(y)
    dydt[0] = y[1]
    
    # Add a small constant to avoid division by zero
    epsilon = 1e-6
    
    dydt[1] = (C * F[int(t / 0.05)] - A * np.sign(y[1]) * np.abs(y[1]) ** alpha - B * y[0]) / (alpha * (np.abs(y[1]) + epsilon) ** (alpha - 1))
    return dydt

# Initial conditions
y0 = [0, 0]

# Solve the Bagley-Torvik equation
solution = spi.odeint(bagley_torvik, y0, T)

# Extract the response variable
y = solution[:, 0]

# Plot the solution
plt.figure()
plt.plot(T, y)
plt.xlabel('Time [sec]')
plt.ylabel('y(t)')
plt.title('Bagley-Torvik Equation')
plt.grid(True)
plt.show()



import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Continuous-time system
num_Gs_DCM = [0.08]
den_Gs_DCM = [0.05, 1, 0]
Gs_DCM = ctrl.TransferFunction(num_Gs_DCM, den_Gs_DCM)

# Fractional-order controller design
omega_b = 1e-2
omega_h = 1e2
N = 6

# Create a fractional-order controller manually
s = ctrl.TransferFunction([1, 0], [1])
num_Cs = (
    0.625 * np.prod([(1 + 2 * np.pi * omega_b / (10 ** (i / N)) + 1j * 2 * np.pi * omega_h) for i in range(N)]) +
    12.5 * np.prod([(1 + 2 * np.pi * omega_b / (10 ** (i / N)) - 1j * 2 * np.pi * omega_h) for i in range(N)])
)
den_Cs = (
    np.prod([(1 + 2 * np.pi * omega_b / (10 ** (i / N)) + 1j * 2 * np.pi * omega_h) for i in range(N)]) +
    np.prod([(1 + 2 * np.pi * omega_b / (10 ** (i / N)) - 1j * 2 * np.pi * omega_h) for i in range(N)])
)
Cs = ctrl.TransferFunction(num_Cs, den_Cs)

# Closed-loop system
Gs_close = Gs_DCM * Cs

# Step response
time = np.linspace(0, 15, 1000)
t, y = ctrl.step_response(Gs_close, time)
plt.figure()
plt.plot(t, y)
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Closed-Loop Step Response')
plt.grid()

# Bode plot
omega = np.logspace(-2, 2, 1000)
mag, phase, omega = ctrl.bode(Gs_DCM * Cs, omega)
plt.figure()
plt.semilogx(omega, mag)
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Magnitude [dB]')
plt.title('Bode Plot (Magnitude)')
plt.grid()
plt.figure()
plt.semilogx(omega, phase)
plt.xlabel('Frequency [rad/s]')
plt.ylabel('Phase [degrees]')
plt.title('Bode Plot (Phase)')
plt.grid()

# Margin calculation
gm, pm, _, _ = ctrl.margin(Gs_DCM * Cs)
print(f"Gain Margin (dB): {20 * np.log10(gm)}")
print(f"Phase Margin (degrees): {pm}")

plt.show()



















