import matplotlib.pyplot as plt
import numpy as np
from fracdiff import FD
from scipy.integrate import solve_ivp

plt.style.use('seaborn-poster')
#%matplotlib inline

# Define the fractional differential equation D^alpha S(t) = -sin(t)
alpha = 0.5  # Fractional order
F = lambda t, s: -np.sin(t)

# Time span and initial condition
t_span = (0, np.pi)
initial_condition = [0]

# Create a time array for evaluation
t_eval = np.arange(t_span[0], t_span[1], 0.1)

# Create a fractional differential operator
fd = FD(alpha, t_span, kernel='gaussian')

# Solve the fractional differential equation using solve_ivp
sol = solve_ivp(fd.diff_eq, t_span, initial_condition, t_eval=t_eval)

# Plot the solution
plt.figure(figsize=(12, 4))
plt.subplot(121)
plt.plot(sol.t, sol.y[0])
plt.xlabel('t')
plt.ylabel('S(t)')
plt.title('Fractional Differential Equation Solution')
plt.subplot(122)
plt.plot(sol.t, sol.y[0] - np.sin(sol.t))
plt.xlabel('t')
plt.ylabel('S(t) - sin(t)')
plt.title('Error')
plt.tight_layout()
plt.show()
