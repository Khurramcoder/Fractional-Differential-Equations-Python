import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, fresnel

# Define fractional cosine function
def fractional_cos(q, t):
    sqrt_pi = np.sqrt(np.pi)
    C, _ = fresnel(np.sqrt(2 * t / np.pi))
    return (1 / gamma(q)) * sqrt_pi * C

# Define fractional sine function
def fractional_sin(q, t):
    sqrt_pi = np.sqrt(np.pi)
    _, S = fresnel(np.sqrt(2 * t / np.pi))
    return (1 / gamma(q)) * sqrt_pi * S

# Define the range of t values
t_values = np.linspace(0, 10, 100)  # Adjust the range as needed

# Choose a value for q (0 < q <= 1)
q = 0.5

# Calculate fractional cosine and sine values
cos_values = [fractional_cos(q, t) for t in t_values]
sin_values = [fractional_sin(q, t) for t in t_values]

# Plot the results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_values, cos_values, label=f'Cos Fractional (q={q})')
plt.title('Fractional Cosine Function')
plt.xlabel('t')
plt.ylabel('Cos Fractional')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_values, sin_values, label=f'Sin Fractional (q={q})', color='orange')
plt.title('Fractional Sine Function')
plt.xlabel('t')
plt.ylabel('Sin Fractional')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, fresnel

# Define fractional cosine function
def fractional_cos(q, t):
    sqrt_pi = np.sqrt(np.pi)
    C, _ = fresnel(np.sqrt(2 * t / np.pi))
    return (1 / gamma(q + 0.5)) * sqrt_pi * C

# Define fractional sine function
def fractional_sin(q, t):
    sqrt_pi = np.sqrt(np.pi)
    _, S = fresnel(np.sqrt(2 * t / np.pi))
    return (1 / gamma(q + 0.5)) * sqrt_pi * S

# Define the range of t values
t_values = np.linspace(0, 10, 100)  # Adjust the range as needed

# Choose a value for q (0 < q <= 1)
q = 0.5

# Calculate fractional cosine and sine values
cos_values = [fractional_cos(q, t) for t in t_values]
sin_values = [fractional_sin(q, t) for t in t_values]

# Plot the results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_values, cos_values, label=f'Cos Fractional (q={q})')
plt.title('Fractional Cosine Function')
plt.xlabel('t')
plt.ylabel('Cos Fractional')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_values, sin_values, label=f'Sin Fractional (q={q})', color='orange')
plt.title('Fractional Sine Function')
plt.xlabel('t')
plt.ylabel('Sin Fractional')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import gamma

# Define fractional cosine function using numerical integration
def fractional_cos(q, t):
    def integrand(s):
        return (t - s)**(q - 1) * np.cos(s**2)
    
    integral, _ = quad(integrand, 0, t)
    sqrt_pi = np.sqrt(np.pi)
    return (1 / gamma(q)) * sqrt_pi * integral

# Define fractional sine function using numerical integration
def fractional_sin(q, t):
    def integrand(s):
        return (t - s)**(q - 1) * np.sin(s**2)
    
    integral, _ = quad(integrand, 0, t)
    sqrt_pi = np.sqrt(np.pi)
    return (1 / gamma(q)) * sqrt_pi * integral

# Define the range of t values
t_values = np.linspace(0, 10, 100)  # Adjust the range as needed

# Choose a value for q (0 < q <= 1)
q = 0.5

# Calculate fractional cosine and sine values
cos_values = [fractional_cos(q, t) for t in t_values]
sin_values = [fractional_sin(q, t) for t in t_values]

# Plot the results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(t_values, cos_values, label=f'Cos Fractional (q={q})')
plt.title('Fractional Cosine Function')
plt.xlabel('t')
plt.ylabel('Cos Fractional')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t_values, sin_values, label=f'Sin Fractional (q={q})', color='orange')
plt.title('Fractional Sine Function')
plt.xlabel('t')
plt.ylabel('Sin Fractional')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()























