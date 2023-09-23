import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from ipywidgets import interact, widgets

# Define the functions 'f[t]' and 'y[t]'
def f(t):
    return t**2 + t**4/6

def y(t, a):
    return 2*t**(2*a)/gamma(2*a + 1) + 0.1*t**(4*a)/gamma(4*a + 1)  # Adjusted the scaling factor for visibility

# Create an interactive plot using ipywidgets
def plot(a=0.5):
    t_values = np.linspace(0, 1, 100)
    f_values = f(t_values)
    y_values = y(t_values, a)

    plt.figure(figsize=(6, 4))
    plt.plot(t_values, f_values, label='f[t]')
    plt.plot(t_values, y_values, label='y[t]')
    plt.xlabel('t')
    plt.ylabel('Function Values')
    plt.legend()
    plt.grid(True)
    plt.show()

interact(plot, a=widgets.FloatSlider(min=0, max=1, step=0.01, value=0.5))
