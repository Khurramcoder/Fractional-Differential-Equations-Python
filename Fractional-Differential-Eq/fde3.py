import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

# Define the symbolic variable 't'
t = sp.symbols('t')

# Define the function 'f[t]'
f = t + t**3/6 + t**4/12 + t**5/120

# Define the 'Manipulate' block with a slider for 'a'
def plot_function(a):
    # Define the function 'y[t]' using the provided formula
    y = t + t**(a + 1)/sp.gamma(a + 2) + 2*t**(a + 2)/sp.gamma(a + 3) + \
        2*(a + 2)*t**(2*a + 2)/sp.gamma(2*a + 3) + \
        4*(a + 3)*t**(2*a + 3)/sp.gamma(2*a + 4)

    # Convert the symbolic functions to Python functions
    f_lambda = sp.lambdify(t, f, modules=['numpy'])
    y_lambda = sp.lambdify(t, y, modules=['numpy'])

    # Generate 't' values for the plot
    t_values = np.linspace(0, 1, 100)

    # Evaluate the functions at 't' values
    f_values = f_lambda(t_values)
    y_values = y_lambda(t_values)

    # Plot the functions
    plt.figure(figsize=(6, 4))
    plt.plot(t_values, f_values, label='f[t]')
    plt.plot(t_values, y_values, label='y[t]')
    plt.xlabel('t')
    plt.ylabel('Function Values')
    plt.legend()
    plt.grid(True)
    plt.show()

# Use the 'interact' function to create a slider for 'a'
from ipywidgets import interact
interact(plot_function, a=(0, 1, 0.01))






