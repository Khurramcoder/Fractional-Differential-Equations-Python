import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk

# Define the symbolic variable 't'
t = sp.symbols('t')

# Define the function 'f[t]'
f = t + t**3/6 + t**4/12 + t**5/120

# Create a function to update the plot based on the slider value
def update_plot(a):
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

    # Clear the previous plot
    ax.clear()

    # Plot the functions
    ax.plot(t_values, f_values, label='f[t]')
    ax.plot(t_values, y_values, label='y[t]')
    ax.set_xlabel('t')
    ax.set_ylabel('Function Values')
    ax.legend()
    ax.grid(True)

# Create a Tkinter window
root = tk.Tk()
root.title("Interactive Plot")

# Create a slider control for 'a' without specifying the resolution
a_slider = ttk.Scale(root, from_=0, to=1, orient="horizontal", length=200)
a_slider.set(0.5)  # Initial value of 'a'
a_slider.pack()

# Create a canvas for the plot using FigureCanvasTkAgg
fig, ax = plt.subplots(figsize=(6, 4))
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack()

# Create a function to update the plot with the current slider value
def update_with_slider(val):
    a = a_slider.get()
    update_plot(a)
    canvas.draw()

# Link the slider to the update function
a_slider.configure(command=update_with_slider)

# Initialize the plot with the initial 'a' value
update_plot(a_slider.get())

# Start the Tkinter main loop
root.mainloop()
