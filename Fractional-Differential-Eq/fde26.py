import numpy as np
import matplotlib.pyplot as plt

def RK4(f, x0, A, B, n, a):
    h = (B - A) / n
    Sol = [[A, *x0]]  # Initialize Sol as a list of lists
    x = x0
    for k in range(1, n+1):
        t = A + k * (h**a) / np.math.gamma(a + 1)
        K1 = (h**a) / np.math.gamma(a + 1) * np.array(f(t, x))
        K2 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (1/2) * (h**a) / np.math.gamma(a + 1), x + (1/2) * K1))
        K3 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (1/2) * (h**a) / np.math.gamma(a + 1), x + (1/2) * K2))
        K4 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (h**a) / np.math.gamma(a + 1), x + K3))
        x = x + (1/6) * K1 + (1/3) * K2 + (1/3) * K3 + (1/6) * K4
        Sol.append([t, *x])  # Append [t, x1, x2] as a sublist
    return Sol

def F(t, x):
    return [x[1], 2 * (1 - x[0]**2) * x[1] - x[0]]

A = 0.0
B = 100.0
n = 5000
a = 0.998
x0 = [0.0, 1.0]

Solution = RK4(F, x0, A, B, n, a)

# Extract x1 and x2 for the ring-like plot
x1_values = [x[1] for x in Solution]
x2_values = [x[2] for x in Solution]

# Create a ring-like plot
plt.figure(figsize=(8, 6))
plt.plot(x1_values, x2_values, color='blue')
plt.xlabel('x1(t)')
plt.ylabel('x2(t)')
plt.title('Ring-Like Plot of the Van der Pol System')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RK4(f, x0, A, B, n, a):
    h = (B - A) / n
    Sol = [[A, *x0]]  # Initialize Sol as a list of lists
    x = x0
    for k in range(1, n+1):
        t = A + k * (h**a) / np.math.gamma(a + 1)
        K1 = (h**a) / np.math.gamma(a + 1) * np.array(f(t, x))
        K2 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (1/2) * (h**a) / np.math.gamma(a + 1), x + (1/2) * K1))
        K3 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (1/2) * (h**a) / np.math.gamma(a + 1), x + (1/2) * K2))
        K4 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (h**a) / np.math.gamma(a + 1), x + K3))
        x = x + (1/6) * K1 + (1/3) * K2 + (1/3) * K3 + (1/6) * K4
        Sol.append([t, *x])  # Append [t, x1, x2, x3] as a sublist
    return Sol

def F(t, x):
    return [-x[1] - x[2], x[0] + 0.2 * x[1], 0.2 + x[0] * x[2] - 5.7 * x[2]]

A = 0.0
B = 200.0
n = 5000
a = 0.998
x0 = [1.0, 0.0, 0.0]

Solution = RK4(F, x0, A, B, n, a)

# Extract x1, x2, and x3 values
x1_values = [x[1] for x in Solution]
x2_values = [x[2] for x in Solution]
x3_values = [x[3] for x in Solution]

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1_values, x2_values, x3_values, color='blue')
ax.set_xlabel('x1(t)')
ax.set_ylabel('x2(t)')
ax.set_zlabel('x3(t)')
ax.set_title('Vector Solution of the Rössler Attractor')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RK4(f, x0, A, B, n, a):
    h = (B - A) / n
    Sol = [[A, *x0]]  # Initialize Sol as a list of lists
    x = x0
    for k in range(1, n+1):
        t = A + k * (h**a) / np.math.gamma(a + 1)
        K1 = (h**a) / np.math.gamma(a + 1) * np.array(f(t, x))
        K2 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (1/2) * (h**a) / np.math.gamma(a + 1), x + (1/2) * K1))
        K3 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (1/2) * (h**a) / np.math.gamma(a + 1), x + (1/2) * K2))
        K4 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (h**a) / np.math.gamma(a + 1), x + K3))
        x = x + (1/6) * K1 + (1/3) * K2 + (1/3) * K3 + (1/6) * K4
        Sol.append([t, *x])  # Append [t, x1, x2, x3] as a sublist
    return Sol

def F(t, x):
    return [-x[0] - 5 * x[1] - x[1] * x[2], -85 * x[0] - x[1] - x[0] * x[2], x[0] * x[1] + 0.5 * x[2] + 1]

A = 0.0
B = 100.0
n = 5000
a = 0.998
x0 = [8.0, 2.0, 1.0]

Solution = RK4(F, x0, A, B, n, a)

# Extract x1, x2, and x3 values
x1_values = [x[1] for x in Solution]
x2_values = [x[2] for x in Solution]
x3_values = [x[3] for x in Solution]

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1_values, x2_values, x3_values, color='blue')
ax.set_xlabel('x1(t)')
ax.set_ylabel('x2(t)')
ax.set_zlabel('x3(t)')
ax.set_title('Vector Solution of the Volta Attractor')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def RK4(f, x0, A, B, n, a):
    h = (B - A) / n
    Sol = [[A, *x0]]  # Initialize Sol as a list of lists
    x = x0
    for k in range(1, n+1):
        t = A + k * (h**a) / np.math.gamma(a + 1)
        K1 = (h**a) / np.math.gamma(a + 1) * np.array(f(t, x))
        K2 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (1/2) * (h**a) / np.math.gamma(a + 1), x + (1/2) * K1))
        K3 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (1/2) * (h**a) / np.math.gamma(a + 1), x + (1/2) * K2))
        K4 = (h**a) / np.math.gamma(a + 1) * np.array(f(t + (h**a) / np.math.gamma(a + 1), x + K3))
        x = x + (1/6) * K1 + (1/3) * K2 + (1/3) * K3 + (1/6) * K4
        Sol.append([t, *x])  # Append [t, x1, x2, x3] as a sublist
    return Sol

def F(t, x):
    return [10 * (x[1] - x[0]), x[0] * (28 - x[2]) - x[1], x[0] * x[1] - 8/3 * x[2]]

A = 0.0
B = 100.0
n = 5000
a = 0.998
x0 = [1.0, 1.0, 1.0]

Solution = RK4(F, x0, A, B, n, a)

# Extract x1, x2, and x3 values
x1_values = [x[1] for x in Solution]
x2_values = [x[2] for x in Solution]
x3_values = [x[3] for x in Solution]

# Create a 3D plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x1_values, x2_values, x3_values, color='blue')
ax.set_xlabel('x1(t)')
ax.set_ylabel('x2(t)')
ax.set_zlabel('x3(t)')
ax.set_title('Vector Solution of the Lorenz Attractor')
plt.show()


