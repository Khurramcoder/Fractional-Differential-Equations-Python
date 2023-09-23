import sympy as sp

# Define symbolic variables and functions
lambda_ = sp.symbols('lambda')

# Create a list of symbolic variables y[0] to y[4]
n = 5  # Number of indexed variables
y = [sp.symbols(f'y{i}') for i in range(n)]

# Define the function f[y]
f = y[0]**2

# Define the series S[lambda]
S = sum([y[i] * lambda_**i for i in range(5)])

# Define the function g[lambda]
g = f.subs({y[0]: S})

# Calculate the derivatives and substitute lambda = 0
ad = []
for j in range(6):  # Use a different variable for the loop
    derivative = sp.diff(g, lambda_, j).subs(lambda_, 0)
    ad.append(derivative)

# Simplify the results
ad = [sp.simplify(derivative) for derivative in ad]

# Print the results as a table
table = sp.Matrix(ad)
print(table)
