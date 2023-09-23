import sympy as sp

# Define the symbolic variables
t = sp.symbols('t')
y = sp.Function('y')(t)

# Define the differential equation
ec = sp.Eq(sp.diff(y, t) + y, 1)

# Solve the differential equation
sol = sp.dsolve(ec, y, ics={y.subs(t, 0): 0})

# Print the solution
print("Solution:")
print(sol)

# Compute the series expansion
series_solution = sp.series(sol.rhs, t, 0, 10)

# Print the series expansion
print("\nSeries Expansion:")
print(series_solution)

import sympy as sp

# Define the symbolic variables
t, s = sp.symbols('t s')

# Define the Laplace transform variable
Y = sp.Function('Y')(t)

# Define the differential equation in the Laplace domain
laplace_eq = sp.Eq(s**2 * sp.LaplaceTransform(Y, t, s) - Y, 0)

# Solve the Laplace-transformed equation for Y(s)
laplace_solution = sp.solve(laplace_eq, sp.LaplaceTransform(Y, t, s))

# Invert the Laplace transform to get the solution in the time domain
solution = sp.inverse_laplace_transform(laplace_solution[0], s, t)

# Print the solution
print("Solution:")
print(solution)





































