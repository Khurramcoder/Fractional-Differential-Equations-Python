import sympy as sp

# Define the symbolic variables 't' and 'y(t)'
t = sp.symbols('t')
y = sp.Function('y')

# Define the differential equation
ec = sp.Eq(y(t).diff(t), 1 + y(t)**2)

# Solve the differential equation
sol = sp.dsolve(ec, y(t), ics={y(0): 0})

# Compute the series solution up to the 10th order term
series_solution = sp.series(sol.rhs, t, 0, 10)

# Print the series solution
print(series_solution)

