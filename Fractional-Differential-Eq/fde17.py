import sympy as sp

# Define the symbolic variable t
t = sp.symbols('t')

# Define the symbolic functions B0, B1, and B2
B0 = 1
B1 = t
B2 = t**2

# Define the symbolic expressions A, B, and C1
A = -t**(3/2)*B0
B = -t**(3/2)*B1
C1 = 4*sp.sqrt(t)*sp.sqrt(sp.pi) - t**(3/2)*B2

# Define the symbolic functions f(beta)
def f(beta):
    return sp.gamma(beta + 1)/sp.gamma(beta + 5/2)

# Compute A11, A12, A13, A21, A22, A23, A31, A32, and A33
A11 = f(3)
A12 = f(4)
A13 = f(5) - 4/sp.sqrt(sp.pi)*f(2)
A21 = A12
A22 = f(5)
A23 = f(6) - 4/sp.sqrt(sp.pi)*f(3)
A31 = A13
A32 = A23
A33 = f(7) - 8/sp.sqrt(sp.pi)*f(4) + 16/sp.pi*f(1)

# Define the symbolic expressions f1, f2, and f3
f1 = (4/sp.sqrt(sp.pi)*t**(1/2) - t**(7/2))*A
f2 = (4/sp.sqrt(sp.pi)*t**(1/2) - t**(7/2))*B
f3 = (4/sp.sqrt(sp.pi)*t**(1/2) - t**(7/2))*C1

# Compute F1, F2, and F3
F1 = f(5) - 4/sp.sqrt(sp.pi)*f(2)
F2 = f(6) - 4/sp.sqrt(sp.pi)*f(3)
F3 = f(7) - 8/sp.sqrt(sp.pi)*f(4)

# Define the symbolic equations ec1, ec2, and ec3
x, y, z = sp.symbols('x y z')
ec1 = A11*x + A12*y + A13*z
ec2 = A21*x + A22*y + A23*z
ec3 = A31*x + A32*y + A33*z

# Solve the system of equations
solutions = sp.solve([ec1 - F1, ec2 - F2, ec3 - F3], (x, y, z))

print(solutions)




from sympy import symbols, sqrt, gamma, Eq, solve, pi

# Define symbols
t, A, B = symbols('t A B')

# Define the function f
def f(x):
    return gamma(x + 1) / gamma(x + 3/2)

# Calculate f1 and f2
f1 = -f(1/2).evalf()
f2 = -f(3/2).evalf()

# Define Ya and DYa
Ya = A*sqrt(t) + B*t**(3/2)
DYa = A*sqrt(t)*sqrt(pi)/2 + B*3*sqrt(pi)/4*t

# Define the equation for g
g = DYa - Ya**2

# Define the equations ec1 and ec2
ec1 = 1/2*A*sqrt(pi)*f(1/2) + 3/4*B*sqrt(pi)*f(3/2) - A**2*f(3/2) - 2*A*B*f(5/2) - B**2*f(7/2)
ec2 = 1/2*A*sqrt(pi)*f(3/2) + 3/4*B*sqrt(pi)*f(5/2) - A**2*f(5/2) - 2*A*B*f(7/2) - B**2*f(9/2)

# Solve the equations
solutions = solve((ec1 - f1, ec2 - f2), (A, B))

print(solutions)




