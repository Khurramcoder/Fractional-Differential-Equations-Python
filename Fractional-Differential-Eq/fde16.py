from sympy import symbols, gamma, Eq, solve, N

a = 1/2
P = 1/gamma(2 - a)
Q = 2/gamma(3 - a)
C1 = 6/gamma(4 - a)

t, B, C11, C2 = symbols('t B C11 C2')
v1 = P*t - B*t**2
v2 = P*t**2 - C1*t**3

B11 = P**2*gamma(3 - 2*a)/gamma(3 - a) - 2*P*Q*gamma(4 - 2*a)/gamma(4 - a) + Q**2*gamma(5 - 2*a)/gamma(5 - a)
B12 = P*Q*gamma(4 - 2*a)/gamma(4 - a) - P*C1*gamma(5 - 2*a)/gamma(5 - a) - Q**2*gamma(5 - 2*a)/gamma(5 - a) + Q*C1*gamma(6 - 2*a)/gamma(6 - a)
B22 = Q**2*gamma(5 - 2*a)/gamma(5 - a) - 2*Q*C1*gamma(6 - 2*a)/gamma(6 - a) + C1**2*gamma(7 - 2*a)/gamma(7 - a)

F1 = -P*gamma(2 - a)/gamma(2) + Q*gamma(3 - a)/gamma(3) + P*(a + 1)*gamma(3 - a)/gamma(3) - Q*(a + 1)*gamma(4 - a)/gamma(4)
F2 = -Q*gamma(3 - a)/gamma(3) + C1*gamma(4 - a)/gamma(4) + Q*(a + 1)*gamma(4 - a)/gamma(4) - C1*(a + 1)*gamma(5 - a)/gamma(5)

ec1 = C11*B11 + C2*B12
ec2 = C11*B12 + C2*B22

solutions = solve([Eq(ec1, F1), Eq(ec2, F2)], (C1, C2))

for solution in solutions:
    C1_value, C2_value = N(solution[C1]), N(solution[C2])
    print(f"For α = {a}, we obtain:")
    print(f"C1 = {C1_value}, C2 = {C2_value}")

