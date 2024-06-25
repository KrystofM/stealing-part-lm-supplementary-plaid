# # Define the system of equations
# def equations(vars):
#     c, d, e, f = vars    
#     eq1 = a * b * c - 0.2
#     eq2 = (1 - a) * d * e - 0.2
#     eq3 = (1 - b) * (1 - d) * f - 0.2
#     eq4 = (1 - c) * (1 - e) * (1 - f) - 0.2
#     return [eq1, eq2, eq3, eq4]

# # Initial guess for the variables
# initial_guess = [0, 0, 0, 0]
# a = 0.5
# b = 0.5
# # Solve the system of equations
# solution = least_squares(equations, initial_guess, bounds=(0, 1), xtol=1, ftol=1, gtol=1, verbose=1)

# # Print the solution
# c, d, e, f = solution.x
# print(f"a = {a}, b = {b}, c = {c}, d = {d}, e = {e}, f = {f}")
# eq1 = a * b * c - 1/5
# eq2 = (1 - a) * d * e - 1/5
# eq3 = (1 - b) * (1 - d) * f - 1/5
# eq4 = (1 - c) * (1 - e) * (1 - f) - 1/5

# print(f"eq1: {eq1}")
# print(f"eq2: {eq2}")
# print(f"eq3: {eq3}")
# print(f"eq4: {eq4}")

from sympy import symbols, Eq, solve

# Define the variables
c, d, e, f = symbols('c, d, e, f')
a = 0.5
b = 0.5

# Define the system of equations
eq1 = Eq(a * b * c, 1/4)
eq2 = Eq((1 - a) * d * e, 1/4)
eq3 = Eq((1 - b) * (1 - d) * f, 1/4)
eq4 = Eq((1 - c) * (1 - e) * (1 - f), 1/4)

# Solve the system of equations
solution = solve([eq1, eq2, eq3, eq4], (c, d, e, f))
print(solution)
# Print the solution
# print(f"a = {a}, b = {b}, c = {solution[0]}, d = {solution[1]}, e = {solution[2]}, f = {solution[3]}")

