#! /usr/bin/env python3

'''
Example of linear programming using PuLP.
'''


from pulp import *


# Create the linear programming problem object
problem = LpProblem('miracle_worker', LpMaximize)
# Create the linear programming variable objects
x = LpVariable('medicine_1_units', 0, None, LpInteger)
y = LpVariable('medicine_2_units', 0, None, LpInteger)
# Add the objective function
problem += 25*x + 20*y, 'health restored, to be maximized'
# Add the constraints
problem += 3*x + 4*y <= 25, 'herb A constraint'
problem += 2*x + y <= 10, 'herb B constraint'
# Write the problem data to a .lp file
problem.writeLP('miracleworker.lp')
# Solve the problem using PuLP's choice of solver
problem.solve()
