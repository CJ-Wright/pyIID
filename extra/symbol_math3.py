__author__ = 'christopher'
from sympy import *


Q = symbols('Q')
r = symbols('r')
gob, x, a, b, r = symbols('Gobs x a b r')

rmin, rmax = symbols('rmin rmax')
gcalc = Function('gcalc')(x, r)
Qmin, Qmax = symbols('Qmin Qmax')
chi = symbols('chi**2')
als = Function('alpha')(x,r)
al = 1/summation(gcalc*gcalc, (r, rmin, rmax)) * summation(gcalc * gob(r), (r, rmin, rmax))
# al = (gcalc*gcalc)**-1*(gcalc*gob)

Chi = summation((gob(r)-als*gcalc)**2, (r, rmin, rmax))
print(latex(Chi))
sol = diff(Chi, x).simplify()
ssol = sol.subs(sqrt(summation((gob(r) - als * gcalc)**2, (r, rmin, rmax))/summation(gob(r)**2, (r, rmin, rmax))), chi)
# ssol = ssol.subs(al, als)
# pprint(ssol)
print(latex(ssol.simplify()))
print('\n\n\n')
print(latex(diff(al, x).subs(al, als).simplify()))