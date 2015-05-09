__author__ = 'christopher'
from sympy import *


Q = symbols('Q')
r = symbols('r')
gob, x, a, b, r = symbols('Gobs x a b r')

rmin, rmax = symbols('rmin rmax')
gcalc = Function('gcalc')(x, r)
Qmin, Qmax = symbols('Qmin Qmax')
RW = symbols('RW')
al = symbols('alpha')
al = 1/summation(gcalc*gcalc, (r, rmin, rmax)) * summation(gcalc * gob(r), (r, rmin, rmax))

Rw = sqrt(summation((gob(r)-al*gcalc)**2, (r, rmin, rmax))/summation(gob(r)**2,
          (r, rmin, rmax)))
# Rw = sqrt(summation((gob(r)-al(x)*gcalc)**2, (r, rmin, rmax))/summation(gob(r)**2,
#           (r, rmin, rmax)))
# pprint(Rw)
# print(latex(Rw))
sol = diff(Rw, x).simplify()
# .subs(sqrt(summation((gob(r) - al * gcalc)**2, (r, rmin, rmax))/summation(gob(r)**2, (r, rmin, rmax))), RW)
pprint(sol)
dg = symbols('Delta_g')

# print(latex(sol))
pprint(sol.subs((al*gcalc-(gob(r))), dg))
pprint(dg)
'''
F = Function('F')(Q, x)
gcalc = 2/pi*Integral(F*sin(Q*r), (Q,Qmin, Qmax))

#print(latex(gcalc))
#print(latex(diff(gcalc, x)))
i, j = symbols(' i j')
N = symbols('N')
FQ = 1/(N*b**2)*summation(b(i,Q)*b(j, Q) * sin(Q*r(i,j, x))/r(i,j, x),(j, 0, j))
#print(latex(FQ))
#print(latex(diff(FQ, x).simplify()))
pprint(diff(FQ, x).simplify())

#print(latex(rij))
#print(latex(diff(rij, sax)))

pprint(FQ)'''