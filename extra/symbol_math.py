__author__ = 'christopher'
from sympy import *

uix, uiy, uiz, ujx, ujy, ujz, sax, say, saz, sbx, sby, sbz = symbols('uix, uiy, uiz, ujx, ujy, ujz, sax, say, saz, sbx, sby, sbz', real = True)
Q = symbols('Q')
r = symbols('r')
sx, sy, sz = symbols('s0 s1 s2')
ux, uy, uz = symbols('u0 u1 u2')
#thermal displacements
ui = Matrix([uix, uiy, uiz])
uj = Matrix([ujx, ujy, ujz])
#position vectors
sa = Matrix([sax, say, saz])
sb = Matrix([sbx, sby, sbz])
d = sa-sb
rij = d.T.norm()
rijv = d/rij
# pprint(rijv)
# pprint(ui-uj)
u = (ui-uj).T
s = (u*rijv)[0]
s2 = s**2
# pprint(s2)

subs_d = {rij: r, sax-sbx: sx, say-sby: sy, saz-sbz: sz,
                    uix-ujx: ux, uiy-ujy: uy, uiz-ujz: uz}
# sub_fq = exp(-.5*s2*Q**2)*sin(Q*rij)/rij
sub_fq = sin(Q*rij)/rij
# pprint(sub_fq.subs(subs_d))

dir_fq = diff(sub_fq, sax)
# pprint(dir_fq)
dfq_subs = dir_fq.subs(subs_d).simplify()
# #print(latex(dfq_subs))
# pprint(dfq_subs)

u_dir = diff(sub_fq, uix)
du_subs = u_dir.subs(subs_d).simplify()
# #print(latex(du_subs))
# pprint(du_subs)


gob, x, a, b, r = symbols('Gobs x a b r')
rmin, rmax = symbols('rmin rmax')
gcalc = Function('gcalc')(x, r)
Qmin, Qmax = symbols('Qmin Qmax')
RW = symbols('RW')
al = symbols('alpha')
Rw = sqrt(summation((gob(r)-al(r)*gcalc)**2, (r, rmin, rmax))/summation(gob(r)**2,
          (r, rmin, rmax)))
# Rw = sqrt(summation((gob(r)-gcalc)**2, (r, rmin, rmax))/summation(gob(r)**2,
#           (r, rmin, rmax)))
# Rw = sqrt(summation((gob(r)-gcalc)**2/gob(r)**2,
#           (r, rmin, rmax)))

Rw = summation((gob(r)-gcalc)**2, (r, rmin, rmax))

pprint(Rw)
# print(latex(Rw))
sol = diff(Rw, x).simplify().subs(sqrt(summation((gob(r)-gcalc)**2, (r, rmin, rmax))/summation(gob(r)**2, (r, rmin, rmax))), RW)
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