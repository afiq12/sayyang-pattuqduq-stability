"""Symbolic check of the Euler-Lagrange equations for the pendulum on a vertically moving pivot (Sec. 2.1)."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
import sympy as sp
t=sp.symbols('t'); m,l,g=sp.symbols('m l g',positive=True)
th=sp.Function('theta')(t); y=sp.Function('y')(t)
def EL(L): return sp.simplify(sp.diff(sp.diff(L,th.diff(t)),t)-sp.diff(L,th))
# Mixing the two forms. With the cross term of L1 (minus sign) plus the ydd term of L2 the drive is doubled;
# with the cross term written with a plus sign the two cancel. Either way the forms must not be mixed.
L_mix_minus=sp.Rational(1,2)*m*l**2*th.diff(t)**2 - m*l*y.diff(t)*th.diff(t)*sp.sin(th) - m*g*l*sp.cos(th) - m*y.diff(t,2)*l*sp.cos(th)
L_mix_plus=sp.Rational(1,2)*m*l**2*th.diff(t)**2 + m*l*y.diff(t)*th.diff(t)*sp.sin(th) - m*g*l*sp.cos(th) - m*y.diff(t,2)*l*sp.cos(th)
L1=sp.Rational(1,2)*m*l**2*th.diff(t)**2 - m*l*y.diff(t)*th.diff(t)*sp.sin(th) - m*g*l*sp.cos(th)
L2=sp.Rational(1,2)*m*l**2*th.diff(t)**2 - m*(g+y.diff(t,2))*l*sp.cos(th)
# check L1 from kinematics
x=l*sp.sin(th); z=y+l*sp.cos(th)
Lk=sp.Rational(1,2)*m*(x.diff(t)**2+z.diff(t)**2)-m*g*z
print("mixed, cross term -:",EL(L_mix_minus))
print("mixed, cross term +:",EL(L_mix_plus))
print("L1 EOM    :",EL(L1))
print("L2 EOM    :",EL(L2))
print("kinematic :",EL(Lk))
