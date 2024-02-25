import sympy
import sympy as sp

t, s, q, cai, Ti, V, p, C, deltaHr, EdivR, k0, UA = sp.symbols('t s q cai Ti V p C deltaHr EdivR k0 UA', real=True)

ca = sp.Function('ca')
temp = sp.Function('temp')

eq = sp.Eq(-V*sp.diff(ca(t), (t, 1))+(q*(cai-ca(t))-V*k0*sp.exp(-EdivR*(1/temp(t)))*ca(t)), 0)

tf = sympy.laplace_transform(eq.lhs, t, s, simplify=True)

print(tf)