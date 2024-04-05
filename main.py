import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import control as ctrl

# Función escalón para representar los caudales de entrada
def escalon(t, t0, t1, a0, a):
    if t >= t0 and t1 > t:
        return a
    else: return a0

def edo(t, y):
    # Igualamos variables a condiciones iniciales
    ca = y[0]
    T = y[1]
    # Definimos parámetros del sistema
    t0 = 0  # Tiempo inicial en minutos
    t1 = 10  # Tiempo final en minutos
    q = 100.  # L/min
    cai = 1.  # mol/L
    Ti = 350.  # K
    V = 100.  # L
    p = 1000.  # g/L
    C = 0.239  # J/g K
    deltaHr = -5e4  # J/mol
    EdivR = 8750.  # K
    k0 = 7.2e10  # min^-1
    UA = 5e4  # J/min K
    num = -EdivR * (1 / T)
    k = k0 * np.exp(num)
    w = q * p
    # Definimos entradas
    Tc = escalon(t, t0, t1, 300, 290)
    return [(q*(cai-ca)-V*k*ca)/V,
           (w*C*(Ti-T)+(-deltaHr)*V*k*ca+UA*(Tc-T))/(V*p*C)]

def lineal(t, y):

    # Igualamos variables a condiciones iniciales
    ca = y[0]
    T = y[1]

    # Definimos parámetros del sistema
    t0 = 0  # Tiempo inicial en minutos
    t1 = 10  # Tiempo final en minutos
    q = 100.  # L/min
    cai = 1.  # mol/L
    Ti = 350.  # K
    V = 100.  # L
    p = 1000.  # g/L
    C = 0.239  # J/g K
    deltaHr = -5e4  # J/mol
    EdivR = 8750.  # K
    k0 = 7.2e10  # min^-1
    UA = 5e4  # J/min K
    num = -EdivR * (1 / T)
#   k = k0 * np.exp(num)
    w = q * p
    cae = 0.9519765894312876
    Te = 313.02857082953074
    f = cae * sp.exp(-EdivR * (1 / Te)) * (EdivR / Te ** 2) * T + sp.exp(-EdivR * (1 / Te)) * ca

    # Definimos entradas
    Tc = escalon(t, t0, t1, 0, 0)

    return [(q*(-ca)-V*k0*f)/V,
           (w*C*(-T)+(-deltaHr)*V*k0*f+UA*(Tc-T))/(V*p*C)]

t_span = (0, 10)
t = np.arange(0, 10, 0.001)
ci = [0.4999182859586579, 350.0055286902126]
ciLineal = [0.4999182859586579-0.9519765894312876, 350.0055286902126-313.02857082953074]

resultado = solve_ivp(edo, t_span, ci, method='RK45', t_eval=t)
resultadoLineal = solve_ivp(lineal, t_span, ciLineal, method='RK45', t_eval=t)

# Ploteamos resultado de CA
plt.plot(resultado.t, resultado.y[0], label='ca no lineal', linewidth=2, color='red')
plt.plot(resultadoLineal.t, resultadoLineal.y[0]+0.9519765894312876, label='ca lineal', linewidth=2, color='green')
plt.ylabel('Ca [mol/L]')
plt.xlabel('Tiempo [min]')
plt.ticklabel_format(useOffset=False)
plt.title('Concentracion molar en 10 minutos de Simulacion')
plt.legend()
plt.show()

# Ploteamos resultado de T
plt.plot(resultado.t, resultado.y[1], label='T no lineal', linewidth=2, color='red')
plt.plot(resultadoLineal.t, resultadoLineal.y[1]+313.02857082953074, label='T lineal', linewidth=2, color='green')
plt.ylabel('T [K]')
plt.xlabel('Tiempo [min]')
plt.ticklabel_format(useOffset=False)
plt.title('Temperatura en 10 minutos de Simulacion')
plt.legend()
plt.show()

# PUNTO B
print('Punto de equilibrio de Ca: ', resultado.y[0][-1])
print('Punto de equilibrio de T', resultado.y[1][-1])

ca = sp.Symbol('\overline{\\rm ca}')
T = sp.Symbol('\overline{\\rm T}')
k0 = sp.Symbol('k0')
p = sp.Symbol('p')
q = sp.Symbol('q')
UA = sp.Symbol('UA')
EdivR = sp.Symbol('E/R')
Tc = sp.Symbol('\overline{\\rm Tc}')
V = sp.Symbol('V')
cai = sp.Symbol('cai')
C = sp.Symbol('C')
deltaHr = sp.Symbol('\Delta Hr')
Ti = sp.Symbol('Ti')
dcadt = sp.Symbol('\\frac{\partial ca}{\partial t}')
dTdt = sp.Symbol('\\frac{\partial T}{\partial t}')
cae = sp.Symbol('{ca}^{e}')
Te = sp.Symbol('{T}^{e}')
s = sp.Symbol('s')
CA = sp.Symbol('CA')
Ts = sp.Symbol('Ts')
f = cae * sp.exp(-EdivR*(1/Te)) * (EdivR/Te**2) * T + sp.exp(-EdivR*(1/Te)) * ca

edo1 = - V * dcadt - q * ca - V * k0 * f
edo2 = - V * p * C * dTdt - p * q * C * T + (-deltaHr) * V * k0 * f + UA * (Tc-T)

edo1 = sp.simplify(edo1)
edo2 = edo2

print('EDO de dCa/dt en Latex: ', sp.latex(edo1))
print('EDO de dT/dt en Latex: ', sp.latex(edo2))

ftca = edo1.subs([(dcadt, s*CA), (ca, CA), (T, Ts)])
ftca = sp.solve(ftca, CA)[0]
ftT = edo2.subs([(dTdt, s*T), (ca, CA), (T, Ts)])
ftT = ftT.subs(CA, ftca)
ftT = sp.solve(ftT, Ts)[0]
ftT = sp.simplify(ftT)
FT = ftT/Tc
FT = sp.simplify(FT)

FT = FT.subs([(q, 100), (V, 100), (p, 1000), (C, 0.239),
              (cai, 1), (Ti, 350), (UA, 5e4), (EdivR, 8750),
              (deltaHr, -5e4), (k0, 7.2e10), (Te, 313.02857082953074),
              (cae, 0.9519765894312876), (Tc, 300)])

print('Transformada de la planta resultante: ', FT)


sys = ctrl.tf([4899344307.77893*137946936672367.0, 145146936672367.0*4899344307.77893],
               [3.23056079623503e+23,1.03895065590711e+24,7.51169855299611e+23])
print('Ceros de la planta: ', sys.zeros())
print('Polos de la planta : ', sys.poles())
ctrl.root_locus(sys)
plt.show()

# Planteamos lazo cerrado con una Gp simplificada con polos en -1 y en -2.12 y un cero en -1
K = 2.1002 # Ver informe. Se obtiene de factor comun en numerador
Gp = (s+1.05219398)/((s+2.11837197)*(s+1.09763517))

kp = sp.Symbol('kp')
Td = sp.Symbol('Td')
Ti = sp.Symbol('Ti')

Gc = kp*(1 + (1/(Ti*s)) + Td*s)
H = 1

LazoAbierto = sp.simplify(Gp*Gc*H)

# Reemplazo Td y Ti por 1
LazoAbierto = LazoAbierto.subs([(Td, 1), (Ti, 1)])

print('Lazo abierto con Td = 1 y Ti = 1: ', LazoAbierto)

# Numerador (s + 1.05219398)*(s*(s + 1) + 1) NOTA: Saco kp para incluirlo en K posteriormente
# Denominador (s*(s + 1.09763517)*(s + 2.11837197))

# print(sp.expand((s + 1.05219398)*(s*(s + 1) + 1)))
# print(sp.expand((s*(s + 1.09763517)*(s + 2.11837197))))

# Numerador Expandido s**3 + 2.05219398*s**2 + 2.05219398*s + 1.05219398
# Denominador Expandido s**3 + 3.21600714*s**2 + 2.32519957741418*s

# Coloco los coeficientes calculados arriba en la FT de control

sysLazoAbierto = ctrl.tf([1, 2.05219398, 2.05219398, 1.05219398], [1, 3.21600714, 2.32519957741418, 0])
print('Ceros con Td = 1 y Ti = 1: ', sysLazoAbierto.zeros())
print('Polos con Td = 1 y Ti = 1:', sysLazoAbierto.poles())
ctrl.root_locus(sysLazoAbierto)
plt.show()

# Pruebo con valores distintos de Td y Ti
# Reemplazo Td por 0.22 y Ti por 1
LazoAbierto2 = sp.simplify(Gp*Gc*H)
LazoAbierto2 = LazoAbierto2.subs([(Td,0.22), (Ti,1)])

print('Lazo abierto con Td = 0.22 y Ti = 1: ', LazoAbierto2)

# Numerador (s + 1.05219398)*(s*(0.22*s + 1) + 1)
# Denominador (s*(s + 1.09763517)*(s + 2.11837197))

#print(sp.expand((s + 1.05219398)*(s*(0.22*s + 1) + 1)))
#print(sp.expand((s*(s + 1.09763517)*(s + 2.11837197))))

# Numerador Expandido 0.22*s**3 + 1.2314826756*s**2 + 2.05219398*s + 1.05219398
# Denominador Expandido s**3 + 3.21600714*s**2 + 2.32519957741418*s

# Coloco los coeficientes calculados arriba en la FT de control

sysLazoAbierto2 = ctrl.tf([0.22, 1.2314826756, 2.05219398, 1.05219398], [1, 3.21600714, 2.32519957741418, 0])
print('Ceros con Td = 0.22 y Ti = 1: ', sysLazoAbierto2.zeros())
print('Polos con Td = 0.22 y Ti = 1:', sysLazoAbierto2.poles())
ctrl.root_locus(sysLazoAbierto2)
plt.show()




# Intento de cálculo de Routh-Hurwitz

# LazoCerrado = sp.simplify((Gc*FT)/(1+H*Gc*FT))
#
# print('\n\nlazo cerrado: \n', LazoCerrado)
#
# numeradorExpandido = sp.simplify(sp.expand(4899344307.77893*kp*(137946936672367.0*s + 145146936672367.0)*(Ti*s*(Td*s + 1) + 1)))
# denominadorExpandido = sp.expand(Ti*s*(3.23056079623503e+23*s**2 + 1.03895065590711e+24*s + 7.51169855299611e+23) + 4899344307.77893*kp*(137946936672367.0*s + 145146936672367.0)*(Ti*s*(Td*s + 1) + 1))
#
# print('numerador expandido: \n', numeradorExpandido)
# print('denominador expandido: \n',denominadorExpandido)
#
# polos = sp.roots(denominadorExpandido, s)
# polos = list(polos)
#
# print('Polos en función de kp, Ti y Td de lazo cerrado: ')
# print('Polo 1: ', sp.nsimplify(polos[0]))
# print('Polo 2: ', sp.nsimplify(polos[1]))
# print('Polo 3: ', sp.nsimplify(polos[2]))