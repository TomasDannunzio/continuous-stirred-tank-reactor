import numpy
import numpy as np
import matplotlib.pyplot as plt
import sympy.solvers.solveset
from scipy.integrate import solve_ivp
import scipy as sp
import sympy as sy

# PUNTO A

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
    num = -EdivR*(1/T)
    k = k0*np.exp(num)
    w = q*p


    # Definimos entradas
    Tc = escalon(t, t0, t1, 300, 290)

    return [(100*(1-ca)-100*7.2e10*sy.exp(-8750*(1/T))*ca)/100,
            (1000*100*0.239*(350-T)+(5e4)*100*7.2e10*sy.exp(-8750*(1/T))*ca+5e4*(Tc-T))/(100*1000*0.239)]
    #return [(q*(cai-ca)-V*k*ca)/V,
    #        (w*C*(Ti-T)+(-deltaHr)*V*k*ca+UA*(Tc-T))/(V*p*C)]

def lineal(t, y):
    # Igualamos variables a condiciones iniciales
    cal = y[0]
    Tl = y[1]

    # Definimos parámetros del sistema
    t0 = 0  # Tiempo inicial en minutos
    t1 = 10  # Tiempo final en minutos
    q = 100  # L/min
    cai = 1  # mol/L
    Ti = 350  # K
    V = 100  # L
    p = 1000  # g/L
    C = 0.239  # J/g K
    deltaHr = -5e4  # J/mol
    EdivR = 8750  # K
    k0 = 7.2e10  # min^-1
    UA = 5e4  # J/min K
    num = -EdivR * (1 / Tl)
    k = k0 * np.exp(num)
    w = q * p

    # Definimos entradas
    Tcd = escalon(t, t0, t1, 0, 5)
    Te = 350.0055286902126
    cae = 0.4999182859586579

    f = (cae * np.exp(-EdivR*(1/Te)) * (EdivR/Te**2) * Tl +
                                               np.exp(-EdivR*(1/Te)) * cal)

    return [(q * (- cal) - V * k0 * f) / V,
            (w * C * (- Tl) + (-deltaHr) * V * k0 * f + UA * Tcd - UA * Tl) / (V * p * C)]

    #return [(100 * (1 - ca) - 100 * 7.2e10 * sy.exp(-8750 * (1 / T)) * ca) / 100,
    #        (1000 * 100 * 0.239 * (350 - T) + (-5e4) * 100 * 7.2e10 * sy.exp(-8750 * (1 / T)) * ca + 5e4 * (Tc - T)) / (
    #                    100 * 1000 * 0.239)]


t_span = (0, 10)
t = np.arange(0, 10, 0.0001)

# Definimos condiciones iniciales
ciLineal = [0, 0]
ciNoLineal = [0.4999182859586579, 350.0055286902126]

resultadoNoLineal = solve_ivp(edo, t_span, ciNoLineal, method='RK45', t_eval=t)
resultadoLineal = solve_ivp(lineal, t_span, ciLineal, method='RK45', t_eval=t)

# Ploteamos resultado de CA
plt.plot(resultadoNoLineal.t, resultadoNoLineal.y[0], label='ca no lineal', linewidth=2, color='red')
plt.plot(resultadoLineal.t, -resultadoLineal.y[0]+0.4999182859586579, label='ca', linewidth=2, color='green')
plt.ylabel('Ca [mol/L]')
plt.xlabel('Tiempo [s]')
plt.title('Concentracion molar en 10 minutos de Simulacion')
plt.axis('equal')
plt.legend()
plt.show()

# Ploteamos resultado de T
plt.plot(resultadoNoLineal.t, resultadoNoLineal.y[1], label='T no lineal', linewidth=2, color='red')
plt.plot(resultadoLineal.t, -resultadoLineal.y[1]+350.0055286902126, label='T lineal', linewidth=2, color='green')
plt.ylabel('T [K]')
plt.xlabel('Tiempo [s]')
plt.title('Temperatura en 10 minutos de Simulacion')
plt.axis('equal')
plt.legend()
plt.show()

# PUNTO B
def funcPE(p):

    Ca, T = p

    q = 100  # L/min
    cai = 1  # mol/L
    Ti = 350  # K
    V = 100  # L
    p = 1000  # g/L
    C = 0.239  # J/g K
    deltaHr = -5e4  # J/mol
    EdivR = 8750
    k0 = 7.2e10  # min^-1
    UA = 5e4  # J/min K
    Tce=290

    # Idealmente seria mejor reemplazar los parametros en el return, pero vemos que al hacerlo obtenemos warnings en runtime
    # porque fsolve termina realizando muchas iteraciones.

    return [100*(1-Ca)-100*7.2e10*sy.exp(-8750*(1/T))*Ca, 1000*100*0.239*(350-T)+(5e4)*100*7.2e10*sy.exp(-8750*(1/T))*Ca+5e4*(290-T)]

condiniciales = np.ndarray((2,), buffer=np.array(ciNoLineal), dtype=float)

print(sp.optimize.fsolve(funcPE,ciNoLineal))
