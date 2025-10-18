"""
Laboratorio 2: Transformaciones Lineales

Funciones del Módulo ALC.
"""

# %% 

# Librerias y Herramientas.

import numpy as np 
from Modulo_ALC import calcularAx


# %% 

'''
Input -> ángulo (en radianes)

Implementa la matríz de rotación [cos(alpha) -sen(alpha)]
                                 [sen(alpha)  cos(alpha)]
                                 
Esta matriz, al multiplicarla por un vector (columna) lo rota en un ángulo 'alpha' (en sentido antiorario, como nos
manejamos siempre).

'''

def rota(theta:float) -> np.ndarray :
    
    c:float = np.cos(theta)
    s:float = np.sin(theta) 
    
    res:np.ndarray = ([[c, -s], 
                       [s, c]])
    
    return res 


# %% 

'''
Si quiero a un vector v = (x, y) modificar su escala según los valores (a, b), debo multiplicar 'v' (en forma de 
columna) por una matriz que tenga todos ceros y los elementos 'a' y 'b' en la diagonal. Ya que al multiplicar 'v' 
por esa "matriz de cambio de escala", obtengo v'=(a * x , b * y).
'''

def escala(s) -> np.ndarray :
    
    # Por las dudas, tomo 's' y lo parseo a un array de numpy (el PDF no especifica bien el tipo de dato que es 's').
    # Con eso en cuenta, puede ser un string, puede ser un array común, puede ser una tupla... por las dudas trato
    # de atajar todo.
    s = np.asarray(s, dtype=float)
    
    n:int = s.size
    
    # Armo la matriz resultado como una matriz 'n x n' vacía (de ceros).
    res:np.ndarray = np.zeros((n, n) , dtype=float) 
    
    # Lleno la diagonal de la matriz, para poder construir la matriz de cambio de escala.
    for i in range(0, len(s)) :
        res[i][i] = s[i]
    
    return res 


# %%

'''
Lo que busco es una composición de transformaciones lineales: f(g(v)) donde 'g()' es la 'TL rotación' y 'f()' es la 
'TL escalado'. Como vengo trabajando con matrices (de las TLs), la composición es la multiplicacion de las 
respectivas matrices. 

IMP -> Si la composición es f(g()), el producto de las matrices es M(f)*M(g). Si hago M(g)*M(f) estoy haciendo 
       la composición g(f())
'''

def rota_y_escala(theta:float, s) -> np.ndarray : 
    
    matriz_escala:np.ndarray = escala(s) 
    matriz_rotacion:np.ndarray = rota(theta) 
    
    # Para la multiplicación de matrices uso la función de numpy (que seguro es mas eficiente que hacerlo a mano).
    res:np.ndarray = matriz_escala @ matriz_rotacion 
    
    return res 


# %% 

'''
La matriz 'afin' se define como [[A, b,  ]
                                 [0, 0, 1]]

Donde 'A' es la "matriz de rotación y cambio de escala", y 'b' es un "vector de desplazamiento". 
Como estamos trabajando en R², la matriz queda: [[A[0][0], A[0][1], b[0]], 
                                                 [A[1][0], A[1][1], b[1]], 
                                                 [0      , 0      , 1]] 
'''

def afin(theta:float, s, b) -> np.ndarray : 
    
    # Como en los puntos anteriores, no se me especifica en qué estrucura vienen dados 's' y 'b'.
    # Por las dudas los parseo a array de numpy.
    s = np.asarray(s, dtype=float) 
    b = np.asarray(b, dtype=float)
    
    m_rotar_escalar:np.ndarray = rota_y_escala(theta, s)
    
    # Contruyo la matriz afin. Como es de '3 x 3' creo que me conviene armarla manualmente (sin ciclos ni nada raro).
    res:np.ndarray = np.array([[m_rotar_escalar[0][0], m_rotar_escalar[0][1], b[0]], 
                               [m_rotar_escalar[1][0], m_rotar_escalar[1][1], b[1]], 
                               [0                    , 0                    , 1]])
    
    return res 


# %% 

'''
Quiero usar el resultado que devuelve la función anterior, la llamo 'res_afin'. 
Aplicar la transformación afin a un vector 'v' se puede calcular como 'res_afin * v'. 
El tema es que para hacer ese producto, debo extender 'v' a un vector de R³ (porque 'res_afin' es de '3 x 3' y 
'v' es de '2 x 1' (si lo pienso como vector columna)). 
Lo que puedo hacer es extender 'v' a un vector [v[0], v[1], 1], hacer el producto de matrices, y retornar los 
primeros dos elementos del resultado.
'''

def trans_afin(v, theta:float, s, b) -> np.ndarray : 
    
    # Como en los puntos anteriores, no se me especifica en qué estrucura vienen dados 'v', 's' y 'b'.
    # Por las dudas los parseo a array de numpy.
    v = np.asarray(v, dtype=float) 
    s = np.asarray(s, dtype=float) 
    b = np.asarray(b, dtype=float)
    
    m_afin:np.ndarray = afin(theta, s, b) 
    v_extendido:np.ndarray = np.array([v[0], v[1], 1.0])
    
    res_aux:np.ndarray = calcularAx(m_afin, v_extendido, vector_fila=True) 
    res:np.ndarray = np.array([res_aux[0], res_aux[1]])
    
    return res 


# %% 

# Test -> 'rota()'

assert(np.allclose(rota(0), np.eye(2)))
assert(np.allclose(rota(np.pi/2), np.array([[0, -1], [1, 0]])))
assert(np.allclose(rota(np.pi), np.array([[-1, 0], [0, -1]]))) 

print("Todos los test de 'rota()' pasados correctamente.")


# %%

# Test -> 'escala()' 

assert(np.allclose(escala([2, 3]), np.array([[2, 0], [0, 3]])))
assert(np.allclose(escala([1, 1, 1]), np.eye(3)))
assert(np.allclose(escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]])))

print("Todos los test de 'escala()' pasados correctamente.")


# %% 

# Test -> 'rota_y_escala'

assert(np.allclose(rota_y_escala(0, [2, 3]), np.array([[2, 0], [0, 3]])))
assert(np.allclose(rota_y_escala(np.pi/2, [1, 1]), np.array([[0, -1], [1, 0]]))) 
assert(np.allclose(rota_y_escala(np.pi, [2, 2]), np.array([[-2, 0], [0, -2]])))

print("Todos los test de 'rota_y_escala' pasados correctamente.") 


# %% 

# Test -> 'afin()'

assert(np.allclose(afin(0, [1, 1], [1, 2]), np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])))
assert(np.allclose(afin(np.pi/2, [1, 1], [0, 0]), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])))
assert(np.allclose(afin(0, [2, 3], [1, 1]), np.array([[2, 0, 1], [0, 3, 1], [0, 0, 1]])))

print("Todos los test de 'afin()' pasados correctamente.")


# %% 

# Test -> 'trans_afin()' 

assert(np.allclose(trans_afin(np.array([1, 0]), np.pi/2, [1, 1], [0, 0]), np.array([0, 1])))
assert(np.allclose(trans_afin(np.array([1, 1]), 0, [2, 3], [0, 0]), np.array([2, 3])))
assert(np.allclose(trans_afin(np.array([1, 0]), np.pi/2, [3, 2], [4, 5]), np.array([4, 7]))) 

print("Todos los test de 'trans_afin()' pasados correctamente.")


# %% 

print("Si se imprime esto, es porque todos los test pasaron exitosamente!")


# Fin.