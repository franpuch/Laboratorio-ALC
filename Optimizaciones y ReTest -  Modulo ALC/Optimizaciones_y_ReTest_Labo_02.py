"""
Laboratorio 2: Transformaciones Lineales.

Funciones del Módulo ALC.
Nuevos Test. 
"""


# %% 

# Librerias y Herramientas.

import numpy as np 
from Modulo_ALC_Casi_Final import multiplicar_matrices, calcularAx


# %% 

def rota(theta:float) -> np.ndarray :
    
    c:float = np.cos(theta)
    s:float = np.sin(theta) 
    
    res:np.ndarray = np.array([[c, -s], 
                               [s, c]])
    
    return res 


def escala(s) -> np.ndarray :
    
    s = np.asarray(s, dtype = float)
    
    n:int = s.size
    
    res:np.ndarray = np.zeros((n, n) , dtype = float) 
    
    np.fill_diagonal(res, s)
    
    return res 


'''
Lo que busco es una composición de transformaciones lineales: f(g(v)) donde 'g()' es la 'TL rotación' y 'f()' es la 
'TL escalado'. Como vengo trabajando con matrices (de las TLs), la composición es la multiplicacion de las 
respectivas matrices. 
'''
def rota_y_escala(theta:float, s) -> np.ndarray : 
    
    matriz_escala:np.ndarray = escala(s) 
    matriz_rotacion:np.ndarray = rota(theta) 
    
    res:np.ndarray = multiplicar_matrices(matriz_escala, matriz_rotacion) 
    
    return res 


def afin(theta:float, s, b) -> np.ndarray : 
    
    s = np.asarray(s, dtype = float) 
    b = np.asarray(b, dtype = float)
    
    m_rotar_escalar:np.ndarray = rota_y_escala(theta, s)
    
    res:np.ndarray = np.array([[m_rotar_escalar[0][0], m_rotar_escalar[0][1], b[0]], 
                               [m_rotar_escalar[1][0], m_rotar_escalar[1][1], b[1]], 
                               [0                    , 0                    , 1]])
    
    return res 


def trans_afin(v, theta:float, s, b) -> np.ndarray : 
    
    v = np.asarray(v, dtype = float) 
    s = np.asarray(s, dtype = float) 
    b = np.asarray(b, dtype = float)
    
    m_afin:np.ndarray = afin(theta, s, b) 
    v_extendido:np.ndarray = np.array([v[0], v[1], 1.0])
    
    res_aux:np.ndarray = calcularAx(m_afin, v_extendido, vector_fila=True) 
    res:np.ndarray = np.array([res_aux[0], res_aux[1]])
    
    return res 


# %% 
# Vamos con los Test. 

# Test -> 'rota()'

assert(np.allclose(rota(0), np.eye(2)))
assert(np.allclose(rota(np.pi/2), np.array([[0, -1], [1, 0]])))
assert(np.allclose(rota(np.pi), np.array([[-1, 0], [0, -1]]))) 

print("Todos los test de 'rota()' pasados correctamente.")


# Test -> 'escala()' 

assert(np.allclose(escala([2, 3]), np.array([[2, 0], [0, 3]])))
assert(np.allclose(escala([1, 1, 1]), np.eye(3)))
assert(np.allclose(escala([0.5, 0.25]), np.array([[0.5, 0], [0, 0.25]])))

print("Todos los test de 'escala()' pasados correctamente.")


# Test -> 'rota_y_escala'

assert(np.allclose(rota_y_escala(0, [2, 3]), np.array([[2, 0], [0, 3]])))
assert(np.allclose(rota_y_escala(np.pi/2, [1, 1]), np.array([[0, -1], [1, 0]]))) 
assert(np.allclose(rota_y_escala(np.pi, [2, 2]), np.array([[-2, 0], [0, -2]])))

print("Todos los test de 'rota_y_escala' pasados correctamente.") 


# Test -> 'afin()'

assert(np.allclose(afin(0, [1, 1], [1, 2]), np.array([[1, 0, 1], [0, 1, 2], [0, 0, 1]])))
assert(np.allclose(afin(np.pi/2, [1, 1], [0, 0]), np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])))
assert(np.allclose(afin(0, [2, 3], [1, 1]), np.array([[2, 0, 1], [0, 3, 1], [0, 0, 1]])))

print("Todos los test de 'afin()' pasados correctamente.")


# Test -> 'trans_afin()' 

assert(np.allclose(trans_afin(np.array([1, 0]), np.pi/2, [1, 1], [0, 0]), np.array([0, 1])))
assert(np.allclose(trans_afin(np.array([1, 1]), 0, [2, 3], [0, 0]), np.array([2, 3])))
assert(np.allclose(trans_afin(np.array([1, 0]), np.pi/2, [3, 2], [4, 5]), np.array([4, 7]))) 

print("Todos los test de 'trans_afin()' pasados correctamente.")


print("Si se imprime esto, es porque todos los test pasaron exitosamente!") 


# Fin. 
