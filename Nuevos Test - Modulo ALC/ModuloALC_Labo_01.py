"""
Laboratorio 1: Numeros de Maquina 

Funciones del Módulo ALC.
Nuevos Test. 
"""


# %% 

# Librerias y Herramientas.

import numpy as np 


# %%

def error(x, y):
    
    x = np.float64(x) 
    y = np.float64(y) 
    
    return np.abs(x - y) 


def error_relativo(x, y):

    x = np.float64(x) 
    y = np.float64(y) 
    
    if x == 0:
        return np.abs(y)
    
    return np.abs(x - y) / np.abs(x) 


def matricesIguales(A, B, tol = 1e-12) :
    
    if (len(A) != len(B)) :
        return False
    
    if (any(len(A[i]) != len(B[i]) for i in range(len(A)))) : # Miro que todas las filas sean de igual tamaño.
        return False

    for i in range(len(A)) :
        for j in range(len(A[i])) :
            if (error(A[i][j], B[i][j]) > tol) :   # Uso la función 'error()' que implemente antes.
                return False
    return True 


# %% 
# Vamos con los Test. 

def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x,y), 0, atol=atol)

assert(not sonIguales(1,1.1))
assert(sonIguales(1,1 + np.finfo('float64').eps))
assert(not sonIguales(1,1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps, atol=1e-3))

assert(np.allclose(error_relativo(1,1.1),0.1))
assert(np.allclose(error_relativo(2,1),0.5))
assert(np.allclose(error_relativo(-1,-1),0))
assert(np.allclose(error_relativo(1,-1),2))

assert(matricesIguales(np.diag([1,1]),np.eye(2)))
assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))

print("Todos los Test pasaron correctamente.") 


# Fin. 
