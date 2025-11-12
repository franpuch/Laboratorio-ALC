"""
Laboratorio 0: Funciones Básicas.
Optimizaciones y nuevos tests.

Funciones del Módulo ALC.
"""


# %% 

# Librerias y Herramientas.

import numpy as np 


# %% 

def esSimetrica(matriz: np.ndarray, atol: float = 1e-12) -> bool :
    
    # Verifico si es cuadrada y algunas cositas más (que me llevaron a errores con el tiempo).
    if (matriz is None) :
        return False
    
    if (len(np.shape(matriz)) != 2) :
        return False
    
    filas, columnas = np.shape(matriz)
    
    if (filas != columnas) :
        return False

    # Compara solo la mitad superior (para ahorrar tiempo).
    for i in range(filas) :
        for j in range(i + 1, columnas) :
            if (abs(matriz[i, j] - matriz[j, i]) > atol) :
                return False
            
    return True 


# %% 

def traspuesta(matriz: np.ndarray) -> np.ndarray :
    
    matriz = np.array(matriz, dtype = np.float64)

    # Caso 1: vector 1D -> lo tratamos como matriz fila (1 × n).
    if (matriz.ndim == 1):
        n = matriz.shape[0]
        res = np.zeros((1, n), dtype = np.float64)
        res[0, :] = matriz[:] 
        return res

    # Caso general: matriz 2D.
    filas, columnas = np.shape(matriz)
    res = np.zeros((columnas, filas), dtype = np.float64)

    for i in range(filas) :
        res[:, i] = matriz[i, :] 

    return res


# Fin. 