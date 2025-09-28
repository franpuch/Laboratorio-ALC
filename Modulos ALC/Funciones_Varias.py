"""
Laboratorio ALC 

Funciones Extra que voy haciendo.
"""


# %% 
# Imports y Herramientas. 

import numpy as np 


# %% 
# Multiplicación General de Matrices. 

def multiplicar_matrices(A:np.ndarray, B:np.ndarray) -> np.ndarray :
    
    # Parseo a arrays de numpy con dtype float64 (por las dudas que venga en otro formato).
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)
    
    # Obtengo dimensiones.
    filas_A, cols_A = A.shape
    filas_B, cols_B = B.shape
    
    # Verifico compatibilidad.
    if (cols_A != filas_B) :
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.")
    
    # Creo matriz resultado con ceros.
    C = np.zeros((filas_A, cols_B), dtype=np.float64)
    
    # La lleno con el resultado de la multiplicación.
    for i in range(0, filas_A) :
        for j in range(0, cols_B) :
            suma:float = np.float64(0.0) 
            for k in range(0, cols_A) :
                suma += A[i, k] * B[k, j]
            C[i, j] = suma 
    
    return C 


# %% 
# Producto interno entre vectores. 

def producto_interno(x1:np.ndarray, x2:np.ndarray) -> float : 
    
    if (len(x1) != len(x2)) : 
        raise ValueError("Las dimensiones de los vectores no son compatibles para el producto inetrno (no son iguales).")
    
    long_vectores:int = len(x1) 
    
    res:float = 0 
    
    for i in range(0, long_vectores) : 
        res += x1[i] * x2[i] 
        
    return res 


# %% 
# TEST PARA multiplicar_matrices().

# Caso 1: Identidad (no cambia la matriz).
A = [[1, 2],
     [3, 4]]
I = [[1, 0],
     [0, 1]]
resultado = multiplicar_matrices(A, I)
esperado = np.array([[1, 2],
                     [3, 4]], dtype=np.float64)
assert np.allclose(resultado, esperado)

# Caso 2: Multiplicación con ceros (todo debe dar cero).
A = [[1, 2, 3],
     [4, 5, 6]]
Z = [[0, 0],
     [0, 0],
     [0, 0]]
resultado = multiplicar_matrices(A, Z)
esperado = np.zeros((2, 2), dtype=np.float64)
assert np.allclose(resultado, esperado)

# Caso 3: Matrices rectangulares (2x3 con 3x2).
A = [[1, 2, 3],
     [4, 5, 6]]
B = [[7, 8],
     [9, 10],
     [11, 12]]
resultado = multiplicar_matrices(A, B)
esperado = np.array([[58, 64],
                     [139, 154]], dtype=np.float64)
assert np.allclose(resultado, esperado)

# Caso 4: Multiplicación con escalares (1x1 * 1x1).
A = [[5]]
B = [[2]]
resultado = multiplicar_matrices(A, B)
esperado = np.array([[10]], dtype=np.float64)
assert np.allclose(resultado, esperado)

# Caso 5: Compatibilidad incorrecta (debe lanzar error).
try:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2, 3]]
    multiplicar_matrices(A, B)
    assert False, "Se esperaba un ValueError por dimensiones incompatibles"
except ValueError:
    pass

# Caso 6: Con números negativos (corregido)
A = [[-1, -2],
     [ 3,  4]]
B = [[ 5, -6],
     [-7,  8]]
resultado = multiplicar_matrices(A, B)
esperado = np.array([[9, -10],
                     [-13, 14]], dtype=np.float64)
assert np.allclose(resultado, esperado)

# Caso 7: Con floats explícitos.
A = [[1.5, 2.5],
     [3.5, 4.5]]
B = [[2.0, 0.5],
     [1.0, 1.5]]
resultado = multiplicar_matrices(A, B)
esperado = np.array([[1.5*2.0 + 2.5*1.0, 1.5*0.5 + 2.5*1.5],
                     [3.5*2.0 + 4.5*1.0, 3.5*0.5 + 4.5*1.5]], dtype=np.float64)
esperado = np.array([[5.5, 4.5],
                     [11.5, 8.5]], dtype=np.float64)
assert np.allclose(resultado, esperado)

print("Todos los tests pasaron correctamente.")


# %% 

# Fin. 