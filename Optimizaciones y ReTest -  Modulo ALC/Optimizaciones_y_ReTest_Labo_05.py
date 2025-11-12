"""
Laboratorio 5: Factorización QR.

Funciones del Módulo ALC.
Nuevos Test. 
"""

import numpy as np 

# Funciones Auxiliares. 

def esCuadrada(matriz:np.ndarray) -> bool : 
    
    if (matriz.shape[0] == 0) :
        return False
    
    return (matriz.shape[0] == matriz.shape[1]) 


def producto_interno(x1:np.ndarray, x2:np.ndarray) -> float : 
    
    if (len(x1) != len(x2)) : 
        raise ValueError("Las dimensiones de los vectores no son compatibles para el producto inetrno (no son iguales).")
    
    long_vectores:int = len(x1) 
    
    res:float = 0 
    
    for i in range(0, long_vectores) : 
        res += np.float64(x1[i] * x2[i]) 
        
    return res 


def norma(x:np.ndarray, p:int) -> float :
    
    x = np.array(x) 
    
    i:int = 0
    suma:float = 0

    while (i < len(x)) :
        
        # Caso especial -> Norma Infinito.
        if (p == "inf") :
            suma = max(suma, abs(x[i]))
            
        else: 
            suma += (abs(x[i]))**p
            
        i += 1
        
    if (p == "inf") : 
        return suma
    
    return np.float64(suma**(1/p)) 


def multiplicar_matrices(A:np.ndarray, B:np.ndarray) -> np.ndarray :
    
    filas_A, cols_A = A.shape
    filas_B, cols_B = B.shape
    
    # Verifico compatibilidad.
    if (cols_A != filas_B) :
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.")
    
    C = np.zeros((filas_A, cols_B), dtype = np.float64)
    
    for i in range(0, filas_A) :
        for j in range(0, cols_B) :
            suma:float = np.float64(0.0) 
            for k in range(0, cols_A) :
                suma += np.float64(A[i, k] * B[k, j])
            C[i, j] = suma 
    
    return C 


def traspuesta(matriz:np.ndarray) -> np.ndarray :

    # Caso 1: vector 1D -> lo tratamos como matriz fila (1 × n).
    if (matriz.ndim == 1):
        n = matriz.shape[0]
        res = np.zeros((1, n), dtype = np.float64)
        res[0, :] = np.float64(matriz[:])
        return res

    # Caso general: matriz 2D.
    filas, columnas = np.shape(matriz)
    res = np.zeros((columnas, filas), dtype = np.float64)

    for i in range(filas) :
        res[:, i] = np.float64(matriz[i, :]) 

    return res


# Ahora sí las funciones principales.

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna_nops permite (opcionalmente) retornar el numero de operaciones realizado
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones).
    Si la matriz A no es de n x n, debe retornar None
    """
        
    # A debe ser cuadrada.
    if (not esCuadrada(A)) :
        return None

    A = np.array(A, dtype=np.float64)
    n:int = A.shape[0]
    Q:np.ndarray = np.zeros((n, n), dtype=np.float64)
    R:np.ndarray = np.zeros((n, n), dtype=np.float64)

    nops:int = 0  # Contador de operaciones.

    for j in range(0, n) :
            
        v:np.ndarray = np.copy(A[:, j])  # Columna j.

        for i in range(0, j) :
            qi = Q[:, i]
                
            r_ij = producto_interno(qi, v) 
            R[i, j] = r_ij
            v = v - r_ij * qi
                
            # Contamos Operaciones: producto interno ('n' multiplicaciones + 'n - 1' sumas), escala y resta.
            nops += (2 * n - 1) + n + n

        r_jj = norma(v, 2)
            
        if (r_jj > tol) :
                
            Q[:, j] = v / r_jj
            R[j, j] = r_jj
                
            # Contamos las operaciones de la normalización: 'n' multiplicaciones + 'n' divisiones.
            nops += n + n
                
        else :
                
            Q[:, j] = 0.0
            R[j, j] = 0.0

    if (retorna_nops) :
        return Q, R, nops
        
    return Q, R 


def QR_con_HH(A,tol=1e-12):
    """
    A una matriz de m x n (m>=n)
    tol la tolerancia con la que se filtran elementos nulos en R
    retorna matrices Q y R calculadas con reflexiones de Householder
    Si la matriz A no cumple m>=n, debe retornar None
    """
    
    A = np.array(A, dtype=np.float64)
    m, n = A.shape
    
    if (m < n) :
        return None

    R = np.copy(A)
    Q = np.eye(m, dtype=np.float64)

    for k in range(0, n) :
        
        x = R[k:, k].copy()
        norm_x = norma(x, 2)
        
        if (norm_x < tol) :
            continue

        # Atajo el caso cuando x[0] == 0 (la función 'np.sign()' devuelve 0 y no quiero eso porque me cancela todo).
        if (abs(x[0]) < tol) :
            alpha = -norm_x
        else:
            alpha = -np.sign(x[0]) * norm_x
        
        # Armo el canónico.
        e1 = np.zeros_like(x)
        e1[0] = 1.0
        
        # Ahora construyo 'u'
        u = x - alpha * e1
        norm_u = norma(u, 2) 
        
        if (norm_u < tol) :
            continue
        
        u = u / norm_u

        # Hk = I - 2 u u^T
        u_col = np.array([[ui] for ui in u], dtype=np.float64)
        u_row = np.array([u], dtype=np.float64)
        uuT = multiplicar_matrices(u_col, u_row)
        Hk = np.eye(len(u), dtype=np.float64) - 2.0 * uuT
        
        # Es este último bloque (de arriba) no puedo utilizar 'multiplicar_matrices()' de una, porque le voy a estar pasando 
        # dos vectores (que para numpy tienen dimensión 1). Entonces, al desempaquetar en dos variables 'np.shape()' (esto es 
        # una parte clave de 'multiplicar_matrices()') se rompe porque numpy interpreta los vectores como de dimensión 1. 
        # Para evitar este problema, fuerzo las dimensiones construyendo las matrices a mano.

        # Extiendo a H̃k en dimensión 'm' y le enchufo Hk donde corresponde.
        H_tilde = np.eye(m)
        H_tilde[k:, k:] = Hk

        # Actualizo R y Q.
        R = multiplicar_matrices(H_tilde, R)
        Q = multiplicar_matrices(Q, traspuesta(H_tilde))

    # Para ser consistente con los test (y evitar dolores de cabeza), "limpio" la parte nula de la matriz.
    # La idea es que no me queden residuos de números muy pequeños QUE NO SON CERO (pero que los tomamos como tal por lo 
    # pequeño que son).
    R[np.abs(R) < tol] = 0.0

    return Q, R 


def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """ 
    
    if (not esCuadrada(A)) :
        return None

    if (metodo == 'GS') :
        return QR_con_GS(A, tol = tol)
    
    elif (metodo == 'RH') :
        return QR_con_HH(A, tol = tol)
    
    else :
        return None
    

# Tests L05-QR:

# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)

print("Si se imprime esto, es porque todo pasó correctamente.")