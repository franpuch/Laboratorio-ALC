"""
Laboratorio 6: Método de la Potencia y Diagonalización. 

Funciones del Módulo ALC.
Nuevos Test. 
"""

import numpy as np 

#### Funciones Auxiliares ---------------------------------------------------------------------------------------------- #

def producto_interno(x1:np.ndarray, x2:np.ndarray) -> float : 
    
    if (len(x1) != len(x2)) : 
        raise ValueError("Las dimensiones de los vectores no son compatibles para el producto inetrno (no son iguales).")
    
    long_vectores:int = len(x1) 
    
    res:float = 0 
    
    for i in range(0, long_vectores) : 
        res += np.float64(x1[i] * x2[i]) 
        
    return res 


def calcularAx(A:np.ndarray, x:np.ndarray, vector_fila:bool = False) -> np.ndarray:

    n_filas, n_cols = np.shape(A)
    forma_x = np.shape(x)

    # Normalizo x a vector 1D.
    if (len(forma_x) == 2) :
        f, c = forma_x
        
        if (c == 1) :          # vector columna
            x = x[:, 0]
            
        elif (f == 1) :        # vector fila
            x = x[0, :]
            
        else:
            raise ValueError("La 'matriz_x' no es un vector válido.")
            
    elif (len(forma_x) != 1) :
        raise ValueError("La 'matriz_x' no es un vector válido.")

    # Verifico de compatibilidad.
    if (len(x) != n_cols) :
        raise ValueError("Dimensiones incompatibles para la multiplicación A * x.")

    # Resultado (vector columna)
    res = np.zeros((n_filas, 1), dtype = np.float64)

    for i in range(n_filas) :
        fila_i = A[i] 
        suma = 0.0
        
        for j in range(n_cols) :
            suma += fila_i[j] * x[j] 
            
        res[i, 0] = suma

    # Fran del Futuro añade esta nueva opción que retorna el resultado en forma de vector fila (porque en el futuro no me sirve
    # que esta función me devuelva el resultado como vector columna).
    if (vector_fila) :
        res = res[:, 0] 

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


def normaExacta(A:np.ndarray, p:int) -> float | None : 
    
    A = np.array(A) 
    
    nro_filas, nro_columnas = np.shape(A) 
    
    if (p == 1) : 
        suma_max:float = 0 
        
        for j in range(0, nro_columnas) :
            suma_actual = 0
            
            for i in range(0, nro_filas) : 
                suma_actual += np.abs(A[i][j]) 
                
            if (suma_max < suma_actual) : 
                suma_max = suma_actual 
        
        return np.float64(suma_max) 
    
    elif (p == 'inf') : 
        suma_max:float = 0 
        
        for i in range(0, nro_filas) : 
            suma_actual = 0 
            
            for j in range(0, nro_columnas) : 
                suma_actual += np.abs(A[i][j]) 
                
            if (suma_max < suma_actual) : 
                suma_max = suma_actual 
        
        return np.float64(suma_max) 
    
    else : 
        return None 


def error(x, y) -> float :
    
    x = np.float64(x) 
    y = np.float64(y) 
    
    return np.abs(x - y)


def matricesIguales(A, B, tol = 1e-12) -> bool :
    
    if (len(A) != len(B)) :
        return False
    
    if (any(len(A[i]) != len(B[i]) for i in range(len(A)))) : # Miro que todas las filas sean de igual tamaño.
        return False

    for i in range(len(A)) :
        for j in range(len(A[i])) :
            if (error(A[i][j], B[i][j]) > tol) :   # Uso la función 'error()' que implemente antes.
                return False
    return True 


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

#### Funciones Principales --------------------------------------------------------------------------------------------- #

def metpot2k(A:np.ndarray, tol:float = 1e-15, K:int = 1000) :

    n, m = np.shape(A)
    
    if (n != m) :
        return None

    # Genero el vector inicial aleatorio (valores entre 0 y 1).
    v = np.random.rand(n).astype(np.float64)
    norma_v = norma(v, 2) 
    
    # Atajo el caso de que la norma del vector aleatorio sea 0 (más adelante no quiero dividir por cero).
    if (norma_v < tol) :
        v[:] = 1.0   # Re-utilizo el mismo vector (para ahorrar memoria).
        norma_v = norma(v, 2) 
        
    v /= norma_v

    # Aplico dos veces la matriz A (la transformación 'f') al vector 'v'.
    # Para evitar hacer una definción de función aparte, lo enchufo de una. 
    Av = calcularAx(A, v, vector_fila = True)
    Av /= norma(Av, 2)
    v_tilde = calcularAx(A, Av, vector_fila = True)
    v_tilde /= norma(v_tilde, 2)

    # Defino la variable 'e'. 
    # Acá me dicen que haga 'transpuesto(v_virulete) * v', es lo mismo que hacer producto interno entre ambos.
    e = producto_interno(v_tilde, v)

    iteraciones = 0

    while ((abs(e - 1.0) > tol) and (iteraciones < K)) :
        v = v_tilde

        Av = calcularAx(A, v, vector_fila=True)
        Av /= norma(Av, 2)
        v_tilde = calcularAx(A, Av, vector_fila=True)
        v_tilde /= norma(v_tilde, 2)

        e = producto_interno(v_tilde, v)
        iteraciones += 1

    # Aproximación del AutoValor. 
    Av_tilde = calcularAx(A, v_tilde, vector_fila=True)
    autovalor = producto_interno(v_tilde, Av_tilde)

    # Calculo el error que dice el pseudo-código (lo dejo como variable sin usar porque en los test no lo piden y se me 
    # rompe la función).
    epsilon = e - 1.0

    return v_tilde, autovalor, iteraciones 


def diagRH(A: np.ndarray, tol: float = 1e-15, K: int = 1000) :

    # Verificación de simetría (numérica).
    if (not matricesIguales(A, traspuesta(A), tol=1e-10)) :
        return None, None

    n = A.shape[0]

    # Caso base 1 -> 'n = 1'.
    if (n == 1) :
        return np.array([[1.0]]), np.array([[A[0, 0]]])

    # Primer AutoVector y AutoValor por Método de la Potencia.
    v1, l1, _ = metpot2k(A, tol, K)

    # Construyo el Reflector de Householder: H = I - 2 * (u u^T) / (u^T u)
    e1 = np.zeros_like(v1)
    e1[0] = 1.0
    u = e1 - v1
    denom = producto_interno(u, u)

    if (denom < tol) :   # No hay que dividir por cero...
        H_v1 = np.eye(n)
        
    else:
        u = u.reshape(-1, 1)   # vector columna
        H_v1 = np.eye(n) - 2.0 * multiplicar_matrices(u, traspuesta(u)) / denom

    # Construyo B = H * A * H^T
    B = multiplicar_matrices(H_v1, multiplicar_matrices(A, traspuesta(H_v1)))
    B[np.abs(B) < tol] = 0.0   # Limpio números que tomamos como cero (los haco cero realmente).

    # Caso base 2 -> 'n = 2'.
    if (n == 2) :
        return H_v1, B

    # Paso recursivo.
    A_tilde = B[1:, 1:]
    S_tilde, D_tilde = diagRH(A_tilde, tol, K)

    # Construcción de D.
    D = np.eye(n)
    D[0, 0] = l1
    D[1:, 1:] = D_tilde

    # Construcción de S.
    S = np.eye(n)
    S[1:, 1:] = S_tilde
    S = multiplicar_matrices(H_v1, S)

    return S, D


#### TESTEOS ----------------------------------------------------------------------------------------------------------- #

# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95 

print("Si se imprime esto, todo salió bien.")


# Tests diagRH
D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95

print("Si se imprime esto, todo salió bien.") 


# Fin. 