"""
Funciones Módulo ALC y Funciones Auxiliares.
"""

# %% Imports.

import numpy as np 

# %% Funciones Auxiliares. 

""" 
Multiplicación de matrices 2D.
"""
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


""" 
Producto interno entre vectores fila.
"""
def producto_interno(x1:np.ndarray, x2:np.ndarray) -> float : 
    
    if (len(x1) != len(x2)) : 
        raise ValueError("Las dimensiones de los vectores no son compatibles para el producto inetrno (no son iguales).")
    
    long_vectores:int = len(x1) 
    
    res:float = 0 
    
    for i in range(0, long_vectores) : 
        res += np.float64(x1[i] * x2[i]) 
        
    return res 


''' 
Recibe una matriz y retorna la matriz con sus columnas normalizadas.
'''
def normaliza_columnas(matriz:np.ndarray, p:int, tol:float = 1e-15) -> np.ndarray : 
    
    filas, columnas = matriz.shape
    res:list[list[float]] = [] 
    
    for i in range(0, columnas) :
        
        columna_actual = matriz[:,i] 
        norma_columna = norma(columna_actual, p) 
        
        if (norma_columna <= tol) : 
            columna_actual = np.zeros(filas, dtype = np.float64) 
            
        else : 
            columna_actual = columna_actual / norma_columna
            
        res.append(columna_actual) 
        
    return traspuesta(np.array(res, dtype = np.float64)) 


# %% Laboratorio 00. 

def esCuadrada(matriz:np.ndarray) -> bool : 
    
    if (matriz.shape[0] == 0) :
        return False
    
    return (matriz.shape[0] == matriz.shape[1]) 


def triangSup(matriz:np.ndarray) -> np.ndarray :
    
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.")
        return matriz 
    
    else :
        res:np.ndarray = matriz.copy() 
        nro_filas:int = res.shape[0]
        
        for i in range(0, nro_filas) :
            for j in range(0, i + 1):
                if (j <= i) :
                    res[i][j] = 0
                    
        return res 


def triangInf(matriz:np.ndarray) -> np.ndarray :
    
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.") 
        return matriz 
    
    else :
        res:np.ndarray = matriz.copy() 
        nro_filas:int = res.shape[0] 
        nro_columnas:int = res.shape[1] 
        
        for i in range(0, nro_filas) :
            for j in range(i, nro_columnas) :
                res[i][j] = 0 
        
        return res 


def diagonal(matriz:np.ndarray) -> np.ndarray : 
    
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.") 
        return matriz 
    
    else :
        res:np.ndarray = matriz.copy() 
        nro_filas:int = matriz.shape[0] 
        nro_columnas:int = matriz.shape[1] 
        
        for i in range(0, nro_filas) :
            for j in range(0, nro_columnas) :
                if (i != j) : 
                    res[i][j] = 0 
                    
        return res 


def traza(matriz:np.ndarray) -> float :
    
    if not(esCuadrada(matriz)):
        print("La matriz no es cuadrada, no puede definirse una diagonal principal")
        return 0
    
    nro_filas, nro_columnas = np.shape(matriz)
    res = 0.0
    
    for i in range(nro_filas):
        res += matriz[i][i]
    
    return res 


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


def esSimetrica(matriz:np.ndarray, atol: float = 1e-12) -> bool :
    
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


def intercambiarFilas(matriz:np.ndarray , i:int , j:int) -> None :
    nro_columnas:int = np.shape(matriz)[1]
    
    for a in range(0, nro_columnas) :
        aux:float = matriz[i][a] 
            
        matriz[i][a] = matriz[j][a]
        matriz[j][a] = aux 


def sumar_fila_multiplo(matriz:np.ndarray , i:int , j:int , s:float) -> None : 
    nro_columnas:int = np.shape(matriz)[1] 
    
    for a in range(0, nro_columnas) :
        matriz[i][a] += matriz[j][a] * s 


def esDiagonalmenteDominante(matriz:np.ndarray) -> bool :
    nro_filas , nro_columnas = np.shape(matriz) 
    
    if (nro_filas != nro_columnas) :
        return False 
    
    for i in range(nro_filas) :
        dominante:float = abs(matriz[i][i]) 
        suma_parcial:float = 0
        
        for j in range(nro_columnas) :
            if (j != i) :
                suma_parcial += abs(matriz[i][j]) 
        
        if (dominante <= suma_parcial) :
            return False
    
    return True 


def matrizCirculante(vector: np.ndarray) -> np.ndarray :
    
    shape = np.shape(vector)
    if len(shape) == 1 :                 # vector 1D: (n,)
        n = shape[0]
        v = [vector[i] for i in range(n)]
    elif shape[0] == 1 :                 # vector fila: (1, n)
        n = shape[1]
        v = [vector[0][j] for j in range(n)]
    elif shape[1] == 1 :                 # vector columna: (n, 1)
        n = shape[0]
        v = [vector[i][0] for i in range(n)]
    else:
        raise ValueError("El parámetro 'vector' debe ser 1D, fila (1×n) o columna (n×1).")

    res = np.array([[0 for _ in range(n)] for _ in range(n)])

    for i in range(n):
        for j in range(n):
            res[i][j] = v[(j - i) % n] 

    return res 


def matrizVandermonde(vector: np.ndarray) -> np.ndarray :
    
    n:int = np.shape(vector)[0]
    
    res:np.ndarray = np.array([[0 for _ in range(n)] for _ in range(n)], dtype=float)
    
    for fila in range(n) : 
        for columna in range(n) :   
            res[fila][columna] = vector[fila] ** columna
    
    return res 


def matriz_Fibonacci(n:int) -> np.ndarray : 
    
    res:np.ndarray = np.array([[0 for _ in range(0, n)] for _ in range(0, n)]) 
    
    fib:list[int] = [0, 1]
    for contador in range(2, 2 * n) :   # Empiezo en 2 el ciclo porque los primeros 2 fibonaccis ya los tengo.
        fib.append(fib[-1] + fib[-2])   # Aprovecho que Python entiende los índices negativos como empezar desde el final.
    
    for fila in range(0, n) :
        for columna in range(0, n) :
            res[fila][columna] = fib[fila + columna]
    
    return res 


def matrizHilbert(n:int) -> np.ndarray : 
    
    res:np.ndarray = np.array([[0 for _ in range(0, n)] for _ in range(0, n)], dtype = float) 
    
    for fila in range(0, n) :
        for columna in range(0, n) :
            res[fila][columna] = (1) / (fila + columna + 1)
    
    return res 


def row_echelon_stable(matriz:np.ndarray) -> np.ndarray :
    nro_filas, nro_columnas = np.shape(matriz) 
    
    for fila in range(0, nro_filas) :
        
        # Verifico si es la de mayor pivot. 
        max_fila:int = fila 
        max_valor:float = abs(matriz[fila][fila]) 
        for i in range(fila, nro_filas) :
            if ((abs(matriz[i][fila])) > max_valor) : 
                max_fila = i 
                max_valor = abs(matriz[i][fila]) 
        
        # Hago el swap de filas (si es necesario).
        if (max_fila != fila) :
            intercambiarFilas(matriz, fila, max_fila) 
            
        # Hago Eliminación Gaussiana para esa columna.
        for columna in range(fila + 1, nro_filas) :
            if (matriz[columna][fila] != 0) :
                factor:float = matriz[columna][fila] / matriz[fila][fila] 
                sumar_fila_multiplo(matriz, columna, fila, (-factor))
    
    return matriz 


# %% Laboratorio 01. 

def error(x, y) -> float :
    
    x = np.float64(x) 
    y = np.float64(y) 
    
    return np.abs(x - y) 


def error_relativo(x, y) -> float :

    x = np.float64(x) 
    y = np.float64(y) 
    
    if x == 0:
        return np.abs(y)
    
    return np.abs(x - y) / np.abs(x) 


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


# %% Laboratorio 02. 

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
    
    for i in range(0, len(s)) :
        res[i][i] = s[i]
    
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


# %% Laboratorio 03 

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


def normaliza(x:np.ndarray, p:int) -> list[list[float]] :
    
    res:list[list[float]] = []
    
    for elemento in x :
        
        if not(np.any(elemento)) :   # Es el vector nulo. 
            res.append(elemento)
        
        else: 
            res.append(elemento/norma(elemento, p))
    
    return res 


def normaMatMC(A:np.ndarray, q:int, p:int, Np:np.ndarray) -> tuple[float, np.ndarray] :
    
    columnas:int = A.shape[1]
    
    x = np.random.rand(Np, columnas)
    
    x = normaliza(x, p)
    
    res:float = 0
    vector_max = x[0] 
    
    for elemento in x :
        
        norma_actual = norma(calcularAx(A, elemento), q) 
        
        if (res <= norma_actual) : 
            res = norma_actual
            vector_max = elemento
            
    return (np.float64(res), vector_max) 


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


def condMC(A:np.ndarray, p:int | str) -> float : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 

    Np:int = 10000 
    
    norma_A:float = normaMatMC(A, p, p, Np)[0] 
    norma_A_inv:float = normaMatMC(A_inversa, p, p, Np)[0] 
    
    return np.float64(norma_A * norma_A_inv) 


def condExacta(A:np.ndarray, p:int | str) -> float | None : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 
    
    norma_A:float = normaExacta(A, p) 
    norma_A_inv:float = normaExacta(A_inversa, p) 
    
    if (norma_A == None) or (norma_A_inv == None) : 
        return None 
    
    return np.float64(norma_A * norma_A_inv) 


# %% Laboratorio 04 

def calculaLU(A: np.ndarray) :
    
    # Validaciones iniciales
    if (A is None) :
        return None, None, 0
    
    if (len(np.shape(A)) != 2) :
        return None, None, 0
    
    filas, columnas = np.shape(A)
    
    if (filas != columnas) :
        return None, None, 0

    U = np.copy(A).astype(np.float64)
    L = np.eye(filas, dtype = np.float64)
    nops = 0

    for k in range(filas - 1) : 

        pivote = U[k, k]
        
        if (pivote == 0.0) :
            return None, None, 0  # No factorizable sin pivoteo.

        for i in range(k + 1, filas) :
            mult = U[i, k] / pivote
            L[i, k] = mult
            nops += 1  # División

            # Guardamos la fila de U[k, :] para reducir accesos y ahorrar tiempo. 
            Uk_fila = U[k, k:filas]
            Ui_fila = U[i, k:filas]

            for j in range(filas - k) :
                Ui_fila[j] = Ui_fila[j] - mult * Uk_fila[j] 
                
                if (j != 0) :
                    nops += 2  # mult + resta

    return L, U, nops 


def res_tri(L:np.ndarray, b:np.ndarray, inferior: bool = True) :
    
    if (L is None or b is None) :
        return None

    filas, columnas = np.shape(L)
    
    if (filas != columnas or b.shape[0] != filas) :
        return None

    res = np.zeros(filas, dtype = np.float64)

    if (inferior) :
        # Algoritmo de Forward Substitution. 
        for i in range(filas) : 
            Li_fila = L[i, :] 
            suma = 0.0
            
            for j in range(i) : 
                suma += Li_fila[j] * res[j] 
                
            diag = Li_fila[i] 
            
            if (diag == 0.0) :   # Ojo con divididr por cero. 
                return None
            
            res[i] = (b[i] - suma) / diag 
            
    else:
        # Algoritmo de Backward Substitution.
        for i in range(filas - 1, -1, -1) :
            Li_fila = L[i, :]
            suma = 0.0
            
            for j in range(i + 1, filas) :
                suma += Li_fila[j] * res[j] 
                
            diag = Li_fila[i]
            
            if (diag == 0.0) :   # Ojo con dividir por cero (otra vez).
                return None
            
            res[i] = (b[i] - suma) / diag

    return res 


def inversa(A: np.ndarray) : 
    
    if (A is None) :
        return None
    
    if (len(np.shape(A)) != 2) :
        return None 
    
    filas, columnas = np.shape(A)
    
    if (filas != columnas) :
        return None

    # Factorización A = L U.
    L, U, nops = calculaLU(A) 
    
    if ((L is None) or (U is None)) :
        return None

    # Construyo identidad y matriz resultado. 
    identidad = np.eye(filas, dtype = np.float64)
    res = np.zeros((filas, filas), dtype = np.float64)

    # A ver si me sale ahorrar memoria...
    for i in range(filas) :
        e = identidad[:, i]
        y = res_tri(L, e, inferior = True) 
        
        if (y is None) :
            return None
        
        x = res_tri(U, y, inferior=False)
        
        if (x is None) :
            return None 
        
        res[:, i] = x

    return res 


def calculaLDV(A: np.ndarray) : 
    
    # Validaciones iniciales.
    if (A is None) :
        return None, None, None 
    
    if (len(np.shape(A)) != 2) :
        return None, None, None 
    
    filas, columnas = np.shape(A)
    
    if (filas != columnas) :
        return None, None, None 

    # Factorizo A = L * U.
    L, U, nops = calculaLU(A) 
    
    if ((L is None) or (U is None)) :
        return None, None, None 

    # Construyo D diagonal (solo con los pivotes de U).
    D = np.zeros((filas, filas), dtype = np.float64)
    
    for i in range(filas) :
        Dii = U[i, i]
        
        if (Dii == 0.0) :
            return None, None, None 
        
        D[i, i] = Dii

    # Construyo V = U normalizado (cada fila dividida por su pivote).
    V = np.zeros_like(U, dtype = np.float64)
    
    for i in range(filas) :
        piv = D[i, i]
        
        if (piv == 0.0) :
            return None, None, None 
        
        Ui_fila = U[i, :]
        Vi_fila = V[i, :]
        for j in range(filas) :
            Vi_fila[j] = Ui_fila[j] / piv

    return L, D, V 


def esSDP(A:np.ndarray, atol:float = 1e-8) -> bool :
    
    if (A is None) :
        return False
    
    if (len(np.shape(A)) != 2) :
        return False
    
    filas, columnas = np.shape(A)
    
    if (filas != columnas) :
        return False

    # Verifico simetría.
    if (not esSimetrica(A, atol = atol)) :
        return False

    # Factorización A = L D V.
    L, D, V = calculaLDV(A)
    
    if ((L is None) or (D is None) or (V is None)) :
        return False

    # Revisa positividad de D (diagonal).
    for i in range(filas) :
        if (D[i, i] < atol) :
            return False

    return True 


# %% Laboratorio 5 

def QR_con_GS(A:np.ndarray, tol:float = 1e-12, retorna_nops:bool = False) :
    
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


def QR_con_HH(A:np.ndarray, tol:float = 1e-12) :
    
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


def calculaQR(A:np.ndarray, metodo:str = 'RH', tol:float = 1e-12, retorna_nops:bool = False) :
    
    if (not esCuadrada(A)) :
        return None

    if (metodo == 'GS') :
        return QR_con_GS(A, tol = tol)
    
    elif (metodo == 'RH') :
        return QR_con_HH(A, tol = tol)
    
    else :
        return None 


# %% Laboratorio 6. 

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


# %% Laboratorio 7. 

# Faltan Re-Testear y Optimizar. 


# %% Laboratorio 8. 

# Falta Re-Testear y Optimizar. 


# Fin. 
