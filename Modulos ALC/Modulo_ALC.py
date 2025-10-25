"""
Módulo ALC

Funciones: 
    - Labo 00 -> esCuadrada() 
    - Labo 00 -> triangSup() 
    - Labo 00 -> triangInf() 
    - Labo 00 -> diagonal() 
    - Labo 00 -> traza() 
    - Labo 00 -> traspuesta() 
    - Labo 00 -> esSimetrica() 
    - Labo 00 -> calcularAx() 
    - Labo 00 -> intercambiarFilas() 
    - Labo 00 -> sumar_fila_multiplo() 
    - Labo 00 -> esDiagonalmenteDominante() 
    - Labo 00 -> matrizCirculante() 
    - Labo 00 -> MatrizVandermonde() 
    - Labo 00 -> matriz_Fibonacci() 
    - Labo 00 -> matrizHilbert() 
    - Labo 00 -> row_echelon_stable()
    
    - Labo 01 -> error() 
    - Labo 01 -> error_relativo() 
    - Labo 01 -> matricesIguales()
    
    - Labo 02 -> rota() 
    - Labo 02 -> escala() 
    - Labo 02 -> rota_y_escala() 
    - Labo 02 -> afin() 
    - Labo 02 -> trans_afin()
    
    - Labo 03 -> norma() 
    - Labo 03 -> normaliza() 
    - Labo 03 -> normaMatMC() 
    - Labo 03 -> normaExacta() 
    - Labo 03 -> condMC() 
    - Labo 03 -> condExacto() 
    
    - Labo 04 -> calculaLU() 
    - Labo 04 -> res_tri() 
    - Labo 04 -> inversa() 
    - Labo 04 -> calculaLDV() 
    - Labo 04 -> esSDP() 
    
    - Labo 05 -> QR_con_GS() 
    - Labo 05 -> QR_con_HH() 
    - Labo 05 -> calculaQR() 
    
    - Labo 06 -> metpot2k() 
    - Labo 06 -> diagRH() 
    
    - Labo 07 -> transiciones_al_azar_continuas() 
    - Labo 07 -> transiciones_al_azar_uniforme() 
    - Labo 07 -> nucleo() 
    - Labo 07 -> crea_rala() 
    - Labo 07 -> multiplica_rala_vector() -----> FALTA AÑADIR. 
    
"""

import numpy as np 


# ----------------------------------- Funciones Extras ------------------------------------------------------ #

def multiplicar_matrices(A:np.ndarray, B:np.ndarray) -> np.ndarray :
    
    # Parseo a arrays de numpy con dtype float64 (por las dudas que venga en otro formato).
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64) 
    
    filas_A, cols_A = A.shape
    filas_B, cols_B = B.shape
    
    # Verifico compatibilidad.
    if (cols_A != filas_B) :
        raise ValueError("Las dimensiones de las matrices no son compatibles para la multiplicación.")
    
    C = np.zeros((filas_A, cols_B), dtype=np.float64)
    
    for i in range(0, filas_A) :
        for j in range(0, cols_B) :
            suma:float = np.float64(0.0) 
            for k in range(0, cols_A) :
                suma += A[i, k] * B[k, j]
            C[i, j] = suma 
    
    return C 


def producto_interno(x1:np.ndarray, x2:np.ndarray) -> float : 
    
    if (len(x1) != len(x2)) : 
        raise ValueError("Las dimensiones de los vectores no son compatibles para el producto inetrno (no son iguales).")
    
    long_vectores:int = len(x1) 
    
    res:float = 0 
    
    for i in range(0, long_vectores) : 
        res += x1[i] * x2[i] 
        
    return res 


# ----------------------------------- Laboratorio 00 -------------------------------------------------------- #

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


def traza(matriz: np.ndarray) -> float:
    if not(esCuadrada(matriz)):
        print("La matriz no es cuadrada, no puede definirse una diagonal principal")
        return 0
    
    nro_filas, nro_columnas = np.shape(matriz)
    res = 0.0
    
    for i in range(nro_filas):
        res += matriz[i][i]
    
    return res 


def traspuesta(matriz: np.ndarray) -> np.ndarray:

    matriz = np.array(matriz, dtype=np.float64)

    # Caso 1: vector 1D -> lo tratamos como (1, n).
    if (matriz.ndim == 1) :
        n = matriz.shape[0]
        return np.array([[matriz[i] for i in range(n)]], dtype=np.float64)

    # Caso 2: vector columna (n x 1) -> pasa a (1, n).
    if ((matriz.ndim == 2) and (matriz.shape[1] == 1)) :
        n = matriz.shape[0]
        return np.array([[matriz[i, 0] for i in range(n)]], dtype=np.float64)

    # Caso general: estandar de matriz 2D.
    filas, columnas = matriz.shape
    res = np.zeros((columnas, filas), dtype=np.float64)
    
    for i in range(filas):
        for j in range(columnas):
            res[j, i] = matriz[i, j]
    return res


def esSimetrica(matriz:np.ndarray) -> bool :
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.") 
        return False
    
    matriz_t:np.ndarray = traspuesta(matriz) 
    
    filas, columnas = np.shape(matriz)
    
    for i in range(0, filas) :
        for j in range(0, columnas) :
            if matriz[i][j] != matriz_t[i][j] :
                return False 
    
    return True


def calcularAx(matriz_A:np.ndarray , matriz_x:np.ndarray, vector_fila:bool = False) -> np.ndarray :
    
    nro_filas, nro_columnas = np.shape(matriz_A) 
    
    if (len(np.shape(matriz_x)) == 2) :
        filas_x, cols_x = np.shape(matriz_x)
        
        if (cols_x == 1) : 
            matriz_x = np.array([matriz_x[i][0] for i in range(filas_x)])
            
        elif (filas_x == 1) : 
            matriz_x = np.array([matriz_x[0][j] for j in range(cols_x)])
            
        else:
            raise ValueError("La 'matriz_x' no es un vector válido.")
            
    elif (len(np.shape(matriz_x)) != 1) :
        raise ValueError("La 'matriz_x' no es un vector válido.")
    
    res:np.ndarray = np.zeros((nro_filas, 1), dtype = np.float64)
    
    for i in range(0, nro_filas) :
        res_parcial: float = 0 
        
        for j in range(0, nro_columnas) :
            res_parcial += matriz_A[i][j] * matriz_x[j]
        
        res[i, 0] = res_parcial 
        
    # Fran del Futuro añade esta nueva opción que retorna el resultado en forma de vector fila (porque en el futuro no me sirve
    # que esta función me devuelva el resultado como vector columna).
    if (vector_fila == True) :
        res = np.array([res[i][0] for i in range(len(res))], dtype=np.float64) if len(np.shape(res))==2 else np.array([res[i] for i in range(len(res))], dtype=np.float64)
    
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


# ----------------------------------- Laboratorio 01 -------------------------------------------------------- #

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


'''
Por lo que ví, la forma mas "correcta" de usar la función "matricesIguales" es ajustando la tolerancia en función del tamaño 
de los numeros (que se estan comparando). Esto se puede hacer multiplicando el Epsilon de Maquina por el modulo del 
comparando mas grande. Si los numeros son muy pequeños, me quedo con el Epsilon de Maquina.
'''

def matricesIguales(A, B, tol = None) :
    
    A = np.array(A, dtype=np.float64)
    B = np.array(B, dtype=np.float64)

    if (A.shape != B.shape) :
        return False

    if (tol is None) :
        tol = np.finfo(np.float64).eps   # Epsilon de máquina.

    for i in range(A.shape[0]) :
        for j in range(A.shape[1]) : 
            tol_ajustada = tol * max(1.0, np.abs(A[i,j]), np.abs(B[i,j]))
            if (np.abs(A[i,j] - B[i,j]) > tol_ajustada) :
                return False
    return True


# ----------------------------------- Laboratorio 02 -------------------------------------------------------- #

def rota(theta:float) -> np.ndarray :
    
    c:float = np.cos(theta)
    s:float = np.sin(theta) 
    
    res:np.ndarray = ([[c, -s], 
                       [s, c]])
    
    return res 


def escala(s) -> np.ndarray :
    
    s = np.asarray(s, dtype=float)
    
    n:int = s.size
    
    res:np.ndarray = np.zeros((n, n) , dtype=float) 
    
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
    
    res:np.ndarray = matriz_escala @ matriz_rotacion 
    
    return res 


def afin(theta:float, s, b) -> np.ndarray : 
    
    s = np.asarray(s, dtype=float) 
    b = np.asarray(b, dtype=float)
    
    m_rotar_escalar:np.ndarray = rota_y_escala(theta, s)
    
    res:np.ndarray = np.array([[m_rotar_escalar[0][0], m_rotar_escalar[0][1], b[0]], 
                               [m_rotar_escalar[1][0], m_rotar_escalar[1][1], b[1]], 
                               [0                    , 0                    , 1]])
    
    return res 


def trans_afin(v, theta:float, s, b) -> np.ndarray : 
    
    v = np.asarray(v, dtype=float) 
    s = np.asarray(s, dtype=float) 
    b = np.asarray(b, dtype=float)
    
    m_afin:np.ndarray = afin(theta, s, b) 
    v_extendido:np.ndarray = np.array([v[0], v[1], 1.0])
    
    res_aux:np.ndarray = calcularAx(m_afin, v_extendido, vector_fila=True) 
    res:np.ndarray = np.array([res_aux[0], res_aux[1]])
    
    return res 


# ----------------------------------- Laboratorio 03 -------------------------------------------------------- # 

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


def condMC(A:np.ndarray, p:int | str, Np:int) -> float : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 
    
    if (Np == None) :
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


# ----------------------------------- Laboratorio 04 -------------------------------------------------------- # 

def calculaLU(A:np.ndarray) :
    
    A = np.array(A, dtype = np.float64)
    filas = A.shape[0]
    
    # Inicializamos L como identidad, U como copia de A.
    L = np.eye(filas, dtype = np.float64)
    U = A.copy().astype(np.float64)
    nops = 0
    
    for k in range(0, filas-1) :
        if abs(U[k, k]) == 0 :  # Pivote cero => no factorizable.
            return None, None, 0
        
        for i in range(k+1, filas) :
            L[i, k] = U[i, k] / U[k, k] 
            nops += 1  # División.
            
            for j in range(k, filas) :
                U[i, j] = U[i, j] - L[i, k] * U[k, j] 
                
                if (i != j) : 
                    nops += 2  # Multiplicación + Resta
    
    return L, U, nops 


def res_tri(L:np.ndarray, b:np.ndarray, inferior:bool = True) : 
    
    L = np.array(L, dtype = np.float64) 
    b = np.array(b, dtype = np.float64) 
    filas = L.shape[0] 
    res = np.zeros(filas, dtype = np.float64) 
    
    if (inferior) :
        for i in range(0, filas) :
            suma = 0.0
            
            for j in range(0, i):
                suma += L[i, j] * res[j]
            res[i] = (b[i] - suma) / L[i, i] 
    
    else :
        for i in reversed(range(0, filas)) :
            suma = 0.0
            
            for j in range(i + 1, filas) :
                suma += L[i, j] * res[j] 
                
            res[i] = (b[i] - suma) / L[i, i]
    
    return res 


def inversa(A:np.ndarray) : 
    
    A = np.array(A, dtype = np.float64)
    filas = A.shape[0]
    
    L, U, nops = calculaLU(A) 
    
    if ((L is None) or (U[filas - 1, filas - 1] == 0)) : 
        print("La matriz es Singular, no tiene Inversa.")
        return None 
    
    identidad:np.ndarray = np.eye(filas, dtype = np.float64)
    res = np.zeros((filas, filas), dtype = np.float64)
    
    for i in range(0, filas) :
        e = identidad[:, i]   # Tomo la columna 'i'.
        
        y = res_tri(L, e)   # L es Triengular Inferior.
        x = res_tri(U, y, inferior = False)   # U es Triangular Superior.
        
        res[:, i] = x   # Reemplazo la columna de 'res' por el vector columna que obtuve de resolver los sistemas triagulares.
    
    return res 


def calculaLDV(A:np.ndarray) :
    
    A = np.array(A, dtype = np.float64)
    filas = A.shape[0]

    # Factorizo A = L U
    L, U, nops = calculaLU(A)
    
    if ((L is None) or (U is None)) :
        return None, None, None, 0

    # D = diagonal con los pivotes de U.
    D = np.zeros((filas, filas), dtype = np.float64)
    for i in range(0, filas) :
        D[i, i] = U[i, i]

    # V = U normalizado (cada fila dividida por su pivote).
    V = np.zeros_like(U, dtype = np.float64)
    for i in range(filas):
        if (abs(D[i, i]) == 0) :  # Pivote Nulo -> No Factorizable. 
            return None, None, None, 0
        V[i, :] = U[i, :] / D[i, i]

    return L, D, V, nops 


def esSDP(A:np.ndarray, atol:float = 1e-8) -> bool : 
    
    A = np.array(A, dtype = np.float64)

    # Chequeo simetría.
    if (not esSimetrica(A)) :
        print("La matriz de entrada no es simétrica.")
        return False

    # Hago la Factorización LDV.
    L, D, V, nops = calculaLDV(A)
    if ((L is None) or (D is None) or (V is None)) : 
        print("La matriz de entrada no tiene descomposición A = L D V.")
        return False

    # Reviso que los elementos de la diagonal de D sean positivos.
    for i in range(0, D.shape[0]) :
        if (D[i, i] <= atol) :
            print("La matriz de entrada NO es Simétrica Definida Positiva.")
            return False

    return True


# ----------------------------------- Laboratorio 05 -------------------------------------------------------- #

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
        return QR_con_GS(A, tol = tol, retorna_nops = retorna_nops)
    
    elif (metodo == 'RH') :
        return QR_con_HH(A, tol = tol)
    
    else :
        return None


# ----------------------------------- Laboratorio 06 -------------------------------------------------------- # 

def metpot2k(A:np.ndarray, tol:float = 1e-15, K:int = 1000) : 
    
    filas, columnas = np.shape(A) 
    
    # Checkeo que la matriz sea cuadrada.
    if (filas != columnas) : 
        return None 
    
    # Genero un vector aleatorio (proveniente de una distribución normal).
    v:np.ndarray = np.random.rand(filas).astype(np.float64)
    norma_v:float = norma(v, 2)
    
    # Atajo el caso de que la norma del vector aleatorio sea 0 (más adelante no quiero dividir por cero).
    if (norma_v < tol) :
        v = np.ones(filas, dtype=np.float64)
    else:
        v = v / norma_v
    
    # Definición de fA función auxiliar.
    def fA(A_local:np.ndarray, vector:np.ndarray) : 
        
        Av = calcularAx(A_local, vector, vector_fila = True)
        
        norma_Av = norma(Av, 2)
        if (norma_Av < tol) :
            return np.zeros_like(Av)
        
        return (Av / norma_Av) 
    
    # Aplico dos veces la matriz A (la transformación 'f') al vector 'v'.
    v_tilde:np.ndarray = fA(A, fA(A, v)) 
    
    # Acá me dicen que haga 'transpuesto(v_virulete) * v', es lo mismo que hacer producto interno entre ambos. 
    e:float = producto_interno(v_tilde, v) 
    
    iteraciones:int = 0 
    
    while ((abs(e - 1) > tol) and (iteraciones < K)) : 
        v = v_tilde 
        v_tilde = fA(A, fA(A, v)) 
        e = producto_interno(v_tilde, v) 
        iteraciones += 1 
    
    Av_tilde:np.ndarray = calcularAx(A, v_tilde, vector_fila = True) 
    
    autovalor:float = producto_interno(v_tilde, Av_tilde) 
    
    return v_tilde, autovalor, iteraciones 


def diagRH(A:np.ndarray, tol:float = 1e-15, K:int = 1000) :

    A = np.array(A, dtype = np.float64)

    # Verificación de simetría (numérica, no exacta).
    if (not matricesIguales(A, traspuesta(A), tol)) :
        return None, None

    n = A.shape[0]

    # Caso Base 1.
    if (n == 1) :
        return np.array([[1.0]], dtype = np.float64), np.array([[A[0, 0]]], dtype = np.float64)

    # Primer Autovector y Autovalor usando método de la potencia.
    v1, l1, _ = metpot2k(A, tol, K)

    # Construyo el reflector de Householder.
    e1 = np.zeros(n, dtype = np.float64)
    e1[0] = 1.0
    u = e1 - v1

    denom = producto_interno(u, u)
    if (denom < tol) :   # No queremos dividir por cero.
        H_v1 = np.eye(n, dtype = np.float64) 
        
    else:
        # H = I - 2 * (u u^T) / (u^T u)
        u_col = np.array([[ui] for ui in u], dtype = np.float64)
        u_row = np.array([u], dtype = np.float64)
        uuT = multiplicar_matrices(u_col, u_row)
        vvT = uuT / denom 
        H_v1 = np.eye(n, dtype = np.float64) - 2.0 * vvT

    # Transformación intermedia B = H · A · H^T
    B = multiplicar_matrices(H_v1, multiplicar_matrices(A, traspuesta(H_v1)))
    B[np.abs(B) < tol] = 0.0  # Si me quedaron números que (por la tolerancia) los consideramos cero, los limpio a cero. 

    # Caso base 2.
    if (n == 2) :
        S = H_v1
        D = B
        return S, D

    # Paso recursivo.
    A_tilde = B[1:, 1:]
    S_tilde, D_tilde = diagRH(A_tilde, tol, K)

    # Construcción de D.
    D = np.eye(n, dtype = np.float64)
    D[0, 0] = l1
    D[1:, 1:] = D_tilde

    # Construcción de S.
    S_Aux = np.eye(n, dtype = np.float64)
    S_Aux[1:, 1:] = S_tilde
    S = multiplicar_matrices(H_v1, S_Aux)

    return S, D 


# ----------------------------------- Laboratorio 07 -------------------------------------------------------- # 

def transiciones_al_azar_continuas(n:int) -> np.ndarray : 
    
    res:np.ndarray = np.random.rand(n, n) 
    
    for columna in range(0, n) : 
            
        suma:float = np.sum(res[:, columna]) 
        
        # Normalizo "a la suma" las columnas.
        for fila in range(0, n) : 
            
            res[fila, columna] = res[fila, columna] / suma 
    
    return res 


def transiciones_al_azar_uniformes(n:int, thres:float) -> np.ndarray : 
    
    res:np.ndarray = np.random.rand(n, n) 
    
    for fila in range(0, n) : 
        
        for columna in range(0, n) : 
            
            if (res[fila, columna] <= thres) : 
                res[fila, columna] = 1 
                
            else : 
                res[fila, columna] = 1 
                
    # Ahora normalizo "a la suma" las columnas.
    for columna in range(0, n) : 
        
        suma:int = np.sum(res[:, columna]) 
        
        for fila in range(0, n) : 
            
            res[fila, columna] = res[fila, columna] / suma 
        
    return res 


def nucleo(A:np.ndarray, tol:float = 1e-15) -> np.ndarray : 
    
    A = multiplicar_matrices(A, traspuesta(A)) 
    n:int = A.shape[0]
    
    # 'vec' matriz de autovectores, 'val' matriz de autovalores. 
    vec, val = diagRH(A) 
    
    # Quiero saber cuántas veces 0 es autovector, y en qué columnas está.
    posiciones:list[int] = [] 
    for i in range(0, val.shape[0]) : 
        if (val[i, i] <= tol) : 
            pos:int = i   # Lo copio para evitar problemas con punteros al meter este valor en una lista. 
            posiciones.append(pos) 
    
    if (len(posiciones) == 0) : 
        return np.zeros((0,))
    
    else : 
        k = len(posiciones)
        res = np.zeros((n, k)) 
        index_aux:int = 0 
        
        for i in posiciones : 
            autovector:np.ndarray = np.array(vec[:, i]) 
            
            res[:,index_aux] = autovector 
            index_aux += 1 
        
        return res 


def crea_rala(listado:list[list[float]], m_filas:int, n_columnas:int, tol:float = 1e-15) -> list[dict[int, tuple[int, float]], tuple[int, int]] : 
    
    dict_res:dict[int, tuple[int, float]] = {} 
    
    # Caso Borde -> listado = [] (lista vacía) 
    if (len(listado) == 0) : 
        return [dict_res, (m_filas, n_columnas)] 
    
    # Ahora sí, vamos con la vaina.
    for index in range(0, len(listado[1])) : 
        
        valor:int = listado[2][index] 
        
        if (valor <= tol) : 
            continue   # Valores menores/iguales a la toleracia se descartan (según enunciado).
        
        fila:int = listado[0][index] 
        columna:int = listado[1][index] 
        
        dict_res[(fila, columna)] = valor 
    
    dim:tuple[int, int] = (m_filas, n_columnas) 
    
    return [dict_res, dim] 


# Fin. 