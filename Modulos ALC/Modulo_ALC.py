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
    
    - Labo 04 -> 
    
"""

import numpy as np 


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


def traspuesta(matriz: np.ndarray) -> np.ndarray :
    
    if (len(np.shape(matriz)) == 1) :
        n = np.shape(matriz)[0]
        
        return np.array([[matriz[i]] for i in range(n)])
    
    filas, columnas = np.shape(matriz)
    res = [[0 for _ in range(filas)] for _ in range(columnas)]

    for i in range(filas):
        for j in range(columnas):
            res[j][i] = matriz[i][j]

    return np.array(res)


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


def calcularAx(matriz_A:np.ndarray , matriz_x:np.ndarray) -> np.ndarray :
    
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
    
    res:np.ndarray = np.array([[0] for _ in range(0, nro_filas)]) 
    
    for i in range(0, nro_filas) :
        res_parcial: float = 0 
        
        for j in range(0, nro_columnas) :
            res_parcial += matriz_A[i][j] * matriz_x[j]
        
        res[i] = res_parcial 
    
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
    for contador in range(2, 2 * n) : 
        fib.append(fib[-1] + fib[-2]) 
    
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
        
        max_fila:int = fila 
        max_valor:float = abs(matriz[fila][fila]) 
        for i in range(fila, nro_filas) :
            if ((abs(matriz[i][fila])) > max_valor) : 
                max_fila = i 
                max_valor = abs(matriz[i][fila]) 
        
        if (max_fila != fila) :
            intercambiarFilas(matriz, fila, max_fila) 
            
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
    
    res_aux:np.ndarray = m_afin @ v_extendido 
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
        
        norma_actual = norma(A @ (elemento.T), q)
        
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

# ----------------------------------- Laboratorio 05 -------------------------------------------------------- #


# Fin. 