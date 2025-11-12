"""
Laboratorio 4: Factorización y Descomposición LU.
Optimizaciones y nuevos tests.

Funciones del Módulo ALC.
"""


# %% 

# Librerias y Herramientas.

import numpy as np 
from Optimizaciones_Labo_00 import esSimetrica 


# %% 

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


# Test -> 'calculaLU()'

print("TESTS calculaLU")

L0 = np.array([[1,0,0],
               [0,1,0],
               [1,1,1]])

U0 = np.array([[10,1,0],
               [0,2,1],
               [0,0,1]])

A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(np.allclose(L,L0))
assert(np.allclose(U,U0))


L0 = np.array([[1,0,0],
               [1,1.001,0],
               [1,1,1]])

U0 = np.array([[1,1,1],
               [0,1,1],
               [0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(not np.allclose(L,L0))
assert(not np.allclose(U,U0))
assert(np.allclose(L,L0,atol=1e-3))
assert(np.allclose(U,U0,atol=1e-3))
assert(nops == 13)

L0 = np.array([[1,0,0],
               [1,1,0],
               [1,1,1]])

U0 = np.array([[1,1,1],
               [0,0,1],
               [0,0,1]])

A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(L is None)
assert(U is None)
assert(nops == 0)

assert(calculaLU(None) == (None, None, 0))

assert(calculaLU(np.array([[1,2,3],[4,5,6]])) == (None, None, 0))

print("-----ÉXITO!!!!\n") 


# %% 

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


# Test LDV:
print("TESTS calculaLDV")

L0 = np.array([[1,0,0],[1,1.,0],[1,1,1]])
D0 = np.diag([1,2,3])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ D0 @ V0
L,D,V = calculaLDV(A)
assert(np.allclose(L,L0))
assert(np.allclose(D,D0))
assert(np.allclose(V,V0))


L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
D0 = np.diag([3,2,1])
V0 = np.array([[1,1,1],[0,1,1],[0,0,1.001]])
A =  L0 @ D0  @ V0
L,D,V = calculaLDV(A)
assert(np.allclose(L,L0,1e-3))
assert(np.allclose(D,D0,1e-3))
assert(np.allclose(V,V0,1e-3))

print("-----ÉXITO!!!!\n") 


# %% 

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


# TESTS SDP
print("TESTS esSDP")

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
D0 = np.diag([1,1,1])
A = L0 @ D0 @ L0.T
assert(esSDP(A))

D0 = np.diag([1,-1,1])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

D0 = np.diag([1,1,1e-16])
A = L0 @ D0 @ L0.T
assert(not esSDP(A))

L0 = np.array([[1,0,0],
               [1,1,0],
               [1,1,1]])
D0 = np.diag([1,1,1])
V0 = np.array([[1,0,0],
               [1,1,0],
               [1,1+1e-3,1]]).T
A = L0 @ D0 @ V0
assert(esSDP(A,1e-3))

print("-----ÉXITO!!!!\n") 


# %% 

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


# TESTS res_tri.
print("TESTS res_tri")

A = np.array([[1,0,0],
              [1,1,0],
              [1,1,1]])

b = np.array([1,1,1])
assert(np.allclose(res_tri(A,b),np.array([1,0,0])))

b = np.array([0,1,0])
assert(np.allclose(res_tri(A,b),np.array([0,1,-1])))

b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b),np.array([-1,2,-2])))

b = np.array([-1,1,-1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([-1,1,-1])))

A = np.array([[3,2,1],[0,2,1],[0,0,1]])
b = np.array([3,2,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1/3,1/2,1])))

A = np.array([[1,-1,1],[0,1,-1],[0,0,1]])
b = np.array([1,0,1])
assert(np.allclose(res_tri(A,b,inferior=False),np.array([1,1,1])))
print("-----ÉXITO!!!!\n") 


# %% 

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


# Test inversa.
print("TESTS inversa")

def esSingular(A):
    try:
        np.linalg.inv(A)
        return False
    except:
        return True

# Por que no siempre es invertible, hacemos varios tests
ntest = 10
for i in range(ntest):
    A = np.random.random((4,4))
    A_ = inversa(A)
    if not esSingular(A):
        inversaConNumpy = np.linalg.inv(A)
        assert(A_ is not None)
        assert(np.allclose(inversaConNumpy,A_))
    else: 
        assert(A_ is None)

# Matriz singular devería devolver None
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
assert(inversa(A) is None)

print("-----ÉXITO!!!!\n")


# Fin. 