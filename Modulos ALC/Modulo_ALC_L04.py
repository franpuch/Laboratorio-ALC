"""
Laboratorio 4: Factorización y Descomposición LU.

Funciones del Módulo ALC.
"""


# %% 

# Librerias y Herramientas.

import numpy as np 


# %% 

def calculaLU(A:np.ndarray) :
    
    A = np.array(A, dtype=np.float64)
    filas = A.shape[0]
    
    # Inicializamos L como identidad, U como copia de A.
    L = np.eye(filas, dtype=np.float64)
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
                nops += 2  # Multiplicación + Resta
    
    return L, U, nops


"""
Versión para ver cómo trabaja iteración a iteración.
"""

def calculaLU_B(A:np.ndarray, debug:bool = True) :
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    
    L = np.eye(n, dtype=np.float64)
    U = A.copy().astype(np.float64)
    nops = 0
    
    if debug:
        print("Matriz inicial A:")
        print(A)
        print("-"*40)
    
    for k in range(n-1):
        if abs(U[k, k]) < 1e-15:
            if debug:
                print(f"Pivote nulo en fila {k}, no se puede factorizar")
            return None, None, 0
        
        if debug:
            print(f"\n== Iteración k = {k} ==")
        
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]; nops += 1
            if debug:
                print(f"m({i},{k}) = U[{i},{k}] / U[{k},{k}] = {L[i,k]}")
            
            for j in range(k, n):
                antes = U[i, j]
                U[i, j] = U[i, j] - L[i, k] * U[k, j]
                nops += 2
                if debug:
                    print(f"U[{i},{j}] = {antes} - ({L[i,k]} * {U[k,j]}) = {U[i,j]}")
        
        if debug:
            print("L parcial:")
            print(L)
            print("U parcial:")
            print(U)
    
    if debug:
        print("\nFactorización final:")
        print("L =")
        print(L)
        print("U =")
        print(U)
        print(f"Total operaciones = {nops}")
    
    return L, U, nops 


# %% 

def res_tri(L:np.ndarray, b:np.ndarray, inferior:bool = True) : 
    
    L = np.array(L, dtype=np.float64) 
    b = np.array(b, dtype=np.float64) 
    filas = L.shape[0] 
    res = np.zeros(filas, dtype=np.float64) 
    
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


"""
Versión para ver cómo trabaja iteración a iteración.
"""

def res_tri_B(L:np.ndarray, b:np.ndarray, inferior:bool = True, debug:bool = True) :
    
    L = np.array(L, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    filas = L.shape[0]
    res = np.zeros(filas, dtype=np.float64)
    
    if (debug) :
        print("\nResolviendo sistema triangular")
        print("Matriz:")
        print(L)
        print("b =", b)
        print("inferior =", inferior)
    
    if (inferior) :
        for i in range(filas) :
            suma = 0.0
            
            for j in range(i) :
                suma += L[i, j] * res[j]
                
                if (debug) :
                    print(f"sumando L[{i},{j}]*res[{j}] = {L[i,j]}*{res[j]}")
                    
            res[i] = (b[i] - suma) / L[i, i]
            
            if (debug) :
                print(f"res[{i}] = (b[{i}] - {suma}) / {L[i,i]} = {res[i]}") 
                
    else :
        for i in reversed(range(filas)) :
            suma = 0.0
            
            for j in range(i+1, filas) :
                suma += L[i, j] * res[j]
                
                if (debug) :
                    print(f"sumando L[{i},{j}]*res[{j}] = {L[i,j]}*{res[j]}")
            res[i] = (b[i] - suma) / L[i, i]
            if (debug) :
                print(f"res[{i}] = (b[{i}] - {suma}) / {L[i,i]} = {res[i]}")
    
    if (debug) :
        print("Solución final res =", res)
    
    return res 


# %% 

def inversa(A:np.ndarray) : 
    
    A = np.array(A, dtype=np.float64)
    filas = A.shape[0]
    
    L, U, nops = calculaLU_B(A) 
    
    if (L is None) : 
        print("La matriz es Singular, no tiene Inversa.")
        return None 
    
    identidad:np.ndarray = np.eye(filas, dtype=np.float64)
    res = np.zeros((filas, filas), dtype=np.float64)
    
    for i in range(0, filas) :
        e = identidad[:, i]   # Tomo la columna 'i'.
        
        y = res_tri(L, e)   # L es Triengular Inferior.
        x = res_tri(U, y, inferior = False)   # U es Triangular Superior.
        
        res[:, i] = x   # Reemplazo la columna de 'res' por el vector columna que obtuve de resolver los sistemas triagulares.
    
    return res 


# %% 

# Test -> 'calculaLU()' 

L0 = np.array([[1,0,0],[0,1,0],[1,1,1]])
U0 = np.array([[10,1,0],[0,2,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(np.allclose(L,L0))
assert(np.allclose(U,U0))

L0 = np.array([[1,0,0],[1,1.001,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,1,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(not np.allclose(L,L0))
assert(not np.allclose(U,U0))
assert(np.allclose(L,L0,atol=1e-3))
assert(np.allclose(U,U0,atol=1e-3))
assert(nops == 13)

L0 = np.array([[1,0,0],[1,1,0],[1,1,1]])
U0 = np.array([[1,1,1],[0,0,1],[0,0,1]])
A =  L0 @ U0
L,U,nops = calculaLU(A)
assert(L is None)
assert(U is None)
assert(nops == 0) 

print("Todos los test de 'calculaLU()' pasados correctamente.") 


# %% 

# Test -> 'res_tri()' 

A = np.array([[1,0,0],[1,1,0],[1,1,1]])
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

print("Todos los test de 'res_tri()' pasados correctamente.")


# %% 

# Test -> 'inversa()' 

ntest = 10
iter = 0
while iter < ntest:
    A = np.random.random((4,4))
    A_ = inversa(A)
    if not A_ is None:
        assert(np.allclose(np.linalg.inv(A),A_))
        iter += 1

# Matriz singular devería devolver None
A = np.array([[1,2,3],[4,5,6],[7,8,9]])
assert(inversa(A) is None) 

print("Todos los test de 'inversa()' pasados correctamente.") 


# %% 

print("Si se imprime esto, es porque todos los test pasaron exitosamente!")


# Fin. 