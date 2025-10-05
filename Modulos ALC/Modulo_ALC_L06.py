"""
Laboratorio 6: Método de la Potencia.

Funciones del Módulo ALC.
"""


# %% 

# Librerias y Herramientas.

import numpy as np 
from Modulo_ALC import calcularAx, norma, traspuesta, esCuadrada, esSimetrica
from Funciones_Varias import producto_interno, multiplicar_matrices
from Modulo_ALC import normaExacta   # Esta la utilizan en los Test.


# %% 

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
    
    # Acá me dicen que haga 'transpuesto(v_virulete) * v', es lo mismo que hacer producto interno entre ambos (esa función ya 
    # la tengo). 
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


# %% 

'''
Hay que corregirla. Hay algo que está funcionando mal.
'''

def diagRH(A:np.ndarray, tol:float = 1e-15, K = 1000) : 
    
    # Chequeo si es cuadrada y simétrica.
    if ((not esCuadrada(A)) or (not esSimetrica(A))) :
        return None, None
    
    n = np.shape(A)[0] 
    
    # Es una función recursiva, necesito un Caso Base.
    if (n == 1) :
        return np.eye(1), np.array([[A[0, 0]]], dtype=np.float64)
    
    v1, l1, _ = metpot2k(A, tol, K) 

    # A 'v1' le resto el vector canónico 'e1'.
    e1 = np.zeros_like(v1)
    e1[0] = 1.0
    v_aux = e1 - v1 
    
    denom = producto_interno(v_aux, v_aux) 
    
    if (denom < tol) :  # Evita división por cero.
        H_v1 = np.eye(n, dtype=np.float64) 
    
    else:
        v_aux = v_aux / denom
        
        v_aux_col = np.array([[ui] for ui in v_aux], dtype=np.float64)
        v_aux_row = np.array([v_aux], dtype=np.float64)
        vvT = multiplicar_matrices(v_aux_col, v_aux_row)
        
        # vvT = multiplicar_matrices(traspuesta(v_aux), v_aux) 
        H_v1 = np.eye(n, dtype=np.float64) - 2.0 * vvT
    
    if (n == 2) : 
        S = H_v1 
        D = multiplicar_matrices(H_v1, multiplicar_matrices(A, traspuesta(H_v1))) 
        return S, D 
        
    else : 
        B = multiplicar_matrices(H_v1, multiplicar_matrices(A, traspuesta(H_v1))) 
        A_tilde = B[1:,1:] 
        
        S_tilde, D_tilde = diagRH(A_tilde, tol, K) 
        
        D = np.eye(n, dtype = np.float64) 
        D[0,0] = l1 
        D[1:, 1:] = D_tilde 
        
        S_Aux = np.eye(n, dtype = np.float64) 
        S_Aux[1:, 1:] = S_tilde
        S = multiplicar_matrices(H_v1, S_Aux) 
        
        return S, D 


# %% 

# Test -> 'metpot2k()' 

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


print("Todos los test de 'metpot2k()' pasados correctamente.") 


# %% 

# Test -> 'diagRH()' 

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


print("Todos los test de 'diagRH()' pasados correctamente.")


# %% 

print("Si se imprime esto, es porque todos los test pasaron exitosamente!")


# Fin. 
