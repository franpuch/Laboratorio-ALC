"""
Laboratorio 6: Método de la Potencia.

Funciones del Módulo ALC.
"""


# %% 

# Librerias y Herramientas.

import numpy as np 
from Modulo_ALC import calcularAx, norma, traspuesta
from Modulo_ALC import producto_interno, multiplicar_matrices, matricesIguales
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
