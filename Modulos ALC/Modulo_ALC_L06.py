"""
Laboratorio 6: Método de la Potencia.

Funciones del Módulo ALC.
"""


# %% 

# Librerias y Herramientas.

import numpy as np 
from Modulo_ALC import calcularAx, norma 
from Funciones_Varias import producto_interno


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

def diagRH(A:np.ndarray, tol:float = 1e-15, K = 1000) : 
    return None


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

print("Si se imprime esto, es porque todos los test pasaron exitosamente!")


# Fin. 
