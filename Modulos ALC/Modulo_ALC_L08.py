"""
Laboratorio 8: Singular Values Descomposition.

Funciones del Módulo ALC.
"""


# %% 

# Librerias y Herramientas.

import numpy as np 
from Modulo_ALC import norma, traspuesta, multiplicar_matrices, diagRH, calcularAx


# %% 

''' 
Recibe una matriz y retorna la matriz con sus columnas normalizadas.
'''
def normaliza_columnas(matriz:np.ndarray, p:int, tol:float = 1e-15) -> np.ndarray : 
    
    matriz = np.array(matriz, dtype = np.float64)
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


def svd_reducida_2(A:np.ndarray, k = "max", tol:float = 1e-15) : 
    
    A = np.array(A, dtype = np.float64)
    filas, columnas = A.shape
    
    trans = False
    if (filas < columnas) :
        A = traspuesta(A)
        trans = True
        filas, columnas = A.shape   # Actualizo las variables de dimension. 
        
    B = multiplicar_matrices(traspuesta(A), A)
    
    V, D = diagRH(B, tol)
    
    valores_singulares:list[float] = []   # No nulos.

    if (k == "max") :
        k = len(D)
   
    cantidad_autovalores:int = 0
    
    for i in range(0, k) :
        
        if (D[i, i] <= tol) :
            break
           
        cantidad_autovalores += 1
        valores_singulares.append(np.sqrt(D[i, i]))

    k = min(k, cantidad_autovalores)
    valores_singulares = np.array(valores_singulares, dtype = np.float64) 
    
    V = V[:, :k]   # Nos quedamos con los primeros 'k' autovectores.
    V = normaliza_columnas(V, 2)   # Normalizamos las columnas (para que quede unitaria). 
    
    U = np.zeros((filas, k), dtype = np.float64) 
    
    for index in range(0, k) : 
        
        if (valores_singulares[index] <= tol) :
            
            U[:, index] = 0
            
        else : 
            
            U[:, index] = calcularAx(A, V[:, index], vector_fila = True) / valores_singulares[index]
            
    U = normaliza_columnas(U, 2)   # Normalizamos (para que quede unitaria). 
    
    if (trans) :
        
        return V, valores_singulares, U
    
    else:
        
        return U, valores_singulares, V 
    

def svd_reducida(A:np.ndarray, k = "max", tol:float = 1e-15, reducida:bool = True) : 
    
    A = np.array(A, dtype = np.float64)
    filas, columnas = A.shape
    
    trans = False
    if (filas < columnas) :
        A = traspuesta(A)
        trans = True
        filas, columnas = A.shape   # Actualizo las variables de dimension. 
        
    B = multiplicar_matrices(traspuesta(A), A)
    
    V, D = diagRH(B, tol)
    
    valores_singulares:list[float] = []   # No nulos.
    
    for i in range(len(D)) :
        
        if (D[i, i] <= tol) :
            break
           
        valores_singulares.append(np.sqrt(D[i, i]))

    if k == "max":
        k = len(valores_singulares)
    else:
        k = min(k, len(valores_singulares))
    
    # Si es reducida, recortar
    if reducida:
        valores_singulares = valores_singulares[:k]
        V = V[:, :k]
    
    V = normaliza_columnas(V, 2)
    U = np.zeros((filas, filas if not reducida else k), dtype=np.float64)
    
    for i in range(k):
        
        if valores_singulares[i] <= tol:
            U[:, i] = 0
            
        else:
            U[:, i] = calcularAx(A, V[:, i], vector_fila=True) / valores_singulares[i]
            
    U = normaliza_columnas(U, 2)   # Normalizamos (para que quede unitaria). 
    
    if (trans) :
        
        return V, valores_singulares, U
    
    else:
        
        return U, valores_singulares, V
    

# %%

# Funciones Auxiliares para Testeo. 

def genera_matriz_para_test(m,n=2,tam_nucleo=0):
    if tam_nucleo == 0:
        A = np.random.random((m,n))
    else:
        A = np.random.random((m,tam_nucleo))
        A = np.hstack([A,A])
    return(A) 


# %% 

# Test -> 'svd_reducida()' 

def test_svd_reducida_mn(A,tol=1e-15):
    m,n = A.shape
    hU,hS,hV = svd_reducida(A,tol=tol)
    nU,nS,nVT = np.linalg.svd(A)
    r = len(hS)+1
    assert np.all(np.abs(np.abs(np.diag(hU.T @ nU))-1)<10**r*tol), 'Revisar calculo de hat U en ' + str((m,n))
    assert np.all(np.abs(np.abs(np.diag(nVT @ hV))-1)<10**r*tol), 'Revisar calculo de hat V en ' + str((m,n))
    assert len(hS) == len(nS[np.abs(nS)>tol]), 'Hay cantidades distintas de valores singulares en ' + str((m,n))
    assert np.all(np.abs(hS-nS[np.abs(nS)>tol])<10**r*tol), 'Hay diferencias en los valores singulares en ' + str((m,n))

for m in [2,5,10,20]:
    for n in [2,5,10,20]:
        for _ in range(10):
            A = genera_matriz_para_test(m,n)
            test_svd_reducida_mn(A)


# Matrices con nucleo

m = 12
for tam_nucleo in [2,4,6]:
    for _ in range(10):
        A = genera_matriz_para_test(m,tam_nucleo=tam_nucleo)
        test_svd_reducida_mn(A)

# Tamaños de las reducidas
A = np.random.random((8,6))
for k in [1,3,5]:
    hU,hS,hV = svd_reducida(A,k=k)
    assert hU.shape[0] == A.shape[0], 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[0] == A.shape[1], 'Dimensiones de hV incorrectas(caso a)'
    assert hU.shape[1] == k, 'Dimensiones de hU incorrectas (caso a)'
    assert hV.shape[1] == k, 'Dimensiones de hV incorrectas(caso a)'
    assert len(hS) == k, 'Tamaño de hS incorrecto'


print("Todos los test de 'svd_reducida()' pasados correctamente. \n")


# %% 

print("Si se imprime esto, es porque todos los test pasaron exitosamente!")


# Fin. 
