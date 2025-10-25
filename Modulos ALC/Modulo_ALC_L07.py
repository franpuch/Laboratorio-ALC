"""
Laboratorio 7: Matrices de Transición - Markov.

Funciones del Módulo ALC.
"""


# %% 

# Librerias y Herramientas.

import numpy as np 
from Modulo_ALC import multiplicar_matrices, traspuesta, diagRH 


# %% 

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


''' 
Falta implementar 'multiplicar_rala_vector(A, v)'
'''


# %% 

# Funciones Auxiliares para Testeo. 

def es_markov(T,tol=1e-6):
    """
    T una matriz cuadrada.
    tol la tolerancia para asumir que una suma es igual a 1.
    Retorna True si T es una matriz de transición de Markov (entradas no negativas y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    n = T.shape[0]
    for i in range(n):
        for j in range(n):
            if T[i,j]<0:
                return False
    for j in range(n):
        suma_columna = sum(T[:,j])
        if np.abs(suma_columna - 1) > tol:
            return False
    return True


def es_markov_uniforme(T,thres=1e-6):
    """
    T una matriz cuadrada.
    thres la tolerancia para asumir que una entrada es igual a cero.
    Retorna True si T es una matriz de transición de Markov uniforme (entradas iguales a cero o iguales entre si en cada columna, y columnas que suman 1 dentro de la tolerancia), False en caso contrario.
    """
    if not es_markov(T,thres):
        return False
    # cada columna debe tener entradas iguales entre si o iguales a cero
    m = T.shape[1]
    for j in range(m):
        non_zero = T[:,j][T[:,j] > thres]
        # all close
        close = all(np.abs(non_zero - non_zero[0]) < thres)
        if not close:
            return False
    return True


def esNucleo(A,S,tol=1e-5):
    """
    A una matriz m x n
    S una matriz n x k
    tol la tolerancia para asumir que un vector esta en el nucleo.
    Retorna True si las columnas de S estan en el nucleo de A (es decir, A*S = 0. Esto no chequea si es todo el nucleo
    """
    for col in S.T:
        res = A @ col
        if not np.allclose(res,np.zeros(A.shape[0]), atol=tol):
            return False
    return True


# %%

# Test -> 'transiciones_al_azar_continuas()' y 'transicion_al_azar_uniforme()' 

for i in range(1,100):
    T = transiciones_al_azar_continuas(i)
    assert es_markov(T), f"transiciones_al_azar_continuas fallo para n={i}"
    
    T = transiciones_al_azar_uniformes(i,0.3)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    # Si no atajan casos borde, pueden fallar estos tests. Recuerden que suma de columnas DEBE ser 1, no valen columnas nulas.
    T = transiciones_al_azar_uniformes(i,0.01)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}"
    T = transiciones_al_azar_uniformes(i,0.01)
    assert es_markov_uniforme(T), f"transiciones_al_azar_uniformes fallo para n={i}" 


print("Todos los test de 'transiciones_al_azar_continuas()' y 'transiciones_al_azar_uniformes()' pasados correctamente. \n") 


# %% 

# Test -> 'nucleo()' 

A = np.eye(3)
S = nucleo(A)
assert S.shape[0]==0, "nucleo fallo para matriz identidad"
A[1,1] = 0
S = nucleo(A)
msg = "nucleo fallo para matriz con un cero en diagonal"
assert esNucleo(A,S), msg
assert S.shape==(3,1), msg
assert abs(S[2,0])<1e-2, msg
assert abs(S[0,0])<1e-2, msg

v = np.random.random(5)
v = v / np.linalg.norm(v)
H = np.eye(5) - np.outer(v, v)  # proyección ortogonal
S = nucleo(H)
msg = "nucleo fallo para matriz de proyeccion ortogonal"
assert S.shape==(5,1), msg
v_gen = S[:,0]
v_gen = v_gen / np.linalg.norm(v_gen)
assert np.allclose(v, v_gen) or np.allclose(v, -v_gen), msg 


print("Todos los test de 'nucleo()' pasados correctamente. \n") 


# %% 

# Test -> 'crea_rala()' 

listado = [[0,17],[3,4],[0.5,0.25]]
A_rala_dict, dims = crea_rala(listado,32,89)
assert dims == (32,89), "crea_rala fallo en dimensiones"
assert A_rala_dict[(0,3)] == 0.5, "crea_rala fallo"
assert A_rala_dict[(17,4)] == 0.25, "crea_rala fallo"
assert len(A_rala_dict) == 2, "crea_rala fallo en cantidad de elementos"

listado = [[32,16,5],[3,4,7],[7,0.5,0.25]]
A_rala_dict, dims = crea_rala(listado,50,50)
assert dims == (50,50), "crea_rala fallo en dimensiones con tol"
assert A_rala_dict.get((32,3)) == 7
assert A_rala_dict[(16,4)] == 0.5
assert A_rala_dict[(5,7)] == 0.25

listado = [[1,2,3],[4,5,6],[1e-20,0.5,0.25]]
A_rala_dict, dims = crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con tol"
assert (1,4) not in A_rala_dict
assert A_rala_dict[(2,5)] == 0.5
assert A_rala_dict[(3,6)] == 0.25
assert len(A_rala_dict) == 2

# caso borde: lista vacia. Esto es una matriz de 0s
listado = []
A_rala_dict, dims = crea_rala(listado,10,10)
assert dims == (10,10), "crea_rala fallo en dimensiones con lista vacia"
assert len(A_rala_dict) == 0, "crea_rala fallo en cantidad de elementos con lista vacia" 


print("Todos los test de 'crea_rala()' pasados correctamente. \n") 


# %% 

print("Si se imprime esto, es porque todos los test pasaron exitosamente!")


# Fin. 