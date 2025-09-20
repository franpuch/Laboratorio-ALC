"""
Laboratorio 3: Norma y Numero de Condicion

Funciones del Módulo ALC.
"""


# %% 

# Librerias y Herramientas.

import numpy as np 


# %% 

def norma(x:np.ndarray, p:int) -> float :
    
    # Por las dudas, parseo 'x' como array de numpy (para atajar si bien en otro tipo de dato).
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


# %% 

def normaliza(x:np.ndarray, p:int) -> list[list[float]] :
    
    res:list[list[float]] = []
    
    for elemento in x :
        res.append(elemento/norma(elemento, p))
    
    return res 


# %% 

def normaMatMC(A:np.ndarray, q:int, p:int, Np:np.ndarray) -> tuple[float, np.ndarray] :
    
    columnas:int = A.shape[1]
    
    # Creo los vectores aleatorios.
    x = np.random.rand(Np, columnas)
    
    # Ahora normalizo todos los vectores creados en el paso anterior.
    x = normaliza(x, p)
    
    res:float = 0
    vector_max = x[0]   # En un principio, le asigno cualquier vector; en este caso el primero.
    
    for elemento in x :
        
        # Calculo la norma del vector que esta trabajando actualmente.
        norma_actual = norma(A @ (elemento.T), q)
        
        if (res <= norma_actual) :   # Si encontré un vector con norma mayor al que tenía taggeado como mayor antes.
            res = norma_actual
            vector_max = elemento
            
    # Armo la tupla con los elementos que obtuve y lo retorno.
    return (np.float64(res), vector_max) 


# %% 

def normaExacta(A:np.ndarray, p:int) -> float | None : 
    
    # Por las dudas, parseo A como un array de numpy.
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


"""
Versión de esta función mas compacta (pero creo que no puedo usar algunas de estas funciones de numpy).
"""

def normaExacta_V2(A: np.ndarray, p) -> float | None:
    
    A = np.array(A)

    if p == 1:
        # Norma 1 = máximo de la suma de valores absolutos por columna
        return np.max(np.sum(np.abs(A), axis=0))
    elif p == "inf":
        # Norma infinito = máximo de la suma de valores absolutos por fila
        return np.max(np.sum(np.abs(A), axis=1))
    else:
        return None 


# %% 

def condMC(A:np.ndarray, p:int | str, Np:int) -> float : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 
    
    if (Np == None) :
        Np:int = 10000   # Defino esa cantidad de vectores aleatorios (para pasarle a mi función de Norma con Monte Carlo) en caso de que no se le pase ese parámetro.
    
    norma_A:float = normaMatMC(A, p, p, Np)[0] 
    norma_A_inv:float = normaMatMC(A_inversa, p, p, Np)[0] 
    
    return np.float64(norma_A * norma_A_inv) 


# %% 

def condExacta(A:np.ndarray, p:int | str) -> float | None : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 
    
    norma_A:float = normaExacta(A, p) 
    norma_A_inv:float = normaExacta(A_inversa, p) 
    
    if (norma_A == None) or (norma_A_inv == None) : 
        return None 
    
    return np.float64(norma_A * norma_A_inv) 


# %% 

# Test -> 'norma()'

assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

print("Todos los test de 'norma()' pasados correctamente.")


# %% 

# Test -> 'normaliza()' 

for x in normaliza([np.array([1]*k) for k in range(1,11)],2):
    assert(np.allclose(norma(x,2),1))
for x in normaliza([np.array([1]*k) for k in range(2,11)],1):
    assert(not np.allclose(norma(x,2),1) )
for x in normaliza([np.random.rand(k) for k in range(1,11)],'inf'):
    assert( np.allclose(norma(x,'inf'),1) )

print("Todos los test de 'normaliza()' pasados correctamente.")


# %% 

# Test -> 'normaMatMC()' 

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
assert(np.allclose(nMC[0],normaExacta(A,'inf'),rtol=2e-1)) 

print("Todos los test de 'normaMatMC()' pasados correctamente.") 


# %% 

# Test -> 'normaExacta()' 

assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]),1),2))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),1),6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]),'inf'),7))
assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None)
assert(normaExacta(np.random.random((10,10)),1)<=10)
assert(normaExacta(np.random.random((4,4)),'inf')<=4) 

print("Todos los test de 'normaExacta()' pasados correctamente.") 


# %% 

# Test -> 'condMC()' 

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2,10000)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-3))

print("Todos los test de 'condMC()' pasados correctamente.") 


# %% 

# Test -> 'condExacta()' 

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,1)
normaA_ = normaExacta(A_,1)
condA = condExacta(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaExacta(A,'inf')
normaA_ = normaExacta(A_,'inf')
condA = condExacta(A,'inf')
assert(np.allclose(normaA*normaA_,condA)) 

print("Todos los test de 'condExacta()' pasados correctamente.") 


# %% 

print("Si se imprime esto, es porque todos los test pasaron exitosamente!")


# Fin.