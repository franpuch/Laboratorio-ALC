"""
Laboratorio 3: Norma y Número de Condición.

Funciones del Módulo ALC.
Nuevos Test. 
"""


# %% 

# Librerias y Herramientas.

import numpy as np 
from Modulo_ALC_Casi_Final import calcularAx


# %%  

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
        
        if not(np.any(elemento)) :   # Es el vector nulo. 
            res.append(elemento)
        
        else: 
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


def condMC(A:np.ndarray, p:int | str) -> float : 
    
    A_inversa:np.ndarray = np.linalg.inv(A) 

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

# Tests norma
print("TESTS NORMA")
assert(np.allclose(norma(np.array([0,0,0,0]),1), 0))
assert(np.allclose(norma(np.array([4,3,-100,-41,0]),"inf"), 100))
assert(np.allclose(norma(np.array([1,1]),2),np.sqrt(2)))
assert(np.allclose(norma(np.array([1]*10),2),np.sqrt(10)))
assert(norma(np.random.rand(10),2)<=np.sqrt(10))
assert(norma(np.random.rand(10),2)>=0)

print("------ÉXITO!!!!\n")

# Tests normaliza
print("TEST NORMALIZA")


# caso borde
print("---TEST NORMALIZA NULO")
test_borde = normaliza([np.array([0,0,0,0])],2)
assert(len(test_borde) == 1)
assert(np.allclose(test_borde[0],np.array([0,0,0,0])))
print("------ÉXITO!!!!")


# normaliza norma 2
print("---TEST NORMALIZA 2")
test_n2 = normaliza([np.array([1]*k) for k in range(1,11)],2)
assert(len(test_n2) != 0)
for x in test_n2:
    assert(np.allclose(norma(x,2),1))
print("------ÉXITO!!!!")

# normaliza norma 1
print("---TEST NORMALIZA 1")
test_n1 = normaliza([np.array([1]*k) for k in range(2,11)],1)
assert(len(test_n1) != 0)
for x in test_n1:
    assert(np.allclose(norma(x,1),1))
print("------ÉXITO!!!!")

# normaliza norma inf
print("---TEST NORMALIZA INF")
test_nInf = normaliza([np.random.rand(k) for k in range(1,11)],'inf')
assert(len(test_nInf) != 0)
for x in test_nInf:
    assert(np.allclose(norma(x,'inf'),1))

print("------ÉXITO!!!!\n")

# Tests normaExacta
print("TEST normaExacta")

# assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]))[0],2))
assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]), 1),2))

# assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]))[1],2)) 
assert(np.allclose(normaExacta(np.array([[1,-1],[-1,-1]]), 'inf'),2))

# assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]))[0] ,6))
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]), 1) ,6))

# assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]))[1],7)) 
assert(np.allclose(normaExacta(np.array([[1,-2],[-3,-4]]), 'inf'),7))

assert(normaExacta(np.array([[1,-2],[-3,-4]]),2) is None) 

# assert(normaExacta(np.random.random((10,10)))[0] <=10)
assert(normaExacta(np.random.random((10,10)), 1) <=10)

# assert(normaExacta(np.random.random((4,4)))[1] <=4) 
assert(normaExacta(np.random.random((4,4)), 'inf') <=4)

print("------ÉXITO!!!!\n")

# Test normaMatMC
print("TEST normaMatMC")

nMC = normaMatMC(A=np.eye(2),q=2,p=1,Np=100000)
assert(np.allclose(nMC[0],1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),0,atol=1e-3) or np.allclose(np.abs(nMC[1][1]),0,atol=1e-3))

nMC = normaMatMC(A=np.eye(2),q=2,p='inf',Np=100000)
assert(np.allclose(nMC[0],np.sqrt(2),atol=1e-3))
assert(np.allclose(np.abs(nMC[1][0]),1,atol=1e-3) and np.allclose(np.abs(nMC[1][1]),1,atol=1e-3))

A = np.array([[1,2],[3,4]])
nMC = normaMatMC(A=A,q='inf',p='inf',Np=1000000)
# assert(np.allclose(nMC[0],normaExacta(A)[1],rtol=1e-1)) 
assert(np.allclose(nMC[0],normaExacta(A, 'inf'),rtol=1e-1)) 

print("------ÉXITO!!!!\n")

# Test condMC
print("TEST condMC")

A = np.array([[1,1],[0,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-2))

A = np.array([[3,2],[4,1]])
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
normaA = normaMatMC(A,2,2,10000)
normaA_ = normaMatMC(A_,2,2,10000)
condA = condMC(A,2)
assert(np.allclose(normaA[0]*normaA_[0],condA,atol=1e-2))

print("------ÉXITO!!!!\n")

# Test condExacta
print("TEST condExacta")

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
# normaA = normaExacta(A)[0] 
normaA = normaExacta(A, 1)

# normaA_ = normaExacta(A_)[0] 
normaA_ = normaExacta(A_, 1) 

condA = condExacta(A,1)
assert(np.allclose(normaA*normaA_,condA))

A = np.random.rand(10,10)
A_ = np.linalg.solve(A,np.eye(A.shape[0]))
# normaA = normaExacta(A)[1] 
normaA = normaExacta(A, 'inf')

# normaA_ = normaExacta(A_)[1]
normaA_ = normaExacta(A_, 'inf')

condA = condExacta(A,'inf')
assert(np.allclose(normaA*normaA_,condA))

print("------ÉXITO!!!!\n")

print("---FINALIZADO LABO 3!---") 


# Fin. 