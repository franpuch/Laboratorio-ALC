"""
Laboratorio 1: Numeros de Maquina 

Funciones del Módulo ALC.
"""

# %% 

# Librerias y Herramientas.

import numpy as np 


# %% 

'''
Calcula el Error Absoluto de aproximar 'x' con 'y' en float64.
'''

def error(x, y):
    
    x = np.float64(x) 
    y = np.float64(y) 
    
    return np.abs(x - y)


'''
Calcula el Error Relativo de aproximar 'x' con 'y' en float64.
   - Definición: |x - y| / |x|
   - Caso especial: si x = 0, se devuelve simplemente |y| (para evitar división por cero).
'''

def error_relativo(x, y):

    x = np.float64(x) 
    y = np.float64(y) 
    
    if x == 0:
        return np.abs(y)
    
    return np.abs(x - y) / np.abs(x) 


''' 
Devuelve 'True' si las matrices A y B son exactamente iguales en dimensiones y valores.
En otro caso, devuelve 'False'.
'''

def matricesIguales(A, B):

    # Paso las matrices a arrays de numpy (en caso que no vengan en ese formato por defecto).
    A = np.array(A)
    B = np.array(B)

    if A.shape != B.shape:
        return False

    return np.allclose(A, B) 


# %% 

'''
Test de la Guia.
'''

def sonIguales(x, y, atol=1e-08):
    return np.allclose(error(x,y), 0, atol=atol)

assert(not sonIguales(1,1.1))
assert(sonIguales(1,1 + np.finfo('float64').eps))
assert(not sonIguales(1,1 + np.finfo('float32').eps))
assert(not sonIguales(np.float16(1),np.float16(1) + np.finfo('float32').eps))
assert(sonIguales(np.float16(1),np.float16(1) + np.finfo('float16').eps, atol=1e-3))

assert(np.allclose(error_relativo(1,1.1),0.1))
assert(np.allclose(error_relativo(2,1),0.5))
assert(np.allclose(error_relativo(-1,-1),0))
assert(np.allclose(error_relativo(1,-1),2))

assert(matricesIguales(np.diag([1,1]),np.eye(2)))
assert(matricesIguales(np.linalg.inv(np.array([[1,2],[3,4]]))@np.array([[1,2],[3,4]]),np.eye(2)))
assert(not matricesIguales(np.array([[1,2],[3,4]]).T,np.array([[1,2],[3,4]])))

print("Todos los Test pasaron correctamente.")


# %% 

'''
Versiones de 'matricesIguales()' que no pasan los test.
'''

def matricesIguales_1(A, B) :

    # Paso las matrices a arrays de numpy (en caso que no vengan en ese formato por defecto).
    A = np.array(A)
    B = np.array(B)

    if (A.shape != B.shape) :
        return False

    # Pruebo con la función 'np.array_equal()' que compara que sean iguales todos los numeros con precision exacta.
    return np.array_equal(A, B) 


def matricesIguales_2(A, B) :
    
    A = np.array(A)
    B = np.array(B)

    if (A.shape != B.shape) :
        return False
    
    # Comparar elemento a elemento manualmente. 
    # Nuevamente, Python compara numeros con precision exacta.
    for i in range(A.shape[0]) :
        for j in range(A.shape[1]) :
            if (A[i, j] != B[i, j]) :
                return False
    return True


# %%
 
'''
Esta version si pasa los test. Hasta donde entendi, 'np.allclose()' hace lo mismo: ajustar el error en funcion del tamaño de 
los elementos que se estan comparando.
'''

'''
En esta version añado una tolerancia, para comparar con el error entre dos valores. 

Leyendo me entere que esto no es una buena idea para usar en general, ya que la tolerancia hay que ajustarla en función de 
la magnitud de los valores que se comparan. Para matrices, se usan 'tol = k * eps' siendo 'eps' el epsilon de maquina, donde 
'k' depende del tamaño del problema (número de operaciones relevantes o el tamaño de la matriz).
'''

def matricesIguales_3(A, B, tol = 1e-12) :
    
    if (len(A) != len(B)) :
        return False
    
    if (any(len(A[i]) != len(B[i]) for i in range(len(A)))) : # Miro que todas las filas sean de igual tamaño.
        return False

    for i in range(len(A)) :
        for j in range(len(A[i])) :
            if (error(A[i][j], B[i][j]) > tol) :   # Uso la función 'error()' que implemente antes.
                return False
    return True


'''
Por lo que ví, la forma mas "correcta" de usar la función anterior es ajustando la tolerancia en función del tamaño de los 
numeros (que se estan comparando). Esto se puede hacer multiplicando el Epsilon de Maquina por el modulo del comparando mas 
grande. Si los numeros son muy pequeños, me quedo con el Epsilon de Maquina.
'''

def matricesIguales_4(A, B, tol = None) :
    
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


# Fin. 