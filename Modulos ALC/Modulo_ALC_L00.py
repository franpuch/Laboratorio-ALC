"""
Álgebra Lineal Computacional.

Laboratorio 0: Manipulación de Matrices y Numpy.
"""


# %% 
'''
Módulos.
'''

import numpy as np 
import matplotlib.pyplot as plt


# %% 
# Variables para Testear.

matriz_test0:np.ndarray = np.array([])

matriz_test1:np.ndarray = np.array([[1, 2, 3] ,
                                    [4, 5, 6]])

matriz_test2:np.ndarray = np.array([[1, 2, 3] ,
                                    [4, 5, 6] ,
                                    [7, 8, 9]])

# Matriz Simétrica.
matriz_test3:np.ndarray = np.array([[1, 2, 3] ,
                                    [2, 7, 1] ,
                                    [3, 1, 0]])

# Matriz Columna.
matriz_test4:np.ndarray = np.array([[1] ,
                                    [0] ,
                                    [-1]])

matriz_test4_prima:np.ndarray = np.array([1, 0, -1])

# Matriz Diagonalmente Dominante.
matriz_test5:np.ndarray = np.array([[4, 1, 1] ,
                                    [2, 6, 1] ,
                                    [1, 1, 5]])

# Matriz NO Diagonalmente Dominante.
matriz_test6:np.ndarray = np.array([[2, 3, 1] ,
                                    [1, 2, 1] ,
                                    [1, 1, 2]])

# Matriz Fila.
matriz_test7:np.ndarray = np.array([1, 2, 3])


# %% 
# Ejercicio 1.

'''
OBS -> No estoy seguro de si la matriz vacía se considera cuadrada, por las dudas lo dejo en False.
'''

def esCuadrada(matriz:np.ndarray) -> bool : 
    if (matriz.shape[0] == 0) :
        return False
    return (matriz.shape[0] == matriz.shape[1])

# Test.
print(esCuadrada(matriz_test1))
print(esCuadrada(matriz_test2))
print(esCuadrada(matriz_test3))


# %% 
# Ejercicio 2.

'''
OBS -> Sólo las matrices cuadradas tienen Diagonal Principal. Así que voy a trabajar sólo con matrices cuadradas. 
       Si la matriz no es cuadrada, retorno la misma matriz pasada (sin cambios).
''' 

def triangSup(matriz:np.ndarray) -> np.ndarray :
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.")
        return matriz 
    else :
        res:np.ndarray = matriz.copy() 
        nro_filas:int = res.shape[0]
        
        for i in range(0, nro_filas) :
            for j in range(0, i + 1):
                if (j <= i) :
                    res[i][j] = 0
                    
        return res 

# Test.
print(triangSup(matriz_test1))
print(triangSup(matriz_test2)) 


# %%
# Ejercicio 3.

'''
OBS -> Misma observación que Ejercicio 2, sólo trabajo con matrices cuadradas.
'''

def triangInf(matriz:np.ndarray) -> np.ndarray :
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.") 
        return matriz 
    else :
        res:np.ndarray = matriz.copy() 
        nro_filas:int = res.shape[0] 
        nro_columnas:int = res.shape[1] 
        
        for i in range(0, nro_filas) :
            for j in range(i, nro_columnas) :
                res[i][j] = 0 
        
        return res 

# Test.
print(triangInf(matriz_test1))
print(triangInf(matriz_test2))                    


# %% 
# Ejercicio 4. 

'''
OBS -> Misma observación que Ejercicio 2, sólo trabajo con matrices cuadradas.
'''

def diagonal(matriz:np.ndarray) -> np.ndarray : 
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.") 
        return matriz 
    else :
        res:np.ndarray = matriz.copy() 
        nro_filas:int = matriz.shape[0] 
        nro_columnas:int = matriz.shape[1] 
        
        for i in range(0, nro_filas) :
            for j in range(0, nro_columnas) :
                if (i != j) : 
                    res[i][j] = 0 
                    
        return res 

# Test. 
print(diagonal(matriz_test1))
print(diagonal(matriz_test2)) 


# %% 
# Ejercicio 5.

'''
OBS -> Misma observación que Ejercicio anterior, sólo trabajo con matrices cuadradas.
'''

def traza(matriz: np.ndarray) -> float:
    if not(esCuadrada(matriz)):
        print("La matriz no es cuadrada, no puede definirse una diagonal principal")
        return 0
    
    nro_filas, nro_columnas = np.shape(matriz)
    res = 0.0
    
    for i in range(nro_filas):
        res += matriz[i][i]
    
    return np.float64(res) 

# Test. 
print(traza(matriz_test1)) 
print(traza(matriz_test2))


# %% 
# Ejercicio 6.

def traspuesta(matriz: np.ndarray) -> np.ndarray :
    
    # Si es un arreglo 1D, lo tratamos como (1, n).
    if (len(np.shape(matriz)) == 1) :
        n = np.shape(matriz)[0]
        
        # Convertir [1,2,3] -> [[1],[2],[3]]
        return np.array([[matriz[i]] for i in range(n)])
    
    # Caso general: matriz 2D.
    filas, columnas = np.shape(matriz)
    res = [[0 for _ in range(filas)] for _ in range(columnas)]

    for i in range(filas):
        for j in range(columnas):
            res[j][i] = matriz[i][j]

    return np.array(res)

# Test.
print(traspuesta(matriz_test1))
print(traspuesta(matriz_test2)) 
print(traspuesta(matriz_test4))
print(traspuesta(matriz_test7)) 


# %%
# Ejercicio 7. 

def esSimetrica(matriz:np.ndarray) -> bool :
    if not(esCuadrada(matriz)) :
        print("La matriz no es cuadrada, no puede definirse una diagonal principal.") 
        return False
    
    matriz_t:np.ndarray = traspuesta(matriz) 
    
    filas, columnas = np.shape(matriz)
    
    for i in range(0, filas) :
        for j in range(0, columnas) :
            if matriz[i][j] != matriz_t[i][j] :
                return False 
    
    return True

# Test.
print(esSimetrica(matriz_test1))
print(esSimetrica(matriz_test2))
print(esSimetrica(matriz_test3))


# %% 
# Ejercicio 8. 

'''
Quiero hacer una función robusta que, venga como venga 'x' (como vector fila, como vector columna o como vector aplanado), 
realice el producto y retorne un vector aplanado (vector fila digamos).
'''

def calcularAx(matriz_A:np.ndarray , matriz_x:np.ndarray) -> np.ndarray :
    
    # Obtengo los tamaños que necesito.
    nro_filas, nro_columnas = np.shape(matriz_A) 
    
    # Normalizo 'x' para que siempre sea un vector 1D de largo 'm'.
    if (len(np.shape(matriz_x)) == 2) :
        filas_x, cols_x = np.shape(matriz_x)
        
        if (cols_x == 1) :   # Vector Columna.
            matriz_x = np.array([matriz_x[i][0] for i in range(filas_x)])
            
        elif (filas_x == 1) :   # Vector Fila.
            matriz_x = np.array([matriz_x[0][j] for j in range(cols_x)])
            
        else:
            raise ValueError("La 'matriz_x' no es un vector válido.")
            
    elif (len(np.shape(matriz_x)) != 1) :
        raise ValueError("La 'matriz_x' no es un vector válido.")
        
    # Ahora la 'matriz_x' es siempre 1D de largo 'm'
    
    # Armo el resultado como un vector columna de ceros.
    res:np.ndarray = np.array([[0] for _ in range(0, nro_filas)]) 
    
    for i in range(0, nro_filas) :
        res_parcial: float = 0 
        
        for j in range(0, nro_columnas) :
            res_parcial += matriz_A[i][j] * matriz_x[j]
        
        res[i] = res_parcial 
    
    return res 

print(calcularAx(matriz_test1, matriz_test4))
print(calcularAx(matriz_test2, matriz_test4))
print(calcularAx(matriz_test3, matriz_test4)) 


# %% 
# Ejercicio 9. 

def intercambiarFilas(matriz:np.ndarray , i:int , j:int) -> None :
    nro_columnas:int = np.shape(matriz)[1]
    
    for a in range(0, nro_columnas) :
            
        aux:float = matriz[i][a] 
            
        matriz[i][a] = matriz[j][a]
        matriz[j][a] = aux

# Test.
intercambiarFilas(matriz_test1, 0, 1)
print(matriz_test1) 


# %% 
# Ejercicio 10. 

def sumar_fila_multiplo(matriz:np.ndarray , i:int , j:int , s:float) -> None : 
    nro_columnas:int = np.shape(matriz)[1] 
    
    for a in range(0, nro_columnas) :
        matriz[i][a] += matriz[j][a] * s 
        
# Test. 
sumar_fila_multiplo(matriz_test1, 0, 1, 2)
print(matriz_test1) 


# %% 
# Ejercicio 11.

def esDiagonalmenteDominante(matriz:np.ndarray) -> bool :
    nro_filas , nro_columnas = np.shape(matriz) 
    
    if (nro_filas != nro_columnas) :
        return False 
    
    for i in range(nro_filas) :
        dominante:float = abs(matriz[i][i]) 
        suma_parcial:float = 0
        
        for j in range(nro_columnas) :
            if (j != i) :
                suma_parcial += abs(matriz[i][j]) 
        
        if (dominante <= suma_parcial) :
            return False
    
    return True

# Test. 
print(esDiagonalmenteDominante(matriz_test5))
print(esDiagonalmenteDominante(matriz_test6))


# %% 
# Ejercicio 12. 

def matrizCirculante(vector: np.ndarray) -> np.ndarray :
    
    # Normalizar a un vector 1D (lista de Python).
    # Esto me lo corrigió el Gordo (mi algoritmo era correcto, pero no funcionaba sin esto).
    shape = np.shape(vector)
    if len(shape) == 1 :                 # vector 1D: (n,)
        n = shape[0]
        v = [vector[i] for i in range(n)]
    elif shape[0] == 1 :                 # vector fila: (1, n)
        n = shape[1]
        v = [vector[0][j] for j in range(n)]
    elif shape[1] == 1 :                 # vector columna: (n, 1)
        n = shape[0]
        v = [vector[i][0] for i in range(n)]
    else:
        raise ValueError("El parámetro 'vector' debe ser 1D, fila (1×n) o columna (n×1).")

    # Genero la matriz resultado llena de ceros.
    res = np.array([[0 for _ in range(n)] for _ in range(n)])

    # Cada fila es una rotación a la derecha de la anterior.
    for i in range(n):
        for j in range(n):
            res[i][j] = v[(j - i) % n]   # Indice circular.

    return res

# Test.
print(matrizCirculante(matriz_test7)) 
print(matrizCirculante(matriz_test4)) 


# %% 
# Ejercicio 13.

def matrizVandermonde(vector: np.ndarray) -> np.ndarray :
    
    # Cantidad de elementos del vector.
    n:int = np.shape(vector)[0]
    
    # Matriz resultado (n x n) llena de ceros.
    res:np.ndarray = np.array([[0 for _ in range(n)] for _ in range(n)], dtype=float)
    
    for fila in range(n) : 
        for columna in range(n) :   
            res[fila][columna] = vector[fila] ** columna
    
    return res

# Test. 
print(matrizVandermonde(matriz_test7))


# %% 
# Ejercicio 14. 

def numeroAureo(n:int) -> list[float] : 
    
    # Matriz de Fibonacci.
    A:np.ndarray = np.array([[1, 1],
                             [1, 0]], dtype=object)
    
    # Vector inicial como columna: traspuesta([F1, F0]) 
    v:np.ndarray = np.array([[1],
                             [0]], dtype=object)
    
    # Aquí voy guardando los números que voy obteniendo.
    aproximaciones:list[float] = []
    
    for k in range(1, n+1):
        v = calcularAx(A, v) 
        Fk1, Fk = v[0,0], v[1,0]
        
        if (Fk != 0) :
            aproximaciones.append(Fk1 / Fk)
    
    # Ahora, grafico los resultados (le pido ayuda al Gordo porque no recuerdo bien cómo graficar con mathplotlib).
    plt.plot(range(1, len(aproximaciones)+1), aproximaciones, marker="o", label="Aproximación")
    plt.axhline((1+np.sqrt(5))/2, color="red", linestyle="--", label="φ real")
    plt.xlabel("Iteraciones (k)")
    plt.ylabel("Aproximación de φ")
    plt.title("Convergencia del Número Áureo usando Fibonacci (Matricial)")
    plt.legend()
    plt.grid()
    plt.show()
    
    return aproximaciones

# Test.
print(numeroAureo(8)) 


# %% 
# Ejercicio 15. 

'''
Idea: 
    - Primero, generar la matriz resultado: matriz 'n x n' llena de ceros.
    - Calcular (en una lista) los números de fibonacci hasta el '2n'. Ya que los números que debo ubicar en la matriz resultado
      son los fibonacci hasta el 'i + j' con 'i' = 'j' = '(n - 1)': i + j = (n - 1) + (n - 1) = 2n - 2 (con calcular 
      hasta el '2n' voy sobrado).
    - Llenar la matriz resultado con los números que fui guardando en la lista en el paso anterior.
'''

def matriz_Fibonacci(n:int) -> np.ndarray : 
    
    res:np.ndarray = np.array([[0 for _ in range(0, n)] for _ in range(0, n)]) 
    
    fib:list[int] = [0, 1]
    for contador in range(2, 2 * n) :   # Empiezo en 2 el ciclo porque los primeros 2 fibonaccis ya los tengo.
        fib.append(fib[-1] + fib[-2])   # Aprovecho que Python entiende los índices negativos como empezar desde el final.
    
    for fila in range(0, n) :
        for columna in range(0, n) :
            res[fila][columna] = fib[fila + columna]
    
    return res 

# Test. 
print(matriz_Fibonacci(3)) 


# %% 
# Ejercicio 16. 

'''
OBS -> Como en algunos ejercicios anteriores, debo settear la matriz resultado como tipo 'float'. Ya que sino se infiere como
       'ints' y me trunca los números racionales a 0 (dejandome una matriz llena de ceros).
'''

def matrizHilbert(n:int) -> np.ndarray : 
    
    res:np.ndarray = np.array([[0 for _ in range(0, n)] for _ in range(0, n)], dtype = float) 
    
    for fila in range(0, n) :
        for columna in range(0, n) :
            res[fila][columna] = (1) / (fila + columna + 1)
    
    return res 

# Test. 
print(matrizHilbert(4)) 


# %% 
# Ejercicio 17.

# Primero, preparo una función para generar los puntos que voy a evaluar en los polinomios.
def generar_puntos(cota_inf:float , cota_sup:float , cantidad:int) -> np.ndarray :
    
    paso:float = (cota_sup - cota_inf) / (cantidad - 1)   # La idea es que los puntos sean equidistantes: entre dos consecutivos hay
                                                   # la misma distancia en todos.
    
    puntos:list[float] = [(cota_inf + (i * paso)) for i in range(cantidad)] 
    
    return np.array(puntos, dtype = float) 


# Ahora, voy a usar la función del Ejercicio 13. Ya que la matriz Vandermonde tiene las potencias que necesito (de cada 'x'
# a evaluar). Pero la voy a modificar un poco, ya que la del Ejercicio 13 me genera las potencias hasta una menos de la que
# necesito: para grado 5 (por ejemplo), el Ejercicio 13 me genera desde la potencia 0 a la 4, necesito que me genere de 0 a 5.

def matrizVandermonde(vector: np.ndarray, grado:int) -> np.ndarray:
    n = np.shape(vector)[0]
    res = np.array([[0 for _ in range(grado+1)] for _ in range(n)], dtype=float)
    
    for fila in range(n):
        for columna in range(grado+1):
            res[fila][columna] = vector[fila] ** columna
    return res 


# Ahora sí, puedo pasar a calcular el polinomio usando mi matriz de Vandermonde.

def evaluar_polinomio_vandermonde(coeficientes:list[float] , dominio:np.ndarray) -> np.ndarray : 
    grado:int = len(coeficientes) - 1
    v:np.ndarray = matrizVandermonde(dominio, grado) 
    
    res:np.ndarray = np.array([0 for _ in range(0, len(dominio))], dtype = float)   # Inicializo una lista llena de ceros, 
                                                                                    # aquí voy a ir almancenando los 
                                                                                    # resultados.
                                                                                    
    n:int = len(coeficientes) 
    
    # Los coeficientes los presento en orden descendiente (por potencia), pero mi función de Vandermonde me genera las 
    # potencias en orden creciente. Así que ojo con eso, tengo que ir indexando 'coeficientes' al revés.
    for i in range(0, n) :
        res += coeficientes[n - i - 1] * v[:,i]   # De la matriz Vandermonde tomo la columna 'i' todas las filas.
                                                  # v[:, i] son las potencias 'x^{i}
    
    return res 


# Ahora paso a graficar los polinomios que me piden.
# Genero 100 puntos entre -1 y 1.
x = generar_puntos(-1, 1, 100)

# Evalúo los tres polinomios.
y1 = evaluar_polinomio_vandermonde([1, -1, 1, -1, 1, -1], x)
y2 = evaluar_polinomio_vandermonde([1, 0, 3], x)
y3 = evaluar_polinomio_vandermonde([1.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 , -2.0], x)

# Grafico.
plt.plot(x, y1, label='x^5 - x^4 + x^3 - x^2 + x - 1')
plt.plot(x, y2, label='x^2 + 3')
plt.plot(x, y3, label='x^10 - 2')
plt.xlabel('x')
plt.ylabel('p(x)')
plt.title('Evaluación de polinomios')
plt.legend()
plt.grid(True)
plt.show()


'''
Voy a analizar la Cantidad de Operaciones y el Espacio en Memoria en términos de Complejidad Algorítmica.

COMPLEJIDAD TEMPORAL -> "cantidad de operaciones"
Tenemos: 
    - m -> cantidad de puntos a considerar.
    - n -> grado del polinomio.
    
Generar la matríz de Vandermonde cuesta O(m * n).
Evaluar el polinomio cuesta O(m * n).

La complejidad temporal total es O(m * n).


COMPLEJIDAD ESPACIAL -> "cantidad de memoria usada"
Tenemos: 
    - m -> cantidad de puntos a considerar.
    - n -> grado del polinomio.

El vector de puntos es O(m).
La matriz de Vandermonde es O(m * n).
El vector de resultados (puntos evaluados) es O(m).

La complejidad espacial total es O(m * n). 
'''

'''
MODIFICACIONES PARA MEJORAR LA EFICIENCIA.

- Creo que si, en vez de utilizar la matríz de Vandermonde, directamente calculo cada uno de los términos del polinomio (para
  cada punto); voy a estar trabajando más eficientemente en memoria (porque sólo estoy almacenando el vector de puntos y el 
  vector de resultados). 
- Con el enfoque anterior, el complejidad temporal estamos igual.
'''


# %% 
# Ejercicio 18. 

def row_echelon_stable(matriz:np.ndarray) -> np.ndarray :
    nro_filas, nro_columnas = np.shape(matriz) 
    
    for fila in range(0, nro_filas) :
        
        # Verifico si es la de mayor pivot. 
        max_fila:int = fila 
        max_valor:float = abs(matriz[fila][fila]) 
        for i in range(fila, nro_filas) :
            if ((abs(matriz[i][fila])) > max_valor) : 
                max_fila = i 
                max_valor = abs(matriz[i][fila]) 
        
        # Hago el swap de filas (si es necesario).
        if (max_fila != fila) :
            intercambiarFilas(matriz, fila, max_fila) 
            
        # Hago Eliminación Gaussiana para esa columna.
        for columna in range(fila + 1, nro_filas) :
            if (matriz[columna][fila] != 0) :
                factor:float = matriz[columna][fila] / matriz[fila][fila] 
                sumar_fila_multiplo(matriz, columna, fila, (-factor))
    
    return matriz 

# Test.

a = np.array([[2, 1],
              [4, 3]], dtype = float)
print(row_echelon_stable(a))   # Test matriz simple.

b = np.array([[0, 2, 1],
              [1, 1, 1],
              [2, 3, 4]], dtype=float)
print(row_echelon_stable(b))   # Test pivoteo obligatorio.

c = np.array([[2, 4, 6],
              [1, 2, 3],
              [3, 6, 9]], dtype=float)
print(row_echelon_stable(c))   # Test matriz con dependencia lineal (Sist Comp Ind)

d = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]], dtype=float)
print(row_echelon_stable(d))   # Test matriz ya escalonada.

e = np.array([[0, 1, 2],
              [1, 0, 3],
              [4, 5, 6]], dtype=float)
print(row_echelon_stable(e))   # Test con un cero en la diagonal.


# %% 

print("Si se imprime esto, es porque todos los test pasaron exitosamente!")


# Fin. 