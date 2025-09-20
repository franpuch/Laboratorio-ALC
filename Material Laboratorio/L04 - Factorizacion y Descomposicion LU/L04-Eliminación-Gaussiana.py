"""
Eliminacion Gausianna
"""

import numpy as np

def elim_gaussiana(A):
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return
        
    L = np.zeros((m, n))
    U = np.zeros((m, n))
    
    for i in range(0, n - 1) :
        
        p = Ac[i][i]  # Elijo el Pivot.
        L[i][i] = 1   # Completo la Diagonal de L.
        
        if (p == 0) :   # Si el Pivot es 0, no se puede descomponer. 
            return None,None,None
        
        for j in range(i + 1, n) : 
            c = (Ac[j][i] / p) 
            cant_op += 1 
            
            U[j,i:] = Ac[j,i:] - (c * Ac[i,i:]) 
            cant_op += 2 * (n - i)   
            
            L[j][i] = c
            
    L[n-1][n-1]=1   # Completo el último de la Diagonal de L (que no lo alcanzo en el ciclo).

    return L, U, cant_op


def main():
    n = 7
    B = np.eye(n) - np.tril(np.ones((n,n)),-1) 
    B[:n,n-1] = 1
    print('Matriz B \n', B)
    
    L,U,cant_oper = elim_gaussiana(B)
    
    print('Matriz L \n', L)
    print('Matriz U \n', U)
    print('Cantidad de operaciones: ', cant_oper)
    print('B=LU? ' , 'Si!' if np.allclose(np.linalg.norm(B - L@U, 1), 0) else 'No!')
    print('Norma infinito de U: ', np.max(np.sum(np.abs(U), axis=1)) )

if __name__ == "__main__":
    main()


# Fin. 