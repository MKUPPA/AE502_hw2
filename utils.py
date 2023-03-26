import numpy as np

def solve_kepler(M, e, tol):
    
    """
    Solves the kepler equation:
    M = E - e*sin(E)
    
    - M is mean anomaly (input, radians)
    - E is eccentric anomaly (output, radians)
    - e is the eccentricity
    - g(E) = E - e*sin(E) - M
    - g'(E) = 1 - e*cos(E)
    
    Method used: Newtons
    - E_ip1 = Ei - (g(Ei)/g'(Ei))
    - tol: tolerance to stop solving
    """
    
    # initial guess
    E0 = M
    g = E0 - e*np.sin(E0) - M
    
    # iteration number 
    p = 0
    
    Ei = E0
    
    while np.abs(g) > tol:
        
        p = p+1
        
        # compute g(E) and g'(E)
        g = Ei - e*np.sin(Ei) - M
        g_der = 1 - e*np.cos(Ei)
        
        # update E
        Eip1 = Ei - (g/g_der)
        
        # Compute g at updated value
        g = Eip1 - e*np.sin(Eip1) - M
        
        Ei = Eip1
        
        
    print(f'solution converged in {p} iterations, E = {Eip1}')
    
    return Eip1
        
    
    
    
    
    