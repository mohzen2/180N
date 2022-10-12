#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
from scipy.stats import linregress
from scipy.optimize import newton
import copy
from scipy import linalg
from scipy.stats import unitary_group
from numpy.polynomial.hermite import hermval
from IPython.display import display, Math
import math

import matplotlib.pyplot as plt
plt.style.use("ggplot")
# In Josh's opinion, this makes plots a bit prettier :) . Feel free to remove if desired.



# In[1]:


def dynamics_solve(f, D = 1, t_0 = 0.0, s_0 = 1, h = 0.1, N = 100, method = "Euler"):
    
    """ Solves for dynamics of a given dynamical system
    
    - User must specify dimension D of phase space.
    - Includes Euler, RK2, RK4, that user can choose from using the keyword "method"
    
    Args:
        f: A python function f(t, s) that assigns a float to each time and state representing
        the time derivative of the state at that time.
        
    Kwargs:
        D: Phase space dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        s_0: Initial state (float for D=1, ndarray for D>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4"
    
    Returns:
        T: Numpy array of times
        S: Numpy array of states at the times given in T
    """
    
    T = np.array([t_0 + n * h for n in range(N + 1)])
    
    if D == 1:
        S = np.zeros(N + 1)
    
    if D > 1:
        S = np.zeros((N + 1, D))
        
    S[0] = s_0
    
    if method == 'Euler':
        for n in range(N):
            S[n + 1] = S[n] + h * f(T[n], S[n])
    
    if method == 'RK2':
        for n in range(N):
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n] + (2.0/3.0)*h, S[n] + (2.0/3.0)*k1)
            S[n + 1] = S[n] + (1.0/4.0) * k1 + (3.0/4.0) * k2
    
    if method == 'RK4':
        for n in range(N):
            k1 = h * f(T[n], S[n])
            k2 = h * f(T[n] + (0.5)*h, S[n] + (0.5)*k1)
            k3 = h * f(T[n] + (0.5)*h, S[n] + (0.5)*k2)
            k4 = h * f(T[n] + h, S[n] + k3)
            S[n + 1] = S[n] + (1.0/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            
    return T, S


# In[2]:


def population(T, S):
  """ Models birth death ratio of a population
  Args:
  T: Time (float)
  S: State of system at T (float)
    
  Variables:
  B = Births per thousand per year (float per months)
  D = Deaths per thouhsand per year (float per months)
    
  Return:
  Sfinal: State after a given cycle (float)
    
  """
  B = ((17./1000.)/12.)
  D = ((32./1000.)/12.)
  Sfinal= (B-D) * S
  return Sfinal



def hamiltonian_solve(d_qH, d_pH, d = 1, t_0 = 0.0, q_0 = 0.0, p_0 = 1.0, h = 0.1, N = 100, method = "Euler",):
    
    """ Solves for dynamics of Hamiltonian system
    
    - User must specify dimension d of configuration space.
    - Includes Euler, RK2, RK4, Symplectic Euler (SE) and Stormer Verlet (SV) 
      that user can choose from using the keyword "method"
    
    Args:
        d_qH: Partial derivative of the Hamiltonian with respect to coordinates (float for d=1, ndarray for d>1)
        d_pH: Partial derivative of the Hamiltonian with respect to momenta (float for d=1, ndarray for d>1)
        
    Kwargs:
        d: Spatial dimension (int) set to 1 as default
        t_0: Initial time (float) set to 0.0 as default
        q_0: Initial position (float for d=1, ndarray for d>1) set to 0.0 as default
        p_0: Initial momentum (float for d=1, ndarray for d>1) set to 1.0 as default
        h: Step size (float) set to 0.1 as default
        N: Number of steps (int) set to 100 as default
        method: Numerical method (string), can be "Euler", "RK2", "RK4", "SE", "SV"
    
    Returns:
        T: Numpy array of times
        Q: Numpy array of positions at the times given in T
        P: Numpy array of momenta at the times given in T
    """
    T = np.array([t_0 + n * h for n in range(N + 1)]) 
    
    if d == 1:
        P = np.zeros(N + 1)
        Q = np.zeros(N + 1)
        
        Q[0] = q_0
        P[0] = p_0
    
    if d > 1:
        P = np.zeros((N + 1, d))
        Q = np.zeros((N + 1, d))
        
        P[0][0] = p_0[0]
        P[0][1] = p_0[1]
        Q[0][0] = q_0[0]
        Q[0][1] = q_0[1]
        
    
    if method == 'Euler':
        for n in range(N):
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P[n])
            P[n + 1] = P[n] - h * d_qH(Q[n], P[n])
    
    if method == 'RK2':
        for n in range(N):
            k1_Q = h * d_pH(Q[n], P[n])
            k1_P = h * (- d_qH(Q[n], P[n]))
            
            
            k2_Q = h * d_pH(Q[n] + (2.0/3.0) * h, P[n] + (2.0/3.0) * k1_Q)
            k2_P = h * -d_qH(Q[n] + (2.0/3.0) * h, P[n] + (2.0/3.0) * k1_P)
            
            Q[n + 1] = Q[n] + 0.25 * k1_Q + 0.75 * k2_Q
            P[n + 1] = P[n] + 0.25 * k1_P + 0.75 * k2_P
        
    if method == 'RK4':
        for n in range(N): 
            k1_Q = h * d_pH(Q[n], P[n])
            k1_P = h * (- d_qH(Q[n], P[n]))
            
            k2_Q = h * d_pH(Q[n] + 0.5 * h, P[n] + 0.5 * k1_Q)
            k2_P = h * -d_qH(Q[n] + 0.5 * h, P[n] + 0.5 * k1_P)
            
            k3_Q = h * d_pH(Q[n] + 0.5 * h, P[n] + 0.5 * k2_Q)
            k3_P = h * -d_qH(Q[n] + 0.5 * h, P[n] + 0.5 * k2_P)
            
            k4_Q = h * d_pH(Q[n] + h, P[n] + k3_Q)
            k4_P = h * -d_qH(Q[n] + h, P[n] + k3_P)
            
            Q[n + 1] = Q[n] + (1.0/6.0) * (k1_Q + 2 * k2_Q + 2 * k3_Q + k4_Q)
            P[n + 1] = P[n] + (1.0/6.0) * (k1_P + 2 * k2_P + 2 * k3_P + k4_P)
        
    if method == 'SE':
         for n in range(N):
            Q[n + 1] = Q[n] + h * d_pH(Q[n], P[n])
            P[n + 1] = P[n] - h * d_qH(Q[n+1], P[n])
        
    if method == "SV" and d > 1:
        for n in range(N):
            Pn0 = P[n][0] - d_qH(Q[n][0], P[n][0])[0]
            Pn1 = P[n][1] - d_qH(Q[n][1], P[n][1])[1]
            
            Q[n + 1][0] = Q[n][0] + d_pH(Q[n][0], Pn0)[0]
            Q[n + 1][1] = Q[n][1] + d_pH(Q[n][1], Pn1)[1]
            
            P[n + 1][0] = Pn0 - d_qH(Q[n + 1][0], P[n][0])[0]
            P[n + 1][1] = Pn1 - d_qH(Q[n + 1][1], P[n][1])[1]
    
    if method == "SV" and d == 1:
        for n in range(N):
            Pn3 = P[n] - h/2 * d_qH(Q[n], P[n])
            Q[n + 1] = Q[n] + h * d_pH(Q[n], Pn3)
            P[n + 1] = Pn3 - h/2 * d_qH(Q[n+1], P[n])
        
        
    return T, Q, P




def d_qH1(q, p):
    
    """ This function finds the derivatice of a Hamiltonian with respect to position
    Args:
        q: position (float in this case)
        p: momentum (Float)
    
    Variables:
        M: Mass (also a float)
        W: Angular frequncy(Float)
    
    Returns:
        deriv: The position dervative of the hamiltonian (float)
    """
    m = 0.5
    w = 1
    deriv = m * w**2 * q
    return deriv


def d_pH1(q, p):
    """ This function finds the derivative of the Hamiltonian with respect to momentum
    
    Args:
        Q: position (float)
        P: momentum (float)
    
    Variables:
        M: Mass (float)
    
    Returns:
        deriv: The derivative of Hamiltonian with respect to momentum (float)
    """
    m = 0.5
    deriv = 1/m * p
    return deriv

def pop2 (P, R = 0.2, K = 10**6 ):
    """ Models population growth given the inital conditions of R and K
    
    
    Args:
        P: Population (float)
        s0 = Initial state of population (float)
    
    Kwargs:
        R: Growth rate (float)
        K: Carrying capacity (float)
    
    Retruns:
        Sfinal: Final state of system over given time period (float)
    """
    Sfinal = R * (1 - (P/K)) * P
    return Sfinal

def pop3 (T, P0, R = 0.2, K = 10**6):
    """ Function for population growth to be given to dynamics_solve
    
    Args:
        P0: Initial population (float)
        T: Time (float)
    
    Kwargs:
        R: Growth rate (float)
        K: Carrying capacity (float)
        
    Returns:
        Sfinal: Final state of system over elapsed time (float)
    """
    Sfinal = R  * (1 - (P0/K)) * P0
    return Sfinal

def popc(T, P):
    """ Models population growth for given function
    
    Args:
        P0: Inital population (Float)
        T: Time (Float)
        
    Variables:
        R: Growth Rate (float)
        K: Carrying capacity (float)
        C: Idk (float)
        Pc: Also idk (float)
        
    Returns:
        Sfinal: final state of Population after elapsed time
    """
    R = 0.2
    K = 1000
    C = 40
    P_c = 100
    Sfinal = R * (1 - (P/K)) * P - C * (P**2 / (P_c**2 + P**2))
    return Sfinal
    

def d_qearth(q, p):
    """ This function finds the derivatice of a Hamiltonian of earth's position
    Args:
        q: position (float in this case)
        p: momentum (Float)
    
    Variables:
        M: Mass (also a float)
        W: Angular frequncy(Float)
    
    Returns:
        deriv: The position dervative of the hamiltonian (float)
    """
    m = 1
    w = 0.0172
    deriv = m * w**2 * q
    return deriv


def d_pearth(q, p):
    """ This function finds the derivative of the Hamiltonian of earth's momentum
    
    Args:
        Q: position (float)
        P: momentum (float)
    
    Variables:
        M: Mass (float)
    
    Returns:
        deriv: The derivative of Hamiltonian with respect to momentum (float)
    """
    m = 1
    deriv = 1/m * p
    return deriv

def print_matrix(array):
    matrix=''
    for row in array:
        try:
            for number in row:
                matrix+=f'{number}&'
        except TypeError:
            matrix +=f'{row}&'
        matrix=matrix[:-1] + r'\\'
    display(Math(r'\begin{bmatrix}' + matrix+r'\end{bmatrix}'))
    
    
def norm (A):
    """ This function takes all elements of a matrtix and finds their total length
    
    Args:
        A: nxn matrix
        
    Variables: 
        A: Matrix
        
    Returns:
        A: Length of matrix elements
    """
    A = np.array(A.flatten())
    A = A*A
    A = np.sum(A)
    A = np.sqrt(A)
    return A


def off (A):
    """ This function takes the off elements of a matrtix and finds their length
    
    Args:
        A: nxn matrix
        
    Variables: 
        b: place holder for reshapped matrix and removing the diagonal
        f: place holder for computing length
        g: final computation of matrix
        
    Returns:
        G: Length of matrix off elements
    """
    off = (A.flatten())
    b = np.delete(off, range(0, len(off), len(A)+1), 0)
    f = b*b
    g = np.sqrt(np.sum(f))
    return g


def jacobi_rotation(A, j, k):
    """ This function finds the rotation matrix and turns selected values to zero
    #Args:
        # A (np.ndarray): n by n real symmetric matrix
        # j (int): row parameter.
        # k (int): column parameter.
 
    #Returns:
        # A (np.ndarray): n by n real symmetric matrix, where the A[j,k] and A[k,j]
            element is zero
        # J (np.ndarray): n by n orthogonal matrix, the jacobi_rotation matrix
    """
    J = np.identity(np.shape(A)[0])
    if (A[j, k] != 0):
      tau = (A[k, k] - A[j, j]) / (2 * A[j, k])

      if tau >= 0:
        t = 1 / (tau + np.sqrt(1 + tau**2))

      else:
        t = 1 / (tau - np.sqrt(1 + tau**2))

      c = 1 / np.sqrt(1 + t**2)
      s = t * c

    else:
      c = 1
      s = 0
    
    J[j, j] = c
    J[j ,k] = s
    J[k, j] = -s
    J[k, k] = c
    A = (np.transpose(J)) @ A @ J

    return A, J


def real_eigen(A, tolerance):
    """
    #Args:
        # A (np.ndarray): n by n real symmetric matrix
        # tolerance (float): the relative precision
    #Returns:
        # d (np.ndarray): n by 1 vector, d[i] is the i-th eigenvalue, repeated according 
        #                 to multiplicity and ordered in non-decreasing order
        # R (np.ndarray): n by n orthogonal matrix, R[:,i] is the i-th eigenvector
    """
    R = np.identity(np.shape(A)[0])
    Delta = tolerance * norm(A)

    size = np.shape(A)[0]
    

    while off(A) > Delta:
      for i in range(size-1):
        for m in range(i+1,size):
          A, J = jacobi_rotation(A, i, m)
          R = R @ J 
          
          
          
          
         

    d0 = A.diagonal()
    d_columns = np.argsort(d0)
    d = d0[d_columns]
    R = R[:, d_columns]
        
   

    return d, R



def hermitian_eigensystem(H, tolerance):
    """ Finds the eigenvalues and eigenvectors of a hermitian matrix
    
    #Args:
        # A (np.ndarray): n by n complex hermitian matrix
        # tolerance (float): the relative precision
    #Returns:
        # d (np.ndarray): n by 1 vector, d[i] is the i-th eigenvalue, repeated according 
        #                 to multiplicity and ordered in non-decreasing order
        # U (np.ndarray): n by n unitary matrix, U[:,i] is the i-th eigenvector
    """
    Q = H
    S = Q.real
    A = Q.imag

    left = np.concatenate((S, A))

    right = np.concatenate((-A, S)) 

    combo = np.concatenate((left, right), axis = 1)

    X, R = real_eigen(combo, tolerance)

    U = R.T @ combo @ R

    d = np.delete(X, range(1, X.shape[0], 2), axis=0)

    U = np.delete(U, range(1, U.shape[1], 2), axis=1)
        
    upper = []
    lower = []
    size = int(len(U)/2)

    for i in range(size):
      upper.append(U[i,:])
        
    for i in range(size,(len(U))):
      lower.append(U[i, :])

    Upper = np.array(upper)
    Lower = np.array(lower)

    U = Upper + 1j*(Lower)







        
        
    return d, U

def real_eigen2(A, tolerance):
    """ Finds the real eigenvalues of a matrix by elimination of off elements
    #Args:
        # A (np.ndarray): n by n real symmetric matrix
        # tolerance (float): the relative precision
    #Returns:
        # d (np.ndarray): n by 1 vector, d[i] is the i-th eigenvalue, repeated according 
        #                 to multiplicity and ordered in non-decreasing order
        # R (np.ndarray): n by n orthogonal matrix, R[:,i] is the i-th eigenvector
    """
    
    length = len(A[0])
    R = np.identity(length)
    
    
    for i in range (0, length):
        j = i + 1
        
        while j < length:
            J = jacobi_rotation(A, i, j)[1] 
            A_r = J.T @ A @ J
            R = R @ J
            A = A_r
            j = j + 1
    
    for i in range (0, length): 
         j = i + 1

         while j < length:
            J = jacobi_rotation(A, i, j)[1] 
            A_r = J.T @ A @ J
            R = R @ J
            A = A_r
            j = j + 1
    
    
    J = jacobi_rotation(A, 0, 1)[1] 
    A_r = J.T @ A @ J
    
    R = R @ J  
    A = A_r
    
    MAX = norm(A)
    off_val = off(A)
        
    if off_val <= MAX:
        return A, R
    
    elif off_val > MAX:
        print("Sorry try again")
        return 0


def hermitian_eigensystem2(H, tolerance):
    """Finds the eigenvalues and eigenvectors of a hermitian matrix
    #Args:
        # A (np.ndarray): n by n complex hermitian matrix
        # tolerance (float): the relative precision
    #Returns:
        # d (np.ndarray): n by 1 vector, d[i] is the i-th eigenvalue, repeated according 
        #                 to multiplicity and ordered in non-decreasing order
        # U (np.ndarray): n by n unitary matrix, U[:,i] is the i-th eigenvector
    """

    length = len(H[0])
    d = np.zeros((length, 1))
  
    for i in range (0, length):
        d[i] = real_eigen2(H, 0)[0][i,i]
    
    U = real_eigen2(H, 0)[1]
    
    return d, U

def anharmonic(N, L):   
    
    """ This creates an anharmonic function of specified size and length that we can feed into hermitian eigensystem to diagonalize

    Args:
        N = Size of the NxN matrix that will be created
        L = Size of the perturbation (lambda)
        
    Returns:
        A = The resulting anharmonic matrix
  
    """
    A = np.zeros((N,N))
    
    for i in range(0, N): 
        a = 6 * (i ** 2)
        b = 6 * (i)
        A[i,i] = ((0.25) * (a + b + 3) * (L)) + (i + 0.5)
    
    for i in range(0, N-2): 
        a = i + 2
        b = i + 1
        A[i,i+2] = A[i,i-2] + np.sqrt(a * b) * (i + 1.5) * (L)
    
    for i in range(2, N): 
        a = i 
        b = i - 1
        m = i-2
        A[i,m] = A[i,m] + np.sqrt(a * b) * (m - 0.5) * (L)
    
    for i in range(0, N-4): 
        a = i + 4
        b = i + 3
        c = i + 2
        d = i + 1
        m = i + 4
        A[i,m] = A[i,m] + 0.25 * np.sqrt(a * b * c * d) * (L)
    
    for i in range(4, N): 
        m = i - 4
        a = i - 3
        b = i - 2
        c = i - 1
        A[i,m] = A[i,m] + 0.25 * np.sqrt(a * b * c * i) * (L)
    
    return A




#weighted coin
# 0 = heads
# 1 = tails
# heads is the cheating side
# f is how many values we want to skip to get a cleaner graph for the
# comparison section. Ugly way to do it i know
def weighted_coin (n, beta = 0.524):
  """ Simulates MCMC for a weighted coin
  - User must specify the number of flips and how much the coin favors one side

  Args:
    beta = Odds of heads (float)
    n = number of flips (int)

  Kwargs: 
    Counter = number of iterations (int)
    State = state of coin (int)
    Counter = iteration counter (int)
    tails = tails counter (int)
    heads = heads counter (int)
    flips = array of counter to graph against earnings (array of ints)
    Earnings = Amount generated per flip (float)
    

  Returns:
    Plot of earnings vs flip, or in other configiration
    Earnings and Counter arrays
  """


  f = 20
  state = 0
  counter = 0
  heads = 0
  tails = 0
  tailsprob = 1.0 - beta
  flips = []
  earnings = []
  for i in range(n):
    flip = np.random.randint(2)
    if flip == state:
      counter += 1
      heads += 1
    else:
      prob = np.random.uniform(0,1)
      probaccept = min(1.0, (tailsprob/beta)) 
      if (prob < probaccept):
        state = 1
        counter += 1
        tails += 1
      else:
        counter += 1
        heads += 1
    flips.append(counter)
    earn = ((heads*(1.00)) + (tails*(-1.00)))/counter
    earnings.append(earn)
  flips2 = flips[f:]
  earnings2 = earnings[f:]
  plt.plot(flips2, earnings2)
  plt.xlabel('Amount of flips (Time)')
  plt.ylabel('Average Earnings per flip')
  plt.title("Average Earnings vs Flips for weighted coin")
  return print("Helena's average earnings on a",beta,"heads favored coin over", n,"flips is", "$",earnings[len(earnings) - 1])
  return plt.show
  #return earnings[len(earnings) - 1]

    
def average_earnings_per_flip(beta):
  """ Calculates theoretical average earnings per flip
  - User enters probabilty for favored side

  Args:
    Beta = probability of favored side (float)

  Kwargs:
    alpha = probability for opposing side (float)
    E = earnings per flip (float)

  Returns:
    E (float)
  """

  alpha = 1 - beta
  E = beta*(1) + alpha*(-1)
  return E





def weighted_die(n):
  """ Perfoms MCMC for a weighted die that favors 1,2 3x as more as 3,4,5,6
  - User defines number of rolls

  Args:
    n = number of rolls

  Kwargs:
    state = where the die lands (int)
    counter = roll iteration (int)
    me = my wins (int)
    them = opposing number counter (int)
    themloosing = opposing roll percentage (float)
    mewin = percentage my numbers are favored (float)
    rolls = array of roll iterations
    earnings = array of earnings won per roll

  Returns:
    Plot of rolls vs Earnings 

  """
  state = 0
  counter = 0
  me = 0
  them = 0
  themlosing = 0.25
  mewin = 0.75
  rolls = []
  earnings3 = []
  for i in range(n):
    roll = np.random.randint(1,6)
    if roll == 1 or roll == 2:
      counter += 1
      me += 1
    else:
      prob = np.random.uniform(0,1)
      probaccept = min(1.0, (themlosing/mewin)) 
      if (prob < probaccept):
        state = 6
        counter += 1
        them += 1
      else:
        counter += 1
        me += 1
    rolls.append(counter)
    earn2 = ((me*(1.00)) + (them*(-1.00)))/counter
    earnings3.append(earn2)
  plt.plot(rolls, earnings3)
  plt.xlabel('Amount of rolls')
  plt.ylabel('Average Earnings per roll')
  plt.title('Average Earnings vs Rolls for weighted Die')
  return print("Rolling my weighted die",n, "times my average earnings is",earnings3[len(earnings3) - 1])
  return plt.show
  #return earnings[len(earnings) - 1]
    
    

    
    
    
def two_dim_ising(L, temp, num_steps):
  """ Simulates 2D ising model for a square latice of given temp and size

  -User must specify latice size, tempurature, and number of iterations

  Args: 
    L - latice size LxL (int)
    temp - float
    num_steps - iterations (int)

  Kwargs:
    N = LxL matrix of spins (array)
    H = magnetic field (float) set to zero for now
    E = total energy of latice float
    E2 = Energy squaref for averaging purposes (float)
    S = total magnetization (float)
    S2 = total magnetization squared for averaging (float)
    U = internal energy per iteration (array)
    M = average magnetization per iteration (array)
    Xt = magnetic susceptibilty per latice site per iteration (array)
    Ch heat susceptibilty per latice site per iteration (array)
    t = time steps of iterations/latice sites (array)

  Returns
    U, M, Xt, Ch, N, t 

  """
  N = np.random.choice([-1,1], size = (L, L))
  H = 0
  E = 0
  E2 = 0
  S = np.sum(N)
  S2 = (np.sum(np.multiply(N,N)))
  U = np.zeros(num_steps)
  M = np.zeros(num_steps)
  Xt = np.zeros(num_steps)
  Ch = np.zeros(num_steps)
  t = np.zeros(num_steps)

  for i in range (L):
    for m in range (L):
      top1 = N[(i - 1) % L][m]
      bottom1 = N[(i + 1) % L][m]
      left1 = N[i][(m - 1) % L]
      right1 = N[i][(m + 1) % L]
      En = -2*N[i][m]*(top1 + bottom1 + left1 +right1 + H)
      E += En
      E2 += En**2

  for i in range(num_steps):
    t[i] = (i)/(L*L) 
    j = random.randint(0,L-1)
    k = random.randint(0,L-1)

    s = N[j][k]

    top = N[(j - 1) % L][k]
    bottom = N[(j + 1) % L][k]
    left = N[j][(k - 1) % L]
    right = N[j][(k + 1) % L]

    deltaE = (-2*s)*(top + bottom + left + right + H)
    dS = 0
  

    if deltaE <= 0:
      N[j][k] *= -1
      dS = 2 * N[j][k]
    else:
      paccept = min(1.0, np.exp(-deltaE/temp))
      r = np.random.rand(1)
      if r < paccept:
        N[j][k] *= -1
        dS = 2*N[j][k]
      else:
          deltaE = 0
    E += deltaE
    S += dS
    E2 += (deltaE**2)
    S2 += (dS**2)
    U[i] = E/(L*L)
    M[i] = S/(L*L)
    Xt[i] = ((S2) - (S**2))/((L*L) * temp)
    Ch[i] = ((E2) - (E**2))/((L*L)*(temp**2))
    

  return U, M, Xt, Ch, N, t

    
    




    