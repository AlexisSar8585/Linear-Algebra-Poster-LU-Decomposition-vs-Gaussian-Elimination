import numpy as np
import time
import matplotlib.pyplot as plt


#------------------------------------------------------------------------------
#Definitions of Functions
#------------------------------------------------------------------------------

def forward_elimination(A, b):
    # A: N×N, b: N×1
    # returns augmented matrix after forward elimination (no back substitution)

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1,1)

    if A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]:
        raise ValueError("Dimension mismatch: A must be square and compatible with b")

    aug = np.concatenate((A, b), axis=1)
    n = A.shape[0]

    for col in range(n - 1):

        # --- PARTIAL PIVOTING ---
        # Find the row with the largest absolute value in this column from col downward
        pivot_row = col + np.argmax(np.abs(aug[col:, col]))

        # If pivot is zero, system is singular or has no unique solution
        if aug[pivot_row, col] == 0:
            raise ValueError(f"Zero pivot encountered at column {col}; matrix is singular.")

        # Swap rows if pivot row is not the current row
        if pivot_row != col:
            aug[[col, pivot_row]] = aug[[pivot_row, col]]

        # --- ELIMINATION ---
        for row in range(col + 1, n):
            multiplier = aug[row, col] / aug[col, col]
            aug[row] -= multiplier * aug[col]

    return aug

def backward_substitution(Ab):
    # Input is a N x (N+1) matrix
    # output is a column vector of size N x 1

    # Initialize your 'x' vector that will contain your solution
    N = np.shape(Ab)[0]
    x = np.zeros((N,1))
    b = Ab[:,-1]
    for i in range(N):
        k = N-1-i #Starts from the last row instead of the first
        substituteSum = 0
        for j in range(k+1, N):
            substituteSum = substituteSum + Ab[k,j]*x[j]
        substitute = b[k] - substituteSum
        x[k] = substitute/Ab[k,k]
    return x

def gaussian_elimination(A, b):
    return(backward_substitution(forward_elimination(A,b)))

#returns the decomposed matrix into [L\U] form. This saves space since 
#the elements in L and U don't overlap, we can group them in a single matrix
def LUdecomp(a):
    n = len(a)
    for k in range(0,n-1):
        for i in range(k+1,n):
            if a[i,k] != 0.0:
                lam = a [i,k]/a[k,k]
                a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                a[i,k] = lam
    return a

def LUsolve(a,b):
    n = len(a)
    for k in range(1,n):
        b[k] = b[k] - np.dot(a[k,0:k], b[0:k])
    b[n-1] = b[n-1]/a[n-1,n-1]
    for k in range(n-2,-1,-1):
        b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
    return b


#------------------------------------------------------------------------------
#Settings for Experiment
#------------------------------------------------------------------------------


n = 300              # size of matrix A (300x300)
num_rhs_list = [1, 5, 10, 20, 40, 80, 160]   # number of b vectors

# generate a fixed random matrix
A = np.random.rand(n, n)

num_trials = 5
mean_times_gauss = []
std_times_gauss = []
mean_times_lu = []
std_times_lu = []

# LU factorization for A (only once)
LU = LUdecomp(A.copy())


#------------------------------------------------------------------------------
# Experiment
#------------------------------------------------------------------------------

for num_rhs in num_rhs_list:
    print(f"Testing {num_rhs} right-hand sides ...")
    gauss_trials = []
    lu_trials = []

    Bs = np.random.rand(n, num_rhs)

    for trial in range(num_trials):
        print(f"current trial: {trial}")
        # Gaussian
        start = time.perf_counter()
        for j in range(num_rhs):
            gaussian_elimination(A.copy(), Bs[:, j].copy())
        gauss_trials.append(time.perf_counter() - start)

        # LU
        start = time.perf_counter()
        for j in range(num_rhs):
            LUsolve(LU, Bs[:, j].copy())
        lu_trials.append(time.perf_counter() - start)

    mean_times_gauss.append(np.mean(gauss_trials))
    std_times_gauss.append(np.std(gauss_trials))
    mean_times_lu.append(np.mean(lu_trials))
    std_times_lu.append(np.std(lu_trials))


#------------------------------------------------------------------------------
# Plotting
#------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.errorbar(num_rhs_list, mean_times_gauss, yerr=std_times_gauss, fmt='o-', capsize=5, elinewidth=2,markeredgewidth=2,label='Gaussian')
plt.errorbar(num_rhs_list, mean_times_lu, yerr=std_times_lu, fmt='s-',elinewidth=2,markeredgewidth=2, label='LU')
plt.xlabel("Number of right-hand sides")
plt.ylabel("Time (seconds)")
plt.title("Timing with Error Bars")
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig("gauss_vs_lu.png", dpi=300)
plt.show()




