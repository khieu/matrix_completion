import numpy as np

# size is number of coin tossed (in matrix it would be n*m or number of cells)
# n = 1 for bernoulli random var
# p is the probability. p = 0.5 is a fair coin tossed.
# np.random.binomial(size=10, n=1, p= 0.5)

def observe_matrix(A, p):
#     get size of A and initialize observation B
    num_rows = len(A); num_cols = len(A[0])
    B = np.zeros((num_rows, num_cols))
#     generate noise of mean 0,standard deviation 1, n*m values
    noise = np.random.normal(0,0.01,num_rows*num_cols)
#     create Chi variable (n=1, p=p, size = number of entries in the matrix)
    chi = np.random.binomial(1, p, size=num_rows*num_cols)
#     create observation matrix
    k = 0
    for i in range (0, num_rows):
        for j in range(0, num_cols):
            B[i][j] = (A[i][j] + noise[k]) * chi[k]
            k +=1
    
    return B

#     finding the projection matrix using formula from http://www.math.lsa.umich.edu/~speyer/417/OrthoProj.pdf
def find_projection_matrix(A):
    A_T = A.transpose()
    A_T_A_inverse = np.linalg.inv(np.matmul(A_T, A))
    res = np.matmul(A,A_T_A_inverse)
    res = np.matmul(res, A_T)

    return res
    
def complete_matrix(A, B,p):
    B_tilde = (1/p)*B
#     find left singular vectors through svd
    # u, s, v_t = np.linalg.svd(A)
    u, s, v_t = np.linalg.svd(B_tilde)
    num_rows = len(B_tilde); num_cols = len(B_tilde[0])
    r = min(num_rows, num_cols)
#     getting the left singular vectors (U) of matrix B_tilde
    left_singular_vectors = u[:, :r]
    # check if vectors are linearly independent ?
    if np.linalg.matrix_rank(left_singular_vectors) < r:
        return -1
    # find the projection matrix of the subspace spanned by the left_singular_vectors 
    P_U_tilde = find_projection_matrix(left_singular_vectors)
    # A_tilde is the approximation of A
    A_tilde = np.matmul(P_U_tilde, B_tilde)

    return (A, A_tilde)


A = np.random.rand(50,50)
p = 0.9
B = observe_matrix(A, p)

(_, A_tilde) = complete_matrix(A, B, p)
print(A-A_tilde)
print(np.linalg.norm(A-A_tilde))
print(np.linalg.norm(A-B))

