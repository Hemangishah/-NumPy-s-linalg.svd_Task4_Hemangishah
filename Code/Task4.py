import numpy as np

# Sample matrix (MxN)
M = np.array([[3, 2, 2], 
              [2, 3, -2]])

# Perform SVD
U, sigma, Vt = np.linalg.svd(M)

# Convert sigma to a diagonal matrix with the correct dimensions
# We need a (MxN) matrix, so we'll pad with zeros if needed.
Sigma = np.zeros((U.shape[1], Vt.shape[0]))
np.fill_diagonal(Sigma, sigma)

# Reconstruct the original matrix M
M_reconstructed = U @ Sigma @ Vt

print("Original Matrix M:\n", M)
print("\nMatrix U (Left Singular Vectors):\n", U)
print("\nSigma (Diagonal Matrix of Singular Values):\n", Sigma)
print("\nMatrix V^T (Right Singular Vectors Transposed):\n", Vt)
print("\nReconstructed Matrix M:\n", M_reconstructed)

