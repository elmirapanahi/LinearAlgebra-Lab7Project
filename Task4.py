import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_face


Average_Face = np.load('Average_landmarks/average_face.npy')
Flattened_Faces = np.load('Average_landmarks/Flattened_Faces.npy')


centered_faces=Flattened_Faces.reshape(42,68,2)-Average_Face
centered_faces_2d=centered_faces.reshape(42,68*2)  # 42,136

cov_matrix=np.cov(centered_faces_2d.T)# transpose to make features columns
eigenvalues_ED,eigenvectors_ED=np.linalg.eigh(cov_matrix)
sorted_eigenvalues_ED=np.sort(eigenvalues_ED)[::-1]

U,S,Vt=np.linalg.svd(centered_faces_2d,full_matrices=False)
singular_values=S**2/(centered_faces_2d.shape[0]-1)

plt.figure(figsize=(10,5))
plt.plot(sorted_eigenvalues_ED[:42],label="Eigenvalues")
plt.plot(singular_values,label="Squared Singular Values")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Comparison of Eigenvalues and Singular Values Squared")
plt.legend()
plt.grid(True)
plt.show()