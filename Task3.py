import numpy as np

Average_Face = np.load('Average_landmarks/average_face.npy')
Flattened_Faces = np.load('Average_landmarks/Flattened_Faces.npy')

# TODO: Flattened_Faces_centered = 
Flattened_Faces_centered=Flattened_Faces-Average_Face.ravel()
print("Flattened_Faces_centered shape:",Flattened_Faces_centered.shape)
# TODO: Perform Singular Value Decomposition
U,S,Vt=np.linalg.svd(Flattened_Faces_centered,full_matrices=False)
print("U shape:",U.shape)
print("S shape:",S.shape)
print("Vt shape:",Vt.shape)

K=16
#U_K = U[:, :K]
U_K=Vt.T[:,:K]
S_K=np.diag(S[:K])

print("U_K shape:",U_K.shape)
print("S_K shape:",S_K.shape)

np.save('U_k.npy', U_K)
np.save('S_k.npy', S_K)

print(f"First {K} principal components and singular values have been saved.")