import numpy as np
import matplotlib.pyplot as plt
from plot_utils import plot_face
import os
# from register import register, plot_transformed_face
from utils import  save_landmarks, FaceRegister



from Task1 import Register,plot_transformed_face
# import utils

Neutral_Face = np.load('images_landmarks/landmarks_0.npy')

landmarks_folder = 'images_landmarks'  
transformed_landmarks_folder = 'transformed_landmarks'
Average_landmarks_folder = 'Average_landmarks'

if not os.path.exists(transformed_landmarks_folder):
    os.makedirs(transformed_landmarks_folder)
    os.makedirs(Average_landmarks_folder)


Flattened_Faces = [Neutral_Face.ravel()]

# TODO: compute Flattened_Faces and plot transformed faces Through all landmarks
# TODO: save transformed face(landmarks)  


for filename in sorted(os.listdir(landmarks_folder)):
    if filename=="landmarks_0.npy":
        continue
    filepath=os.path.join(landmarks_folder,filename)
    Face=np.load(filepath,allow_pickle=True)
    face_register=FaceRegister(Face,Neutral_Face,method="similarity")
    transformed_face=face_register.register()@Face.T
    transformed_face=transformed_face.T

    save_landmarks(transformed_face,filename.split('_')[1].split('.')[0],transformed_landmarks_folder)

    Flattened_Faces.append(transformed_face.ravel())
    # plot_transformed_face(Face,Neutral_Face,transformed_face)

print(len(Flattened_Faces))
# TODO: compute average face(landmarks) from transformed faces
average_face=np.mean(np.array(Flattened_Faces),axis=0).reshape(-1,2)


filename=os.path.join(Average_landmarks_folder,"average_face.npy")
np.save(filename,average_face)  

filename=os.path.join(Average_landmarks_folder,"Flattened_Faces.npy")
np.save(filename,Flattened_Faces)  
# TODO: Plot the average_face 
plt.figure(figsize=(8,8))
ax=plt.gca()
plot_face(ax,average_face)
plt.title("Average Face")
plt.show()