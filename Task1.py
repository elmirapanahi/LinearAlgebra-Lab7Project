import numpy as np
import matplotlib.pyplot as plt
from utils import FaceRegister
from plot_utils import plot_face

def Register(Neutral_Face, Face, Method="affine"):
    # Face is 68*2
    face_register=FaceRegister(Face,Neutral_Face,method=Method)
    transformation_matrix=face_register.register()
    transformed_face=(transformation_matrix @ Face.T).T
    return transformed_face


def plot_transformed_face(Face, Neutral_Face, transformed_face):
    plt.figure(figsize=(12,5))
    ax1=plt.subplot(1,3,1)
    plot_face(ax1,Face)
    plt.title("Original Face")
    ax2=plt.subplot(1,3,2)
    plot_face(ax2,Neutral_Face)
    plt.title("Neutral Face")
    ax3=plt.subplot(1,3,3)
    plot_face(ax3,transformed_face)
    plt.title("Transformed Face")

    plt.show()
