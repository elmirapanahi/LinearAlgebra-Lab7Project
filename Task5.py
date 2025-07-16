import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import plot_utils

Average_Face = np.load('Average_landmarks/average_face.npy')  
U_k = np.load('U_k.npy').reshape(68,2,16)
S_k = np.load('S_k.npy')
Mode = 1

print("Shape of Average_Face:", Average_Face.shape)
print("Shape of U_k:",U_k.shape)
print("Shape of S_k:",S_k.shape)
print(np.diag(S_k).shape)
# S_k=np.diag(S_k)

# TODO: loop in ranges for each mode and plot avtive shape model (use plot_face())
def animate_mode(mode_index):
    fig,ax=plt.subplots()
    
    U_i=U_k[:,:,mode_index]
    print("Shape of U_i:", U_i.shape)

    sigma=np.sqrt(S_k[mode_index][mode_index])  # standard deviation of the mode
    print("sigma:",sigma)
    a_values=np.linspace(-2*sigma,2*sigma,30)

    def update(frame):
        a_t=a_values[frame]
        Face_t=Average_Face+a_t*U_i
        ax.clear()
        plot_utils.plot_face(ax,Face_t)
        ax.set_title(f"Mode {mode_index+1},a={a_t:.2f}")
        ax.set_xlim([100,500])
        ax.set_ylim([100,500])

    ani=animation.FuncAnimation(fig,update,frames=len(a_values),interval=100)
    plt.show()

for mode in range(16):
    animate_mode(mode)
    
    
    
    
    
    
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation

# # Load the average face and flattened faces
# Average_Face = np.load('Average_landmarks/average_face.npy')  
# U_k = np.load('U_k.npy').reshape(68,2,16)
# S_k = np.load('S_k.npy')
# Mode = 1

# print("Shape of Average_Face:", Average_Face.shape)
# print("Shape of U_k:", U_k.shape)
# print("Shape of S_k:", S_k.shape)
# print(np.diag(S_k).shape)
# # S_k=np.diag(S_k)

# # TODO: loop in ranges for each mode and plot avtive shape model (use plot_face())
# # Generate animation for each mode
# def animate_mode(mode_index):
#     fig, ax = plt.subplots()
    
#     # Select principal component
#     U_i = U_k[:,:,mode_index]  # Select the i-th mode
#     print("Shape of U_i:", U_i.shape)

#     # Define range for a (from -2σ to +2σ)
#     sigma = np.sqrt(S_k[mode_index][mode_index])  # Standard deviation of the mode
#     print("sigma:",sigma)
#     a_values = np.linspace(-2 * sigma, 2 * sigma, 30)  # Smooth transition

#     # Function to update frame
#     def update(frame):
#         a_t = a_values[frame]
#         Face_t = Average_Face + a_t * U_i  # Apply transformation  (68,2) (42,)
#         ax.clear()
#         ax.scatter(Face_t[::2], Face_t[1::2])  # Plot landmarks
#         ax.set_title(f"Mode {mode_index+1}, a={a_t:.2f}")
#         # ax.set_xlim([-500,500])  # Adjust based on face scale
#         # ax.set_ylim([500,500])

#     ani = animation.FuncAnimation(fig, update, frames=len(a_values), interval=100)
#     plt.show()


# for mode in range(16):
#     animate_mode(mode)
