import numpy as np
import matplotlib.pyplot as plt

# Define edges for the facial landmarks
Edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8),
         (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),
         (17, 18), (18, 19), (19, 20), (20, 21),
         (22, 23), (23, 24), (24, 25), (25, 26),
         (27, 28), (28, 29), (29, 30), (30, 33), (31, 32), (32, 33), (33, 34), (34, 35),
         (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),
         (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),
         (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54),
         (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),
         (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)]

def plot_face(ax, X, edges=Edges, color='b', margin=50, title='Facial Landmarks Plot', grid = True, marker_size=5):
    """
    Plots a face using landmarks and edges on a given Axes object.

    Parameters:
    - ax: matplotlib.axes.Axes object where the face will be plotted.
    - X: numpy.ndarray of shape (N, 2), containing landmark coordinates.
    - edges: list of tuples, each containing indices of landmarks to connect.
    - color: color of the points and lines.
    - margin: integer, margin added to the axis limits.
    - title: string, title of the subplot.
    """
    # Plot the landmarks
    ax.plot(X[:, 0], X[:, 1], 'o', color=color, markersize=marker_size)
    
    # Draw edges between landmarks
    for i, j in edges:
        xi, yi = X[i, 0], X[i, 1]
        xj, yj = X[j, 0], X[j, 1]
        ax.plot([xi, xj], [yi, yj], '-', color=color)
    
    # Dynamically set axis limits based on landmarks with some margin
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Invert y-axis if needed
    
    ax.set_title(title)
    ax.grid(grid)
    ax.axis('equal') 