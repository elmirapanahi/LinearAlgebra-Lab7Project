import numpy as np
import cv2
import os
import dlib


class FaceRegister:
    """
    A unified class to compute and solve the registration problem (affine or similarity) for face data.
    """

    def __init__(self, Face, Neutral_Face, method="affine"):
        """
        Initialize with the input matrices Face and Neutral_Face and the registration method.

        Parameters:
            Face (np.ndarray): Shape (N, 2), where the first column is x_i and the second column is y_i.
            Neutral_Face (np.ndarray): Shape (N, 2), where the first column is x'_i and the second column is y'_i.
            method (str): The registration method to use ("affine" or "similarity").
        """
        self.Face = Face
        self.Neutral_Face = Neutral_Face
        self.method = method.lower()
        self.coefficients = {}

    def normalize_data(self):
        """
        Normalizes Face and Neutral_Face by centering them to (0, 0) and scaling to unit variance.
        """
        # Center the data (subtract the mean)
        self.Face = self.Face - np.mean(self.Face, axis=0)
        self.Neutral_Face = self.Neutral_Face - \
            np.mean(self.Neutral_Face, axis=0)

        # Scale the data to unit variance (L2-norm scaling)
        self.Face = self.Face / np.linalg.norm(self.Face, axis=0)
        self.Neutral_Face = self.Neutral_Face / \
            np.linalg.norm(self.Neutral_Face, axis=0)

    def extract_columns(self):
        """
        Extracts x_i, y_i, x'_i, and y'_i from the input matrices.
        """
        self.x_i_prime = self.Face[:, 0]
        self.y_i_prime = self.Face[:, 1]
        self.x_i = self.Neutral_Face[:, 0]
        self.y_i = self.Neutral_Face[:, 1]

    def compute_affine_coefficients(self):
        """
        Computes the coefficients A, B, C, D, E, F, G needed for affine registration.
        """
        self.coefficients = {
            'A': np.sum(self.x_i_prime**2),
            'B': np.sum(self.x_i_prime * self.y_i_prime),
            'C': np.sum(self.y_i_prime**2),
            'D': np.sum(self.x_i_prime * self.x_i),
            'E': np.sum(self.y_i_prime * self.x_i),
            'F': np.sum(self.x_i_prime * self.y_i),
            'G': np.sum(self.y_i_prime * self.y_i),
        }

    def compute_similarity_coefficients(self):
        """
        Computes the coefficients A, B, C, D needed for similarity registration.
        """
        self.coefficients = {
            'A': np.sum(self.x_i_prime**2 + self.y_i_prime**2),
            'B': np.sum(-2 * (self.x_i_prime * self.y_i_prime)),
            'C': np.sum(self.x_i * self.x_i_prime + self.y_i * self.y_i_prime),
            'D': np.sum(self.y_i * self.x_i_prime - self.x_i * self.y_i_prime),
        }

    def construct_affine_matrices(self):
        """
        Constructs the coefficient matrix and constants vector for affine registration.

        Returns:
            tuple: Coefficient matrix and constants vector.
        """
        A, B, C = self.coefficients['A'], self.coefficients['B'], self.coefficients['C']
        D, E, F, G = self.coefficients['D'], self.coefficients['E'], self.coefficients['F'], self.coefficients['G']

        coeff_matrix = np.array([[A, B, 0, 0],
                                 [B, C, 0, 0],
                                 [0, 0, A, B],
                                 [0, 0, B, C]])

        constants = np.array([D, E, F, G])

        return coeff_matrix, constants

    def construct_similarity_matrices(self):
        """
        Constructs the coefficient matrix and constants vector for similarity registration.

        Returns:
            tuple: Coefficient matrix and constants vector.
        """
        A, B = self.coefficients['A'], self.coefficients['B']
        C, D = self.coefficients['C'], self.coefficients['D']

        coeff_matrix = np.array([[A, 0], [B, A]])

        constants = np.array([C, D])

        return coeff_matrix, constants

    def solve_linear_system(self):
        """
        Solves the system of equations for the registration parameters.

        Returns:
            np.ndarray: The solution array array [a, b] or array [a, b, c, d].
        """
        if self.method == "affine":
            coeff_matrix, constants = self.construct_affine_matrices()
        elif self.method == "similarity":
            coeff_matrix, constants = self.construct_similarity_matrices()
        else:
            raise ValueError("Invalid method. Use 'affine' or 'similarity'.")
        solution = np.linalg.solve(coeff_matrix, constants)

        # Solve the system of equations
        # A, B = self.coefficients['A'], self.coefficients['B']
        # C, D = self.coefficients['C'], self.coefficients['D']
        # a = C/A
        # b = (D-B*a)/A
        # solution = np.array([a, b])
        return solution

    def register(self):
        """
        Performs the full registration process: normalization, extraction, coefficient computation, and solving.

        Returns:
            dict: The computed coefficients and the solution for the parameters.
        """
        # Step 1: Normalize data
        self.normalize_data()

        # Step 2: Extract columns
        self.extract_columns()

        # Step 3: Compute coefficients
        if self.method == "affine":
            self.compute_affine_coefficients()
        elif self.method == "similarity":
            self.compute_similarity_coefficients()
        else:
            raise ValueError("Invalid method. Use 'affine' or 'similarity'.")

        # Step 4: Solve the system of equations
        solution = self.solve_linear_system()
        if self.method == "affine":
            a, b, c, d = solution
            return np.array([[a, b], [c, d]])
        elif self.method == "similarity":
            a, b = solution
            return np.array([[a, -b], [b, a]])


def process_frame(detector, predictor, frame):
    """
    Detects faces and landmarks, and draws them on the frame.

    Args:
        frame: Current video frame.

    Returns:
        Annotated frame and facial landmarks as a list of arrays.
    """

    dets = detector(frame, 1)
    landmarks = []  # Initialize landmarks as an empty list

    for k, d in enumerate(dets):
        shape = predictor(frame, d)
        landmarks = np.array([(shape.part(i).x, shape.part(i).y)
                             for i in range(shape.num_parts)])
        # Draw landmarks
        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 2, (0, 0, 255))

    return frame, landmarks


def save_landmarks(landmarks, frame_number, output_folder):
    """
    Saves the landmarks as a text file.

    Args:
        landmarks: List of facial landmarks.
        frame_number: Current frame number for file naming.
    """
    filename = os.path.join(output_folder, f"landmarks_{frame_number}.npy")
    np.save(filename, landmarks)  # Save as .npy file
    print(f"Saved: {filename}")


def save_image(frame, frame_number, output_folder):
    """
    Saves the current frame as an image file.

    Args:
        frame: The frame to be saved.
        frame_number: The frame number for file naming.
    """
    filename = os.path.join(output_folder, f"frame_{frame_number}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Saved image: {filename}")
