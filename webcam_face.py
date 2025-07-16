import cv2
import os
import numpy as np
import dlib
from utils import process_frame, save_landmarks, save_image 


OUTPUT_FOLDER = "images_landmarks"
PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
FRAME_SKIP_RATE = 2


if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

vc = cv2.VideoCapture(0)

if not vc.isOpened():
    print("Error: Cannot access the camera.")
else:
    print("Press 'q' to quit or 'space' to print landmarks.")


frame_count = 0
frame_number = 0

# Video Capture and Processing Loop
while True:
    rval, frame = vc.read()

    frame_count += 1

    # Process every 3rd frame
    if frame_count % FRAME_SKIP_RATE == 1:
        processed_frame, landmarks = process_frame(detector, predictor, frame)

        # Display the processed frame
        cv2.imshow('Facial Landmark Detection', processed_frame)
        
        # Handle Keyboard Input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            print("Exiting...")
            break
        elif key == ord('s'):  # Save the current frame and landmarks
            save_image(processed_frame, frame_number, OUTPUT_FOLDER)
            save_landmarks(landmarks, frame_number, OUTPUT_FOLDER)
            frame_number += 1
        elif key == ord(' '):  # Print landmarks
            print('Facial Landmarks:', landmarks)
            print('images shape =',processed_frame.shape)


vc.release()
cv2.destroyAllWindows()