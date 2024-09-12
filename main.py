
# Mediapipe ability demonstration

import mediapipe as mp
import cv2

# Initialize Mediapipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize Video Capture (webcam)
cap = cv2.VideoCapture(0)

# Set up Mediapipe Hands model (detection confidence and tracking confidence are adjustable)
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a more natural interaction (optional)
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB (Mediapipe processes RGB images)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and detect hands
        result = hands.process(frame_rgb)

        # Check if any hand is detected
        if result.multi_hand_landmarks:
            # Alert if a hand is detected
            print("Hand detected!")
        else:
            print("No hand detected!")

        # Draw the hand landmarks on the frame if detected
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show the frame with hand landmarks (if any)
        cv2.imshow('Hand Detection', frame)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()




#
# Test the camera access
# import cv2
#
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break
#
#     cv2.imshow('Camera Feed', frame)
#
#     if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# code for processing the image
# import os
# import mediapipe as mp
# import cv2
#
# # Initialize Mediapipe Hands and Drawing modules
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
#
# # Paths
# input_dir = 'sampleImage'  # Directory with input images
# output_dir = 'result'  # Directory to save output images
#
# # Create the result directory if it doesn't exist
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
#
# # Initialize the Mediapipe Hands module
# with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
#     # Loop through each image in the sampleImage directory
#     for img_name in os.listdir(input_dir):
#         # Construct the full input image path
#         img_path = os.path.join(input_dir, img_name)
#
#         # Read the image using OpenCV
#         image = cv2.imread(img_path)
#         if image is None:
#             print(f"Failed to read image: {img_name}")
#             continue
#
#         # Convert the image to RGB (Mediapipe processes RGB images)
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#         # Process the image with Mediapipe to detect hands
#         result = hands.process(image_rgb)
#
#         # If hand landmarks are detected
#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 # Draw the hand landmarks on the image
#                 mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#             # Construct the full output image path
#             output_path = os.path.join(output_dir, img_name)
#
#             # Save the image with the hand landmarks
#             cv2.imwrite(output_path, image)
#             print(f"Processed and saved: {output_path}")
#         else:
#             print(f"No hand detected in image: {img_name}")
