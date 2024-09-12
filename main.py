# import mediapipe as mp
# import cv2
#
# # Initialize Mediapipe Hands and Drawing modules
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
#
# # Initialize Video Capture (webcam)
# cap = cv2.VideoCapture(0)
#
# # Set up Mediapipe Hands model (detection confidence and tracking confidence are adjustable)
# with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         if not ret:
#             print("Failed to grab frame")
#             break
#
#         # Flip the frame horizontally for a more natural interaction (optional)
#         frame = cv2.flip(frame, 1)
#
#         # Convert the frame to RGB (Mediapipe processes RGB images)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#         # Process the frame and detect hands
#         result = hands.process(frame_rgb)
#
#         # Check if any hand is detected
#         if result.multi_hand_landmarks:
#             # Alert if a hand is detected
#             print("Hand detected!")
#         else:
#             print("No hand detected!")
#
#         # Draw the hand landmarks on the frame if detected
#         if result.multi_hand_landmarks:
#             for hand_landmarks in result.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#
#         # Show the frame with hand landmarks (if any)
#         cv2.imshow('Hand Detection', frame)
#
#         # Break the loop if 'Esc' key is pressed
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#
# # Release the webcam and close the OpenCV window
# cap.release()
# cv2.destroyAllWindows()
#



import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('Camera Feed', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
