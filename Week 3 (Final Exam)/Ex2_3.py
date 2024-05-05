import cv2
import numpy as np

# Load the image of Ronaldo's face
face_img = cv2.imread('ronaldo2.jpg', cv2.IMREAD_GRAYSCALE)

# Load the video
video_path = 'ronaldo_video4.mp4'
cap = cv2.VideoCapture(video_path)

# Initiate ORB detector
orb = cv2.ORB_create()

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Process each frame of the video
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # If faces are detected, proceed with keypoint matching
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract the detected face region
            detected_face = gray_frame[y:y+h, x:x+w]

            # Resize the detected face region to match the size of Ronaldo's face image
            resized_detected_face = cv2.resize(detected_face, (face_img.shape[1], face_img.shape[0]))

            # Find the keypoints and descriptors of the detected face
            face_keypoints, face_descriptors = orb.detectAndCompute(resized_detected_face, None)

            # Convert descriptors to float32
            if face_descriptors is not None:
             face_descriptors = face_descriptors.astype(np.float32)

            # Match descriptors of Ronaldo's face with faace descriptors
            matches = flann.knnMatch(face_descriptors, face_descriptors, k=2)

            # Apply ratio test as per Lowe's paper
            good_matches = []
            for m, n in matches:
                if m.distance < 0.001 * n.distance:
                    good_matches.append(m)

            # Draw matches on the frame
            matched_frame = cv2.drawMatches(face_img, orb.detect(face_img, None), resized_detected_face, face_keypoints,
                                            good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            # Display the frame with matches
            cv2.imshow('Matches', matched_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
