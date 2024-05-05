import cv2
import numpy as np

# Load the image of Ronaldo's face
face_img = cv2.imread('ronaldo.png', cv2.IMREAD_GRAYSCALE)

# Load the video
video_path = 'ronaldo_video.mp4'
cap = cv2.VideoCapture(video_path)

# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors of Ronaldo's face
face_keypoints, face_descriptors = orb.detectAndCompute(face_img, None)

# Convert descriptors to float32
face_descriptors = face_descriptors.astype(np.float32)

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

    # Find the keypoints and descriptors of the frame
    frame_keypoints, frame_descriptors = orb.detectAndCompute(gray_frame, None)

    # Convert descriptors to float32
    frame_descriptors = frame_descriptors.astype(np.float32)

    # Match descriptors of Ronaldo's face with frame descriptors
    matches = flann.knnMatch(face_descriptors, frame_descriptors, k=2)

    # Apply ratio test as per Lowe's paper
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    # Draw matches on the frame
    matched_frame = cv2.drawMatches(face_img, face_keypoints, frame, frame_keypoints,
                                    good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the frame with matches
    cv2.imshow('Matches', matched_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
