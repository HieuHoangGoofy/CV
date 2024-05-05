import cv2
import numpy as np

# Load the image of Ronaldo's face
face_img = cv2.imread('ronaldo2.jpg')

# Resize the image to half its original size
resized_face_img = cv2.resize(face_img, None, fx=0.5, fy=0.5)

# Load the pre-trained face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the video
video_path = 'ronaldo_video4.mp4'
cap = cv2.VideoCapture(video_path)

# Get the original video width and height
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate the new width and height for resizing (half the original size)
new_width = original_width // 2
new_height = original_height // 2

# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors of Ronaldo's face
face_keypoints, face_descriptors = orb.detectAndCompute(resized_face_img, None)

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

    # Resize the frame to half its original size
    resized_frame = cv2.resize(frame, (new_width, new_height))

    # Convert the resized frame to grayscale
    gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Extract the detected face region
        face_region = gray_frame[y:y + h, x:x + w]

        # Find the keypoints of the face region
        region_keypoints = orb.detect(face_region)

        # Compute descriptors if keypoints are found
        if region_keypoints:
            region_keypoints, region_descriptors = orb.compute(face_region, region_keypoints)

            # Convert descriptors to float32
            region_descriptors = region_descriptors.astype(np.float32)

            # Match descriptors of Ronaldo's face with face region descriptors
            if len(region_descriptors) > 0:
                matches = flann.knnMatch(face_descriptors, region_descriptors, k=min(2, len(region_descriptors)))
                # Apply ratio test as per Lowe's paper
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)

        # Draw matches on the frame
        matched_frame = cv2.drawMatches(resized_face_img, face_keypoints, resized_frame, region_keypoints,
                                    good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # Display the frame with matches
        cv2.imshow('Matches', matched_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
