import cv2
import numpy as np

# Load the input image of your face
img = cv2.imread('MyFace.jpg', cv2.IMREAD_GRAYSCALE)
sift = cv2.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img, None)

# Initialize the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    keypoints_2, descriptors_2 = sift.detectAndCompute(gray, None)

    # Match keypoints between the input image and the camera frame
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)
    
    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    if len(good) >= 10:
        src_pts = np.float32([keypoints_1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints_2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Draw matches
        img3 = cv2.drawMatchesKnn(img, keypoints_1, frame, keypoints_2, [good], None, flags=2)

        # Connect keypoints between the input image and the camera frame
        for i in range(len(src_pts)):
            x1, y1 = src_pts[i][0]
            x2, y2 = dst_pts[i][0]
            cv2.line(img3, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        # Display the result
        cv2.imshow('Flann Match', img3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
