import cv2
import numpy as np

# Load the image of the human face
img1 = cv2.imread('lhhieu.jpg')

# Initiate ORB detector
orb = cv2.ORB_create()

# Convert image to float32
img1 = img1.astype(np.float32)

# Start video capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to grayscale
    img2 = cv2.cvtColor(frame)

    # Find the keypoints and descriptors with ORB
    keypoints_1, descriptors_1 = orb.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img2, None)

    # Check if descriptors are not None
    if descriptors_1 is not None and descriptors_2 is not None:

        # Convert descriptors to float32
        descriptors_1 = descriptors_1.astype(np.float32)
        descriptors_2 = descriptors_2.astype(np.float32)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

        # Need to draw only good matches, so create a mask
        matchesMask = [[0, 0] for _ in range(len(matches))]

        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.9 * n.distance:
                matchesMask[i] = [1, 0]

        draw_params = dict(matchColor=(0, 255, 0),
                           singlePointColor=(255, 0, 0),
                           matchesMask=matchesMask,
                           flags=cv2.DrawMatchesFlags_DEFAULT)

        img4 = cv2.drawMatchesKnn(img1, keypoints_1, img2, keypoints_2, matches, None, **draw_params)

        # Display the images
        cv2.imshow('FLANNMatcher', img4)

        # Exit loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("No keypoints or descriptors found in image.")

# Release the capture
cap.release()
cv2.destroyAllWindows()
