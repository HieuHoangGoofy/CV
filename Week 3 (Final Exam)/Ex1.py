import cv2
import numpy as np

# Load the images
img1 = cv2.imread('box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('box_in_img.png', cv2.IMREAD_GRAYSCALE)

# Initiate ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
keypoints_1, descriptors_1 = orb.detectAndCompute(img1, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(img2, None)

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
cv2.waitKey(0)
cv2.destroyAllWindows()
