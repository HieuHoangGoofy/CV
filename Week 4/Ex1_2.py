import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils
import glob
import warnings
import os
import tqdm
import random
random.seed(42)
np.random.seed(42)
cv.setRNGSeed(42)

# Function to read images from file paths
def read_files(paths):
    images = []
    for path in paths:
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype('uint8')
        images.append(img)
    return images

# Function to display images
def show(obj, now=True):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(obj)
    ax.axis('off')
    if now:
        plt.show()
    else:
        return fig, ax

# Constants for RANSAC algorithm
TOL = 5
NUM_ITR = 1000

# Function to find matches between keypoints in two images
def get_matches(img1, img2):
    # SIFT feature detection and matching with Lowe's Test
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Flann-based matching
    flann = cv.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    matches_raw = flann.knnMatch(des1, des2, k=2)

    # Lowe's Test
    matches = []
    for m in matches_raw:
        if m[0].distance < 0.7 * m[1].distance:
            matches.append(m)
    
    points1 = np.array([kp1[m[0].queryIdx].pt for m in matches], dtype=np.float64)
    points2 = np.array([kp2[m[0].trainIdx].pt for m in matches], dtype=np.float64)

    return points1, points2

# Function to estimate homography matrix
def find_homography(src, dst):
    # Homography estimation using an algorithm similar to DLT
    A = [[] for i in range(2 * src.shape[0])]

    for i in range(src.shape[0]):
        Xi = src[i]
        xi = dst[i]
        A[2 * i] = [Xi[0], Xi[1], 1, 0, 0, 0, -xi[0]*Xi[0], -xi[0]*Xi[1], -xi[0]]
        A[(2 * i) + 1] = [0, 0, 0, Xi[0], Xi[1], 1, -xi[1]*Xi[0], -xi[1]*Xi[1], -xi[1]]

    A = np.array(A, dtype=np.float64)
    u, d, vh = np.linalg.svd(A)

    if vh is None:
        warnings.warn('Could not compute H')
        return np.eye(3)

    H = vh[8].reshape((3, 3))
    H = H / H[2, 2]
    return H

# Function to convert points to homogeneous coordinates
def to_homogenous(x):
    return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)

# Function to project 3D points to 2D image plane
def project_points(P, X):
    X_ = to_homogenous(X)
    x = P @ X_.T
    x = x / x[2]
    return x.T[:, :2]

# Function to estimate homography matrix using RANSAC
def find_homography_ransac(src, dst, tol=TOL, num_itr=NUM_ITR):
    best_H = None
    best_thresh = 0

    for i in range(num_itr):
        rand_idx = np.random.choice(dst.shape[0], size=5, replace=False)
        dst_r = dst[rand_idx, :]
        src_r = src[rand_idx, :]

        H = find_homography(src_r, dst_r)
        pred_dst = project_points(H, src)

        count = 0
        for i in range(src.shape[0]):
            e = np.linalg.norm(pred_dst[i, :] - dst[i, :])
            if e < tol:
                count += 1
        
        if count/src.shape[0] >= best_thresh:
            best_H = H
            best_thresh = count/src.shape[0]

    return best_H

# Function to crop the image based on the largest contour
def crop(img_):
    try:
        img = img_.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)

        cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)
        (x, y, w, h) = cv.boundingRect(c)
        img = img[y:y + h, x:x + w]
        return img
    except:
        return img_

# Function to overlap two images
def overlap(img_bot, img_top):
    for i in range(img_top.shape[0]):
        for j in range(img_top.shape[1]):
            if list(img_bot[i, j, :]) == [0, 0, 0]:
                img_bot[i, j, :] = img_top[i, j, :]
            elif list(img_top[i, j, :]) != [0, 0, 0]:
                img_bot[i, j, :] = img_top[i, j, :]
    return img_bot

# Function to calculate dimensions for stitching images
def get_dim(img1, H, img2):
    corners1 = np.array([
        [0,                         0],
        [0,             img1.shape[0]],
        [img1.shape[1], img1.shape[0]],
        [img1.shape[1],             0],
    ], dtype=np.float64)
    
    corners1 = project_points(H, corners1)
    
    corners2 = np.array([
        [0,                         0],
        [0,             img2.shape[0]],
        [img2.shape[1], img2.shape[0]],
        [img2.shape[1],             0],
    ], dtype=np.float64)
    
    x_cords = np.concatenate((corners1[:, 0], corners2[:, 0]))
    y_cords = np.concatenate((corners1[:, 1], corners2[:, 1]))
    
    width = int(np.ceil(np.max(x_cords) - np.min(x_cords)))
    height = int(np.ceil(np.max(y_cords) - np.min(y_cords)))
    return width, height, (np.min(x_cords), np.min(y_cords))

# Function to stitch two images
def stitch_two(img1, img2):
    points1, points2 = get_matches(img1, img2)
    H = find_homography_ransac(points1, points2)
    
    width, height, min_point = get_dim(img1, H, img2)
    T = np.array([
        [1, 0, -min_point[0]],
        [0, 1, -min_point[1]],
        [0, 0,             1],
    ])
    
    img1_ = cv.warpPerspective(img1, T @ H, (width, height))
    img2_ = cv.warpPerspective(img2, T, (width, height))
    return overlap(img1_, img2_)

# Function to create panorama from multiple images
def panaroma(images):
    if len(images) == 0:
        return
    
    base = images[0]
    for img in tqdm.tqdm(images[1:]):
        base = crop(stitch_two(img, base))
    return crop(base)

# Function to create panorama from image paths
def panorama_from_paths(image_paths):
    images = [cv.resize(x, dsize=(0, 0), fx=0.3, fy=0.3) for x in read_files(image_paths)]
    img = panaroma(images)
    show(img)

# Example usage
image_paths = [
    "1_1.jpg",
    "1_2.jpg",
    "1_3.jpg",
    "1_4.jpg"
    # Add more image paths as needed
]
panorama_from_paths(image_paths)
