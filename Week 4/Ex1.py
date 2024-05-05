def warpImages (img1, img2, H):

 rows1, cols1 img1.shape[:2]
 rows2, cols2 img2.shape[:2]
 listPoints1 = np.float32([[0,0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2)
 tempPoints = np.float32([[0,0], [0, rows2], [cols2, rows2], [cols2,0]]).reshape(-1,1,2)

 listPoints2 = cv2.perspectiveTransform (tempPoints, H)

 listPoints = np.concatenate((listPoints1, listPoints2), axis=0)

 [xMin, yMin] = np.int32(listPoints.min(axis=0).ravel() 0.5)

 [xMax, yMax] = np.int32(listPoints.max(axis=0).ravel() +0.5)

 translationDist = [-xMin,-yMin]

 HTranslation = np.array([[1, 0, translationDist [0]], [0, 1, translationDist[1]], [0, 0, 1]])

 output_img = cv2.warpPerspective (img2, HTranslation.dot(H), (xMax-xMin, yMax-yMin))

 output_img[translationDist [1]: rows1+translationDist [1], translationDist[0]: cols1+translation Dist [0]] = img1

 return output_img

def getTransform (src, dst, method='affine'):

 pts1, pts2 featureMatching(src, dst)

 srcPts = np.float32(pts1).reshape(-1,1,2)

 dstPts np.float32(pts2).reshape(-1,1,2)

 if method == 'affine':

 H, mask cv2.estimateAffine2D (srcPts, dstPts, cv2.RANSAC, ransacReproj Threshold=5.0)

 if method == 'homography':

 H, mask = cv2.findHomography (srcPts, dstPts, cv2.RANSAC, 5.0)

 matchesMask mask.ravel().tolist()
 return (H, pts1, pts2, mask)