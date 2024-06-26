

import cv2
import numpy as np


def detect_logo(video_path, logo_path, threshold=10):
    cap = cv2.VideoCapture(video_path)
    roi = cv2.imread(logo_path)

    hsvr = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    M = cv2.calcHist([hsvr], [0, 1], None, [180, 256], [0, 180, 0, 256])

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        blur = cv2.GaussianBlur(frame, (5, 5), 0)

        hsvt = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])

        R = M / (I + 1)

        h, s, v = cv2.split(hsvt)

        B = R[h.ravel(), s.ravel()]
        B = np.minimum(B, 1)
        B = B.reshape(hsvt.shape[:2])

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cv2.filter2D(B, -1, disc, B)
        B = np.uint8(B)
        cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)

        
        ret, thresh = cv2.threshold(B, threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        largest_contour = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                largest_contour = cnt
                max_area = area

        
        if largest_contour is not None:
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(
                frame, (x, y), (x + w, y + h), (0, 255, 255), 3
            )  

        
        cv2.imshow("Logo Detection (Improved)", frame)

        
        if cv2.waitKey(4) & 0xFF == ord("q"):
            break

    
    cap.release()
    cv2.destroyAllWindows()


video_path = "output_video.mp4"
logo_path = "Logo.png"

detect_logo(video_path, logo_path)