
import cv2
import numpy as np

target_img = cv2.imread("Logo.png")
target_hsv = cv2.cvtColor(target_img, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([target_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])

cv2.normalize(roi_hist, roi_hist, 0, 180, cv2.NORM_MINMAX)


cap = cv2.VideoCapture("output_video.mp4")

channels = [0, 1]
scale = 1

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    backproj = cv2.calcBackProject([hsv], channels, roi_hist, [0, 180, 0, 256], scale)

    _, thresh = cv2.threshold(backproj, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video with Logo Detection", frame)
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()