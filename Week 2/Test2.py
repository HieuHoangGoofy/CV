import cv2
import numpy as np

# Video path and logo path
video_path = "output_video.mp4"
logo_path = "Logo.png"
threshold = 10

# Load the logo and calculate its histogram
roi = cv2.imread(logo_path)
hsvr = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
M = cv2.calcHist([hsvr], [0, 1], None, [180, 256], [0, 180, 0, 256])

# Open the video file
cap = cv2.VideoCapture(video_path)

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    hsvt = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    I = cv2.calcHist([hsvt], [0, 1], None, [180, 256], [0, 180, 0, 256])
    R = M / (I + 1)

    h, s, v = cv2.split(hsvt)
    B = R[h.ravel(), s.ravel()]
    B = np.minimum(B, 1)
    B = B.reshape(hsvt.shape[:2])

    # Apply morphological operations
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(B, -1, disc, B)
    B = np.uint8(B)
    cv2.normalize(B, B, 0, 255, cv2.NORM_MINMAX)

    # Apply thresholding to get a binary image
    ret, thresh = cv2.threshold(B, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            largest_contour = cnt
            max_area = area

    # Draw a bounding box around the largest contour
    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)

    # Display the frame
    cv2.imshow("Logo Detection (Improved)", frame)

    # Check for key press
    if cv2.waitKey(4) & 0xFF == ord("q"):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
