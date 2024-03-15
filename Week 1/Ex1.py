import cv2
import numpy as np

cap = cv2.VideoCapture('video.mp4')  

def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)
   
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)

# Load your icon image
icon = read_transparent_png('Logo.png')  

icon_width_percentage = 10  
icon_height_percentage = 10 

icon_position = (10, 10)  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    icon_width = 100
    icon_height = 40
    
    icon_resized = cv2.resize(icon, (icon_width, icon_height))

    # Overlay your icon onto the frame
    frame[icon_position[1]:icon_position[1] + icon_height, icon_position[0]:icon_position[0] + icon_width] = icon_resized
    
    text = "Cocogoat"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (120, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Frame', frame)

    if cv2.waitKey(24) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
