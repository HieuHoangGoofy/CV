import cv2 
image_paths=['1_1.jpg','1_2.jpg','1_3.jpg','1_4.jpg'] 
# initialized a list of images 
imgs = [] 
  
for i in range(len(image_paths)): 
    imgs.append(cv2.imread(image_paths[i])) 
    imgs[i]=cv2.resize(imgs[i],(0,0),fx=0.1,fy=0.1) 
   

  
stitchy=cv2.Stitcher.create() 
(dummy,output)=stitchy.stitch(imgs) 
  
if dummy != cv2.STITCHER_OK: 
 
    print("stitching ain't successful") 
else:  
    print('Your Panorama is ready!!!') 
  
# final output 
cv2.imshow('final result',output) 
  
cv2.waitKey(0)