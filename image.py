import cv2
import numpy as np
 

 
 
img = cv2.imread('d.png')
 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
 
#Red color rangle  169, 100, 100 , 189, 255, 255
 
 
lower_range = np.array([100,100,0])
upper_range = np.array([255,255,255])
 
mask = cv2.inRange(hsv, lower_range, upper_range)
 
mask = cv2.resize(mask, (960, 540))     
#cv2.imshow('image', img)
cv2.imshow('mask', mask)
 
 
cv2.waitKey(0)
cv2.destroyAllWindows()
#%%

cv2.imshow('w',hsv)