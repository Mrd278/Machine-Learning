# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 18:29:46 2019

@author: Mridul
"""

import cv2

face_cascade = cv2.CascadeClassifier('C:\\Users\\mridu\\Downloads\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_default.xml')
img = cv2.imread('C:\\Users\\mridu\\OneDrive\\Pictures\\Saved Pictures\\brothers.JPG')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, scaleFactor = 1.05, minNeighbors = 5)

for x,y,w,h in faces:
    img = cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0),3)

resized = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))
cv2.imshow('Brothers', resized)

cv2.waitKey(0)
cv2.destroyAllWindows()

