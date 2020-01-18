import tkinter
import cv2
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

model = load_model('mrd.h5')
model.summary()

def test():
    stream = cv2.VideoCapture(0)
    grabbed, frame = stream.read()
    if grabbed:
        cv2.imshow('test',frame)
        cv2.imwrite('frame1.jpg',frame)
        print(frame.shape)
        img = image.load_img('frame1.jpg', target_size = (500,500,3))
        img = image.img_to_array(img)
        img = img/255
        img = np.asarray(img).reshape(1,500,500,3)
        pred = model.predict(img)
        print(pred)
        

window = tkinter.Tk()

btn = tkinter.Button(text = 'test', width = 30, command = test).pack()

window.mainloop()
