from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
import numpy as np

dataset = []
for i in range(1,15):
    img = image.load_img('My_pictures/'+str(i)+'.jpg', target_size = (500,500,3))
    img = image.img_to_array(img)
    img = img/255
    dataset.append(img)

dataset = np.asarray(dataset)
print(dataset.shape)

y = np.ones(14).T

model = Sequential()
model.add(Conv2D(filters = 16, kernel_size = (5,5), activation = 'relu',input_shape = (500,500,3)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (5,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (5,5), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
          
model.add(Dense(32, activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(16, activation = 'relu'))
model.add(Dropout(0.25))

model.add(Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(dataset,y, epochs = 100, batch_size = 2)

model.save('mrd.h5', overwrite = True)
