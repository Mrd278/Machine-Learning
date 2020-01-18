# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 19:16:20 2019

@author: Mridul
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers.core import Dense,Activation,Dropout
from keras.models import Sequential

dataset = pd.read_csv('Diabetestype.csv')

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
X[:,6] = le.fit_transform(X[:,6])
onehotencoder=OneHotEncoder(categorical_features=[6])
X= onehotencoder.fit_transform(X).toarray()

from sklearn.preprocessing import StandardScaler
ssc = StandardScaler()
X = ssc.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

model = Sequential()

model.add(Dense(30, input_shape = (8,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit(X_train,y_train, epochs = 100, batch_size = 100, verbose = 2, validation_data = (X_test, y_test))

predict = model.predict_classes(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predict)