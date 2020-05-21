import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential,model_from_json
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D,Flatten,Dropout
from tensorflow.keras.callbacks import EarlyStopping


def train_model():
    data = np.load('mnist_data.npz')
    X_train, Y_train, X_test, Y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']
    X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.15)
    
    # Scaling Data
    X_train = X_train/255
    X_val = X_val/255
    X_test = X_test/255
    
    # Reshape Data
    X_train = X_train.reshape(X_train.shape[0], 28*28)
    X_val = X_val.reshape(X_val.shape[0], 28*28)
    X_test = X_test.reshape(X_test.shape[0], 28*28)
    
    # Model Building
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(30,activation='relu'))
    model.add(Dropout(0.2))    
    model.add(Dense(10,activation='softmax'))
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    early_stop = EarlyStopping(patience=2)
    
    model.fit(X_train,Y_train,validation_data=(X_val,Y_val),epochs=10,batch_size=100,verbose=2,callbacks=[early_stop])
    
    model_json = model.to_json()
    with open("model/tr_model_1.json", "w") as json_file:
        json_file.write(model_json)
        
    model.save("model/tr_model_weights_1.h5")


def train_cnn_model():
    data = np.load('mnist_data.npz')
    X_train, Y_train, X_test, Y_test = data['x_train'],data['y_train'],data['x_test'],data['y_test']
    X_train,X_val,Y_train,Y_val = train_test_split(X_train,Y_train,test_size=0.15)
    
    
    # Scaling Data
    X_train = X_train/255
    X_val = X_val/255
    X_test = X_test/255
    
    
    # Reshape Data
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    
    # Model Building
    model = Sequential()
    
    model.add(Conv2D(filters=10,kernel_size=(4,4),input_shape=(28, 28, 1),padding='valid',activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(20,activation='relu'))
    model.add(Dense(10,activation='softmax'))
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    early_stop = EarlyStopping(patience=2)
    
    
    model.fit(X_train,Y_train,validation_data=(X_val,Y_val),epochs=20,batch_size=100,verbose=2,callbacks=[early_stop])
   
    model_json = model.to_json()
    with open("model/tr_model_2.json", "w") as json_file:
        json_file.write(model_json)
        
    model.save("model/tr_model_weights_2.h5")
    
    
def load_model():
    #load model
    model = model_from_json(open("model/tr_model_1.json", "r").read())
    #load weights
    model.load_weights('model/tr_model_weights_1.h5')
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

def load_cnn_model():
    #load model
    model = model_from_json(open("model/tr_model_2.json", "r").read())
    #load weights
    model.load_weights('model/tr_model_weights_2.h5')
    
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model


def predict_digit(test_x):
    model = load_model()
    
    test_x = test_x/255
    test_x = test_x.reshape(1, 28*28)

    p = model.predict(test_x).argmax(axis=1)
    
    print('Model 1 Output - ',p[0])
    
    return p[0]


def predict_digit_cnn(test_x):
    model = load_cnn_model()
    
    test_x = test_x/255
    test_x = test_x.reshape(1, 28, 28, 1)

    p = model.predict(test_x).argmax(axis=1)
    
    print('Model 2 Output - ',p[0])
    
    return p[0]


