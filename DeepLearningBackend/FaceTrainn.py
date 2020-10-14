import os
import glob
import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow.keras.backend as K

def loadModel():
    #Define VGG_FACE_MODEL architecture
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    # Load VGG Face model weights
    model.load_weights('vgg_face_weights.h5')

    # Remove Last Softmax layer and get model upto last flatten layer with outputs 2622 units
    vgg_face=Model(inputs=model.layers[0].input,outputs=model.layers[-2].output)
    return vgg_face

def classifier_model():
    #Prepare Training Data
    vgg_face=loadModel()
    x_train=[]
    y_train=[]
    person_folders=os.listdir('./Images_crop/')
    person_rep=dict()
    for i,person in enumerate(person_folders):
        person_rep[i]=person
        image_names=os.listdir('./Images_crop/'+person+'/')
        for image_name in image_names:
            img=load_img('./Images_crop/'+person+'/'+image_name,target_size=(224,224))
            img=img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img=preprocess_input(img)
            img_encode=vgg_face(img)
            x_train.append(np.squeeze(K.eval(img_encode)).tolist())
            y_train.append(i)

    x_train=np.array(x_train)
    y_train=np.array(y_train)
    np.save('train_data',x_train)
    np.save('train_labels',y_train)

    #Prepare Test Data
    x_test=[]
    y_test=[]
    person_folders=os.listdir('./Test_Images_crop/')
    for i,person in enumerate(person_folders):
        image_names=os.listdir('./Test_Images_crop/'+person+'/')
        for image_name in image_names:
            img=load_img('./Test_Images_crop/'+person+'/'+image_name,target_size=(224,224))
            img=img_to_array(img)
            img=np.expand_dims(img,axis=0)
            img=preprocess_input(img)
            img_encode=vgg_face(img)
            x_test.append(np.squeeze(K.eval(img_encode)).tolist())
            y_test.append(i)
    print(x_train)
    print(y_train)

    x_test=np.array(x_test)
    y_test=np.array(y_test)
    np.save('test_data',x_test)
    np.save('test_labels',y_test)
    # x_train=np.load('train_data.npy')
    # y_train=np.load('train_labels.npy')

    # Softmax regressor to classify images based on encoding 
    classifier_model=Sequential()
    classifier_model.add(Dense(units=100,input_dim=x_train.shape[1],kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.3))
    classifier_model.add(Dense(units=10,kernel_initializer='glorot_uniform'))
    classifier_model.add(BatchNormalization())
    classifier_model.add(Activation('tanh'))
    classifier_model.add(Dropout(0.2))
    classifier_model.add(Dense(units=7,kernel_initializer='he_uniform'))
    classifier_model.add(Activation('softmax'))
    classifier_model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer='nadam',metrics=['accuracy'])

    classifier_model.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test))
    tf.keras.models.save_model(classifier_model,'face_classifier_model.h5')

# classifier_model()