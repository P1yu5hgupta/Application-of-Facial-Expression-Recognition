import cv2
import sys
from keras.models import load_model
import time
import numpy as np
from decimal import Decimal
from model_utils import define_model, model_weights
from FaceTrainn import loadModel
import pickle
import os.path
from datetime import datetime
import json
import requests
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow.keras.backend as K
from numba import jit, cuda

url = "http://localhost:9000/emotion-detector/user/data"
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

# loads and resizes an image
def resize_img(image_path):
    img = cv2.imread(image_path, 1)
    img = cv2.resize(img, (48, 48))
    return True

# runs the realtime emotion detection 
def realtime_emotions():
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)
    person_folders=os.listdir('./Images_crop/')
    person_rep=dict()
    for i,person in enumerate(person_folders):
        person_rep[i]=person
    # load keras model
    model = define_model()
    model = model_weights(model)
    print('Model loaded')

    # Load saved model
    classifier_model=tf.keras.models.load_model('face_classifier_model.h5')

    input_video="akshay_mov.mp4"
    # save location for image
    save_loc = 'save_loc/1.jpg'
    # numpy matrix for stroing prediction
    result = np.array((1,7))
    # for knowing whether prediction has started or not
    once = False
    # load haar cascade for face
    faceCascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
    # list of given emotions
    EMOTIONS = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Sad', 'Surprised', 'Neutral']

    # store the emoji coreesponding to different emotions
    emoji_faces = []
    for index, emotion in enumerate(EMOTIONS):
        emoji_faces.append(cv2.imread('emojis/' + emotion.lower()  + '.png', -1))
    # set video capture device , webcam in this case
    video_capture = cv2.VideoCapture(0)
    video_capture.set(3, 640)  # WIDTH
    video_capture.set(4, 480)  # HEIGHT

    # save current time
    prev_time = time.time()

    # start webcam feed
    data=[]
    while True:
         # Capture frame-by-frame
        df={}
        ret, frame = video_capture.read()
        # mirror the frame
        frame = cv2.flip(frame, 1, 0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # find face in the frame
        faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
        if(len(data)>0 and len(faces)==0):
            try:
                requests.post(url, data=json.dumps(data), headers=headers,timeout=0.0000001)
            except requests.exceptions.ReadTimeout: 
                pass
            data=[]

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            # required region for the face
            cv2.rectangle(frame, (x-10, y-70),(x+w+20, y+h+40), (15, 175, 61), 4)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y-90:y+h+70, x-50:x+w+50]

            # save the detected face
            cv2.imwrite(save_loc, roi_color)
            # draw a rectangle bounding the face
            curr_time = time.time()
            # keeps track of waiting time for emotion recognition
             # do prediction only when the required elapsed time has passed 
            
            # read the saved image
            img = cv2.imread(save_loc, 0)
        
            if img is not None:
                if curr_time - prev_time >=1:
                    vgg_face=loadModel()
                    # id_, conf = recognizer.predict(img)
                    df['name']=None
                    df['time']=str(datetime.now())
                    # crop_img=cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                    crop_img=cv2.resize(roi_color,(224,224))
                    crop_img=img_to_array(crop_img)
                    crop_img=np.expand_dims(crop_img,axis=0)
                    crop_img=preprocess_input(crop_img)
                    img_encode=vgg_face(crop_img)

                    # Make Predictions
                    embed=K.eval(img_encode)
                    person=classifier_model.predict(embed)
                    namme=person_rep[np.argmax(person)]
                    #print(5: #id_)
                    #print(labels[id_])
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    df['name']=namme
                    color = (255, 255, 255)
                    stroke = 2
                    cv2.putText(frame, namme, (x,y), font, 1, color, stroke, cv2.LINE_AA)
                    # indicates that prediction has been done atleast once
                    once = True

                    # resize image for the model
                    img = cv2.resize(img, (48, 48))
                    img = np.reshape(img, (1, 48, 48, 1))
                    # do prediction
                    result = model.predict(img)
                    # print(result)
                    df['emotion']=EMOTIONS[np.argmax(result[0])]
                    print(json.dumps(df))
                    data.append(df)
                    
                    prev_time = time.time()
                    total_sum = np.sum(result[0])
                    # select the emoji face with highest confidence
                    emoji_face = emoji_faces[np.argmax(result[0])]
            if once==True:
                for index, emotion in enumerate(EMOTIONS):
                    text = str(round(Decimal(result[0][index]/total_sum*100), 2) ) + "%"
                    # for drawing progress bar
                    cv2.rectangle(frame, (100, index * 20 + 10), (100 +int(result[0][index] * 100), (index + 1) * 20 + 4),
                                    (255, 0, 0), -1)
                    # for putting emotion labels
                    cv2.putText(frame, emotion, (10, index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
                    # for putting percentage confidence
                    cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
              
                for c in range(0, 3):
                    # for doing overlay we need to assign weights to both foreground and background
                    foreground = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0)
                    background = frame[350:470, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
                    frame[350:470, 10:130, c] = foreground + background
            break
        # Display the resulting frame
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()

# def prediction_path(path):
#     # load keras model
#     model = define_model()
#     model = model_weights(model)
#     classifier_model=tf.keras.models.load_model('face_classifier_model.h5')

#     data=[]
#     df={}
#     faceCascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')
    
#     # list of given emotions
#     EMOTIONS = ['Angry', 'Disgusted', 'Fearful',
#                 'Happy', 'Sad', 'Surprised', 'Neutral']

#     if os.path.exists(path):
#         # read the image
#         frame = cv2.imread(path, 0)
#         # check if image is valid or not
#         if frame is None:
#             print('Invalid image !!')
#             return 
#         # else:
#         #     print('Image not found')
#         #     return
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = faceCascade.detectMultiScale(
#                 gray,
#                 scaleFactor=1.1,
#                 minNeighbors=5,
#                 minSize=(30, 30),
#                 flags=cv2.CASCADE_SCALE_IMAGE)

#         for (x, y, w, h) in faces:
#             # required region for the face
#             cv2.rectangle(frame, (x-10, y-70),(x+w+20, y+h+40), (15, 175, 61), 4)
#             roi_gray = gray[y:y+h, x:x+w]
#             roi_color = frame[y-90:y+h+70, x-50:x+w+50]
#             cv2.imwrite(save_loc, roi_color)
#             img = cv2.imread(save_loc, 0)
            
#             if img is not None:
#                 vgg_face=loadModel()
#                 # id_, conf = recognizer.predict(img)
#                 df['name']=None
#                 df['time']=str(datetime.now())
#                 # crop_img=cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
#                 crop_img=cv2.resize(roi_color,(224,224))
#                 crop_img=img_to_array(crop_img)
#                 crop_img=np.expand_dims(crop_img,axis=0)
#                 crop_img=preprocess_input(crop_img)
#                 img_encode=vgg_face(crop_img)

#                 # Make Predictions
#                 embed=K.eval(img_encode)
#                 person=classifier_model.predict(embed)
#                 namme=person_rep[np.argmax(person)]

#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 df['name']=namme
#                 color = (255, 255, 255)
#                 stroke = 2
#                 cv2.putText(frame, namme, (x,y), font, 1, color, stroke, cv2.LINE_AA)

#                 # resize image for the model
#                 img = cv2.resize(img, (48, 48))
#                 img = np.reshape(img, (1, 48, 48, 1))
#                 # do prediction
#                 result = model.predict(img)
#                 # print(result)
#                 df['emotion']=EMOTIONS[np.argmax(result[0])]
#                 print(json.dumps(df))
#                 data.append(df)
#                 total_sum = np.sum(result[0])
#                 # select the emoji face with highest confidence
#                 emoji_face = emoji_faces[np.argmax(result[0])]

#                 for index, emotion in enumerate(EMOTIONS):
#                     text = str(round(Decimal(result[0][index]/total_sum*100), 2) ) + "%"
#                     # for drawing progress bar
#                     cv2.rectangle(frame, (100, index * 20 + 10), (100 +int(result[0][index] * 100), (index + 1) * 20 + 4),
#                                     (255, 0, 0), -1)
#                     # for putting emotion labels
#                     cv2.putText(frame, emotion, (10, index * 20 + 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (7, 109, 16), 2)
#                     # for putting percentage confidence
#                     cv2.putText(frame, text, (105 + int(result[0][index] * 100), index * 20 + 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                
#                 for c in range(0, 3):
#                     # for doing overlay we need to assign weights to both foreground and background
#                     foreground = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0)
#                     background = frame[350:470, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
#                     frame[350:470, 10:130, c] = foreground + background
                
#                 cv2.imshow('image', frame)

#                 # if cv2.waitKey(1) & 0xFF == ord('q'):
#                 #     break
#                 print("hi")
#         if(len(data)>0):
#             try:
#                 requests.post(url, data=json.dumps(data), headers=headers,timeout=0.0000001)
#             except requests.exceptions.ReadTimeout: 
#                 pass
#     return