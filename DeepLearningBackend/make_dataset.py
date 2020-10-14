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
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K

image_path_names=[]
person_names=set()
# for file_name in glob.glob('./FaceData/*_[1-9]*.jpg'):
#     image_path_names.append(file_name)
#     person_names.add(image_path_names[-1].split('/')[-1].split('_')[0])

faceCascade = cv2.CascadeClassifier(r'haarcascades/haarcascade_frontalface_default.xml')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "FaceData")

for root, dirs, files in os.walk(image_dir):
    	for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                person_names.add(os.path.basename(root))
                image_path_names.append(path)
                # print(os.path.join(root, file))
# print(person_names)
# print(image_path_names)
# dnnFaceDetector=dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

if not os.path.exists('./Images_crop'):
        os.mkdir('./Images_crop')

for person in person_names:
    if os.path.exists('./Images_crop/'+person+'/'):
        os.rmdir('./Images_crop/'+person+'/')
    os.mkdir('./Images_crop/'+person+'/')


for file_name in image_path_names:
    print(file_name)
    img=cv2.imread(file_name)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # rects=dnnFaceDetector(gray,1)
    # left,top,right,bottom=0,0,0,0
    # for (i,rect) in enumerate(rects):
    #     left=rect.rect.left() #x1
    #     top=rect.rect.top() #y1
    #     right=rect.rect.right() #x2
    #     bottom=rect.rect.bottom() #y2
    # width=right-left
    # height=bottom-top
    faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
    # print("hello")
    for (x, y, w, h) in faces:
        img_crop = img[y-90:y+h+70, x-50:x+w+50]
        img_path='./Images_crop/'+file_name.split('\\')[-2]+'/'+file_name.split('\\')[-1] 
        print(img_path)
        cv2.imwrite(img_path,img_crop)
        # print("hi")
        # cv2.imshow('image',img_crop)

#Test data generation
test_image_path_names=[]
image_dir = os.path.join(BASE_DIR, "TestData")

for root, dirs, files in os.walk(image_dir):
    	for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                test_image_path_names.append(os.path.join(root, file))

if not os.path.exists('./Test_Images_crop'):
        os.mkdir('./Test_Images_crop')

for person in person_names:
    if os.path.exists('./Test_Images_crop/'+person+'/'):
        os.rmdir('./Test_Images_crop/'+person+'/')
    os.mkdir('./Test_Images_crop/'+person+'/')

for file_name in test_image_path_names:
    img=cv2.imread(file_name)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
    for (x, y, w, h) in faces:
        img_crop = img[y-90:y+h+70, x-50:x+w+50]
        img_path='./Test_Images_crop/'+file_name.split('\\')[-2]+'/'+file_name.split('\\')[-1]
        print(img_path)
        cv2.imwrite(img_path,img_crop)
