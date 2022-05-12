import numpy as np
from numpy.linalg import norm
import math
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import re
import dlib
import traceback
import ctypes
from PIL import Image
import PIL
import datetime
from scipy import spatial
from pymongo import MongoClient
import streamlit as st

INPUT_SIZE = (112,112)

def predictMask(mask_model,img):
    # index == 0 : with mask, index != 1 without mask.
    IMG_DIM = (300,300)
    resized = cv2.resize(img,IMG_DIM,interpolation = cv2.INTER_AREA)
    img_array = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    img_array = img_array / 255
    img_array = np.expand_dims(img_array,axis=0)
    prediction = mask_model.predict(img_array)
    index = np.argmax(prediction)
    return index

def dynamicBrightness(img,minimum_brightness = 0.4):
    # Dynamically change the brightness of the input from stream, for better performance
    brightness = np.average(norm(img, axis=2)) / np.sqrt(3) / 255
    ratio = brightness / minimum_brightness
    if ratio >= 1:
        return img
    # Otherwise, if ratio is low, adjust brightness to get the target brightness
    return cv2.convertScaleAbs(img, alpha = 1/ratio, beta = 10)

def isMouthVisible(cascadeMouth_detect,face):
    mouths = cascadeMouth_detect.detectMultiScale(face, 1.3, 3)
    if len(mouths):
        return True
    return False

def euclidean_distance(a, b):
    x1 = a[0]; y1 = a[1]
    x2 = b[0]; y2 = b[1]
    return math.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))

def alignFace(img,eye_1,eye_2,mtcnn_detector):
    # Align the face in respect to the eyes-plane, both eyes should be perpendicular to the x-axis of the image
    # In order to get a face with no rotation/ tilt.
    try:
        img_raw = img.copy()

        if eye_1[0] <= eye_2[0]:
            left_eye = eye_1
            right_eye = eye_2
        else:
            right_eye = eye_1
            left_eye = eye_2

        left_eye_center = (int(left_eye[0] + (left_eye[2] / 2)), int(left_eye[1] + (left_eye[3] / 2)))
        left_eye_x = left_eye_center[0]; left_eye_y = left_eye_center[1]
        
        right_eye_center = (int(right_eye[0] + (right_eye[2]/2)), int(right_eye[1] + (right_eye[3]/2)))
        right_eye_x = right_eye_center[0]; right_eye_y = right_eye_center[1]
        
        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1 #rotate in clock direction
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1 #rotate inverse clock direction

        a = euclidean_distance(left_eye_center, point_3rd)
        b = euclidean_distance(right_eye_center, left_eye_center)
        c = euclidean_distance(right_eye_center, point_3rd)
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        
        angle = np.arccos(cos_a)
        
        angle = (angle * 180) / math.pi
        # because we need the angle on the left eye (the small angle), we need to subtract 90 incase the direction is -1 (since in the case we calculate the bigger degree)
        if direction == -1:
            angle = 90 - angle

        rot_img = Image.fromarray(img_raw)
        '''
        Using resampling filter in the rotation : PIL.Image.BILINEAR, since during rotation,
        the pixels get deformed/distorted, so we apply the resampling filter to sample pixels,
        in a way that will fit the change of pixels in rotation.
        '''
        rot_img = np.array(rot_img.rotate(direction * angle,PIL.Image.BILINEAR))
        rgb_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB)
        r_faces = mtcnn_detector.detect_faces(rgb_img)
        face_x, face_y, face_w, face_h = r_faces[0]['box']
        rot_face = rot_img[int(face_y):int(face_y+face_h), int(face_x):int(face_x+face_w)]
        return rot_face,True
    except:
        return 0,False

def days_between(d1, d2):
    # difference between days for DB embeddings update.
    d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)

def get_crop_face(face,landmark_predictor):
    landmark = landmark_predictor(face,dlib.rectangle(0,0,face.shape[0],face.shape[1])) # face frame and it's dlib frame
    if(landmark.num_parts > 0):
    # use the middle point of the nose (dynamically changes in accordance to the mask position on the nose)
    # point 29 was found to be the ideal point for cropping the part outside the mask
        last_nose_pt = landmark.part(29)
        cropped_face = face[ 0 : last_nose_pt.y +3, 0 : face.shape[1] ]

    return cropped_face

def predict_embedding(model,img):
    im_input = cv2.resize(img,INPUT_SIZE)
    im_input = cv2.cvtColor(im_input,cv2.COLOR_BGR2RGB)
    im_input1 = (im_input.astype(float) - 127.5) / 127
    im_input1 = np.expand_dims(im_input1,axis=0)
    embedding = model.predict(im_input1)
    return embedding

def get_dbCollection():
    client = MongoClient("mongodb+srv://liorsl:636600636@cluster0.yq8sm.mongodb.net/test?authSource=admin&replicaSet=atlas-11k2m2-shard-0&readPreference=primary&appname=mongodb-vscode%200.5.0&ssl=true")
    db = client["MFR_DB"]
    print(db.collection_names())
    collection = db["PersonaEmbeddings"]
    return collection

def insert_persona(collection,embedding):
    name = st.session_state.NAME
    print(name)
    if len(name) > 2:
        person_entry = {"name":name,"embedding":list(embedding[0].astype(float)),"date":str(datetime.date.today())}
        collection.insert_one(person_entry)
    st.session_state.NAME = ''

def recognize_persona(collection,embedding,is_undetected,persona_msg,new_entry):
    DATE_ADDED = None
    persona = None
    cos_thresh = 0.66
    euc_thresh = 4.2
    prob = 0
    persona_data = collection.find({})
    iden_list = []
    for identity in persona_data:
        data_embed = identity["embedding"]
        cos_dist = spatial.distance.cosine(data_embed,embedding)
        cos_dist = 1 - cos_dist
        print(cos_dist)
        iden_list.append([identity["name"],cos_dist,identity["date"]])
    if not any(iden_list):
        # list is empty, no entries in db
        NAME=str(input("DB is empty, Enter your full name: "))
        if NAME:
            insert_persona(collection,embedding,NAME)
    else:
        iden_list.sort(key=lambda x:x[1], reverse=True)
        cos_i = 0
        print(iden_list[cos_i])
        if iden_list[cos_i][1] > cos_thresh:
            print(iden_list[0],iden_list[1])
            persona = iden_list[cos_i][0]
            prob = iden_list[cos_i][1]
            DATE_ADDED = iden_list[cos_i][2]
        elif is_undetected:
            recog_msg = f'<p style="text-indent: -4em;font-family:sans-serif; color:Red; font-size: 20px; text-align: center; font-weight: bold;">Was Not Recognized ✗</p>'
            persona_msg.markdown(recog_msg, unsafe_allow_html=True)
            # calling insert_persona when entering the full name, where the input given is associated with the session state of the given key "NAME"
            new_entry.text_input(label='Enter Full Name',on_change=insert_persona, key = 'NAME', args = (collection,embedding))
            st.stop()
    return persona, np.array(iden_list[:6])
    # if persona is None - try again -  2 times overall, after 2 times of no prediction -  add new image with name to the database
    # returns 6 highest possible personas
        
