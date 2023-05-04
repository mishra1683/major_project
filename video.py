import pandas as pd
# Importing necessary modules
import cv2
import numpy as np
from keras.models import load_model
import streamlit as st
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
def func():
    p=[]
    FRAME_WINDOW = st.image([])

    # Face Classifier
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Loading the model
    classifier = load_model(r'major_project.h5',compile=False)
    classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

    # Different types of emotions
    emotion_labels = ['Not_Stress','Stress']

    # Capturing video using webcam
    cap = cv2.VideoCapture(0)
    # run = st.checkbox("Click this checkbox to run")
    q=0
    while (q!=50):
        # reading the video frame-by-frame
        q+=1
        _, flipped = cap.read()
        frame = cv2.flip(flipped, 1)
        labels = []
        # converting each frame from BGR to Gray
        gray = frame
        
        faces = face_classifier.detectMultiScale(gray)

        # adding rectangle box outside faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            # converting image to an image array
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # predicting the emotion
                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                if label=='Stress':
                    p.append('S')
                label_position = (x, y)
                cv2.putText(frame, label, label_position,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    return p
