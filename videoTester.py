import os
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image
import csv
import datetime

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')
#threshold
emotion_thld = 0.8


face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap=cv2.VideoCapture(0)

# Create a csv file to write the output
now = datetime.datetime.now()
csv_name = 'test_' + now.strftime("%Y%m%d_%H%M%S") + '.csv'
with open(csv_name, 'w', newline='') as f:
    thewriter = csv.writer(f)
    thewriter.writerow(['timestamp', 'emotion'])

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)


    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv2.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        # Save frames of extreme emotions
        if (predictions[0][max_index] > emotion_thld) & (max_index != 6): #omit neutral emotions
            now = datetime.datetime.now()
            frame_name = 'frame_' + predicted_emotion + now.strftime("%Y%m%d_%H%M%S") + '.jpg'
            cv2.imwrite(frame_name, test_img)

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        with open(csv_name, 'a') as fd:
            writer = csv.writer(fd)
            timestamp = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
            row = [timestamp, predicted_emotion]
            writer.writerow(row)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ',resized_img)



    if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows