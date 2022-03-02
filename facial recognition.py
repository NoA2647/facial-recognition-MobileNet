import numpy as np
import cv2
from keras.models import load_model
import pandas as pd
import datetime
import os

classifier = load_model('facialRecognitionModel.h5')

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# insert name of class
faces_dict = {"[0]": "amir ",
              "[1]": "arshia", }


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is tuple():
        return None

    return faces


def preprocess(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = image / 255.
    image = image.reshape(1, 224, 224, 3)
    return image


def show(faces, predict, original):
    for i in range(len(faces)):
        result = faces_dict[str(predict[i])]
        x, y, w, h = faces[i]
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original, result, (x + w + 6, y + h + 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("prediction", original)


def writeLog(path, log):
    if not os.path.isfile(path):
        f = open(path, 'w')
        f.close()

    try:
        history = pd.read_csv(path, index_col=0)
    except pd.errors.EmptyDataError as e:
        history = pd.DataFrame(columns=['Name', 'StartTime', 'EndTime'])
    df = pd.DataFrame(log, columns=['Name', 'StartTime', 'EndTime'])
    history = pd.concat([history, df])
    history.to_csv(path)


def run():
    cap = cv2.VideoCapture(0)
    lock = True
    logs = []
    lastlog = []
    while True:
        ret, original = cap.read()
        faces = face_extractor(original)
        predict = []
        if faces is not None:
            for face in faces:
                x, y, w, h = face
                crop = original[y: y + h, x: x + w]
                crop = preprocess(crop)
                # Get Prediction
                predict.append(np.argmax(classifier.predict(crop, 1, verbose=0), axis=1))

            # add new person or don't do anything if person already there
            templog = lastlog.copy()
            for pred in predict:
                if lastlog.count(pred) == 0:
                    startTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    logs.append([faces_dict[str(pred)], startTime, None])
                    lastlog.append(pred)
                else:
                    templog.remove(pred)

            # remove person fom lastlog and save endtime in log
            for temp in templog:
                endTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                lastlog.remove(temp)
                for i in range(len(logs)):
                    if logs[i][0] == temp:
                        logs[i][2] = endTime

            if lock:
                lock = False
            show(faces, predict, original)

        else:
            if not lock:
                endTime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                for i in range(len(logs)):
                    if logs[i][2] is None:
                        logs[i][2] = endTime
                writeLog('history.csv', logs)
                lastlog.clear()
                logs.clear()
                lock = True
            cv2.imshow("prediction", original)

        cv2.waitKey(1)


run()
