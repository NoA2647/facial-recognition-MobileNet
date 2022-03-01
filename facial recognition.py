import numpy as np
import cv2
from keras.models import load_model

classifier = load_model('facialRecognitionModel.h5')

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is tuple():
        return None

    for (x, y, w, h) in faces:
        cropped_face = img[y: y + h, x: x + w]

    return cropped_face


def preprocess(image):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    image = image / 255.
    image = image.reshape(1, 224, 224, 3)
    return image


def draw_test(name, predict, im):
    faces = "None"

    # insert name of class
    faces_dict = {"[0]": "amir ",
                  "[1]": "arshia", }

    if result is not None:
        faces = faces_dict[str(predict)]
    BLACK = [0, 0, 0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100, cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, faces, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow(name, expanded_image)


cap = cv2.VideoCapture(0)
while True:
    ret, img1 = cap.read()
    img = face_extractor(img1)
    result = None
    if img is not None:
        img = preprocess(img)
        # Get Prediction
        x = classifier.predict(img, 1, verbose=0)
        print(x)
        result = np.argmax(x, axis=1)

    # Show image with predicted class
    draw_test("Prediction", result, img1)
    cv2.waitKey(1)
