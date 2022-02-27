import cv2
import os

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Load functions
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is tuple():
        return None

    for (x, y, w, h) in faces:
        cropped_face = img[y: y + h, x: x + w]

    return cropped_face


def collectFromStorage(sourcePath, destPath):
    images = os.listdir(sourcePath)
    count = 0
    for img in images:
        oldImg = cv2.imread(img)
        newImg = face_extractor(oldImg)
        if newImg is not None:
            cv2.imwrite(destPath+f'/face_{count}', newImg)
            count += 1
    print(f"Collecting Samples Complete\n{count} images saved in {destPath}")


def collectFromWebcam(n=20, path='.', show=True):
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    count = 0
    i = 0
    while True:
        ret, img = cap.read()
        cv2.putText(img, str(i), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('origin', img)
        i += 1
        if i == 100:
            i = 0
            face = face_extractor(img)
            if face is not None:
                count += 1
                face = cv2.resize(face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                file_name_path = f'{path}/face_{count}.jpg'

                cv2.imwrite(file_name_path, face)

                if show:
                    # Put count on images and display live count
                    cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Face Cropper', face)

            else:
                print("Face not found")
                pass

            if count == n:
                break

        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Collecting Samples Complete\nimages saved in {path}")

