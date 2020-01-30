import keras_applications
import keras_preprocessing
from keras.models import load_model
import cv2
import numpy as np

loaded_model = load_model("saved_models/face_detection_model9.h5")

def classify(a):
    batch_size = 1
    label = loaded_model.predict(a, batch_size)
    act = np.argmax(label)
    em = " "
    if act == 0:
        em = "angry"
    elif act == 1:
        em = "disgust"
    elif act == 2:
        em = "fear"
    elif act == 3:
        em = "happy"
    elif act == 4:
        em = "sad"
    elif act == 5:
        em = "surprise"
    elif act == 6:
        em = "neutral"

    return em

def detect():
    min_neighbors = 3

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    video_cap = cv2.VideoCapture(0)
    video_cap.set(cv2.CAP_PROP_FPS, 30)

    while True:

        ret, img = video_cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = face_cascade.detectMultiScale(gray, minNeighbors=min_neighbors)

        if len(rects) >= 0:
            for (x, y, w, h) in rects:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                img_array = np.array(gray[y:y+h, x:x+w])
                dim = (48, 48)
                resized = cv2.resize(img, dim)
                img_reshape = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                img_reshape = img_reshape.reshape(1, 48, 48, 1)
                id = classify(img_reshape)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(img, str(id), (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Emotion detection", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    video_cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect()
