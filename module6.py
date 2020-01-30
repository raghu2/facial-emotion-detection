import cv2

#min_neighbors = 3

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
video_cap = cv2.VideoCapture(0)

while True:

    ret, img = video_cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    rects = face_cascade.detectMultiScale(gray)#, minNeighbors=min_neighbors)

    if len(rects) >= 0:
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('Face Detection on Video', img)
        if cv2.waitKey(1) & 0xFF == 27:
           break
video_cap.release()

cv2.destroyAllWindows()

