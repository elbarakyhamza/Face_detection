import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
eye_glasses_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
)

while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)

        region_gray = gray[y : y + h, x : x + w]
        region_color = frame[y : y + h, x : x + w]

        eyes = eye_cascade.detectMultiScale(region_gray, 1.3, 5)
        eye_glasses = eye_glasses_cascade.detectMultiScale(region_gray, 1.3, 5)

        for ex, ey, ew, eh in eyes:
            cv2.rectangle(region_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

        for ex, ey, ew, eh in eye_glasses:
            cv2.rectangle(region_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

    cv2.imshow("Window", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
