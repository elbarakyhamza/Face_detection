import numpy as np
import cv2

prototext_path = "./models/MobileNetSSD_deploy.prototxt"
model_path = "./models/MobileNetSSD_deploy.caffemodel"
min_confidence = 0.3

classes = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

np.random.seed(543210)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNetFromCaffe(prototext_path, model_path)

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
            cv2.rectangle(region_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 5)

    height, width = frame.shape[0], frame.shape[1]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007, (300, 300), 130)
    net.setInput(blob)
    detected_objects = net.forward()

    for i in range(detected_objects.shape[2]):
        confidence = detected_objects[0][0][i][2]
        if confidence > min_confidence:
            class_index = int(detected_objects[0, 0, i, 1])

            upper_left_x = int(detected_objects[0, 0, i, 3] * width)
            upper_left_y = int(detected_objects[0, 0, i, 4] * height)
            lower_right_x = int(detected_objects[0, 0, i, 5] * width)
            lower_right_y = int(detected_objects[0, 0, i, 6] * height)

            prediction_text = f"{classes[class_index]}: {100*confidence:.2f}%"

            cv2.rectangle(
                frame,
                (upper_left_x, upper_left_y),
                (lower_right_x, lower_right_y),
                colors[class_index],
            )
            cv2.putText(
                frame,
                prediction_text,
                (
                    upper_left_x,
                    upper_left_y - 15 if upper_left_y > 30 else upper_left_y + 15,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                colors[class_index],
                2,
            )

    cv2.imshow("Window", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
