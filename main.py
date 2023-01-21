import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

mp_hands = mp.solutions.hands
hand_model = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gesture_model = load_model("model/mp_hand_gesture")
classes = open("model/gesture.names", "r").read().split("\n")

capture = cv2.VideoCapture(0)
while True:
    _, frame = capture.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hand_model.process(framergb)

    pred_class = ""

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)

            prediction = gesture_model.predict([landmarks])
            pred_classes = []
            for preds in prediction:
                pred_classes.append(classes[np.argmax(preds)])
            print(pred_classes)

    cv2.imshow("Gesture AI", frame)

    if cv2.waitKey(1) == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
