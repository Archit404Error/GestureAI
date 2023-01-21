from collections import deque, namedtuple
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from volume_utils import volume_handler

GestureAction = namedtuple("GestureAction", "time handler pred_classes")


action_queue: deque[GestureAction] = deque([])
gesture_handlers = {"fist": volume_handler}
GESTURE_TIMEOUT = 1500

mp_hands = mp.solutions.hands
hand_model = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gesture_model = load_model("model/mp_hand_gesture")
classes = open("model/gesture.names", "r").read().split("\n")

capture = cv2.VideoCapture(0)
while cv2.waitKey(1) != ord("q"):
    _, frame = capture.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hand_model.process(framergb)

    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            mp_draw.draw_landmarks(frame, handslms, mp_hands.HAND_CONNECTIONS)
            prediction = gesture_model.predict([landmarks], verbose=0)
            pred_classes = []
            for preds in prediction:
                pred_classes.append(classes[np.argmax(preds)])

            for req, handler in gesture_handlers.items():
                if req in pred_classes:
                    action_queue.append(
                        GestureAction(datetime.now(), handler, pred_classes)
                    )

            while action_queue and (
                (datetime.now() - action_queue[0].time) / timedelta(milliseconds=1)
                > GESTURE_TIMEOUT
            ):
                cur_event = action_queue.popleft()
                cur_event.handler(
                    cur_event.pred_classes, landmarks[:21], landmarks[21:]
                )

    cv2.imshow("Gesture AI", frame)

capture.release()
cv2.destroyAllWindows()
