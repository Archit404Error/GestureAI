from collections import deque, namedtuple
from datetime import datetime, timedelta

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from constants import CANCEL_GESTURE, EXEC_GESTURE, HAND_POINTS
from screenshot_utils import screenshot_handler
from utils import draw_preds, draw_queue, split_landmarks
from volume_utils import volume_between, volume_handler

GestureAction = namedtuple("GestureAction", "req handler pred_classes")


action_queue: deque[GestureAction] = deque()
plaintext_queue = deque()
# Actions carried out on execute (thumbs up)
gesture_handlers = {
    "okay": (volume_handler, "volume"),
    "peace": (screenshot_handler, "screenshot"),
}
# Actions carried out after schedule and before execute
middle_handlers = {"okay": volume_between}
last_gestures = []

mp_hands = mp.solutions.hands
hand_model = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

gesture_model = load_model("model/mp_hand_gesture")
classes = open("model/gesture.names", "r").read().split("\n")

capture = cv2.VideoCapture(1)
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
        pred_classes = [classes[np.argmax(preds)] for preds in prediction]

        draw_preds(frame, pred_classes)
        draw_queue(frame, plaintext_queue)

        for req, (handler, plaintext) in gesture_handlers.items():
            if req in pred_classes and req not in last_gestures:
                action_queue.append(GestureAction(req, handler, pred_classes))
                plaintext_queue.append(plaintext)

        if action_queue and (cur_event := action_queue[0]).req in middle_handlers:
            left, right = split_landmarks(landmarks)
            middle_handlers[cur_event.req](
                frame, action_queue[0].pred_classes, left, right, cur_event.req
            )

        execute = EXEC_GESTURE in pred_classes and EXEC_GESTURE not in last_gestures
        remove = CANCEL_GESTURE in pred_classes and CANCEL_GESTURE not in last_gestures

        if action_queue and (execute or remove):
            cur_event = action_queue.popleft()
            plaintext_queue.popleft()
            left, right = split_landmarks(landmarks)
            if execute:
                cur_event.handler(
                    cur_event.pred_classes,
                    left,
                    right,
                    cur_event.req,
                )

        last_gestures = pred_classes

    cv2.imshow("Gesture AI", frame)

capture.release()
cv2.destroyAllWindows()
