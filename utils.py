from math import sqrt
from subprocess import call

import cv2

from constants import HAND_POINTS


def euclidean_dist(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def run_osascript(code):
    call([f"osascript -e '{code}'"], shell=True)


def draw_str(frame, strval, coords):
    cv2.putText(
        frame,
        strval,
        coords,
        cv2.FONT_HERSHEY_COMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_preds(frame, pred_classes):
    draw_str(frame, str(pred_classes), (50, 50))


def draw_queue(frame, actions):
    draw_str(frame, str(list(actions)), (50, 100))


def split_landmarks(landmarks):
    if len(landmarks) == 2 * HAND_POINTS:
        return landmarks[:HAND_POINTS], landmarks[HAND_POINTS:]
    return landmarks, landmarks
