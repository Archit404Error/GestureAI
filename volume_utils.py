import cv2

from utils import draw_str, euclidean_dist, run_osascript


def set_volume(level):
    run_osascript(f"set volume output volume {level}")
    print(f"Volume was set to {level}")


def volume_gesture(landmarks):
    dist = euclidean_dist(landmarks[4], landmarks[8]) - 100
    set_volume(dist // 10)


def volume_handler(preds, left, right, keyword):
    landmarks = right if preds[0] == keyword else left
    volume_gesture(landmarks)


def volume_between(frame, preds, left, right, keyword):
    landmarks = right if preds[0] == keyword else left
    thumb_pt = landmarks[4]
    finger_pt = landmarks[8]
    mid_pt = ((thumb_pt[0] + finger_pt[0]) // 2, (thumb_pt[1] + finger_pt[1]) // 2)
    dist = euclidean_dist(thumb_pt, finger_pt) - 100
    draw_str(frame, f"Vol: {dist // 10}", mid_pt)
    cv2.circle(frame, thumb_pt, 15, (255, 255, 255))
    cv2.circle(frame, finger_pt, 15, (255, 255, 255))
    cv2.line(frame, thumb_pt, finger_pt, (255, 255, 255), 13)
