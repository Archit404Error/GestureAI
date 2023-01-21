from utils import euclidean_dist, run_osascript


def set_volume(level):
    run_osascript(f"set volume output volume {level}")
    print(f"Volume was set to {level}")


def volume_gesture(landmarks):
    dist = euclidean_dist(landmarks[4], landmarks[8]) - 100
    set_volume(dist // 10)


def volume_handler(preds, left, right):
    landmarks = left if preds[0] == "fist" else right
    volume_gesture(landmarks)
