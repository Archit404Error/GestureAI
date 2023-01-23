from utils import euclidean_dist, run_osascript


def set_volume(level):
    run_osascript(f"set volume output volume {level}")
    print(f"Volume was set to {level}")


def volume_gesture(landmarks):
    dist = euclidean_dist(landmarks[4], landmarks[8]) - 100
    set_volume(dist // 10)


def volume_handler(preds, left, right, keyword):
    landmarks = right if preds[0] == keyword else left
    volume_gesture(landmarks)
