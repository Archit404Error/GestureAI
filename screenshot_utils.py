import os
import os.path


def screenshot_handler(preds, left, right):
    img_path = "./pics"
    if not os.path.exists(img_path):
        os.mkdir(img_path, 0o666)
    suffix = sum([1 for _ in os.listdir(img_path)])
    image_file = f"{img_path}/test{suffix}.png"
    os.system(f"screencapture {image_file}")
    print(f"took picture {image_file}")
