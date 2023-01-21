from math import sqrt
from subprocess import call


def euclidean_dist(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def run_osascript(code):
    call([f"osascript -e '{code}'"], shell=True)
