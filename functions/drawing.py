import cv2
import numpy as np


def draw_bounding_boxes(image: np.ndarray, stats: np.ndarray) -> np.ndarray:
    for i in range(1, stats.shape[0]):
        x, y, w, h, area = stats[i]

        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image
