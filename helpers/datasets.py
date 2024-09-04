import cv2
import json
import os
import torch
import numpy as np

from torch.utils.data import Dataset, default_collate


class CrackDataset(Dataset):
    def __init__(self, coco_file_path: str, images_dir: str):
        with open(coco_file_path, "r") as f:
            self.coco_data = json.load(f)

        self.images_dir = images_dir
        self.image_data = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = self._group_annotations_by_image(self.coco_data["annotations"])

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray, list[int], np.ndarray]:
        """
        Returns:
        - image_path: path to the image
        - image: the image as a numpy array
        - labels: a list of foreground/background labels (1 for crack, 0 for background)
        - bboxes: a list of bounding boxes in [x_min, y_min, x_max, y_max] format
        """
        image_info = self.image_data[idx + 1]
        image_path = os.path.join(self.images_dir, image_info["file_name"])
        image = CrackDataset._load_image(image_path)
        annotations = self.annotations.get(image_info["id"], [])
        labels, bboxes = self._parse_annotations(annotations)

        return image_path, image, labels, bboxes

    @staticmethod
    def _group_annotations_by_image(annotations: list[dict]) -> dict[int, list[dict]]:
        grouped_annotations = {}

        for annotation in annotations:
            image_id = annotation["image_id"]

            if image_id not in grouped_annotations:
                grouped_annotations[image_id] = []

            grouped_annotations[image_id].append(annotation)

        return grouped_annotations

    @staticmethod
    def _parse_annotations(annotations: list[dict]) -> tuple[list[int], np.ndarray]:
        """
        Convert the annotations into binary labels (1 for crack, 0 for background) and bounding boxes.

        Returns:
        - labels: List of 1s (for cracks, i.e., foreground) and 0s (for background)
        - bboxes: A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        """
        labels = []
        bboxes = []

        for annotation in annotations:
            bbox = annotation.get("bbox", [])

            if not bbox or len(bbox) != 4:
                continue

            x_min, y_min, width, height = map(int, bbox)

            if width <= 0 or height <= 0:
                continue

            x_max, y_max = x_min + width, y_min + height

            labels.append(1)
            bboxes.append([x_min, y_min, x_max, y_max])

        if not bboxes:
            return [], np.zeros((0, 4), dtype=np.float32)

        return labels, np.array(bboxes, dtype=np.float32)

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        return image


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-size bounding boxes.
    Args:
    - batch: list of tuples (image_path, image, labels, bboxes)
    """
    image_paths, images, labels, bboxes = zip(*batch)
    images = default_collate(images)
    max_num_boxes = max([len(b) for b in bboxes])
    padded_bboxes = []
    padded_labels = []

    for label, bbox in zip(labels, bboxes):
        padded_labels.append(label + [0] * (max_num_boxes - len(label)))
        padded_bboxes.append(np.pad(bbox, ((0, max_num_boxes - len(bbox)), (0, 0)), 'constant'))

    padded_labels = torch.tensor(padded_labels)
    padded_bboxes = torch.tensor(padded_bboxes, dtype=torch.float32)

    return image_paths, images, padded_labels, padded_bboxes
