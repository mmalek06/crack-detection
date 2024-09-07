import cv2
import json
import os
import torch
import PIL.Image
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, coco_file_path: str, images_dir: str):
        with open(coco_file_path, "r") as f:
            self.coco_data = json.load(f)

        self.images_dir = images_dir
        self.image_data = {img["id"]: img for img in self.coco_data["images"]}
        self.annotations = self._group_annotations_by_image(self.coco_data["annotations"])

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray, np.ndarray]:
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
        bboxes = self._parse_annotations(annotations)

        return image_path, image, bboxes

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
    def _parse_annotations(annotations: list[dict]) -> np.ndarray:
        """
        Convert the annotations into binary labels (1 for crack, 0 for background) and bounding boxes.

        Returns:
        - bboxes: A list of bounding boxes in the format [x_min, y_min, x_max, y_max].
        """
        bboxes = []

        for annotation in annotations:
            bbox = annotation.get("bbox", [])

            if not bbox or len(bbox) != 4:
                continue

            x_min, y_min, width, height = map(int, bbox)

            if width <= 0 or height <= 0:
                continue

            x_max, y_max = x_min + width, y_min + height

            bboxes.append([x_min, y_min, x_max, y_max])

        if not bboxes:
            return np.zeros((0, 4), dtype=np.float32)

        return np.array(bboxes, dtype=np.float32)

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        return image


class CrackDatasetForClassification(Dataset):
    def __init__(self, images_dir, transform: transforms.Compose):
        self.images_dir = images_dir
        self.image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = PIL.Image.open(img_path).convert("RGB")
        label = 0 if "noncrack" in img_name else 1
        image = self.transform(image)

        return image, label


def custom_collate_fn(batch) -> tuple[list[str], list[np.ndarray], list[int]]:
    image_paths, images, bboxes = zip(*batch)
    images = list(images)
    bboxes = list(bboxes)

    return list(image_paths), images, bboxes
