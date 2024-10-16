from typing import Callable

import cv2
import json
import os
import torch
import PIL.Image
import numpy as np
import torchvision

from torchvision import transforms
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, coco_file_path: str, images_dir: str):
        with open(coco_file_path, "r") as f:
            self.coco_data = json.load(f)

        self.images_dir = images_dir
        self.image_data = [
            img
            for img in self.coco_data["images"]
            if CrackDataset._load_image(os.path.join(self.images_dir, img["file_name"])) is not None
        ]
        self.annotations = self._group_annotations_by_image(self.coco_data["annotations"])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray, np.ndarray]:
        """
        Returns:
        - image_path: path to the image
        - image: the image as a numpy array
        - bboxes: a list of bounding boxes in [x_min, y_min, x_max, y_max] format
        """
        image_info = self.image_data[idx]
        image_path = os.path.join(self.images_dir, image_info["file_name"])
        image = CrackDataset._load_image(image_path)
        image = self.transform(image)
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


class CrackDatasetForClassificationWithProposals(Dataset):
    def __init__(
            self,
            selective_search_runner: Callable,
            coco_file_path: str,
            images_dir: str
    ):
        with open(coco_file_path, "r") as f:
            self.coco_data = json.load(f)

        self.selective_search_runner = selective_search_runner
        self.images_dir = images_dir
        self.image_data = [
            img
            for img in self.coco_data["images"]
            if CrackDatasetForClassificationWithProposals._load_image(os.path.join(self.images_dir, img["file_name"])) is not None
        ]
        self.annotations = self._group_annotations_by_image(self.coco_data["annotations"])
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self) -> int:
        return len(self.image_data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_info = self.image_data[idx]
        image_path = os.path.join(self.images_dir, image_info["file_name"])
        image = CrackDatasetForClassificationWithProposals._load_image(image_path)
        image = self.transform(image)
        proposals = self.selective_search_runner(image.permute(1, 2, 0).numpy(), image_path)
        ground_truth_boxes = torch.tensor(
            self._parse_annotations(self.annotations.get(image_info["id"], [])),
            dtype=torch.float32)

        if ground_truth_boxes.numel() == 0:
            return image, proposals, torch.zeros(proposals.size(0))

        ious = torchvision.ops.box_iou(proposals, ground_truth_boxes)
        labels = (ious.max(dim=1)[0] > 0.01).float()

        return image, proposals, labels

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


def custom_collate_fn(batch) -> tuple[list[str], list[np.ndarray], list[int]]:
    image_paths, images, bboxes = zip(*batch)
    images = list(images)
    bboxes = list(bboxes)

    return list(image_paths), images, bboxes


def custom_collate_proposals_fn(batch) -> tuple[torch.Tensor, list, list]:
    image, images, label_groups = zip(*batch)
    images = list(images)
    label_groups = list(label_groups)

    return image, images, label_groups
