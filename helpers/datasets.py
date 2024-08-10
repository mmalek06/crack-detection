import json
import os
import PIL.Image
import cv2
import numpy as np
import torch

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

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        image_info = self.image_data[idx + 1]
        image_path = os.path.join(self.images_dir, image_info["file_name"])
        image = CrackDataset._load_image(image_path)
        annotations = self.annotations.get(image_info["id"], [])
        labels, bboxes = self._parse_annotations(annotations, image.shape[0], image.shape[1])

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
    def _parse_annotations(annotations: list[dict], img_height: int, img_width: int) -> tuple[np.ndarray, np.ndarray]:
        labels = np.zeros((img_height, img_width), dtype=np.int64)
        bboxes = []

        for idx, annotation in enumerate(annotations):
            bbox = annotation["bbox"]
            x_min, y_min, width, height = map(int, bbox)
            x_max, y_max = x_min + width, y_min + height
            labels[y_min:y_max, x_min:x_max] = idx + 1

            bboxes.append([x_min, y_min, x_max, y_max])

        return labels, np.array(bboxes, dtype=np.float32)

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


def collate_variable_size_bboxes(
        batch: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function takes the batch of data and handles the variable-size bounding boxes by padding them
    to the maximum number of bounding boxes in the batch
    :param batch:
    :return:
    """
    image_paths, images, labels, bboxes = zip(*batch)
    images = torch.stack([torch.from_numpy(img) for img in images])
    labels = torch.stack([torch.from_numpy(lbl) for lbl in labels])
    max_num_bboxes = max(len(bbox) for bbox in bboxes)
    padded_bboxes = torch.zeros((len(bboxes), max_num_bboxes, 4), dtype=torch.float32)

    for i, bbox in enumerate(bboxes):
        if len(bbox) > 0:
            padded_bboxes[i, :len(bbox)] = torch.from_numpy(bbox)

    return image_paths, images, labels, padded_bboxes
