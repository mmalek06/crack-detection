import pickle

import cv2
import numpy as np

from torch.utils.data import Dataset


class CrackDataset(Dataset):
    def __init__(self, image_paths: list[str], labels_path: str, stats_path: str):
        self.image_paths = image_paths
        self.all_labels = CrackDataset._load_incremental_data(labels_path, np.int64, trim=False)
        self.all_stats = CrackDataset._load_incremental_data(stats_path, np.float32, trim=True)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx: int) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        image = CrackDataset._load_image(self.image_paths[idx])
        labels = self.all_labels[idx]
        stats = self.all_stats[idx]

        return self.image_paths[idx], image, labels, stats

    @staticmethod
    def _load_incremental_data(file_path: str, dtype, trim: bool) -> list[np.ndarray]:
        data = []

        with open(file_path, "rb") as f:
            while True:
                try:
                    batch = pickle.load(f)

                    if trim:
                        # skip the first row and the last column in each row
                        # it's done like this because find_boxes function saved all that cv2 module spat out
                        processed_batch = [np.array(item[1:, :-1], dtype=dtype) for item in batch]
                    else:
                        processed_batch = [np.array(item, dtype=dtype) for item in batch]

                    data.extend(processed_batch)
                except EOFError:
                    break

        return data

    @staticmethod
    def _load_image(image_path: str) -> np.ndarray:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        return image
