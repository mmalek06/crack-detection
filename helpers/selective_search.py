import numpy as np
import torch
import torchvision

from skimage.segmentation import felzenszwalb
from skimage.measure import regionprops
from torchvision import transforms


def generate_proposals(image: np.ndarray, transform: transforms.Compose) -> list[tuple[torch.Tensor, list[int]]]:
    segments = felzenszwalb(image, scale=100, sigma=0.8, min_size=50)
    regions = []

    for region in regionprops(segments):
        minr, minc, maxr, maxc = region.bbox
        proposal = image[minr:maxr, minc:maxc]

        if proposal.shape[0] > 0 and proposal.shape[1] > 0:
            proposal_tensor = transform(proposal)
            regions.append((proposal_tensor, [minc, minr, maxc, maxr]))

    return regions


def get_label_for_proposal(labels: torch.Tensor, proposal_box: torch.Tensor) -> torch.Tensor:
    """
    Extract the most frequent label within the region corresponding to the proposal box.

    :param labels: The full label grid (e.g., (1, 448, 448))
    :param proposal_box: The bounding box of the proposal in (x_min, y_min, x_max, y_max) format
    :return: The most frequent label within the proposal box
    """
    x_min, y_min, x_max, y_max = proposal_box.int().tolist()
    label_region = labels[:, y_min:y_max, x_min:x_max]
    # this needs to be cast to long because that's what CrossEntropyLoss expects
    aggregated_label = label_region.reshape(-1).mode().values

    return aggregated_label


def get_best_bbox_for_proposal(stats: torch.Tensor, bbox_preds: torch.Tensor) -> torch.Tensor:
    """
    Find the best matching bounding box from the stats based on IoU.

    :param stats: Tensor of bounding boxes (e.g., (1, x, 4)) where x is the number of bounding boxes
    :param bbox_preds: Predicted bounding box from the model in (x_min, y_min, x_max, y_max) format
    :return: The best matching bounding box from stats, or an empty tensor if none are valid
    """
    if stats.size(1) == 0 or bbox_preds.size(0) == 0:
        return torch.empty(0, 4, dtype=stats.dtype, device=stats.device)

    stats = stats.squeeze(0)

    print(f"Stats shape: {stats.shape}")
    print(f"bbox_preds shape: {bbox_preds.shape}")

    valid_indices = ~(stats == 0).all(dim=1)
    print(f"Valid indices: {valid_indices}")

    stats = stats[valid_indices]
    print(f"Filtered stats shape: {stats.shape}")

    # Further filter out bounding boxes where x_min >= x_max or y_min >= y_max
    invalid_indices = (stats[:, 0] >= stats[:, 2]) | (stats[:, 1] >= stats[:, 3])
    if torch.any(invalid_indices):
        stats = stats[~invalid_indices]

    if stats.size(0) == 0:
        # No valid ground truth bounding boxes
        return torch.empty(0, 4, dtype=stats.dtype, device=stats.device)

    # Clamp bbox_preds to ensure valid values (using out-of-place operations)
    bbox_preds = bbox_preds.clone()  # Clone to avoid modifying the original tensor
    bbox_preds[:, 0] = torch.clamp(bbox_preds[:, 0], min=0)  # x_min >= 0
    bbox_preds[:, 1] = torch.clamp(bbox_preds[:, 1], min=0)  # y_min >= 0
    bbox_preds[:, 2] = torch.clamp(bbox_preds[:, 2], min=bbox_preds[:, 0] + 1)  # x_max > x_min
    bbox_preds[:, 3] = torch.clamp(bbox_preds[:, 3], min=bbox_preds[:, 1] + 1)  # y_max > y_min

    if torch.any(bbox_preds[:, 0] >= bbox_preds[:, 2]) or torch.any(bbox_preds[:, 1] >= bbox_preds[:, 3]):
        return torch.empty(0, 4, dtype=stats.dtype, device=stats.device)

    ious = torchvision.ops.box_iou(bbox_preds, stats)
    if ious.numel() == 0 or torch.any(torch.isnan(ious)):
        return torch.empty(0, 4, dtype=stats.dtype, device=stats.device)

    best_stat_idx = ious.argmax().item()
    best_stat = stats[best_stat_idx].unsqueeze(0)

    return best_stat

