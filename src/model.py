from __future__ import annotations

import copy
import logging
from collections import defaultdict

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.dataset import DetracDataset

NUM_CLASSES = 4  # background + car + van + bus
_DEVICE = torch.device("cpu")

log = logging.getLogger(__name__)


def create_model(pretrained: bool = True) -> torch.nn.Module:
    log.info("Loading pretrained fasterrcnn_mobilenet_v3_large_fpn weights ...")
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights="DEFAULT" if pretrained else None,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(_DEVICE)
    log.info("Model ready.")
    return model


def train_local(
    model: torch.nn.Module,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
) -> tuple[dict[str, torch.Tensor], float]:
    model.train()
    device = _DEVICE
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    total_loss = 0.0
    num_batches = 0
    for epoch in range(epochs):
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
    avg_loss = total_loss / max(num_batches, 1)
    return copy.deepcopy(model.state_dict()), avg_loss


@torch.inference_mode()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    iou_thresholds: list[float] | None = None,
) -> dict[str, float]:
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.75]

    model.eval()
    device = _DEVICE

    all_predictions: dict[int, list[dict]] = defaultdict(list)
    all_targets: dict[int, list[dict]] = defaultdict(list)

    n_batches = len(dataloader)
    for batch_idx, (images, targets) in enumerate(dataloader, 1):
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == n_batches:
            log.info("  Evaluating batch %d/%d ...", batch_idx, n_batches)
        images = [img.to(device) for img in images]
        preds = model(images)
        for i, (pred, tgt) in enumerate(zip(preds, targets)):
            img_id = tgt["image_id"].item() if tgt["image_id"].dim() > 0 else i
            all_predictions[img_id].append({
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu(),
            })
            all_targets[img_id].append({
                "boxes": tgt["boxes"].cpu(),
                "labels": tgt["labels"].cpu(),
            })

    results = {}
    for iou_thresh in iou_thresholds:
        ap = _compute_mAP(all_predictions, all_targets, iou_thresh)
        results[f"mAP_{iou_thresh:.2f}".replace(".", "_")] = ap
    results["mAP"] = sum(results.values()) / max(len(results), 1)
    return results


def _compute_mAP(
    all_predictions: dict,
    all_targets: dict,
    iou_threshold: float,
) -> float:
    preds_list = []
    targs_list = []

    for img_id in all_targets:
        tgt_boxes = torch.cat([t["boxes"] for t in all_targets[img_id]], dim=0) if all_targets[img_id] else torch.zeros((0, 4))
        tgt_labels = torch.cat([t["labels"] for t in all_targets[img_id]], dim=0) if all_targets[img_id] else torch.zeros(0, dtype=torch.int64)

        if img_id not in all_predictions or not all_predictions[img_id]:
            preds_list.append({
                "boxes": torch.zeros((0, 4)),
                "scores": torch.zeros(0),
                "labels": torch.zeros(0, dtype=torch.int64),
            })
            targs_list.append({"boxes": tgt_boxes, "labels": tgt_labels})
            continue

        pred = all_predictions[img_id][0]
        preds_list.append(pred)
        targs_list.append({"boxes": tgt_boxes, "labels": tgt_labels})

    scores_all = []
    tp_all = []
    fp_all = []
    num_gts = 0

    for pred, tgt in zip(preds_list, targs_list):
        num_gts += tgt["boxes"].size(0)
        if pred["boxes"].size(0) == 0:
            continue

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]
        pred_labels = pred["labels"]

        sort_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sort_idx]
        pred_scores = pred_scores[sort_idx]
        pred_labels = pred_labels[sort_idx]

        detected = torch.zeros(gt_boxes.size(0), dtype=torch.bool)

        for i in range(pred_boxes.size(0)):
            scores_all.append(pred_scores[i].item())
            if gt_boxes.size(0) == 0:
                tp_all.append(0)
                fp_all.append(1)
                continue

            ious = torchvision.ops.box_iou(pred_boxes[i:i+1], gt_boxes)[0]
            best_idx = torch.argmax(ious).item()
            best_iou = ious[best_idx].item()

            if best_iou >= iou_threshold and not detected[best_idx] and pred_labels[i] == gt_labels[best_idx]:
                tp_all.append(1)
                fp_all.append(0)
                detected[best_idx] = True
            else:
                tp_all.append(0)
                fp_all.append(1)

    if num_gts == 0:
        return 0.0

    tp_cum = torch.tensor(tp_all, dtype=torch.float32).cumsum(0)
    fp_cum = torch.tensor(fp_all, dtype=torch.float32).cumsum(0)

    recall = tp_cum / num_gts
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)

    ap = _voc_ap(recall.numpy(), precision.numpy())
    return float(ap)


def _voc_ap(recall, precision) -> float:
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(mpre.size - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


import numpy as np
