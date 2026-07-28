from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms import v2 as T


VEHICLE_LABELS = {"car": 1, "van": 2, "bus": 3}


def _load_annotation_file(xml_path: str) -> dict[int, list[dict[str, Any]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    result: dict[int, list[dict[str, Any]]] = {}
    for frame in root.findall("frame"):
        frame_num = int(frame.get("num", 0))
        targets = []
        for target in frame.findall(".//target"):
            box = target.find("box")
            attr = target.find("attribute")
            if box is None:
                continue
            left = float(box.get("left", 0))
            top = float(box.get("top", 0))
            w = float(box.get("width", 0))
            h = float(box.get("height", 0))
            vtype = attr.get("vehicle_type", "car") if attr is not None else "car"
            targets.append({
                "bbox": [left, top, left + w, top + h],
                "label": VEHICLE_LABELS.get(vtype, 1),
                "area": w * h,
            })
        result[frame_num] = targets
    return result


def _resolve_image_root(base: str) -> str:
    p = Path(base)
    if p.exists() and p.is_dir():
        children = list(p.iterdir())
        if children and all(c.is_dir() for c in children):
            return str(p)
    nested = p / p.name
    if nested.exists() and nested.is_dir():
        return str(nested)
    parent = p.parent
    nested_parent = parent / parent.name
    if nested_parent.exists() and nested_parent.is_dir():
        return str(nested_parent)
    return base


_IMAGE_TRANSFORM = T.Compose([
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class DetracDataset(Dataset):
    def __init__(
        self,
        image_root: str,
        annotation_root: str,
        image_size: int = 480,
    ):
        self.image_root = _resolve_image_root(image_root)
        self.annotation_root = annotation_root
        self.image_size = image_size
        self.samples: list[dict[str, Any]] = []

        image_dir = Path(self.image_root)
        ann_dir = Path(annotation_root)

        seq_dirs = sorted(d for d in image_dir.iterdir() if d.is_dir())

        ann_cache: dict[str, dict[int, list[dict[str, Any]]]] = {}

        for seq_dir in seq_dirs:
            seq_name = seq_dir.name
            ann_file = ann_dir / f"{seq_name}.xml"
            if not ann_file.exists():
                continue

            if seq_name not in ann_cache:
                ann_cache[seq_name] = _load_annotation_file(str(ann_file))
            annotations = ann_cache[seq_name]

            image_files = sorted(seq_dir.glob("*.jpg"))
            for img_path in image_files:
                frame_num = int(img_path.stem.replace("img", ""))
                targets = annotations.get(frame_num, [])
                boxes = [t["bbox"] for t in targets]
                labels = [t["label"] for t in targets]
                areas = [t["area"] for t in targets]
                self.samples.append({
                    "image_path": str(img_path),
                    "boxes": boxes,
                    "labels": labels,
                    "areas": areas,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        image = read_image(sample["image_path"], mode=ImageReadMode.RGB)
        orig_h, orig_w = image.shape[1], image.shape[2]
        scale = self.image_size / max(orig_h, orig_w)
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)
        image = T.functional.resize(image, [new_h, new_w], antialias=True)
        image = _IMAGE_TRANSFORM(image)
        if new_h < self.image_size or new_w < self.image_size:
            pad_h = self.image_size - new_h
            pad_w = self.image_size - new_w
            image = T.functional.pad(image, [0, 0, pad_w, pad_h], fill=0)

        boxes = torch.tensor(sample["boxes"], dtype=torch.float32) if sample["boxes"] else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(sample["labels"], dtype=torch.int64) if sample["labels"] else torch.zeros(0, dtype=torch.int64)

        if boxes.numel() > 0:
            boxes = boxes * scale

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": torch.tensor(sample["areas"], dtype=torch.float32) * (scale ** 2) if sample["areas"] else torch.zeros(0, dtype=torch.float32),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image, target

    @staticmethod
    def split_for_edge(
        dataset: Dataset,
        edge_id: int,
        num_edges: int,
        seed: int = 42,
    ) -> Subset:
        n = len(dataset)
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=rng).tolist()
        per_edge = n // num_edges
        start = edge_id * per_edge
        end = start + per_edge if edge_id < num_edges - 1 else n
        return Subset(dataset, perm[start:end])

    @staticmethod
    def get_validation_split(
        dataset: Dataset,
        val_ratio: float,
        seed: int = 42,
    ) -> Subset:
        n = len(dataset)
        n_val = max(1, int(n * val_ratio))
        rng = torch.Generator().manual_seed(seed + 1)
        perm = torch.randperm(n, generator=rng).tolist()
        return Subset(dataset, perm[:n_val])

    @staticmethod
    def get_edge_image_paths(
        dataset: Dataset,
        edge_id: int,
        num_edges: int,
        seed: int = 42,
    ) -> list[str]:
        subset = DetracDataset.split_for_edge(dataset, edge_id, num_edges, seed)
        return [dataset.samples[subset.indices[i]]["image_path"] for i in range(len(subset))]

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))
