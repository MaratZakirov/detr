# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
#from pycocotools import mask as coco_mask

import datasets.transforms as T
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class YoloDetection(Dataset):
    def __init__(self, img_folder, transforms, return_masks):
        self._transforms = transforms
        self.prepare = ConvertYoloPolysToMask(return_masks)

        with open(img_folder, "r") as file:
            self.img_files = [f.rstrip() for f in file.readlines()]

            # TODO get rid off all empty labels
            self.img_files = self.filterEmpty(self.img_files)

        self.label_files = [
            path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.batch_count = 0

    def filterEmpty(self, img_files):
        new_files = []
        for fi in img_files:
            fl = fi.rstrip().replace('.jpg', '.txt').replace('images', 'labels')
            if len(open(fl).readlines()) > 0:
                new_files.append(fi)
        img_files = new_files
        return img_files

    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx])
        target = np.loadtxt(self.label_files[idx]).astype(np.float32)
        target = {'image_id': -1, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.img_files)

class ConvertYoloPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        boxes = torch.tensor(anno[:, -4:])
        boxes[:, :2] = boxes[:, :2] - boxes[:, 2:]/2
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
        boxes[:, 0::2] *= w
        boxes[:, 1::2] *= h
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = torch.tensor(anno[:, 2], dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = (boxes[:, 2:] - boxes[:, :2])[:, 0] * (boxes[:, 2:] - boxes[:, :2])[:, 1]#obj["area"] for obj in anno])
        iscrowd = torch.tensor([0] * len(anno))#obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        # TODO hack currently filter out all ignore boxes
        if target["labels"].min().item() < 0:
            keep = target["labels"] >= 0
            target["boxes"] = target["boxes"][keep]
            target["labels"] = target["labels"][keep]
            target["area"] = target["area"][keep]
            target["iscrowd"] = target["iscrowd"][keep]

        return image, target


def make_yolo_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640]#, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomGrayscale(),
            T.RandomShuffle(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomRotation(40),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.yolo_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    PATHS = {"train": root / "train.txt",
             "val": root / "test.txt"}
    ann_file = PATHS[image_set]
    dataset = YoloDetection(ann_file, transforms=make_yolo_transforms(image_set), return_masks=args.masks)
    return dataset
