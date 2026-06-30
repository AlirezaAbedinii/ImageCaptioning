"""Dataset utilities: COCO caption loading, train/test image split and the
serialisable dataframe of (image id, captions, url).

The logic mirrors the data-preparation cells of ``Image_Captioning.ipynb``.
Building the dataframe once and serialising it to CSV lets later runs skip the
(slow) COCO traversal entirely.
"""
from __future__ import annotations

import os
import random
from shutil import copyfile

import pandas as pd
from pycocotools.coco import COCO

from . import config
from .config import CONFIG


def load_coco(annotations_file: str = config.ANNOTATIONS_FILE) -> COCO:
    """Initialise the COCO caption API."""
    return COCO(annotations_file)


def img_id_to_url(img_id: int, img_type: str = "jpg") -> str:
    """COCO stores files as zero-padded 12-digit ids, e.g. ``139 -> 000000000139.jpg``."""
    return str(img_id).zfill(12) + "." + img_type


def img_url_to_id(img_url: str, img_type: str = "jpg") -> int:
    return int(img_url[: -len(img_type) - 1])


def split_data(source_dir: str, train_dir: str, test_dir: str,
               split_size: float = CONFIG.train_split) -> None:
    """Randomly copy images from ``source_dir`` into train/test folders.

    Run once; afterwards the two folders define a fixed, repeatable split.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    items = os.listdir(source_dir)
    items = random.sample(items, len(items))
    cutoff = split_size * len(items)
    for i, item in enumerate(items):
        dest = train_dir if i < cutoff else test_dir
        copyfile(os.path.join(source_dir, item), os.path.join(dest, item))


def load_img_captions(coco_caps: COCO, img_id: int) -> list[str]:
    """All ground-truth captions for one image."""
    ann_ids = coco_caps.getAnnIds(imgIds=img_id)
    return [ann["caption"] for ann in coco_caps.loadAnns(ann_ids)]


def build_dataframe(coco_caps: COCO, source_dir: str, is_training: bool) -> pd.DataFrame:
    """Walk ``source_dir`` and collect (id, is_training, captions, url) rows.

    The captions of an image are concatenated into a single string so that the
    frame round-trips cleanly through CSV.
    """
    ids, trainings, all_caps, urls = [], [], [], []
    for img_url in os.listdir(source_dir):
        img_id = img_url_to_id(img_url)
        captions = load_img_captions(coco_caps, img_id)
        ids.append(img_id)
        trainings.append(is_training)
        all_caps.append(" ".join(captions))
        urls.append(img_url)

    return pd.DataFrame(
        {"id": ids, "is_training": trainings, "captions": all_caps, "url": urls}
    )


def read_dataframes(train_path: str = config.TRAIN_DF,
                    test_path: str = config.TEST_DF) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the previously serialised train/test dataframes."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    for df in (train_df, test_df):
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
    return train_df, test_df


if __name__ == "__main__":
    # One-off: build and serialise the dataframes.
    coco = load_coco()
    train = build_dataframe(coco, config.TRAIN_DIR, is_training=True)
    test = build_dataframe(coco, config.TEST_DIR, is_training=False)
    train.to_csv(config.TRAIN_DF)
    test.to_csv(config.TEST_DF)
    print(f"train rows: {len(train)}  test rows: {len(test)}")
