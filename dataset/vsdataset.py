# -*- coding: utf-8 -*-
from pathlib import Path
from pprint import pprint
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from lightning.pytorch import LightningDataModule


class VideoData(Dataset):
    def __init__(self, mode, root_path, dataset_name, split_index):
        """Custom Dataset class wrapper for loading the frame features and ground truth importance scores.
        Args:
            mode (str): The mode of the dataset, either 'train' or 'test'.
            root_path (str): The root path of the dataset.
            dataset_name (str): The name of the dataset.
            split_index (int): The index of the split to be loaded.
        """
        self.mode = mode
        self.root_path = root_path
        self.dataset_name = dataset_name
        self.split_index = split_index

        dataset_name_lower = dataset_name.lower()
        dataset_dir = Path(root_path, dataset_name)
        dataset_file = Path(
            dataset_dir, f"eccv16_dataset_{dataset_name_lower}_google_pool5.h5"
        )
        splits_filename = Path(dataset_dir, f"{dataset_name_lower}_splits.json")

        with open(splits_filename) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        hdf = h5py.File(dataset_file, "r")
        self.list_data = {}

        for video_name in self.split[self.mode + "_keys"]:
            frame_features = torch.Tensor(np.array(hdf[video_name + "/features"]))
            gtscore = torch.Tensor(np.array(hdf[video_name + "/gtscore"]))
            user_summary = torch.Tensor(np.array(hdf[video_name + "/user_summary"]))
            change_points = torch.Tensor(
                np.array(hdf.get(video_name + "/change_points"))
            )
            n_frames = int(hdf.get(video_name + "/n_frames")[()])
            positions = torch.Tensor(np.array(hdf.get(video_name + "/picks")))

            data_dict = {
                "frame_features": frame_features,
                "gtscore": gtscore,
                "user_summary": user_summary,
                "change_points": change_points,
                "n_frames": n_frames,
                "positions": positions,
                "video_name": video_name,
            }

            self.list_data[video_name] = data_dict

        hdf.close()

    def __len__(self):
        """Function to be called for the `len` operator of `VideoData` Dataset."""
        self.len = len(self.split[self.mode + "_keys"])
        return self.len

    def __getitem__(self, index):
        """Function to be called for the index operator of `VideoData` Dataset."""
        video_name = self.split[self.mode + "_keys"][index]
        return self.list_data[video_name]


class VideoSumDataModule(LightningDataModule):
    def __init__(self, root_path, dataset_name, split_index, batch_size):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.train_set = VideoData("train", root_path, dataset_name, split_index)
        self.val_set = VideoData("test", root_path, dataset_name, split_index)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
        )


if __name__ == "__main__":
    root_path = Path("data_source")

    dataset = VideoData(
        mode="train", root_path=root_path, dataset_name="SumMe", split_index=0
    )

    summe_module = VideoSumDataModule(
        root_path=root_path, dataset_name="SumMe", split_index=0, batch_size=1
    )

    tvsum_module = VideoSumDataModule(
        root_path=root_path, dataset_name="TVSum", split_index=0, batch_size=1
    )

    train_loader = tvsum_module.train_dataloader()
    val_loader = tvsum_module.val_dataloader()

    for batch in train_loader:
        pprint(batch)
        break
