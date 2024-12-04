# -*- coding: utf-8 -*-
from pathlib import Path
from pprint import pprint
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json


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
        self.list_frame_features = []
        self.list_gtscores = []
        self.list_user_summary = []

        for video_name in self.split[self.mode + "_keys"]:
            frame_features = torch.Tensor(np.array(hdf[video_name + "/features"]))
            gtscore = torch.Tensor(np.array(hdf[video_name + "/gtscore"]))
            user_summary = torch.Tensor(np.array(hdf[video_name + "/user_summary"]))

            self.list_frame_features.append(frame_features)
            self.list_gtscores.append(gtscore)
            self.list_user_summary.append(user_summary)

        hdf.close()

    def __len__(self):
        """Function to be called for the `len` operator of `VideoData` Dataset."""
        self.len = len(self.split[self.mode + "_keys"])
        return self.len

    def __getitem__(self, index):
        """Function to be called for the index operator of `VideoData` Dataset.
        """
        video_name = self.split[self.mode + "_keys"][index]
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]
        user_summary = self.list_user_summary[index]

        if self.mode == "test":
            return frame_features, video_name, user_summary
        else:
            return frame_features, gtscore



if __name__ == "__main__":
    root_path = Path("data_source")

    dataset = VideoData(
        mode="test", root_path=root_path, dataset_name="SumMe", split_index=0
    )

    frame_features, gtscore, user_summary = dataset[2]

    print(frame_features.shape)
    print(gtscore)
    print(dataset.__len__())
