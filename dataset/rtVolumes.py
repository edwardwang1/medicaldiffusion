from torch.utils.data import Dataset
import torchio as tio
import os
from typing import Optional
import argparse
import numpy as np


PREPROCESSING_TRANSORMS = tio.Compose([
    tio.RescaleIntensity(out_min_max=(-1, 1)),
    #tio.CropOrPad(target_shape=(256, 256, 32))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

#Todo: add transforms for preprocessing
class RTDataset(Dataset):
    def __init__(self, data_directory, patient_list_directory, val_fold, testing_holdout_fold, test=False, holdout=False):
        self.preprocessing = PREPROCESSING_TRANSORMS
        self.transforms = TRAIN_TRANSFORMS
        self.dataDirectory = data_directory
        self.test = test
        self.holdout = holdout
        self.trainIDs = []
        self.valIDs = []
        self.testIDs = []
        for i in range(5):
            filePath = os.path.join(patient_list_directory, "fold" + str(i) + ".txt")
            with open(filePath) as f:
                lines = f.read().splitlines()
            if i == val_fold:
                for l in lines:
                    self.valIDs.append(l)
            elif i == testing_holdout_fold:
                for l in lines:
                    self.testIDs.append(l)
            else: #train
                for l in lines:
                    self.trainIDs.append(l)

    def __len__(self):  # The length of the dataset is important for iterating through it
        if self.test:
            if self.holdout:
                return len(self.testIDs)
            else:
                return len(self.valIDs)
        else:
            return len(self.trainIDs)

    def __getitem__(self, idx):
        if self.test:
            if self.holdout:
                #print("test holdout")
                volumes = np.load(os.path.join(self.dataDirectory, self.testIDs[idx] + ".npy"))
            else:
                volumes = np.load(os.path.join(self.dataDirectory, self.valIDs[idx] + ".npy"))
        else:
            volumes = np.load(os.path.join(self.dataDirectory, self.trainIDs[idx] + ".npy"))

        return {'data': volumes[0][np.newaxis, ...].astype(np.float32)}

    def getValIDs(self):
        return self.valIDs

    def getTestIDs(self):
        return self.testIDs