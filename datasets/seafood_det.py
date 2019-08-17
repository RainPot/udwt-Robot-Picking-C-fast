import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import re
from datasets.transforms import *
import cv2


class SeafoodDET(Dataset):
    def __init__(self, root_dir, split, transforms=None):
        '''
        :param root_dir: root of annotations and image dirs
        :param transform: Optional transform to be applied
                on a sample.
        '''
        # get the csv
        self.images_dir = os.path.join(root_dir, split+'_data', 'images')
        self.annotations_dir = os.path.join(root_dir, split+'_data', 'annotations')
        mdf = os.listdir(self.images_dir)
        restr = r'\w+?(?=(.jpg))'
        for index, mm in enumerate(mdf):
            mdf[index] = re.match(restr, mm).group()
        self.mdf = mdf
        self.transforms = transforms

    def __len__(self):
        return len(self.mdf)

    def __getitem__(self, item):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        raise NotImplementedError
