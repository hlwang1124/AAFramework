import os.path
import torchvision.transforms as transforms
import torch
import cv2
import numpy as np
import glob
from data.base_dataset import BaseDataset


class potholedataset(BaseDataset):
    """dataloader for pothole-600 dataset"""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.root = opt.dataroot # path for the dataset
        self.num_labels = 2
        self.use_size = (opt.useSize, opt.useSize)

        if opt.phase == "train":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'training', 'rgb', '*.png')))
        elif opt.phase == "val":
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'validation', 'rgb', '*.png')))
        else:
            self.image_list = sorted(glob.glob(os.path.join(self.root, 'testing', 'rgb', '*.png')))

    def __getitem__(self, index):
        useDir = "/".join(self.image_list[index].split('/')[:-2])
        name = self.image_list[index].split('/')[-1]

        rgb_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'rgb', name)), cv2.COLOR_BGR2RGB)
        tdisp_image = cv2.cvtColor(cv2.imread(os.path.join(useDir, 'tdisp', name)), cv2.COLOR_BGR2RGB)
        label = cv2.imread(os.path.join(useDir, 'label', name), cv2.IMREAD_ANYDEPTH)
        label[label > 0] = 1
        oriHeight, oriWidth, _ = rgb_image.shape

        # resize image to enable sizes divide 16 for AA-UNet, and to enable divide 32 for AA-RTFNet
        rgb_image = cv2.resize(rgb_image, self.use_size)
        tdisp_image = cv2.resize(tdisp_image, self.use_size)
        label = cv2.resize(label, self.use_size, interpolation=cv2.INTER_NEAREST)

        rgb_image = rgb_image.astype(np.float32) / 255
        tdisp_image = tdisp_image.astype(np.float32) / 255
        rgb_image = transforms.ToTensor()(rgb_image)
        tdisp_image = transforms.ToTensor()(tdisp_image)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        # return a dictionary containing useful information
        # input rgb images, tdisp images and labels for training;
        # 'path': image name for saving predictions
        # 'oriSize': original image size for evaluating and saving predictions
        return {'rgb_image': rgb_image, 'tdisp_image': tdisp_image, 'label': label,
                'path': name, 'oriSize': (oriWidth, oriHeight)}

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'pothole-600'
