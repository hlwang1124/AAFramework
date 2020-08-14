import torch
from .base_model import BaseModel
from . import networks
import numpy as np
import os


class AAUNetModel(BaseModel):
    def name(self):
        return 'AAUNetModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        # changing the default values
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def initialize(self, opt, dataset):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.input = opt.input
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['segmentation']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names = ['rgb_image', 'tdisp_image', 'label', 'output']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        self.model_names = ['AAUNet']

        # load/define networks
        self.netAAUNet = networks.define_AAUNet(dataset.num_labels, init_type=opt.init_type, init_gain= opt.init_gain, gpu_ids= self.gpu_ids)
        # define loss functions
        self.criterionSegmentation = networks.SegmantationLoss(class_weights=None).to(self.device)

        if self.isTrain:
            # initialize optimizers
            self.optimizers = []
            self.optimizer_AAUNet = torch.optim.SGD(self.netAAUNet.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            #self.optimizer_AAUNet = torch.optim.Adam(self.netAAUNet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer_AAUNet)
            self.set_requires_grad(self.netAAUNet, True)

    def set_input(self, input):
        self.rgb_image = input['rgb_image'].to(self.device)
        self.tdisp_image = input['tdisp_image'].to(self.device)
        self.label = input['label'].to(self.device)
        self.image_names = input['path']
        self.image_oriSize = input['oriSize']

    def forward(self):
        if self.input == 'rgb':
            self.output = self.netAAUNet(self.rgb_image)
        else:
            self.output = self.netAAUNet(self.tdisp_image)

    def get_loss(self):
        self.loss_segmentation = self.criterionSegmentation(self.output, self.label)

    def backward(self):
        self.loss_segmentation.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer_AAUNet.zero_grad()
        self.get_loss()
        self.backward()
        self.optimizer_AAUNet.step()
