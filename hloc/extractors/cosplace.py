"""
Code to use CosPlace as a global features extractor.

CosPlace paper: https://arxiv.org/abs/2204.02287 (CVPR 2022)
"""

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class CosPlace(BaseModel):
    required_inputs = ["image"]

    def _init(self, conf):
        backbone = conf.get("backbone", "ResNet50")
        fc_output_dim = conf.get("fc_output_dim", 2048)
        self.net = torch.hub.load(
            "gmberton/cosplace",
            "get_trained_model",
            backbone=backbone,
            fc_output_dim=fc_output_dim,
        ).eval()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data["image"])
        desc = self.net(image)
        return {
            "global_descriptor": desc,
        }
