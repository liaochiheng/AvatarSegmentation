import cv2, torch, argparse
from time import time
import numpy as np
from torch.nn import functional as F

from models import UNet
from dataloaders import transforms
from utils import utils

model_path = "pretrained/model_best_0704.pth"
simple_path = "pretrained/model_best_0704_simple.pth"

model = UNet(
    backbone="mobilenetv2",
    num_classes=2,
	pretrained_backbone=None
)
trained_dict = torch.load(model_path, map_location='cpu')
# state = {
# 			'arch': arch,
# 			'epoch': epoch,
# 			'logger': self.train_logger,
# 			'state_dict': self.model.state_dict(),
# 			'optimizer': self.optimizer.state_dict(),
# 			'monitor_best': self.monitor_best,
# }

torch.save(trained_dict['state_dict'], simple_path)



# model.load_state_dict(trained_dict, strict=True)
# model.eval()